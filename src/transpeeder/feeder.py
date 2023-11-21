""" feader.py """

import copy
import json
from pathlib import Path
from functools import cache
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List
from collections import defaultdict

import torch
import deepspeed
import transformers
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split

from .utils import is_rank_0
from .utils import logger_rank0 as logger


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PREFIX_BEGIN_TOKEN = "<|prefix_begin|>"
PREFIX_END_TOKEN   = "<|prefix_end|>"
PROMPTER_TOKEN     = "<|prompter|>"
ASSISTANT_TOKEN    = "<|assistant|>"
ENDOFTEXT_TOKEN    = "<|endoftext|>"

PROMPT_FIELD = 'prompt'
OUTPUT_FIELD = 'output'


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    batch_tokenized = tokenizer(
        strings,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    input_ids = labels = batch_tokenized
    input_ids_lens = labels_lens = [
        tokenized.ne(tokenizer.pad_token_id).sum().item() for tokenized in batch_tokenized
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _make_labels(input_ids, tokenizer: transformers.PreTrainedTokenizer, mode: str = "sft", **kwargs):
    if mode == "sft":
        assert "source_lens" in kwargs, f"miss parameter: source_lens"
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, kwargs["source_lens"]):
            label[: source_len] = IGNORE_INDEX
        return labels
    elif mode == "pretrain":
        return copy.deepcopy(input_ids)
    elif mode == "dialog":
        labels = torch.full_like(input_ids, IGNORE_INDEX, dtype=input_ids.dtype)
        # <|assistant|> ... <|endoftext|>
        ASSISTANT_TOKEN_ID = tokenizer.convert_tokens_to_ids(ASSISTANT_TOKEN)
        ENDOFTEXT_TOKEN_ID = tokenizer.convert_tokens_to_ids(ENDOFTEXT_TOKEN)
        PROMPTER_TOKEN_ID = tokenizer.convert_tokens_to_ids(PROMPTER_TOKEN)
        for input_row, label_row in zip(input_ids, labels):
            begin_indices = torch.nonzero(input_row == ASSISTANT_TOKEN_ID)
            for idx in begin_indices:
                edi = idx + 1
                while edi < len(input_row) and input_row[edi] != ENDOFTEXT_TOKEN_ID:
                    edi += 1
                if edi < len(input_row) and \
                        input_row[edi + 1] != PROMPTER_TOKEN_ID:
                    logger.warning(f'expect {PROMPTER_TOKEN} after {ENDOFTEXT_TOKEN}, get {input_row[edi + 1]}.')
                label_row[idx + 1: edi + 1] = input_row[idx + 1: edi + 1]

        return labels
    else:
        raise ValueError('Unvalid training mode.')


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    mode: str
) -> Dict:
    """Preprocess the data by tokenizing."""
    samples = [s + t for s, t in zip(sources, targets)]
    samples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (samples, sources)]
    input_ids = samples_tokenized["input_ids"]
    labels = _make_labels(input_ids, tokenizer, mode,
                          source_lens=sources_tokenized["input_ids_lens"])

    # shift
    return dict(
        input_ids=[ids[: -1] for ids in input_ids],
        labels=[lbs[1: ]for lbs in labels]
    )


class PromptDataset(Dataset):
    """ Dataset for prompt-tuning. """

    def __init__(self, data_path: Union[str, Path], eos: str = ""):
        super().__init__()
        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert data_path.exists(), f'{data_path} does not exists.'

        self.samples = []
        all_files = list(data_path.glob('**/*.json') if data_path.is_dir() else [data_path])

        error_count = defaultdict(int)
        ERROR_THRESHOLD = 10
        for single_file in tqdm(all_files, disable=not is_rank_0()):
            with (single_file).open(encoding='utf-8') as f:
                for lnum, ln in enumerate(f):
                    try:
                        sample = json.loads(ln)
                        prompt, output = sample[PROMPT_FIELD], sample[OUTPUT_FIELD]
                        if not isinstance(prompt, str) or not isinstance(output, str):
                            raise ValueError()
                        self.samples.append(dict(
                            prompt=prompt,
                            output=output + eos,
                        ))
                    except:
                        logger.warning(f'{single_file}: {lnum} unvalid.')
                        error_count[str(single_file)] += 1

                    if error_count[str(single_file)] > ERROR_THRESHOLD:
                        logger.warning(f'{single_file} exceeds max error number. skipped.')
                        break

        logger.info(f'total samples num: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, str]:
        # TODO: preprocess here and caching on the fly.
        return self.samples[index]


@dataclass
class DataCollatorForPromptDataset(object):
    """Collate for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    mode: str

    @cache
    @staticmethod
    def get_attn_mask(bs, seq_length):
        """
        Get triangular attention mask.
        """
        # lower triangular attention mask
        mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
            bs, 1, seq_length, seq_length
        )
        # convert to binary
        return mask < 0.5

    @staticmethod
    def get_position_ids(input_ids):
        seq_length = input_ids.shape[1]
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = [sample[PROMPT_FIELD] for sample in samples]
        targets = [sample[OUTPUT_FIELD] for sample in samples]

        data_dict = preprocess(sources, targets, self.tokenizer, self.mode)
        input_ids = data_dict["input_ids"]
        labels = data_dict["labels"]

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        labels = torch.where(labels == self.tokenizer.pad_token_id, IGNORE_INDEX, labels)

        return (
            (
                input_ids,
                DataCollatorForPromptDataset.get_position_ids(input_ids),
                DataCollatorForPromptDataset.get_attn_mask(input_ids.shape[0], input_ids.shape[1]),
            ),
            labels
        )


class TokenizedDataset(Dataset):
    def __init__(self, data_path: Union[str, Path]):
        super().__init__()
        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert data_path.exists(), f'{data_path} does not exists.'

        self.samples = []
        all_files = list(data_path.glob('**/*.pt') if data_path.is_dir() else [data_path])

        for single_file in tqdm(all_files, disable=not is_rank_0()):
            self.samples.extend(torch.load(single_file))

        logger.info(f'total samples num: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, str]:
        return self.samples[index]


@dataclass
class DataCollatorForTokenizedDataset(DataCollatorForPromptDataset):

    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([s['input_ids'] for s in samples])
        labels = torch.stack([s['labels'] for s in samples])
        return (
            (
                input_ids,
                self.get_position_ids(input_ids.shape[0], input_ids.shape[1]),
                self.get_attn_mask(input_ids),
            ),
            labels
        )


def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split, random_state=42, shuffle=True
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def make_prompt_dataloader(tokenizer: transformers.PreTrainedTokenizer, data_args, engine, val_split=None) -> Dict:
    # TODO add eval dataloader
    assert val_split is None
    dataset = PromptDataset(data_path=data_args.data_path, eos=tokenizer.eos_token)
    data_collator = DataCollatorForPromptDataset(tokenizer=tokenizer, mode=data_args.mode)
    g = torch.Generator()
    train_sampler = DistributedSampler(dataset,
                    num_replicas=engine.dp_world_size,
                    rank=engine.mpu.get_data_parallel_rank(),
                    shuffle=True)
    dataloader = DataLoader(dataset,
                            collate_fn=data_collator,
                            num_workers=data_args.num_workers,
                            batch_size=data_args.batch_size,
                            sampler=train_sampler,
                            drop_last=True,
                            generator=g,)
    return iter(deepspeed.utils.RepeatingLoader(dataloader))


def make_tokenized_dataloader(tokenizer: transformers.PreTrainedTokenizer, data_args, engine, val_split=None) -> Dict:
    dataset = TokenizedDataset(data_path=data_args.data_path)
    data_collator = DataCollatorForTokenizedDataset(tokenizer=tokenizer, mode=data_args.mode)
    g = torch.Generator()
    train_sampler = DistributedSampler(dataset,
                    num_replicas=engine.dp_world_size,
                    rank=engine.mpu.get_data_parallel_rank(),
                    shuffle=True)
    dataloader = DataLoader(dataset,
                            collate_fn=data_collator,
                            num_workers=data_args.num_workers,
                            batch_size=data_args.batch_size,
                            sampler=train_sampler,
                            drop_last=True,
                            generator=g,)
    return iter(deepspeed.utils.RepeatingLoader(dataloader))
