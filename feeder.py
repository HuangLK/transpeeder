""" feader.py """

import copy
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, Union
from collections import defaultdict

import torch
import deepspeed
import transformers
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split

import utils
from utils import is_rank_0
from utils import logger_rank0 as logger


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

PROMPT_FIELD = 'prompt'
OUTPUT_FIELD = 'output'


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    # TODO: batch encode
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            #padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


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
    # FIXME: sentencepiece case
    if mode == "sft":
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
    elif mode == "pretrain":
        labels = copy.deepcopy(input_ids)
    else:
        raise ValueError('Unvalid training mode.')

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

    def get_attn_mask(self, input_ids):
        """
        Get triangular attention mask for a given sequence length / device.
        """
        bs = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        # lower triangular attention mask
        mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
            bs, 1, seq_length, seq_length
        )
        # convert to binary
        return mask < 0.5

    def get_position_ids(self, input_ids):
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
                self.get_position_ids(input_ids),
                self.get_attn_mask(input_ids),
            ),
            labels
        )


def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split, random_state=42, shuffle=True
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def make_prompt_dataloader(tokenizer: transformers.PreTrainedTokenizer, data_args, val_split=None) -> Dict:
    # TODO add eval dataloader
    assert val_split is None
    dataset = PromptDataset(data_path=data_args.data_path, eos=tokenizer.eos_token)
    data_collator = DataCollatorForPromptDataset(tokenizer=tokenizer, mode=data_args.mode)
    g = torch.Generator()

    dataloader = DataLoader(dataset,
                            collate_fn=data_collator,
                            num_workers=data_args.num_workers,
                            batch_size=data_args.batch_size,
                            shuffle=True,
                            generator=g,)
    return iter(deepspeed.utils.RepeatingLoader(dataloader))
