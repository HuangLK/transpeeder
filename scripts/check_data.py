import json
import multiprocessing
from pprint import pprint, pformat
from pathlib import Path
from typing import Optional, Literal
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import transformers
import pandas as pd
from tqdm import tqdm
from loguru import logger

from .pretokenize import load_samples, Encoder, _chunk
from .data_utils import LinguaLid

logger.add('./check_report.log')

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PREFIX_BEGIN_TOKEN = "<|prefix_begin|>"
PREFIX_END_TOKEN   = "<|prefix_end|>"
PROMPTER_TOKEN     = "<|prompter|>"
ASSISTANT_TOKEN    = "<|assistant|>"
ENDOFTEXT_TOKEN    = "<|endoftext|>"


@dataclass
class Arguments:
    seed: int = field(default=42)
    tokenizer_name_or_path: str = field(default="/path/to/llama-7b-hf")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    output_path: str = field(default="/path/to/output.pt")
    mode: Literal['sft', 'pretrain', 'dialog'] = 'sft'
    max_seq_len: int = field(default=8192)
    batch_size: int = field(default=16)
    workers: int = field(default=64)


class Summarizer:
    def __init__(self):
        self.lines = defaultdict(list)

    def update(self, info):
        for lnum, msg in info.items():
            self.lines[lnum].append(msg)

    def summary(self):
        msgs = []
        for li in range(999999):
            if li not in self.lines:
                continue
            msg = "; ".join(self.lines[li])
            msgs.append(f'line {li}: {msg}')

        return '\n'.join(msgs)


def _count_tokens(input_ids):
    return [len(x) for x in input_ids]


def _check_oasst_format(tokenizeds, tokenizer):
    BOS_TOKEN_ID = tokenizer.convert_tokens_to_ids(DEFAULT_BOS_TOKEN)
    EOS_TOKEN_ID = tokenizer.convert_tokens_to_ids(DEFAULT_EOS_TOKEN)
    PREFIX_BEGIN_TOKEN_ID = tokenizer.convert_tokens_to_ids(PREFIX_BEGIN_TOKEN)
    PREFIX_END_TOKEN_ID = tokenizer.convert_tokens_to_ids(PREFIX_END_TOKEN)
    PROMPTER_TOKEN_ID = tokenizer.convert_tokens_to_ids(PROMPTER_TOKEN)
    ASSISTANT_TOKEN_ID = tokenizer.convert_tokens_to_ids(ASSISTANT_TOKEN)
    ENDOFTEXT_TOKEN_ID = tokenizer.convert_tokens_to_ids(ENDOFTEXT_TOKEN)

    special_tokens = (
        BOS_TOKEN_ID, EOS_TOKEN_ID, PREFIX_BEGIN_TOKEN_ID, PREFIX_END_TOKEN_ID,
        PROMPTER_TOKEN_ID, ASSISTANT_TOKEN_ID, ENDOFTEXT_TOKEN_ID,
    )
    def _find_next_special_token(input_ids, st):
        while st < len(input_ids) - 1 and input_ids[st] not in special_tokens:
            st += 1
        if input_ids[st] in special_tokens:
            return st
        else:
            return -1

    msgs = {}
    for lnum, item in tqdm(enumerate(tokenizeds), desc='checking oasst format'):
        input_ids, labels = item['input_ids'], item['labels']
        if input_ids[0] != BOS_TOKEN_ID:
            msgs[lnum + 1] = 'bos error'
            continue
        if input_ids[1] != PREFIX_BEGIN_TOKEN_ID:
            msgs[lnum + 1] = 'prefix begin error'
            continue
        tidx = _find_next_special_token(input_ids, 2)
        if tidx < 0 or input_ids[tidx] != PREFIX_END_TOKEN_ID:
            msgs[lnum + 1] = 'prefix end error'
            continue
        # 检查多轮对话
        for _ in range(100):
            tidx += 1
            if tidx < 0 or input_ids[tidx] != PROMPTER_TOKEN_ID:
                msgs[lnum + 1] = 'prompter error'
                break
            tidx += 1
            tidx = _find_next_special_token(input_ids, tidx)
            if tidx < 0 or input_ids[tidx] != ENDOFTEXT_TOKEN_ID:
                msgs[lnum + 1] = 'endoftext error'
                break
            tidx += 1
            if input_ids[tidx] != ASSISTANT_TOKEN_ID:
                msgs[lnum + 1] = 'assistant error'
                break
            tidx += 1
            tidx = _find_next_special_token(input_ids, tidx)
            if tidx > 0 and input_ids[tidx] not in (ENDOFTEXT_TOKEN_ID, EOS_TOKEN_ID,):
                msgs[lnum + 1] = 'endoftext error'
                break
            if tidx < 0 or tidx >= len(input_ids) - 1 or input_ids[tidx] == EOS_TOKEN_ID:
                break

    return msgs


def _check_language(samples):
    allowed_langs = ('en', 'zh',)
    lid = LinguaLid()
    lang_lst = []
    for s in tqdm(samples, desc="checking language"):
        text = s['output']
        lang = lid.detect(text)
        lang_lst.append(lang)

    df = pd.DataFrame({'lang_stat': lang_lst})
    logger.info(f'lang stat:\n' + pformat(df.value_counts()))
    unvalid = {lnum + 1: f'unallowed language [{lang}] detected.' \
               for lnum, lang in enumerate(lang_lst) if lang not in allowed_langs}
    return unvalid


def _check_length():
    pass


def _check_encoding():
    pass


def _tokenize(samples, args):
    batches = _chunk(samples, args.batch_size)
    encoder = Encoder(args)
    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers, encoder.initializer)
        encoded_rlt = pool.imap(encoder.batch_encode, batches)
    else:
        encoder.initializer()
        encoded_rlt = (encoder.batch_encode(batch) for batch in batches)

    data = []
    for encoded_batch in tqdm(encoded_rlt, total=len(samples) // args.batch_size + 1, desc='tokenizing'):
        data.extend(encoded_batch)
    return data


def main():
    parser = transformers.HfArgumentParser((Arguments,))
    args, = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        model_max_length=args.max_seq_len,
        padding_side="right",
        use_fast=True,
    )

    samples = load_samples(args, tokenizer.eos_token)
    tokenizeds = _tokenize(samples, args)

    summarizer = Summarizer()
    summarizer.update(_check_language(samples))
    summarizer.update(_check_oasst_format(tokenizeds, tokenizer))

    logger.info(f"summary: \n" + summarizer.summary())



    token_nums = []
    df = pd.DataFrame({'token_num': token_nums})
    desc = df.describe(
        percentiles=[.5, .75, .85, .90, .95],
    )
    pprint(desc)


if __name__ == "__main__":
    main()
