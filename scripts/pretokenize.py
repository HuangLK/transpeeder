import json
import random
import multiprocessing
from pprint import pprint
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field

import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm

from transpeeder.feeder import (
    preprocess, PROMPT_FIELD, OUTPUT_FIELD, IGNORE_INDEX
)

def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


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


def load_samples(args, eos=''):
    samples = []
    data_path = Path(args.data_path)
    all_files = list(data_path.glob('**/*.json') if data_path.is_dir() else [data_path])

    for single_file in tqdm(all_files):
        with (single_file).open(encoding='utf-8') as f:
            for lnum, ln in enumerate(f):
                sample = json.loads(ln)
                prompt, output = sample[PROMPT_FIELD], sample[OUTPUT_FIELD]
                if not isinstance(prompt, str) or not isinstance(output, str):
                    raise ValueError()
                samples.append(dict(
                    prompt=prompt,
                    output=output + eos,
                ))

    print(f'total samples num: {len(samples)}')
    return samples


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.args.tokenizer_name_or_path,
            model_max_length=self.args.max_seq_len,
            padding_side="right",
            use_fast=True,
        )

    def batch_encode(self, batch):
        sources = [sample[PROMPT_FIELD] for sample in batch]
        targets = [sample[OUTPUT_FIELD] for sample in batch]

        data_dict = preprocess(sources, targets, Encoder.tokenizer, self.args.mode)
        input_ids = data_dict["input_ids"]
        labels = data_dict["labels"]

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        labels = torch.where(labels == Encoder.tokenizer.pad_token_id, IGNORE_INDEX, labels)

        return [
            dict(
                input_ids=iid,
                labels=lbl,
            ) for iid, lbl in zip(input_ids, labels)
        ]


def main():
    parser = transformers.HfArgumentParser((Arguments,))
    args, = parser.parse_args_into_dataclasses()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        model_max_length=args.max_seq_len,
        padding_side="right",
        use_fast=True,
    )

    samples = load_samples(args, tokenizer.eos_token)
    batches = _chunk(samples, args.batch_size)

    encoder = Encoder(args)
    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers, encoder.initializer)
        encoded_rlt = pool.imap(encoder.batch_encode, batches)
    else:
        encoder.initializer()
        encoded_rlt = (encoder.batch_encode(batch) for batch in batches)

    data = []
    for encoded_batch in tqdm(encoded_rlt, total=len(samples) // args.batch_size + 1):
        data.extend(encoded_batch)
    torch.save(data, args.output_path)


if __name__ == "__main__":
    main()
