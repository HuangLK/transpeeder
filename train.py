
import time
import random
import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import transformers
import numpy as np
import deepspeed

from models.llama_pipeline_model import get_model
from models.patching import (
    replace_llama_attn_with_flash_attn,
)
from feeder import (
    make_prompt_dataloader,
)
from common.utils import jload
from common.log import logger_rank0 as logger

warnings.filterwarnings("ignore")

@dataclass
class ModelArguments:
    init_ckpt: str = field(default="llama-7B-init-test-ckpt")
    use_flash_attn: Optional[bool] = field(default=False)

@dataclass
class DeepspeedArguments:
    use_deepspeed: Optional[bool] = field(default=True)
    rank: int = field(default=None)
    local_rank: int = field(default=None)
    pipe_parallel_size: int = field(default=1)
    model_parallel_size: int = field(default=1)
    world_size: int = field(default=None)
    seed: int = field(default=42)
    deepspeed_config: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    mode: Literal['sft', 'pretrain'] = 'sft'
    num_workers: int = field(default=1)


@dataclass
class TrainerArguments:
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./output")
    max_seq_len: int = field(default=128)
    train_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=100)
    log_steps: int = field(default=1)


def read_ds_config(config_path):
    config = jload(config_path)
    return config


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainerArguments, DeepspeedArguments))
    model_args, data_args, trainer_args, ds_args = parser.parse_args_into_dataclasses()

    # setup deepspeed and other stuff
    assert ds_args.use_deepspeed
    deepspeed.init_distributed(dist_backend="nccl")
    ds_args.world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(ds_args.local_rank)

    ds_config = read_ds_config(ds_args.deepspeed_config)
    data_args.num_workers = 2 * ds_args.world_size // ds_args.pipe_parallel_size // ds_args.model_parallel_size
    data_args.batch_size = ds_config.get("train_micro_batch_size_per_gpu", 1)
    activation_checkpointing_config = ds_config.pop("activation_checkpointing", None)

    random.seed(ds_args.seed)
    np.random.seed(ds_args.seed)
    torch.manual_seed(ds_args.seed)
    deepspeed.runtime.utils.set_random_seed(ds_args.seed)

    if model_args.use_flash_attn:
        logger.info("⚡⚡⚡ enable flash attention.")
        replace_llama_attn_with_flash_attn()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.init_ckpt,
        model_max_length=trainer_args.max_seq_len,
        padding_side="right",
        use_fast=True,
    )
    model_config = transformers.AutoConfig.from_pretrained(model_args.init_ckpt)

    # dataset
    train_dataloader = make_prompt_dataloader(tokenizer=tokenizer, data_args=data_args)
    # pipeline model
    model = get_model(model_config, ds_args, activation_checkpointing_config)

    engine, _, _, _ = deepspeed.initialize(
        ds_args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad]
    )

    # use `convert2ckpt.py`
    engine.load_checkpoint(model_args.init_ckpt, load_module_only=True)

    start = time.time()
    for step in range(1, trainer_args.train_steps + 1):
        loss = engine.train_batch(data_iter=train_dataloader)
        if ds_args.local_rank == 0:
            if step % trainer_args.log_steps == 0:
                now = time.time()
                avg_time = (now-start) / trainer_args.log_steps
                logger.info(f"Step={step:>6}, loss={loss.item():.4f}, {avg_time:.2f} it/s")
                start = now

        if step % trainer_args.eval_steps == 0:
            # TODO
            pass

        if step % trainer_args.save_steps == 0:
            logger.info(f"Saving at step {step}")
            engine.save_checkpoint(trainer_args.output_dir)


if __name__ == "__main__":
    main()
