
import time
import random
import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import transformers
import numpy as np
import deepspeed

from transpeeder.models.llama_pipeline_model import get_model
from transpeeder.models.patching import (
    replace_llama_attn_with_flash_attn,
    refine_rope,
)
from transpeeder.feeder import (
    make_prompt_dataloader,
    make_tokenized_dataloader,
)
from transpeeder.utils import jload
from transpeeder.utils import logger_rank0 as logger

warnings.filterwarnings("ignore")

@dataclass
class TrainerArguments:
    init_ckpt: str = field(default="llama-7B-init-test-ckpt")
    use_flash_attn: Optional[bool] = field(default=False)

    rank: int = field(default=None)
    local_rank: int = field(default=None)
    pipe_parallel_size: int = field(default=1)
    model_parallel_size: int = field(default=1)
    world_size: int = field(default=None)
    seed: int = field(default=42)
    deepspeed_config: Optional[str] = field(default=None)

    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    input_format: Literal['raw', 'tokenized'] = 'raw'
    mode: Literal['sft', 'pretrain', 'dialog'] = 'sft'
    num_workers: int = field(default=1)

    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./output")
    max_seq_len: int = field(default=128)
    train_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=100)
    log_steps: int = field(default=1)

    resume_step: int = field(default=-1)
    resume_ckpt: str = field(default="llama-7B-init-test-ckpt")
    ntk : Optional[bool] = field(default=False)

def read_ds_config(config_path):
    config = jload(config_path)
    return config


def main():
    parser = transformers.HfArgumentParser(TrainerArguments)
    args, = parser.parse_args_into_dataclasses()

    # setup deepspeed and other stuff
    deepspeed.init_distributed(dist_backend="nccl")
    args.world_size = torch.distributed.get_world_size()

    ds_config = read_ds_config(args.deepspeed_config)
    args.num_workers = 2 * args.world_size // args.pipe_parallel_size // args.model_parallel_size
    args.batch_size = ds_config.get("train_micro_batch_size_per_gpu", 1)
    activation_checkpointing_config = ds_config.pop("activation_checkpointing", None)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    if args.use_flash_attn:
        logger.info("⚡⚡⚡ enable flash attention.")
        replace_llama_attn_with_flash_attn()
        refine_rope()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.init_ckpt,
        model_max_length=args.max_seq_len,
        padding_side="right",
        use_fast=False,
    )
    model_config = transformers.AutoConfig.from_pretrained(args.init_ckpt)

    if args.ntk:
        rope_scaling = {
            "type": "dynamic",
            "factor": 2,
        }
        model_config.rope_scaling = rope_scaling
        logger.info(f"Turn on dynamic rope for llama2")
        
    # pipeline model
    model = get_model(model_config, args, activation_checkpointing_config, partition_method="type:ParallelTransformerLayerPipe")

    engine, _, _, _ = deepspeed.initialize(
        args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )

    # dataset
    dataloader_maker = make_tokenized_dataloader if args.input_format == 'tokenized' else make_prompt_dataloader
    train_dataloader = dataloader_maker(tokenizer=tokenizer, data_args=args, engine=engine)

    # use `convert2ckpt.py`
    if args.resume_step < 0:
        engine.load_checkpoint(args.init_ckpt,
                            load_module_only=True,
                            load_optimizer_states=False,
                            load_lr_scheduler_states=False,
        )
    else:
        engine.load_checkpoint(args.resume_ckpt)

    start = time.time()
    for step in range(1, args.train_steps + 1):
        if step <= args.resume_step:
            micro_batch_num = ds_config['train_batch_size'] // ds_config['train_micro_batch_size_per_gpu']
            [next(train_dataloader) for _ in range(micro_batch_num)]
            logger.info(f"Step={step:>6}, skipped.")
            continue

        loss = engine.train_batch(data_iter=train_dataloader)
        if args.local_rank == 0:
            if step % args.log_steps == 0:
                now = time.time()
                avg_time = (now-start) / args.log_steps
                logger.info(f"Step={step:>6}, loss={loss.item():.4f}, {avg_time:.2f} it/s")
                start = now

        if step % args.eval_steps == 0:
            # TODO
            pass

        if step % args.save_steps == 0:
            logger.info(f"Saving at step {step}")
            engine.save_checkpoint(args.output_dir)


if __name__ == "__main__":
    main()
