
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
    smart_tokenizer_and_embedding_resize,
    replace_llama_attn_with_flash_attn,
)
from feeder import (
    make_prompt_dataloader,
    DEFAULT_BOS_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_UNK_TOKEN,
)
from utils import jload
from utils import logger_rank0 as logger

warnings.filterwarnings("ignore")

@dataclass
class ModelArguments:
    tokenizer_name_or_path: Optional[str] = field(default='')
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
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


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path or model_args.model_name_or_path,
        model_max_length=trainer_args.max_seq_len,
        padding_side="right",
        use_fast=False,
    )

    if model_args.use_flash_attn:
        logger.info("⚡⚡⚡ enable flash attention.")
        replace_llama_attn_with_flash_attn()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    # dataset
    train_dataloader = make_prompt_dataloader(tokenizer=tokenizer, data_args=data_args)
    # pipeline model
    model = get_model(model, ds_args, activation_checkpointing_config)

    engine, _, _, _ = deepspeed.initialize(
        ds_args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad]
    )

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
