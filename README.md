# llama-deepspeed
This is a project under development, which aims to fine-tune the llama (7-65B) model based on the 🤗transformers and 🚀deepspeed, and provide simple and convenient training scripts.

## requirement
```
pip install -r requirements.txt
```
I have made some minor modifications to `deepspeed` and my [PR](https://github.com/microsoft/DeepSpeed/pull/3064) has not been accepted yet, so if there is any problem you can consider using my own [dev branch](https://github.com/HuangLK/DeepSpeed/tree/dev).

## finetune
llama-7B
```bash
deepspeed --include localhost:0,1,2,3  --master_port 22384 train.py \
    --output_dir /path/to/output \
    --model_name_or_path /path/to/llama-7b-hf \
    --data_path ./data/alpaca_data_sample_oneline_format.json \
    --max_seq_len 1024 \
    --train_steps 1000 \
    --eval_steps 10 \
    --save_steps 200 \
    --log_steps 1 \
    --pipe_parallel_size 4 \
    --model_parallel_size 1 \
    --deepspeed_config ./configs/ds_config.json
```

llama-30B
```bash
deepspeed --master_port 22384 train.py \
    --output_dir /path/to/output \
    --model_name_or_path /path/to/llama-30b-hf \
    --data_path ./data/alpaca_data_sample_oneline_format.json \
    --max_seq_len 1024 \
    --train_steps 1000 \
    --eval_steps 10 \
    --save_steps 200 \
    --log_steps 1 \
    --pipe_parallel_size 8 \
    --model_parallel_size 1 \
    --deepspeed_config ./configs/ds_config_zero1.json
```


## todo
* [X] checkpoint activations
* [ ] add wandb
* [ ] add eval stage
* [ ] add flash-attn
