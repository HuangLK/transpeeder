# llama-deepspeed


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
