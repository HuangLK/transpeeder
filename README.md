# llama-deepspeed
This is a project under development, which aims to fine-tune the llama (7-65B) model based on the 🤗transformers and 🚀deepspeed, and provide simple and convenient training scripts.

## requirement
```
pip install -r requirements.txt
```

## data
Each line is a **JSON string**, as the JSON object must have `prompt` and `output` fields.
```
{
    "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:",
    "output": "The capital of France is Paris."
}
```

## convert hf model to ckpt
```bash
# llama-7B
python convert2ckpt.py --mp_world_size 4 \
    --model_name_or_path /path/to/llama-7b-hf \
    --output_dir /path/to/llama-7b-init-ckpt

# llama-30B
python convert2ckpt.py --mp_world_size 8 \
    --model_name_or_path /path/to/llama-30b-hf \
    --output_dir /path/to/llama-30b-init-ckpt
```

## finetune
llama-7B
```bash
deepspeed --include localhost:0,1,2,3  --master_port 22384 train.py \
    --output_dir /path/to/output \
    --init_ckpt /path/to/llama-7b-init-ckpt/ \
    --data_path ./data/alpaca_data_sample_oneline_format.json \
    --max_seq_len 1024 \
    --train_steps 1000 \
    --eval_steps 10 \
    --save_steps 200 \
    --log_steps 1 \
    --pipe_parallel_size 4 \
    --model_parallel_size 1 \
    --use_flash_attn true \
    --deepspeed_config ./configs/ds_config.json
```

llama-30B
```bash
deepspeed --master_port 22384 train.py \
    --output_dir /path/to/output \
    --init_ckpt /path/to/llama-30b-init-ckpt/ \
    --data_path ./data/alpaca_data_sample_oneline_format.json \
    --max_seq_len 1024 \
    --train_steps 1000 \
    --eval_steps 10 \
    --save_steps 200 \
    --log_steps 1 \
    --pipe_parallel_size 8 \
    --model_parallel_size 1 \
    --use_flash_attn true \
    --deepspeed_config ./configs/ds_config_zero1.json
```

## convert ckpt to hf model
```bash
python convert2hf.py --model_size 7B \
    --input_dir ./output/llama-7B-ckpt/global_step1000/ \
    --output_dir ./output/llama_hf_7B
cp /path/to/llama-7b-hf/*.json ./output/llama_hf_7B
cp /path/to/llama-7b-hf/tokenizer.model ./output/llama_hf_7B
```
