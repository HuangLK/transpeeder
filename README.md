# transpeeder
This is a project under development, which aims to fine-tune the llama (7-70B) model based on the 🤗transformers and 🚀deepspeed, and provide simple and convenient training scripts.

## installation
```
pip install -e .
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
python -m scripts.convert2ckpt --mp_world_size 4 \
    --model_name_or_path /path/to/llama-7b-hf \
    --output_dir /path/to/llama-7b-init-ckpt

# llama-30B
python -m scripts.convert2ckpt --mp_world_size 8 \
    --model_name_or_path /path/to/llama-30b-hf \
    --output_dir /path/to/llama-30b-init-ckpt
```

## finetune
See `examples/train_llama_deepspeed.sh`.


## convert ckpt to hf model
```bash
python -m scripts.convert2hf --model_size 7B \
    --input_dir ./output/llama-7B-ckpt/global_step1000/ \
    --output_dir ./output/llama_hf_7B \
    --tokenizer_size 32001
cp /path/to/llama-7b-hf/*.json ./output/llama_hf_7B
cp /path/to/llama-7b-hf/tokenizer.model ./output/llama_hf_7B
```
