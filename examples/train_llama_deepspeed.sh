#!/bin/bash
# Train script.
set -eux

name=$1
gpus=$2

export MASTER_PORT=23857
export WORK_DIR=`pwd`

#ds_report

OUTPUT=${WORK_DIR}/output/${name}
if [ -d $OUTPUT ]; then
    # rm
    echo "${OUTPUT} exist."
else
    mkdir -p ${OUTPUT}
fi

echo "conda env: $CONDA_PREFIX"
deepspeed --include localhost:$2  --master_port ${MASTER_PORT} ${WORK_DIR}/train.py \
    --output_dir ${OUTPUT} \
    --init_ckpt /path/to/models/llama-30b-init-ckpt/ \
    --data_path /path/to/alpaca_en_zh_oneline_format.json \
    --max_seq_len 8192 \
    --train_steps 1000 \
    --eval_steps 10 \
    --save_steps 200 \
    --log_steps 1 \
    --pipe_parallel_size 8 \
    --model_parallel_size 1 \
    --use_flash_attn true \
    --ntk true \
    --deepspeed_config ${WORK_DIR}/../configs/ds_config_zero1.json
