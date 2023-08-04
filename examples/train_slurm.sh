#!/bin/bash
# Usage: srun train_slurm.sh

set -eux

# 任务名
name=llama-30B-slurm-test

export MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
export MASTER_PORT=23857
export OMP_NUM_THREADS=8
#export CUDA_LAUNCH_BLOCKING=1
export WORK_DIR=`pwd`

# 日志路径
LOG_PATH=./logs/slurm_log_$(date '+%m%d%H%M').txt
GPUS_PER_NODE=8
# 节点数
NNODES=4
# 总gpu数
N_GPUS=32

# testing for potential faulty nodes
# srun --jobid $SLURM_JOB_ID bash -c 'python -c "import torch, socket; print(socket.gethostname(), torch.cuda.is_available())"'
# exit 0

# 模型保存路径
OUTPUT=${WORK_DIR}/output/${name}
if [ -d $OUTPUT ]; then
    # rm
    echo "${OUTPUT} exist."
else
    mkdir -p ${OUTPUT}
fi

echo "conda env: $CONDA_PREFIX"

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 0 \
    --tee 3 \
    "

# 训练任务
export CMD=" \
    ${WORK_DIR}/train.py \
    --output_dir ${OUTPUT} \
    --init_ckpt /path/to/llama-30b-init-ckpt/ \
    --data_path /path/to/alpaca_en_zh_oneline_format.json \
    --max_seq_len 8192 \
    --train_steps 1000 \
    --eval_steps 10 \
    --save_steps 200 \
    --log_steps 1 \
    --pipe_parallel_size 8 \
    --model_parallel_size 1 \
    --use_flash_attn true \
    --deepspeed_config ${WORK_DIR}/../configs/ds_config_zero1.json
    "

# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

echo "START TIME: $(date)"

bash -c "$LAUNCHER --node_rank $SLURM_PROCID $CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"
