#!/bin/bash

# Parameters
#SBATCH --account=your_slurm_cluster_account # Fill in your slurm cluster account
#SBATCH --exclusive
#SBATCH --job-name=your_job_name             # Fill in your job name
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/absolute/path/to/your/output/file/convert_DS_slurm_log/sbatch_general_sa-sa.convert_DS_hf_to_bf16_%j.out
#SBATCH --partition=your_slurm_cluster_partition # Fill in your slurm cluster partition
#SBATCH --time=01:00:00

set -evx

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export TORCHX_MAX_RETRIES=0

set +e

# setup

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NEMO_HOME=/nemo_practice/NEMO_HOME


# Command 1
run_cmd="cd /DeepSeek-V3/inference; python fp8_cast_bf16.py --input-fp8-hf-path /DeepSeek-V3-Base --output-bf16-hf-path /DeepSeek-V3-Base-BF16"

srun --output /absolute/path/to/your/output/file/convert_DS_slurm_log/log-sbatch_general_sa-sa.convert_DS_hf_to_bf16_%j_${SLURM_RESTART_COUNT:-0}.out \
     --container-image nvcr.io/nvidia/nemo:25.02.01 \
     --container-mounts /absolute/path/of/DeepSeek/github/file/position/DeepSeek-V3:/DeepSeek-V3,/absolute/path/of/HuggingFace/ckpt/DeepSeek-V3-Base:/DeepSeek-V3-Base,/absolute/path/of/converted/ckpt/DeepSeek-V3-Base-BF16:/DeepSeek-V3-Base-BF16 \
     bash -c "${run_cmd}"

exitcode=$?

set -e

echo "job exited with code $exitcode"
if [ $exitcode -ne 0 ]; then
    if [ "$TORCHX_MAX_RETRIES" -gt "${SLURM_RESTART_COUNT:-0}" ]; then
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit $exitcode
fi
