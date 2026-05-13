#!/bin/bash
#SBATCH --job-name=pinqi-mrf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=4-00:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100mig:4
#SBATCH --partition=equipment_typeG
#SBATCH --cpus-per-task=3
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

JOB_NAME="pinqi-trainings"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

echo "=== Job Setup ==="
echo "Host: $(hostname -s)"
echo "Script dir: ${SCRIPT_DIR}"

echo "=== nvidia-smi ==="
nvidia-smi

mkdir -p logs

ulimit -n 16384

# Triton cache in shared memory for speed
export TRITON_CACHE_DIR="/dev/shm/${USER}/${JOB_NAME}-triton"
export TORCHINDUCTOR_CACHE_DIR="/dev/shm/${USER}/${JOB_NAME}-torchinductor"
mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

# Python interpreter
if [[ -z "${PYTHON:-}" && -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python"
fi
if [[ -z "${PYTHON:-}" ]]; then
    echo "Error: PYTHON not set and .venv/bin/python not found"
    exit 1
fi
echo "Python: ${PYTHON}"

# Proxy (proxychains for outbound connections)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PROXYCHAINS_QUIET_MODE=1
export PROXYCHAINS_CONF_FILE="/scratch/${USER}/proxychains/proxychains.conf"
export LD_PRELOAD="/scratch/${USER}/proxychains/libproxychains.so.4"

# PyTorch / NCCL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_X_REQUIRE_LEGACY_SERVICE=true  # Rust core incompatible with proxy
export NCCL_SOCKET_IFNAME=en

# Run
CHECKPOINT_NAME="${JOB_NAME}-${SLURM_JOB_ID}"
echo "=== Starting Training ==="
echo "Checkpoint name: ${CHECKPOINT_NAME}"

srun --unbuffered "${PYTHON}" -u -m pinqi_mrf.cli fit \
    --trainer.num_nodes="${SLURM_JOB_NUM_NODES}" \
    --trainer.devices="${SLURM_NTASKS_PER_NODE}" \
    --checkpoint_name "${CHECKPOINT_NAME}" \
    "$@"
