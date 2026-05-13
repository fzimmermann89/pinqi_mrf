#!/bin/bash

JOB_NAME="pinqi-training"

HOST="$(hostname -s)"
export TRITON_CACHE_DIR=/dev/shm/${USER}/${JOB_NAME}-triton
export TORCHINDUCTOR_CACHE_DIR="/dev/shm/${USER}/${JOB_NAME}-torchinductor"
mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

mkdir -p logs

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ -z "${PYTHON}" && -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python"
fi
if [[ -z "${PYTHON}" ]]; then
    echo "Error: PYTHON not set and ${SCRIPT_DIR}/.venv/bin/python not found"
    exit 1
fi

# Proxy setup
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
export PROXYCHAINS_QUIET_MODE=1
export PROXYCHAINS_CONF_FILE="/scratch/${USER}/proxychains/proxychains.conf"
export LD_PRELOAD="/scratch/${USER}/proxychains/libproxychains.so.4"

# PyTorch/CUDA config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_X_REQUIRE_LEGACY_SERVICE=true  # Rust core does not work with proxy
export NCCL_SOCKET_IFNAME=en

ulimit -n 16384

echo "Starting training..."
"${PYTHON}" -u -m pinqi_mrf.cli fit "$@"
