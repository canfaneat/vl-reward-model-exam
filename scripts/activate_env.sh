#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fast path: reuse the already validated environment. To switch to a dedicated
# environment later, run:
#   export REWARD_ENV_PREFIX=/root/private_data/envs/reward-model-exam
ENV_PREFIX="${REWARD_ENV_PREFIX:-/root/private_data/envs/sai-text-slot}"

mkdir -p \
  /root/private_data/cache/huggingface \
  /root/private_data/cache/huggingface/transformers \
  /root/private_data/cache/huggingface/datasets \
  /root/private_data/cache/torch \
  /root/private_data/cache/pip \
  /root/private_data/models/reward-model-exam \
  /root/private_data/datasets/reward-model-exam \
  "$PROJECT_ROOT/outputs/logs" \
  "$PROJECT_ROOT/outputs/eval" \
  "$PROJECT_ROOT/outputs/checkpoints" \
  "$PROJECT_ROOT/artifacts/screenshots"

if [ -f /public/home/yuqiangwang/proxy.sh ]; then
  # shellcheck disable=SC1091
  source /public/home/yuqiangwang/proxy.sh
  export HTTP_PROXY="${http_proxy:-}"
  export HTTPS_PROXY="${https_proxy:-}"
  export FTP_PROXY="${ftp_proxy:-}"
fi

export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"

export HF_HOME="/root/private_data/cache/huggingface"
export HF_HUB_CACHE="/root/private_data/cache/huggingface/hub"
export HF_DATASETS_CACHE="/root/private_data/cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/root/private_data/cache/huggingface/transformers"
export TORCH_HOME="/root/private_data/cache/torch"
export PIP_CACHE_DIR="/root/private_data/cache/pip"

export REWARD_PROJECT_ROOT="$PROJECT_ROOT"
export REWARD_MODELS_DIR="/root/private_data/models/reward-model-exam"
export REWARD_DATASETS_DIR="/root/private_data/datasets/reward-model-exam"
export REWARD_OUTPUTS_DIR="$PROJECT_ROOT/outputs"

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
fi

if [ -d "$ENV_PREFIX" ]; then
  conda activate "$ENV_PREFIX"
else
  echo "Reward env does not exist yet: $ENV_PREFIX"
  echo "Set REWARD_ENV_PREFIX to an existing env, or create it first."
fi

cd "$PROJECT_ROOT"

