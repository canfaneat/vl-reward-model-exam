#!/usr/bin/env bash
set -euo pipefail

cd "/root/private_data/projects/reward model"

MLP_SUMMARY="outputs/eval/A_MLPHead_PromptCap20NoBench_4k_vlrb_full_summary.json"
OUT_DIR="outputs/checkpoints/D_PromptCap20NoBench_4k_Linear"
TRAIN_LOG="outputs/logs/train_D_PromptCap20NoBench_4k_Linear.jsonl"
TMUX_TRAIN_LOG="outputs/logs/tmux_train_D_PromptCap20NoBench_4k_Linear.log"
TMUX_EVAL_LOG="outputs/logs/tmux_eval_D_PromptCap20NoBench_4k_Linear_full.log"
EVAL_JSONL="outputs/eval/D_PromptCap20NoBench_4k_Linear_vlrb_full.jsonl"
EVAL_SUMMARY="outputs/eval/D_PromptCap20NoBench_4k_Linear_vlrb_full_summary.json"

while [[ ! -s "${MLP_SUMMARY}" ]]; do
  echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') waiting for ${MLP_SUMMARY}"
  sleep 120
done

source scripts/activate_env.sh
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

python -u scripts/train_reward_head.py \
  --include-indices artifacts/report_assets/rlaifv_promptcap20_nobench_4k.indices.txt \
  --limit 4096 \
  --shard-limit 2 \
  --epochs 1 \
  --max-tiles 2 \
  --lr 1e-4 \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --pooling final \
  --score-head-type linear \
  --output-dir "${OUT_DIR}" \
  --log-path "${TRAIN_LOG}" \
  2>&1 | tee "${TMUX_TRAIN_LOG}"

python -u scripts/eval_reward_head.py \
  --score-head "${OUT_DIR}/score_head_final.pt" \
  --lora-adapter "${OUT_DIR}/lora_final" \
  --limit 0 \
  --max-tiles 2 \
  --pooling final \
  --score-head-type linear \
  --output "${EVAL_JSONL}" \
  --summary "${EVAL_SUMMARY}" \
  2>&1 | tee "${TMUX_EVAL_LOG}"

python scripts/summarize_eval_summary.py \
  --summary "${EVAL_SUMMARY}" \
  --output-prefix artifacts/report_assets/D_PromptCap20NoBench_4k_Linear_vlrb_full \
  2>&1 | tee -a "${TMUX_EVAL_LOG}"

python scripts/collect_results.py \
  --eval-dir outputs/eval \
  --output-prefix artifacts/report_assets/results \
  2>&1 | tee -a "${TMUX_EVAL_LOG}"
