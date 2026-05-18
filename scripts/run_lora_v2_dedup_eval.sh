#!/usr/bin/env bash
set -euo pipefail

cd "/root/private_data/projects/reward model"
source scripts/activate_env.sh

python -u scripts/eval_reward_head.py \
  --score-head outputs/checkpoints/lora_v2_dedup_1k/score_head_final.pt \
  --lora-adapter outputs/checkpoints/lora_v2_dedup_1k/lora_final \
  --limit 0 \
  --max-tiles 2 \
  --output outputs/eval/lora_v2_dedup_1k_vlrb_full.jsonl \
  --summary outputs/eval/lora_v2_dedup_1k_vlrb_full_summary.json

python scripts/summarize_eval_summary.py \
  --summary outputs/eval/lora_v2_dedup_1k_vlrb_full_summary.json \
  --output-prefix artifacts/report_assets/lora_v2_dedup_1k_vlrb_full

python scripts/collect_results.py \
  --eval-dir outputs/eval \
  --output-prefix artifacts/report_assets/results
