#!/usr/bin/env bash
set -euo pipefail

cd "/root/private_data/projects/reward model"

WAIT_SUMMARY="outputs/eval/D_PromptCap20NoBench_4k_Linear_vlrb_full_summary.json"

while [[ ! -s "${WAIT_SUMMARY}" ]]; do
  echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') waiting for ${WAIT_SUMMARY}"
  sleep 120
done

source scripts/activate_env.sh
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

run_one() {
  local name="$1"
  local indices="$2"
  local out_dir="outputs/checkpoints/${name}"
  local train_log="outputs/logs/train_${name}.jsonl"
  local tmux_train_log="outputs/logs/tmux_train_${name}.log"
  local eval_jsonl="outputs/eval/${name}_vlrb_full.jsonl"
  local eval_summary="outputs/eval/${name}_vlrb_full_summary.json"
  local tmux_eval_log="outputs/logs/tmux_eval_${name}_full.log"

  python -u scripts/train_reward_head.py \
    --include-indices "${indices}" \
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
    --output-dir "${out_dir}" \
    --log-path "${train_log}" \
    2>&1 | tee "${tmux_train_log}"

  python -u scripts/eval_reward_head.py \
    --score-head "${out_dir}/score_head_final.pt" \
    --lora-adapter "${out_dir}/lora_final" \
    --limit 0 \
    --max-tiles 2 \
    --pooling final \
    --score-head-type linear \
    --output "${eval_jsonl}" \
    --summary "${eval_summary}" \
    2>&1 | tee "${tmux_eval_log}"

  python scripts/summarize_eval_summary.py \
    --summary "${eval_summary}" \
    --output-prefix "artifacts/report_assets/${name}_vlrb_full" \
    2>&1 | tee -a "${tmux_eval_log}"

  python scripts/collect_results.py \
    --eval-dir outputs/eval \
    --output-prefix artifacts/report_assets/results \
    2>&1 | tee -a "${tmux_eval_log}"

  python scripts/data_similarity_and_diversity_audit.py \
    2>&1 | tee -a "${tmux_eval_log}"
}

run_one "D_PromptCap50NoBench_4k_Linear" "artifacts/report_assets/rlaifv_promptcap50_nobench_4k.indices.txt"
run_one "D_PromptCap10NoBench_4k_Linear" "artifacts/report_assets/rlaifv_promptcap10_nobench_4k.indices.txt"
