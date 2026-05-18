# VL Reward Model Exam

本仓库用于 2026-05 Research 能力考核：基于 InternVL2.5-2B 构建视觉语言 Reward Model，在 VLRewardBench 上评估 `Score_base` 与 `Score_reward`。

## 方法概览

主线模型：

```text
OpenGVLab/InternVL2_5-2B
```

Reward 模型结构：

```text
image + query + candidate response
  -> InternVL2.5-2B backbone
  -> final valid-token hidden state
  -> linear score head
  -> scalar reward score
```

训练目标使用 pairwise Bradley-Terry loss：

```text
loss = -log sigmoid(score_chosen - score_rejected - margin)
```

当前实现支持两种训练方式：

- `head-only`：冻结 InternVL backbone，只训练线性 score head。
- `LoRA + score head`：冻结 base 权重，对 language model 线性层加 LoRA，同时训练 score head。

## 环境

本机路径约定：

```text
/root/private_data/models/reward-model-exam
/root/private_data/datasets/reward-model-exam
/root/private_data/projects/reward model/outputs
```

激活环境：

```bash
cd "/root/private_data/projects/reward model"
source scripts/activate_env.sh
python scripts/check_env.py
```

已验证环境：

```text
Python 3.10.20
PyTorch 2.7.0+cu126
Transformers 4.46.3
GPU: NVIDIA A800 80GB PCIe
```

如果 Hugging Face 官方入口不可用，可设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 数据与模型

下载 InternVL2.5-2B 与 VLRewardBench：

```bash
source scripts/activate_env.sh
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_assets.py --assets model benchmark --max-workers 8
```

当前使用的训练数据：

```text
trl-lib/rlaif-v
```

本项目严格不使用 VLRewardBench 训练 reward model。

## 训练

Head-only 小样本 sanity：

```bash
source scripts/activate_env.sh
python -u scripts/train_reward_head.py \
  --limit 8 \
  --epochs 2 \
  --max-tiles 2 \
  --lr 1e-3 \
  --output-dir outputs/checkpoints/reward_head_overfit_8 \
  --log-path outputs/logs/train_reward_head_overfit_8.jsonl
```

LoRA v1 训练示例：

```bash
tmux new-session -d -s train_lora_v1_1k "\
cd '/root/private_data/projects/reward model' && \
source scripts/activate_env.sh && \
export HF_ENDPOINT=https://hf-mirror.com && \
python -u scripts/train_reward_head.py \
  --limit 1024 \
  --epochs 1 \
  --max-tiles 2 \
  --lr 1e-4 \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --output-dir outputs/checkpoints/lora_v1_1k \
  --log-path outputs/logs/train_lora_v1_1k.jsonl \
  2>&1 | tee outputs/logs/tmux_train_lora_v1_1k.log"
```

监控：

```bash
tmux capture-pane -pt train_lora_v1_1k:0 -S -80
tail -n 20 outputs/logs/train_lora_v1_1k.jsonl
nvidia-smi
```

LoRA v2 去重消融训练：

```bash
source scripts/activate_env.sh
python scripts/build_dedup_exclude_indices.py \
  --mode union \
  --output-prefix artifacts/report_assets/dedup_rlaifv1024_vlrb

tmux new-session -d -s train_lora_v2_dedup_1k "\
cd '/root/private_data/projects/reward model' && \
source scripts/activate_env.sh && \
python -u scripts/train_reward_head.py \
  --limit 1024 \
  --epochs 1 \
  --max-tiles 2 \
  --lr 1e-4 \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --exclude-indices artifacts/report_assets/dedup_rlaifv1024_vlrb.union.exclude_indices.txt \
  --limit-after-exclude \
  --output-dir outputs/checkpoints/lora_v2_dedup_1k \
  --log-path outputs/logs/train_lora_v2_dedup_1k.jsonl \
  2>&1 | tee outputs/logs/tmux_train_lora_v2_dedup_1k.log"
```

## 评测

Scalar reward 评测：

```bash
source scripts/activate_env.sh
python -u scripts/eval_reward_head.py \
  --score-head outputs/checkpoints/lora_v1_1k/score_head_final.pt \
  --lora-adapter outputs/checkpoints/lora_v1_1k/lora_final \
  --limit 0 \
  --max-tiles 2 \
  --output outputs/eval/lora_v1_1k_vlrb_full.jsonl \
  --summary outputs/eval/lora_v1_1k_vlrb_full_summary.json
```

生成报告用表格：

```bash
python scripts/summarize_training_log.py \
  --log outputs/logs/train_lora_v1_1k.jsonl \
  --window 32 \
  --output-prefix artifacts/report_assets/lora_v1_1k_train

python scripts/summarize_eval_summary.py \
  --summary outputs/eval/lora_v1_1k_vlrb_full_summary.json \
  --output-prefix artifacts/report_assets/lora_v1_1k_vlrb_full
```

LoRA v2 去重消融评测：

```bash
source scripts/activate_env.sh
bash scripts/run_lora_v2_dedup_eval.sh
```

生成报告图表和 HTML：

```bash
source scripts/activate_env.sh
python scripts/plot_report_assets.py
python scripts/render_report_html.py
```

## 当前结果

已跑通：

| setting | train data | VLRewardBench | accuracy |
| --- | ---: | ---: | ---: |
| score head overfit sanity | RLAIF-V 8 pairs | first 10 | 60.00% |
| base InternVL2.5-2B generative judge | none | full 1247 | 46.51% |
| head-only v0 | RLAIF-V 128 pairs | full 1247 | 47.79% |
| LoRA v1 | RLAIF-V 1024 pairs | first 100 | 81.00% |
| LoRA v1 | RLAIF-V 1024 pairs | full 1247 | 74.66% |

`Score_base` 使用原始 InternVL2.5-2B 作为 generative judge，strict accuracy 为 `46.51%`，parse rate 为 `83.40%`。`head-only v0` 属于 DOCX 要求口径里的 `Score_reward`，但只是弱 baseline。当前主结果是 `LoRA v1`，全量 VLRewardBench accuracy 为 `74.66%`。相比 v0 full 的 `47.79%`，提升 `26.86` 个百分点。

轻量泄漏/去重检查：

- RLAIF-V 训练前 1024 条与 VLRewardBench full 的 image md5 overlap 为 0。
- exact query overlap 为 38，主要是通用描述模板。
- dHash 近重复检查发现少量近似图像，需在报告中如实说明，并可作为后续 dedup ablation。

去重消融：

- `configs/lora_v2_dedup_1k.json` 已配置 exact query overlap 与 dHash near overlap union 排除。
- 训练策略是先排除 476 个 overlap indices，再取 1024 条有效 RLAIF-V 样本。
- 该实验用于 robustness 分析，不替代当前主结果。

## 提交说明

GitHub 仓库：

```text
https://github.com/canfaneat/vl-reward-model-exam.git
```

Hugging Face model repo：

```text
https://huggingface.co/canfaneat/internvl2-5-2b-vl-reward-model
```

若最终采用 LoRA，应上传：

- `lora_final/`
- `score_head_final.pt`
- 训练配置与模型说明
- 加载和评测命令

不上传原始训练数据、VLRewardBench 数据、私有 token 或本机缓存。
