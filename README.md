# InternVL2.5-2B 多模态奖励模型训练与数据选择

本仓库是 2026 年 5 月多模态 Reward Model 考核项目的公开代码与实验材料。项目基于 `OpenGVLab/InternVL2_5-2B` 构建视觉语言奖励模型，在 RLAIF-V 图文偏好数据上训练 LoRA adapter 与线性 score head，并在 VLRewardBench 上评测 `Score_base` 与 `Score_reward`。

核心结果如下：原始 InternVL2.5-2B 以生成式 judge 方式在 VLRewardBench full set 上得到 `46.51%`；最终提交模型 `D_PromptCap50NoBench_4k_Linear` 得到 `71.69%`。第一天晚间完成的 raw RLAIF-V 1024 pairs 训练已经达到 `74.66%`，后续工作重点转向数据相似性审计、PromptCap 数据选择、结构消融与训练稳定性诊断。

## 交付内容

- 最终中文报告：[`reports/VL_REWARD_MODEL_REPORT_CN.pdf`](reports/VL_REWARD_MODEL_REPORT_CN.pdf)
- 报告 LaTeX 源文件：[`reports/VL_REWARD_MODEL_REPORT_CN.tex`](reports/VL_REWARD_MODEL_REPORT_CN.tex)
- 实验结果汇总：[`RESULTS.md`](RESULTS.md)
- Hugging Face 模型仓库：<https://huggingface.co/canfaneat/internvl2-5-2b-vl-reward-model>
- GitHub 代码仓库：<https://github.com/canfaneat/vl-reward-model-exam>

## 主要结果

| 设置 | 训练数据 | VLRewardBench full accuracy |
| --- | --- | ---: |
| InternVL2.5-2B base generative judge | 无 reward 训练 | 46.51% |
| Head-only sanity model | RLAIF-V 128 pairs | 47.79% |
| Raw RLAIF-V 1k reward model | 原始前 1024 pairs | 74.66% |
| Strict query+image audit ablation | 4096 pairs | 70.17% |
| PromptCap50 reward model | RLAIF-V 4096 pairs | 71.69% |
| PromptCap50 reward model, seed 123 | 相同数据与配置 | 61.27% |

最终提交 checkpoint 对应：

```text
outputs/checkpoints/D_PromptCap50NoBench_4k_Linear
```

主模型配置：

- base model：`OpenGVLab/InternVL2_5-2B`
- 训练数据：`trl-lib/rlaif-v`
- 数据选择：RLAIF-V 内部 prompt 频率控制，PromptCap50
- LoRA：`r=8`，`alpha=16`，`dropout=0.05`
- score head：linear numeric score head
- pooling：final valid token
- training pairs：4096
- epochs：1
- learning rate：`1e-4`
- max image tiles：2
- seed：42

VLRewardBench 只用于评测和事后审计分析；训练样本不使用 VLRewardBench 的样本、human ranking 或候选回答。

## 实验推进与提交时间线

本项目不是最后一次性整理出的结果，而是按“跑通闭环、发现高分、审计数据、选择主模型、补充稳定性分析、整理交付”的顺序推进。

| 时间 | Git 提交 | 内容 |
| --- | --- | --- |
| 2026-05-19 00:01 +0800 | `7ecb478` | 初始 reward model deliverable；已包含 base judge、head-only、raw RLAIF-V 1k full eval 结果，其中 raw 1k 达到 74.66% |
| 2026-05-19 00:59 +0800 | `3a209a5` | 增加多 shard overlap audit，开始系统检查训练集与 benchmark 的显式相似性 |
| 2026-05-20 15:27 +0800 | `e1d3032` | 加入 PromptCap50 主模型相关结果与数据选择 artifacts |
| 2026-05-20 16:50 +0800 | `8f41cb8` | 优化数据中心实验图表，整理 prompt concentration 与 source-level 结果 |
| 2026-05-20 22:27 +0800 | `c183a8e` | 加入 PromptCap50 seed 诊断，说明单 epoch LoRA reward 训练的优化敏感性 |
| 2026-05-21 | `2dec500`、`be1677b`、`16f8a73` | 完成最终报告、model card、公开仓库门面与结果汇总 |

第一条提交中已经包含 `outputs/eval/lora_v1_1k_vlrb_full_summary.json`、`artifacts/report_assets/lora_v1_1k_train.summary.csv`、训练/评测脚本和模型封装。这说明项目第一天已经跑通训练评测闭环并得到高分结果，后续工作主要是在此基础上补充数据边界、结果解释和稳定性分析。

## 方法概述

Reward model 每次评价一个候选回答：

```text
image + question + candidate response
  -> InternVL2.5-2B backbone
  -> final valid-token hidden state
  -> Linear(hidden_size, 1)
  -> reward score
```

对于 RLAIF-V 中的偏好样本，模型分别计算：

```text
score_chosen = RM(image, question, chosen_response)
score_rejected = RM(image, question, rejected_response)
```

训练目标为 pairwise preference loss：

```text
loss = -log sigmoid(score_chosen - score_rejected - margin)
```

主实验中 `margin=0.0`。

## 数据处理与 PromptCap

早期 raw RLAIF-V 1024 pairs 结果很高，但该子集存在较强的 prompt 模板集中度。为了让最终主模型的数据选择逻辑更清晰，我们做了图像 MD5 重复、dHash 近似图像、query overlap、response overlap、prompt frequency concentration 和 effective prompt count 等审计。

最终主模型采用 PromptCap50：按 RLAIF-V 原始顺序扫描样本，同一个归一化 prompt 最多保留 50 条，直到得到 4096 条 preference pairs。这个策略只使用 RLAIF-V 内部 prompt 频率，不使用 VLRewardBench 字段来选择训练样本。

PromptCap50 将 top-20 prompt mass 从 raw first 1024 的 `41.50%` 降到 `23.63%`，effective prompt 从 `216.31` 提高到 `811.89`，同时保持 `71.69%` 的 full benchmark accuracy。

## 代码结构

| 路径 | 作用 |
| --- | --- |
| `src/internvl_reward.py` | InternVL reward model 封装 |
| `scripts/train_reward_head.py` | reward head / LoRA 训练 |
| `scripts/eval_reward_head.py` | VLRewardBench 数值打分评测 |
| `scripts/build_prompt_cap_indices.py` | PromptCap 数据选择 |
| `scripts/data_similarity_and_diversity_audit.py` | 数据相似性与多样性审计 |
| `scripts/plot_data_centric_figures.py` | 报告图表生成 |
| `scripts/prepare_hf_upload.py` | 整理 Hugging Face 上传包 |

## 环境

实验环境：

```text
Python 3.10.20
PyTorch 2.7.0+cu126
Transformers 4.46.3
GPU: NVIDIA A800 80GB PCIe
平台: SCNet 超算互联网平台
```

激活环境：

```bash
source scripts/activate_env.sh
python scripts/check_env.py
```

本地路径约定：

```text
models/reward-model-exam/OpenGVLab/InternVL2_5-2B
datasets/reward-model-exam/rlaif-v
datasets/reward-model-exam/VL-RewardBench
```

下载辅助命令：

```bash
source scripts/activate_env.sh
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_assets.py --assets model benchmark --max-workers 8
```

## 构建 PromptCap50 indices

```bash
source scripts/activate_env.sh
python scripts/build_prompt_cap_indices.py \
  --cap 50 \
  --limit 4096 \
  --output-prefix artifacts/report_assets/rlaifv_promptcap50_nobench_4k
```

输出：

```text
artifacts/report_assets/rlaifv_promptcap50_nobench_4k.indices.txt
```

## 训练主模型

```bash
source scripts/activate_env.sh
python -u scripts/train_reward_head.py \
  --include-indices artifacts/report_assets/rlaifv_promptcap50_nobench_4k.indices.txt \
  --epochs 1 \
  --max-tiles 2 \
  --lr 1e-4 \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --score-head-type linear \
  --pooling final \
  --output-dir outputs/checkpoints/D_PromptCap50NoBench_4k_Linear \
  --log-path outputs/logs/train_D_PromptCap50NoBench_4k_Linear.jsonl
```

## 评测主模型

```bash
source scripts/activate_env.sh
python -u scripts/eval_reward_head.py \
  --score-head outputs/checkpoints/D_PromptCap50NoBench_4k_Linear/score_head_final.pt \
  --lora-adapter outputs/checkpoints/D_PromptCap50NoBench_4k_Linear/lora_final \
  --limit 0 \
  --max-tiles 2 \
  --score-head-type linear \
  --pooling final \
  --output outputs/eval/D_PromptCap50NoBench_4k_Linear_vlrb_full.jsonl \
  --summary outputs/eval/D_PromptCap50NoBench_4k_Linear_vlrb_full_summary.json
```

预期结果：

```text
n = 1247
accuracy = 0.7169206094627105
accuracy_percent = 71.69%
```

## 结果与图表

生成 compact CSV summary：

```bash
python scripts/summarize_eval_summary.py \
  --summary outputs/eval/D_PromptCap50NoBench_4k_Linear_vlrb_full_summary.json \
  --output-prefix artifacts/report_assets/D_PromptCap50NoBench_4k_Linear_vlrb_full

python scripts/data_similarity_and_diversity_audit.py
```

生成报告图表：

```bash
python scripts/plot_data_centric_figures.py
```

图表位于：

```text
artifacts/figures/data_centric/benchmark_style_table.png
artifacts/figures/data_centric/prompt_concentration_vs_accuracy.png
artifacts/figures/data_centric/promptcap_strength_curve.png
artifacts/figures/data_centric/data_selection_source_accuracy.png
```

## Hugging Face 权重包

Hugging Face 上传包包含 LoRA adapter、score head、训练元信息和 model card，不重复上传 InternVL2.5-2B 基础模型权重。

```bash
source scripts/activate_env.sh
python scripts/prepare_hf_upload.py \
  --checkpoint-dir outputs/checkpoints/D_PromptCap50NoBench_4k_Linear \
  --model-card model_cards/internvl2-5-2b-vl-reward-model.md \
  --output-dir outputs/hf_upload/internvl2-5-2b-vl-reward-model
```

期望文件结构：

```text
README.md
config.json
training_meta.json
score_head_final.pt
lora_final/adapter_config.json
lora_final/adapter_model.safetensors
```

上传命令：

```bash
cd outputs/hf_upload/internvl2-5-2b-vl-reward-model
hf upload canfaneat/internvl2-5-2b-vl-reward-model . . \
  --repo-type model \
  --commit-message "Upload PromptCap50 InternVL2.5-2B reward model"
```

使用时需要先加载 `OpenGVLab/InternVL2_5-2B`，再加载本仓库中的 LoRA adapter 与 `score_head_final.pt`。

## 说明

raw RLAIF-V 1k 的高分体现了同域偏好数据对 reward model adaptation 的有效性。最终选择 PromptCap50，是因为它在保持 70% 以上 full benchmark accuracy 的同时，降低了高频 prompt 模板集中度，使训练数据选择更容易解释和复查。

本 GitHub 仓库保留代码、配置、报告、图表和摘要级实验结果；不上传完整模型权重、原始数据集、完整训练日志和私密考核材料。模型 checkpoint 文件放在 Hugging Face 模型仓库中。

## License and Attribution

本仓库代码使用 MIT License。基础模型、数据集和 benchmark 遵循各自上游许可证与使用条款。
