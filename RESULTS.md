# 实验结果汇总

本文档汇总 InternVL2.5-2B 多模态奖励模型项目的主要公开结果。完整分析见最终中文报告：[`reports/VL_REWARD_MODEL_REPORT_CN.pdf`](reports/VL_REWARD_MODEL_REPORT_CN.pdf)。

## 主结果

本项目使用 VLRewardBench full set 进行评测，共 1247 条样本。

| 设置 | 训练数据 | Accuracy |
| --- | --- | ---: |
| InternVL2.5-2B base generative judge | 无 reward 训练 | 46.51% |
| Head-only sanity model | RLAIF-V 128 pairs | 47.79% |
| Raw RLAIF-V reward model | 原始前 1024 pairs | 74.66% |
| Image-only refined ablation | 4096 pairs | 66.72% |
| Strict query+image audit ablation | 4096 pairs | 70.17% |
| PromptCap50 reward model | RLAIF-V 4096 pairs | 71.69% |
| PromptCap10 reward model | RLAIF-V 4096 pairs | 70.73% |
| PromptCap20 reward model | RLAIF-V 4096 pairs | 60.63% |
| MLP-head ablation | PromptCap20 4096 pairs | 50.44% |

最终提交模型为 PromptCap50 seed-42 run：

```text
outputs/checkpoints/D_PromptCap50NoBench_4k_Linear
```

其 full benchmark summary 位于：

```text
outputs/eval/D_PromptCap50NoBench_4k_Linear_vlrb_full_summary.json
```

其中：

```json
"n": 1247,
"accuracy": 0.7169206094627105
```

## 第一阶段结果

项目第一阶段已经跑通 base judge、head-only sanity 与 LoRA reward model 训练评测闭环。北京时间 2026-05-19 00:01 的第一条公开提交 `7ecb478` 中，已经包含 raw RLAIF-V 1024 pairs 的 full VLRewardBench 评测结果：

```text
outputs/eval/lora_v1_1k_vlrb_full_summary.json
```

该结果为：

```json
"n": 1247,
"accuracy": 0.7465918203688853
```

也就是 `74.66%`。由于该早期分数较高，后续实验重点转向数据相似性审计和数据选择，而不是只追求单点最高分。

## 数据选择结果

PromptCap 只使用 RLAIF-V 内部 prompt 频率，不使用 VLRewardBench 样本、标签、human ranking 或候选回答作为训练样本。

| 数据策略 | Effective prompt | Top-20 prompt mass | Query overlap rate | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Raw first 1024 | 216.31 | 41.50% | 46.39% | 74.66% |
| Image-only refined 4096 | 407.21 | 40.16% | 44.95% | 66.72% |
| Strict query+image 4096 | 1768.71 | 16.67% | 0.00% | 70.17% |
| PromptCap50 4096 | 811.89 | 23.63% | 30.88% | 71.69% |
| PromptCap20 4096 | 1733.88 | 9.77% | 16.04% | 60.63% |
| PromptCap10 4096 | 2440.44 | 4.88% | 8.96% | 70.73% |

Raw first 1024 的分数最高，但 prompt 模板集中度也最高。PromptCap50 在保持 70% 以上准确率的同时降低了高频 prompt 集中度，因此被选为最终提交模型。

## 结构与稳定性

| 实验 | 数据 | Accuracy | 说明 |
| --- | --- | ---: | --- |
| Final-token pooling + linear head | PromptCap20 1k | 61.03% | 使用最后一个有效 token 的 hidden state |
| Final+mean concat pooling + linear head | PromptCap20 1k | 60.14% | 拼接 final token 与文本 mean pooling |
| MLP score head | PromptCap20 4k | 50.44% | 在当前设置下复杂 head 未带来收益 |
| PromptCap50 seed 42 | PromptCap50 4k | 71.69% | 最终提交 checkpoint |
| PromptCap50 seed 123 | PromptCap50 4k | 61.27% | 诊断实验，reward scale 未稳定形成 |

seed=123 诊断实验最后 512 step 的 `score_chosen - score_rejected` 平均值为 `0.0008`，positive gap rate 为 `50.98%`；seed=42 主模型对应平均 gap 为 `0.3411`，positive gap rate 为 `63.67%`。该结果说明单 epoch LoRA reward 训练存在优化敏感性，后续可以通过 warmup、gradient clipping、checkpoint selection 和多 seed 验证进一步提升稳定性。

## 提交时间线

| 时间 | 提交 | 内容 |
| --- | --- | --- |
| 2026-05-19 00:01 +0800 | `7ecb478` | 初始 deliverable，包含 base judge、head-only、raw RLAIF-V 1k full eval 与训练摘要 |
| 2026-05-19 00:59 +0800 | `3a209a5` | 增加 overlap audit，检查训练集与 benchmark 的显式相似性 |
| 2026-05-20 15:27 +0800 | `e1d3032` | 加入 PromptCap50 主模型结果与 artifacts |
| 2026-05-20 16:50 +0800 | `8f41cb8` | 整理 prompt concentration、source-level accuracy 等数据中心图表 |
| 2026-05-20 22:27 +0800 | `c183a8e` | 加入 seed=123 稳定性诊断 |
| 2026-05-21 | `2dec500`、`be1677b`、`16f8a73` | 完成最终报告、model card、仓库门面和结果汇总 |

这条提交线体现了项目推进顺序：第一阶段快速跑通训练评测并得到高分；第二阶段围绕高分来源做数据审计；第三阶段选择 PromptCap50 作为主模型；第四阶段补充结构消融、稳定性诊断和最终报告。

## 公开 artifacts

仓库保留了便于检查的 compact summary 和图表：

- `artifacts/report_assets/results.csv`
- `artifacts/report_assets/D_PromptCap50NoBench_4k_Linear_vlrb_full.overview.csv`
- `artifacts/report_assets/D_PromptCap50NoBench_4k_Linear_vlrb_full.by_source.csv`
- `artifacts/report_assets/data_similarity_diversity.summary.csv`
- `artifacts/figures/data_centric/benchmark_style_table.png`
- `artifacts/figures/data_centric/prompt_concentration_vs_accuracy.png`
- `artifacts/figures/data_centric/promptcap_strength_curve.png`
- `artifacts/figures/data_centric/data_selection_source_accuracy.png`

完整训练日志、原始数据集、本地 checkpoint 和私密考核材料不放在 GitHub；模型权重通过 Hugging Face 模型仓库交付。
