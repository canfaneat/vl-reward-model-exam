# Experiment Results

This file collects the public-facing experiment results for the InternVL2.5-2B vision-language reward model project. The full analysis and Chinese report are in [`reports/VL_REWARD_MODEL_REPORT_CN.pdf`](reports/VL_REWARD_MODEL_REPORT_CN.pdf).

## Main Score

VLRewardBench full set contains 1247 evaluation examples in this setup.

| Setting | Training data | Accuracy |
| --- | --- | ---: |
| InternVL2.5-2B base generative judge | none | 46.51% |
| Head-only sanity model | RLAIF-V 128 pairs | 47.79% |
| Raw RLAIF-V reward model | first 1024 pairs | 74.66% |
| Image-only refined ablation | 4096 pairs | 66.72% |
| Strict query+image audit ablation | 4096 pairs | 70.17% |
| PromptCap50 reward model | RLAIF-V 4096 pairs | 71.69% |
| PromptCap10 reward model | RLAIF-V 4096 pairs | 70.73% |
| PromptCap20 reward model | RLAIF-V 4096 pairs | 60.63% |
| MLP-head ablation | PromptCap20 4096 pairs | 50.44% |

The submitted checkpoint is the PromptCap50 seed-42 run:

```text
outputs/checkpoints/D_PromptCap50NoBench_4k_Linear
```

## Data Selection Summary

PromptCap selects training samples using only RLAIF-V internal prompt frequencies. It does not use VLRewardBench samples, labels, human rankings, or candidate responses as reward-model training data.

| Data strategy | Effective prompt | Top-20 prompt mass | Query overlap rate | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Raw first 1024 | 216.31 | 41.50% | 46.39% | 74.66% |
| Image-only refined 4096 | 407.21 | 40.16% | 44.95% | 66.72% |
| Strict query+image 4096 | 1768.71 | 16.67% | 0.00% | 70.17% |
| PromptCap50 4096 | 811.89 | 23.63% | 30.88% | 71.69% |
| PromptCap20 4096 | 1733.88 | 9.77% | 16.04% | 60.63% |
| PromptCap10 4096 | 2440.44 | 4.88% | 8.96% | 70.73% |

The raw 1k run reached the highest single accuracy early in the project, but it also had stronger prompt-template concentration. PromptCap50 was selected for the final checkpoint because it keeps the score above 70% while reducing high-frequency prompt concentration.

## Structure and Stability

| Experiment | Data | Accuracy | Note |
| --- | --- | ---: | --- |
| Final-token pooling + linear head | PromptCap20 1k | 61.03% | Uses the final valid-token hidden state |
| Final+mean concat pooling + linear head | PromptCap20 1k | 60.14% | Adds text mean pooling |
| MLP score head | PromptCap20 4k | 50.44% | More complex head did not help in this setup |
| PromptCap50 seed 42 | PromptCap50 4k | 71.69% | Submitted checkpoint |
| PromptCap50 seed 123 | PromptCap50 4k | 61.27% | Diagnostic rerun, weak reward-scale formation |

The seed-123 diagnostic had mean `score_chosen - score_rejected` of `0.0008` over the last 512 training steps and a positive-gap rate of `50.98%`. The seed-42 submitted checkpoint had mean gap `0.3411` and positive-gap rate `63.67%` over the same window.

## Public Artifacts

Compact result summaries and figures are included for inspection:

- `artifacts/report_assets/results.csv`
- `artifacts/report_assets/D_PromptCap50NoBench_4k_Linear_vlrb_full.overview.csv`
- `artifacts/report_assets/D_PromptCap50NoBench_4k_Linear_vlrb_full.by_source.csv`
- `artifacts/report_assets/data_similarity_diversity.summary.csv`
- `artifacts/figures/data_centric/benchmark_style_table.png`
- `artifacts/figures/data_centric/prompt_concentration_vs_accuracy.png`
- `artifacts/figures/data_centric/promptcap_strength_curve.png`
- `artifacts/figures/data_centric/data_selection_source_accuracy.png`

Full training logs, raw datasets, local checkpoints, and private assessment files are intentionally excluded from GitHub. The Hugging Face model repository is used for the LoRA adapter and score-head checkpoint files.
