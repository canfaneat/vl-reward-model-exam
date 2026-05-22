---
base_model: OpenGVLab/InternVL2_5-2B
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - reward-model
  - internvl
  - lora
  - vlrewardbench
---

# InternVL2.5-2B 多模态奖励模型

本仓库包含一个基于 `OpenGVLab/InternVL2_5-2B` 的视觉语言奖励模型，用于图像、问题与候选回答之间的 pairwise response ranking。模型保留 InternVL2.5-2B 的基础图文表征能力，在语言模型线性层上训练 LoRA adapter，并在最后一个有效 token 的 hidden state 后接线性 score head 输出 reward score。

上传文件只包含 LoRA adapter、score head、训练配置和元信息，不重复上传 InternVL2.5-2B 基础模型权重。使用时需要先加载基础模型，再加载本仓库中的 `lora_final/` 与 `score_head_final.pt`。

## 模型结构

- Base model: `OpenGVLab/InternVL2_5-2B`
- Reward head: linear numeric score head
- Pooling: final valid-token hidden state
- Adapter: LoRA on language-model linear modules
- LoRA r: 8
- LoRA alpha: 16
- LoRA dropout: 0.05

## 训练数据与边界

- Training data: `trl-lib/rlaif-v`
- Training subset: PromptCap50 选择的 4096 条 preference pairs
- Benchmark: `MMInstruction/VL-RewardBench`

PromptCap50 指按 RLAIF-V 原始顺序扫描样本，并限制同一个归一化 prompt 最多保留 50 条样本。该数据选择策略只使用 RLAIF-V 内部 prompt 频率；VLRewardBench 字段不参与最终训练子集选择。

训练中没有使用 VLRewardBench 的样本、human ranking 或候选回答作为 reward-model training samples。

## 训练配置

```json
{
  "limit": 4096,
  "epochs": 1,
  "lr": 0.0001,
  "max_tiles": 2,
  "margin": 0.0,
  "seed": 42,
  "use_lora": true,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "score_head_type": "linear",
  "pooling": "final"
}
```

## 评测结果

VLRewardBench full set，共 1247 条样本：

| 模型 / 实验设置 | Accuracy |
| --- | ---: |
| Base InternVL2.5-2B generative judge | 46.51% |
| Head-only sanity model, RLAIF-V 128 pairs | 47.79% |
| Raw RLAIF-V 1k reward model | 74.66% |
| Strict query+image audit ablation, 4096 pairs | 70.17% |
| PromptCap50 reward model, 4096 pairs | 71.69% |
| PromptCap50 reward model, seed 123 | 61.27% |

本仓库上传的 checkpoint 为 PromptCap50 seed-42 reward model，对应 full benchmark accuracy `71.69%`。

主模型 source-level 结果节选：

| Source | N | Accuracy |
| --- | ---: | ---: |
| POVID_preference_data_for_VLLMs | 448 | 81.03% |
| empty query_source | 317 | 59.94% |
| wildvision-battle | 171 | 63.16% |
| COCO | 63 | 82.54% |
| OK-VQA | 32 | 87.50% |
| VQAv2 | 35 | 80.00% |

## 使用方式

代码仓库位于：

```text
https://github.com/canfaneat/vl-reward-model-exam
```

评测时加载基础模型、LoRA adapter 和 score head。示例命令：

```bash
python -u scripts/eval_reward_head.py \
  --score-head score_head_final.pt \
  --lora-adapter lora_final \
  --limit 0 \
  --max-tiles 2 \
  --score-head-type linear \
  --pooling final
```

完整复现命令、数据审计脚本、PromptCap 构造脚本和报告见 GitHub 仓库。

## 文件结构

```text
README.md
config.json
training_meta.json
score_head_final.pt
lora_final/README.md
lora_final/adapter_config.json
lora_final/adapter_model.safetensors
```

## 说明

该模型是考核项目中的研究型交付物。RLAIF-V 与 VLRewardBench 同处多模态偏好判断任务生态，因此结果应解释为同域 preference data 下的 reward-model adaptation，而不是广泛跨域泛化能力的结论。

项目同时保留了数据相似性审计、PromptCap 数据选择实验和 seed 诊断。seed=123 在相同数据和超参数下得到 `61.27%`，最后 512 step 的 `score_chosen - score_rejected` 平均值为 `0.0008`，positive gap rate 为 `50.98%`，说明单 epoch LoRA reward 训练存在优化敏感性。后续改进方向包括 learning-rate scheduling、warmup、gradient clipping、checkpoint selection 和多 seed 验证。
