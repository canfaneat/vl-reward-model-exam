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

# InternVL2.5-2B VL Reward Model

This repository contains a LoRA-based vision-language reward model for pairwise response ranking.

The model is built from `OpenGVLab/InternVL2_5-2B`. It adds a scalar score head on the final valid-token hidden state and trains LoRA adapters on the language-model linear layers. The uploaded files contain only the LoRA adapter, score head, training metadata, and configuration. The base model weights are not duplicated.

## Model

- Base model: `OpenGVLab/InternVL2_5-2B`
- Reward head: linear scalar score head
- Pooling: final valid-token hidden state
- Adapter: LoRA on language-model linear modules
- LoRA r: 8
- LoRA alpha: 16
- LoRA dropout: 0.05

## Training Data

- Training data: `trl-lib/rlaif-v`
- Training subset: 4096 preference pairs selected by PromptCap50
- Benchmark: `MMInstruction/VL-RewardBench`

PromptCap50 means that the training subset is selected by scanning RLAIF-V and limiting each normalized prompt to at most 50 examples. This is an internal training-set frequency control strategy; VLRewardBench fields are not used to select this final training subset.

VLRewardBench samples, human rankings, and candidate responses were not used as reward-model training samples.

## Training Configuration

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

## Evaluation

VLRewardBench full set, 1247 examples:

| Model | Accuracy |
| --- | ---: |
| Base InternVL2.5-2B generative judge, strict parser | 46.51% |
| Head-only sanity model, RLAIF-V 128 pairs | 47.79% |
| Raw RLAIF-V 1k reward model | 74.66% |
| Strict query+image audit ablation, 4096 pairs | 70.17% |
| PromptCap50 reward model, 4096 pairs | 71.69% |

The uploaded checkpoint is the PromptCap50 reward model.

Source-level highlights for the uploaded checkpoint:

| Source | N | Accuracy |
| --- | ---: | ---: |
| POVID_preference_data_for_VLLMs | 448 | 81.03% |
| empty query_source | 317 | 59.94% |
| wildvision-battle | 171 | 63.16% |
| COCO | 63 | 82.54% |
| OK-VQA | 32 | 87.50% |
| VQAv2 | 35 | 80.00% |

## Usage

Use the repository code at `https://github.com/canfaneat/vl-reward-model-exam` to load the base model, LoRA adapter, and score head.

Example evaluation command:

```bash
python -u scripts/eval_reward_head.py \
  --score-head score_head_final.pt \
  --lora-adapter lora_final \
  --limit 0 \
  --max-tiles 2 \
  --score-head-type linear \
  --pooling final
```

## Limitations

This model is a research assessment artifact. RLAIF-V and VLRewardBench are in the same broad multimodal preference ecosystem, so the score should be interpreted as in-domain reward-model adaptation rather than proof of broad zero-shot generalization.

Data auditing found no direct use of VLRewardBench samples or labels as training samples, but prompt templates and task types are naturally similar across public multimodal preference datasets. This is why the repository includes data similarity audits and PromptCap sampling experiments.

## Files

- `lora_final/adapter_model.safetensors`
- `lora_final/adapter_config.json`
- `score_head_final.pt`
- `training_meta.json`
- `config.json`
