
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


@dataclass
class PreferenceSample:
    image_bytes: bytes
    question: str
    chosen: str
    rejected: str
    source: str = ''


def build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios, width: int, height: int, image_size: int):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def image_bytes_to_tensor(image_bytes: bytes, input_size=448, max_num=6) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(img) for img in images])


def _message_text(message_list: Any) -> str:
    message = message_list[0] if isinstance(message_list, (list, np.ndarray)) else message_list
    content = message['content']
    parts = content.tolist() if hasattr(content, 'tolist') else list(content)
    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get('type') == 'text' and part.get('text'):
            texts.append(str(part['text']))
    return '\n'.join(texts).strip()


def parse_rlaif_row(row: pd.Series) -> PreferenceSample:
    images = row['images']
    image0 = images[0] if isinstance(images, (list, np.ndarray)) else images
    image_bytes = image0['bytes']
    return PreferenceSample(
        image_bytes=image_bytes,
        question=_message_text(row['prompt']),
        chosen=_message_text(row['chosen']),
        rejected=_message_text(row['rejected']),
        source='rlaif-v',
    )


class RLAIFVPreferenceDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        limit: int | None = None,
        shard_limit: int | None = None,
        exclude_indices: set[int] | None = None,
        limit_after_exclude: bool = False,
    ):
        self.data_dir = Path(data_dir)
        files = sorted((self.data_dir / 'data').glob('train-*.parquet'))
        if shard_limit is not None:
            files = files[:shard_limit]
        frames = []
        remaining = limit
        offset = 0
        for file in files:
            df = pd.read_parquet(file)
            df = df.copy()
            df['_global_idx'] = range(offset, offset + len(df))
            offset += len(df)
            if remaining is not None and not limit_after_exclude:
                df = df.head(remaining)
                remaining -= len(df)
            frames.append(df)
            if remaining is not None and not limit_after_exclude and remaining <= 0:
                break
        self.df = pd.concat(frames, ignore_index=True)
        if exclude_indices:
            self.df = self.df[~self.df['_global_idx'].isin(exclude_indices)].reset_index(drop=True)
        if limit is not None and limit_after_exclude:
            self.df = self.df.head(limit).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> PreferenceSample:
        return parse_rlaif_row(self.df.iloc[idx])


class InternVLRewardModel(nn.Module):
    def __init__(
        self,
        model_path: str | Path,
        dtype=torch.bfloat16,
        freeze_backbone=True,
        use_flash_attn=False,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple[str, ...] = ("wqkv", "wo", "w1", "w2", "w3"),
    ):
        super().__init__()
        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False, local_files_only=True
        )
        self.backbone = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
            local_files_only=True,
        )
        hidden_size = self.backbone.config.llm_config.hidden_size
        self.score_head = nn.Linear(hidden_size, 1)
        self.dtype = dtype
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.backbone.img_context_token_id = self.img_context_token_id
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(lora_target_modules),
            )
            self.backbone.language_model = get_peft_model(self.backbone.language_model, lora_config)
        self.score_head.to(dtype=dtype)

    def load_lora_adapter(self, adapter_path: str | Path, trainable: bool = False) -> None:
        self.backbone.language_model = PeftModel.from_pretrained(
            self.backbone.language_model,
            str(adapter_path),
            is_trainable=trainable,
            local_files_only=True,
        )

    def trainable_parameters_summary(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def build_query(self, question: str, answer: str, num_patches: int) -> str:
        template = self.backbone.conv_template.copy()
        template.system_message = self.backbone.system_message
        user_text = (
            '<image>\n'
            f'Question: {question}\n'
            f'Candidate answer: {answer}\n'
            'Evaluate the candidate answer for visual accuracy, completeness, clarity, and relevance.'
        )
        template.append_message(template.roles[0], user_text)
        template.append_message(template.roles[1], 'Reward score:')
        query = template.get_prompt()
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.backbone.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)
        return query

    def encode_one(self, image_bytes: bytes, question: str, answer: str, max_tiles: int = 6) -> dict[str, torch.Tensor]:
        pixel_values = image_bytes_to_tensor(image_bytes, max_num=max_tiles).to(dtype=self.dtype)
        query = self.build_query(question, answer, num_patches=pixel_values.shape[0])
        inputs = self.tokenizer(query, return_tensors='pt')
        return {
            'pixel_values': pixel_values,
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
        }

    def forward_encoded(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        pixel_values = pixel_values.to(device=device, dtype=self.dtype)
        input_ids = input_ids.to(device=device).unsqueeze(0)
        attention_mask = attention_mask.to(device=device).unsqueeze(0)
        image_flags = torch.ones((pixel_values.shape[0], 1), dtype=torch.long, device=device)
        outputs = self.backbone(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        last_idx = attention_mask.sum(dim=1).long() - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=device), last_idx]
        score = self.score_head(pooled).squeeze(-1)
        return score

    def score(self, image_bytes: bytes, question: str, answer: str, max_tiles: int = 6) -> torch.Tensor:
        encoded = self.encode_one(image_bytes, question, answer, max_tiles=max_tiles)
        return self.forward_encoded(**encoded)


def pairwise_loss(score_chosen: torch.Tensor, score_rejected: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    return -torch.nn.functional.logsigmoid(score_chosen - score_rejected - margin).mean()
