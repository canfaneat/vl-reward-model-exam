from __future__ import annotations

import argparse
import io
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

JUDGE_PATTERN = re.compile(
    r"(?:Overall Judgment|Therefore)\s*.*\s*-*\s*Answer\s*(\d+)\s*is\s*(?:the\s*)?(?:slightly\s*)?better",
    re.IGNORECASE,
)


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
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


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
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


def load_image_from_bytes(image_bytes: bytes, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def build_official_prompt(
    row: dict[str, Any],
    random_number: int,
    short_output: bool = False,
) -> tuple[str, list[str]]:
    responses = list(row["response"])
    answers = responses if random_number == 0 else [responses[1], responses[0]]
    if short_output:
        prompt = f"""
You are a multimodal judge. Look at the image, question, and two candidate answers.
Question: {row["query"]}
Answer 1: {answers[0]}
Answer 2: {answers[1]}
Choose the better answer based on visual accuracy, completeness, clarity, and relevance.
You must output exactly one line in this format:
Overall Judgment: Answer X is better.
""".strip()
        return prompt, answers

    prompt = f"""
You are a highly capable multimodal AI assistant tasked with evaluating answers to visual questions. Please analyze the following image and question, then determine which of the two provided answers is better.
Question: {row["query"]}
Answer 1: {answers[0]}
Answer 2: {answers[1]}
Please evaluate both answers based on the following criteria:
1. Accuracy: How well does the answer align with the visual information in the image?
2. Completeness: Does the answer fully address all aspects of the question?
3. Clarity: Is the answer easy to understand and well-articulated?
4. Relevance: Does the answer directly relate to the question and the image?
After your evaluation, please:
1. Explain your reasoning for each criterion.
2. Provide an overall judgment on which answer is better (Answer 1 or Answer 2).
For example: Overall Judgment: Answer X is better.
Your response should be structured and detailed, demonstrating your understanding of both the visual and textual elements of the task.
""".strip()
    return prompt, answers


def parse_choice(text: str) -> int:
    match = JUDGE_PATTERN.search(text.replace("\n", "").replace("*", ""))
    if match:
        choice = int(match.group(1))
        if choice in (1, 2):
            return choice
    lowered = text.lower()
    if "answer 1" in lowered and "answer 2" not in lowered:
        return 1
    if "answer 2" in lowered and "answer 1" not in lowered:
        return 2
    return -1


def correct_display_choice(human_ranking: Any, random_number: int) -> int:
    ranking = np.asarray(human_ranking)
    chosen_original_idx = int(np.where(ranking == 0)[0][0])
    display_to_original = [0, 1] if random_number == 0 else [1, 0]
    return display_to_original.index(chosen_original_idx) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/private_data/models/reward-model-exam/OpenGVLab/InternVL2_5-2B")
    parser.add_argument(
        "--data-path",
        default="/root/private_data/datasets/reward-model-exam/VL-RewardBench/data/test-00000-of-00001.parquet",
    )
    parser.add_argument("--output", default="/root/private_data/projects/reward model/outputs/eval/base_sanity_internvl2_5_2b.jsonl")
    parser.add_argument("--summary", default="")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-tiles", type=int, default=6)
    parser.add_argument("--short-output", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading data: {args.data_path}", flush=True)
    df = pd.read_parquet(args.data_path)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    print(f"loading tokenizer/model: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        local_files_only=True,
    ).eval().cuda()

    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
    correct = 0
    parsed = 0
    records = []
    start = time.time()

    with output_path.open("w", encoding="utf-8") as f:
        for local_idx, (_, row) in enumerate(df.iterrows(), start=1):
            item = row.to_dict()
            random_number = random.choice([0, 1])
            prompt, displayed_answers = build_official_prompt(item, random_number, short_output=args.short_output)
            question = "<image>\n" + prompt
            pixel_values = load_image_from_bytes(
                item["image"]["bytes"],
                max_num=args.max_tiles,
            ).to(torch.bfloat16).cuda()
            with torch.inference_mode():
                response = model.chat(tokenizer, pixel_values, question, generation_config)
            pred = parse_choice(response)
            target = correct_display_choice(item["human_ranking"], random_number)
            is_correct = pred == target
            if pred != -1:
                parsed += 1
            correct += int(is_correct)
            record = {
                "id": item["id"],
                "query": item["query"],
                "response": list(item["response"]),
                "displayed_answers": displayed_answers,
                "human_ranking": np.asarray(item["human_ranking"]).tolist(),
                "random_number": random_number,
                "target_choice": target,
                "pred_choice": pred,
                "correct": is_correct,
                "model_output": response,
                "query_source": item.get("query_source", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)
            print(
                f"[{local_idx}/{len(df)}] id={item['id']} pred={pred} target={target} correct={is_correct}",
                flush=True,
            )

    elapsed = time.time() - start
    accuracy = correct / max(1, len(df))
    by_source = {}
    for record in records:
        source = str(record.get("query_source", ""))
        item = by_source.setdefault(source, {"n": 0, "correct": 0})
        item["n"] += 1
        item["correct"] += int(bool(record["correct"]))
    for item in by_source.values():
        item["accuracy"] = item["correct"] / max(1, item["n"])
        del item["correct"]
    summary = {
        "n": len(records),
        "parsed": parsed,
        "parse_rate": parsed / max(1, len(records)),
        "accuracy": accuracy,
        "model_path": args.model_path,
        "short_output": args.short_output,
        "max_new_tokens": args.max_new_tokens,
        "max_tiles": args.max_tiles,
        "elapsed_sec": elapsed,
        "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "by_query_source": by_source,
    }
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"parsed={parsed}/{len(df)} accuracy={accuracy:.4f} elapsed_sec={elapsed:.1f}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {output_path}", flush=True)
    if args.summary:
        print(f"summary {args.summary}", flush=True)
    print(f"cuda_max_mem_gb={summary['cuda_max_mem_gb']:.3f}", flush=True)


if __name__ == "__main__":
    main()
