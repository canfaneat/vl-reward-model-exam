from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def message_text(message_list: Any) -> str:
    message = message_list[0] if isinstance(message_list, (list, tuple)) or hasattr(message_list, "__len__") else message_list
    try:
        content = message["content"]
    except Exception:
        return ""
    parts = content.tolist() if hasattr(content, "tolist") else list(content)
    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
            texts.append(str(part["text"]))
    return "\n".join(texts).strip()


def norm_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def text_hash(text: str) -> str:
    return hashlib.sha1(norm_text(text).encode("utf-8")).hexdigest()


def read_indices(path_text: str) -> list[int]:
    if not path_text:
        return []
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(path)
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_train_idx_csv(path_text: str) -> set[int]:
    path = Path(path_text)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    indices: set[int] = set()
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "train_idx" not in (reader.fieldnames or []):
            return set()
        for row in reader:
            indices.add(int(row["train_idx"]))
    return indices


def select_after_exclude(limit: int, exclude: set[int], total: int) -> list[int]:
    selected: list[int] = []
    for idx in range(total):
        if idx in exclude:
            continue
        selected.append(idx)
        if len(selected) >= limit:
            break
    return selected


def load_train(data_dir: Path, total_needed: int) -> pd.DataFrame:
    frames = []
    offset = 0
    for file in sorted((data_dir / "data").glob("train-*.parquet")):
        df = pd.read_parquet(file)
        df = df.copy()
        df["_global_idx"] = range(offset, offset + len(df))
        offset += len(df)
        frames.append(df)
        if offset >= total_needed:
            break
    if not frames:
        raise FileNotFoundError(f"No parquet files under {data_dir / 'data'}")
    return pd.concat(frames, ignore_index=True)


def load_bench_response_hashes(path: Path) -> tuple[set[str], int]:
    df = pd.read_parquet(path)
    hashes: set[str] = set()
    n = 0
    for _, row in df.iterrows():
        for response in list(row["response"]):
            hashes.add(text_hash(response))
            n += 1
    return hashes, n


def entropy_effective_n(counts: Counter[str]) -> tuple[float, float]:
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy, math.exp(entropy)


def top_mass(counts: Counter[str], k: int) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(count for _, count in counts.most_common(k)) / total


def summary_for_selection(
    name: str,
    indices: list[int],
    row_map: dict[int, pd.Series],
    query_overlap_indices: set[int],
    dhash_near_indices: set[int],
    bench_response_hashes: set[str],
    accuracy: float | None = None,
) -> dict[str, Any]:
    prompts: list[str] = []
    chosen_lens: list[int] = []
    rejected_lens: list[int] = []
    response_exact_rows = 0
    for idx in indices:
        row = row_map[idx]
        prompt = norm_text(message_text(row["prompt"]))
        chosen = message_text(row["chosen"])
        rejected = message_text(row["rejected"])
        prompts.append(prompt)
        chosen_lens.append(len(chosen))
        rejected_lens.append(len(rejected))
        if text_hash(chosen) in bench_response_hashes or text_hash(rejected) in bench_response_hashes:
            response_exact_rows += 1

    counts = Counter(prompts)
    entropy, effective_n = entropy_effective_n(counts)
    selected_set = set(indices)
    return {
        "selection": name,
        "accuracy_percent": "" if accuracy is None else round(accuracy * 100, 4),
        "n": len(indices),
        "min_idx": min(indices),
        "max_idx": max(indices),
        "mean_idx": round(sum(indices) / len(indices), 2),
        "unique_prompt_n": len(counts),
        "unique_prompt_ratio": round(len(counts) / len(indices), 4),
        "prompt_entropy": round(entropy, 4),
        "effective_prompt_n": round(effective_n, 2),
        "top1_prompt_count": counts.most_common(1)[0][1],
        "top10_prompt_mass": round(top_mass(counts, 10), 4),
        "top20_prompt_mass": round(top_mass(counts, 20), 4),
        "query_overlap_train_idx_n": len(selected_set & query_overlap_indices),
        "query_overlap_train_idx_rate": round(len(selected_set & query_overlap_indices) / len(indices), 4),
        "dhash_near_train_idx_n": len(selected_set & dhash_near_indices),
        "dhash_near_train_idx_rate": round(len(selected_set & dhash_near_indices) / len(indices), 4),
        "response_exact_overlap_row_n": response_exact_rows,
        "response_exact_overlap_row_rate": round(response_exact_rows / len(indices), 4),
        "chosen_len_mean": round(sum(chosen_lens) / len(chosen_lens), 2),
        "rejected_len_mean": round(sum(rejected_lens) / len(rejected_lens), 2),
        "length_gap_mean": round(sum(c - r for c, r in zip(chosen_lens, rejected_lens)) / len(indices), 2),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_accuracy(path_text: str) -> float | None:
    path = Path(path_text)
    if not path.exists():
        return None
    return float(json.loads(path.read_text(encoding="utf-8")).get("accuracy", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/private_data/datasets/reward-model-exam/rlaif-v")
    parser.add_argument(
        "--bench-path",
        default="/root/private_data/datasets/reward-model-exam/VL-RewardBench/data/test-00000-of-00001.parquet",
    )
    parser.add_argument("--total-needed", type=int, default=8192)
    parser.add_argument("--output-prefix", default="artifacts/report_assets/data_similarity_diversity")
    args = parser.parse_args()

    query_overlap_indices = read_train_idx_csv("artifacts/report_assets/overlap_rlaifv8192_vlrb.query_overlap.csv")
    dhash_near_indices = read_train_idx_csv("artifacts/report_assets/overlap_rlaifv8192_vlrb.dhash_near.csv")
    strict_exclude = read_indices("artifacts/report_assets/dedup_rlaifv8192_vlrb.union.exclude_indices.txt")
    refined_exclude = read_indices("artifacts/report_assets/dedup_rlaifv8192_vlrb.refined.exclude_indices.txt")
    promptcap20 = read_indices("artifacts/report_assets/rlaifv_promptcap20_4k.indices.txt")
    promptcap20_nobench = read_indices("artifacts/report_assets/rlaifv_promptcap20_nobench_4k.indices.txt")
    promptcap50 = read_indices("artifacts/report_assets/rlaifv_promptcap50_4k.indices.txt")
    promptcap50_nobench = read_indices("artifacts/report_assets/rlaifv_promptcap50_nobench_4k.indices.txt")
    promptcap10_nobench = read_indices("artifacts/report_assets/rlaifv_promptcap10_nobench_4k.indices.txt")

    selections = {
        "raw_first_1024": list(range(1024)),
        "raw_first_4096": list(range(4096)),
        "strict_query_image_4096": select_after_exclude(4096, set(strict_exclude), args.total_needed + 2048),
        "image_only_refined_4096": select_after_exclude(4096, set(refined_exclude), args.total_needed + 2048),
        "promptcap50_imageonly_4096": promptcap50,
        "promptcap50_nobench_4096": promptcap50_nobench,
        "promptcap20_imageonly_4096": promptcap20,
        "promptcap20_nobench_4096": promptcap20_nobench,
        "promptcap10_nobench_4096": promptcap10_nobench,
    }
    accuracy_paths = {
        "raw_first_1024": "outputs/eval/lora_v1_1k_vlrb_full_summary.json",
        "strict_query_image_4096": "outputs/eval/lora_v3_dedup_4k_vlrb_full_summary.json",
        "image_only_refined_4096": "outputs/eval/lora_v4_refined_dedup_4k_vlrb_full_summary.json",
        "promptcap20_imageonly_4096": "outputs/eval/D_PromptCap20_4k_vlrb_full_summary.json",
        "promptcap20_nobench_4096": "outputs/eval/D_PromptCap20NoBench_4k_Linear_vlrb_full_summary.json",
        "promptcap50_nobench_4096": "outputs/eval/D_PromptCap50NoBench_4k_Linear_vlrb_full_summary.json",
        "promptcap10_nobench_4096": "outputs/eval/D_PromptCap10NoBench_4k_Linear_vlrb_full_summary.json",
    }

    max_idx = max(max(indices) for indices in selections.values())
    train = load_train(Path(args.data_dir), max(max_idx + 1, args.total_needed))
    row_map = {int(row["_global_idx"]): row for _, row in train.iterrows()}
    bench_response_hashes, bench_response_n = load_bench_response_hashes(Path(args.bench_path))

    rows = []
    for name, indices in selections.items():
        acc = read_accuracy(accuracy_paths[name]) if name in accuracy_paths else None
        rows.append(
            summary_for_selection(
                name,
                indices,
                row_map,
                query_overlap_indices,
                dhash_near_indices,
                bench_response_hashes,
                acc,
            )
        )

    prefix = Path(args.output_prefix)
    write_csv(prefix.with_suffix(".summary.csv"), rows)
    write_md(prefix.with_suffix(".summary.md"), rows)
    meta = {
        "query_overlap_unique_train_idx_8192": len(query_overlap_indices),
        "dhash_near_unique_train_idx_8192": len(dhash_near_indices),
        "bench_response_n": bench_response_n,
        "bench_unique_response_hash_n": len(bench_response_hashes),
        "output_csv": str(prefix.with_suffix(".summary.csv")),
        "output_md": str(prefix.with_suffix(".summary.md")),
    }
    prefix.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
