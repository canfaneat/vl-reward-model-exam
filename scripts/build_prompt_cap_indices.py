from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def message_text(message_list: Any) -> str:
    message = message_list[0] if isinstance(message_list, (list, tuple)) or hasattr(message_list, "__len__") else message_list
    content = message["content"]
    parts = content.tolist() if hasattr(content, "tolist") else list(content)
    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
            texts.append(str(part["text"]))
    return "\n".join(texts).strip()


def norm_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def read_indices(path_text: str) -> set[int]:
    if not path_text:
        return set()
    path = Path(path_text)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    return {int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_rows(data_dir: Path, candidate_limit: int) -> pd.DataFrame:
    frames = []
    offset = 0
    remaining = candidate_limit
    for file in sorted((data_dir / "data").glob("train-*.parquet")):
        df = pd.read_parquet(file)
        df = df.copy()
        df["_global_idx"] = range(offset, offset + len(df))
        offset += len(df)
        if remaining is not None:
            df = df.head(remaining)
            remaining -= len(df)
        frames.append(df)
        if remaining is not None and remaining <= 0:
            break
    return pd.concat(frames, ignore_index=True)


def bucket(idx: int, width: int = 512) -> str:
    start = (idx // width) * width
    return f"{start}-{start + width - 1}"


def write_txt(path: Path, indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(i) for i in indices) + ("\n" if indices else ""), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/private_data/datasets/reward-model-exam/rlaif-v")
    parser.add_argument("--candidate-limit", type=int, default=8192)
    parser.add_argument("--target-size", type=int, default=4096)
    parser.add_argument("--prompt-cap", type=int, required=True)
    parser.add_argument("--exclude-indices", default="", help="Optional RLAIF-V-only exclusion list, e.g. image near-duplicate indices.")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    exclude = read_indices(args.exclude_indices)
    df = load_rows(Path(args.data_dir), args.candidate_limit)

    selected = []
    skipped_excluded = 0
    skipped_prompt_cap = 0
    prompt_counts = defaultdict(int)
    selected_rows = []
    candidate_prompt_counter = Counter()

    for _, row in df.iterrows():
        idx = int(row["_global_idx"])
        prompt = message_text(row["prompt"])
        norm_prompt = norm_text(prompt)
        candidate_prompt_counter[norm_prompt] += 1
        if idx in exclude:
            skipped_excluded += 1
            continue
        if prompt_counts[norm_prompt] >= args.prompt_cap:
            skipped_prompt_cap += 1
            continue
        prompt_counts[norm_prompt] += 1
        selected.append(idx)
        selected_rows.append(
            {
                "global_idx": idx,
                "idx_bucket": bucket(idx),
                "prompt": prompt,
                "norm_prompt": norm_prompt,
                "prompt_words": len(prompt.split()),
                "chosen_len": len(message_text(row["chosen"])),
                "rejected_len": len(message_text(row["rejected"])),
            }
        )
        if len(selected) >= args.target_size:
            break

    if len(selected) < args.target_size:
        raise SystemExit(f"Only selected {len(selected)} rows; increase --candidate-limit or --prompt-cap.")

    top_selected = Counter(row["prompt"] for row in selected_rows).most_common(20)
    idx_counter = Counter(row["idx_bucket"] for row in selected_rows)
    summary = {
        "selection": f"prompt_cap_{args.prompt_cap}",
        "candidate_limit": args.candidate_limit,
        "target_size": args.target_size,
        "prompt_cap": args.prompt_cap,
        "exclude_indices": args.exclude_indices,
        "exclude_indices_n": len(exclude),
        "selected_n": len(selected),
        "min_idx": min(selected),
        "max_idx": max(selected),
        "mean_idx": sum(selected) / len(selected),
        "unique_prompt_n": len(set(row["norm_prompt"] for row in selected_rows)),
        "top20_prompt_mass": sum(count for _, count in top_selected) / len(selected_rows),
        "prompt_word_mean": sum(row["prompt_words"] for row in selected_rows) / len(selected_rows),
        "chosen_len_mean": sum(row["chosen_len"] for row in selected_rows) / len(selected_rows),
        "rejected_len_mean": sum(row["rejected_len"] for row in selected_rows) / len(selected_rows),
        "skipped_excluded_before_target": skipped_excluded,
        "skipped_prompt_cap_before_target": skipped_prompt_cap,
        "top_selected_prompts": top_selected,
        "index_bucket_counts": idx_counter.most_common(),
    }

    prefix = Path(args.output_prefix)
    write_txt(prefix.with_suffix(".indices.txt"), selected)
    write_csv(prefix.with_suffix(".rows.csv"), selected_rows)
    prefix.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
