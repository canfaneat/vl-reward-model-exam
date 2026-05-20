from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def message_text(message_list: Any) -> str:
    message = message_list[0] if isinstance(message_list, (list, tuple)) or hasattr(message_list, "__len__") else message_list
    content = message["content"]
    parts = content.tolist() if hasattr(content, "tolist") else list(content)
    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
            texts.append(str(part["text"]))
    return "\n".join(texts).strip()


def read_exclude(path: str) -> set[int]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return {int(line.strip()) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def select_indices(total: int, limit: int, exclude: set[int]) -> list[int]:
    selected = []
    for idx in range(total):
        if idx in exclude:
            continue
        selected.append(idx)
        if len(selected) >= limit:
            break
    return selected


def load_rows(data_dir: Path, total_needed: int) -> pd.DataFrame:
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
    return pd.concat(frames, ignore_index=True)


def image_quick_stats(image_obj: Any) -> dict[str, Any]:
    image0 = image_obj[0] if isinstance(image_obj, (list, tuple)) or hasattr(image_obj, "__len__") else image_obj
    blob = image0["bytes"]
    md5_prefix = hashlib.md5(blob).hexdigest()[:8]
    try:
        image = Image.open(io.BytesIO(blob))
        w, h = image.size
    except Exception:
        w, h = 0, 0
    return {"image_md5_prefix": md5_prefix, "image_w": w, "image_h": h, "image_aspect": round(w / h, 4) if h else 0}


def bucket(idx: int, width: int = 512) -> str:
    start = (idx // width) * width
    return f"{start}-{start + width - 1}"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = [r["prompt"] for r in records]
    chosen_lens = [r["chosen_len"] for r in records]
    rejected_lens = [r["rejected_len"] for r in records]
    idxs = [r["global_idx"] for r in records]
    prompt_counts = Counter(prompts)
    return {
        "n": len(records),
        "min_idx": min(idxs),
        "max_idx": max(idxs),
        "mean_idx": sum(idxs) / len(idxs),
        "unique_prompt_n": len(prompt_counts),
        "top_prompts": prompt_counts.most_common(12),
        "prompt_word_mean": sum(r["prompt_words"] for r in records) / len(records),
        "chosen_len_mean": sum(chosen_lens) / len(chosen_lens),
        "rejected_len_mean": sum(rejected_lens) / len(rejected_lens),
        "length_gap_mean": sum(r["chosen_len"] - r["rejected_len"] for r in records) / len(records),
        "index_bucket_counts": Counter(bucket(i) for i in idxs).most_common(),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/private_data/datasets/reward-model-exam/rlaif-v")
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument("--total-needed", type=int, default=8192)
    parser.add_argument("--strict-exclude", default="artifacts/report_assets/dedup_rlaifv8192_vlrb.union.exclude_indices.txt")
    parser.add_argument("--refined-exclude", default="artifacts/report_assets/dedup_rlaifv8192_vlrb.refined.exclude_indices.txt")
    parser.add_argument("--output-prefix", default="artifacts/report_assets/dedup_strategy_audit")
    args = parser.parse_args()

    data = load_rows(Path(args.data_dir), args.total_needed + 2048)
    selections = {
        "raw_first_4096": list(range(args.limit)),
        "strict_union_4096": select_indices(len(data), args.limit, read_exclude(args.strict_exclude)),
        "refined_4096": select_indices(len(data), args.limit, read_exclude(args.refined_exclude)),
    }
    selected_union = sorted(set().union(*[set(v) for v in selections.values()]))
    row_map = {int(row["_global_idx"]): row for _, row in data[data["_global_idx"].isin(selected_union)].iterrows()}

    summaries = {}
    rows = []
    for name, indices in selections.items():
        records = []
        for idx in indices:
            row = row_map[idx]
            prompt = message_text(row["prompt"])
            chosen = message_text(row["chosen"])
            rejected = message_text(row["rejected"])
            stat = image_quick_stats(row["images"])
            record = {
                "selection": name,
                "global_idx": idx,
                "idx_bucket": bucket(idx),
                "prompt": prompt,
                "prompt_words": len(prompt.split()),
                "chosen_len": len(chosen),
                "rejected_len": len(rejected),
                "length_gap": len(chosen) - len(rejected),
                **stat,
            }
            records.append(record)
            rows.append(record)
        summaries[name] = summarize(records)

    prefix = Path(args.output_prefix)
    write_csv(prefix.with_suffix(".rows.csv"), rows)
    prefix.with_suffix(".summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
