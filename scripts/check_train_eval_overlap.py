from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


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


def md5_bytes(blob: bytes) -> str:
    return hashlib.md5(blob).hexdigest()


def dhash_bytes(image_bytes: bytes, hash_size: int = 8) -> int:
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((hash_size + 1, hash_size))
    pixels = list(image.getdata())
    bits = []
    for y in range(hash_size):
        row = pixels[y * (hash_size + 1) : (y + 1) * (hash_size + 1)]
        for x in range(hash_size):
            bits.append(1 if row[x] > row[x + 1] else 0)
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for key in headers:
            value = str(row[key]).replace("\n", " ")
            values.append(value[:240])
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="/root/private_data/datasets/reward-model-exam/rlaif-v")
    parser.add_argument(
        "--bench-path",
        default="/root/private_data/datasets/reward-model-exam/VL-RewardBench/data/test-00000-of-00001.parquet",
    )
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--output-prefix", default="artifacts/report_assets/overlap_rlaifv1024_vlrb")
    args = parser.parse_args()

    train_files = sorted((Path(args.train_dir) / "data").glob("train-*.parquet"))
    if not train_files:
        raise SystemExit(f"No train parquet files found under {args.train_dir}")

    train = pd.read_parquet(train_files[0]).head(args.train_limit)
    bench = pd.read_parquet(args.bench_path)

    bench_query = {}
    bench_md5 = {}
    bench_dhash = []
    for _, row in bench.iterrows():
        query = norm_text(row["query"])
        bench_query.setdefault(query, []).append(row)
        image_bytes = row["image"]["bytes"]
        bench_md5.setdefault(md5_bytes(image_bytes), []).append(row)
        bench_dhash.append(
            {
                "hash": dhash_bytes(image_bytes),
                "id": row["id"],
                "query_source": row.get("query_source", ""),
                "query": row["query"],
            }
        )

    query_overlaps = []
    unique_query_overlaps = set()
    image_md5_overlaps = []
    dhash_near = []

    for train_idx, row in train.iterrows():
        prompt = message_text(row["prompt"])
        train_query = norm_text(prompt)
        for bench_row in bench_query.get(train_query, []):
            unique_query_overlaps.add(train_query)
            query_overlaps.append(
                {
                    "train_idx": int(train_idx),
                    "bench_id": bench_row["id"],
                    "bench_query_source": bench_row.get("query_source", ""),
                    "query": prompt,
                }
            )

        images = row["images"]
        image0 = images[0] if isinstance(images, (list, tuple)) or hasattr(images, "__len__") else images
        image_bytes = image0["bytes"]
        train_md5 = md5_bytes(image_bytes)
        for bench_row in bench_md5.get(train_md5, []):
            image_md5_overlaps.append(
                {
                    "train_idx": int(train_idx),
                    "bench_id": bench_row["id"],
                    "bench_query_source": bench_row.get("query_source", ""),
                    "query": bench_row["query"],
                }
            )

        train_dhash = dhash_bytes(image_bytes)
        best = None
        for item in bench_dhash:
            dist = hamming(train_dhash, int(item["hash"]))
            if best is None or dist < best["hamming"]:
                best = {
                    "train_idx": int(train_idx),
                    "hamming": dist,
                    "bench_id": item["id"],
                    "bench_query_source": item["query_source"],
                    "bench_query": item["query"],
                    "train_prompt": prompt,
                }
        if best and best["hamming"] <= args.near_threshold:
            dhash_near.append(best)

    prefix = Path(args.output_prefix)
    write_csv(prefix.with_suffix(".query_overlap.csv"), query_overlaps)
    write_md(prefix.with_suffix(".query_overlap.md"), query_overlaps, "Exact Query Overlap")
    write_csv(prefix.with_suffix(".image_md5_overlap.csv"), image_md5_overlaps)
    write_md(prefix.with_suffix(".image_md5_overlap.md"), image_md5_overlaps, "Image MD5 Overlap")
    write_csv(prefix.with_suffix(".dhash_near.csv"), dhash_near)
    write_md(prefix.with_suffix(".dhash_near.md"), dhash_near, f"dHash Near Overlap <= {args.near_threshold}")

    summary = {
        "train_n": int(len(train)),
        "bench_n": int(len(bench)),
        "exact_query_overlap_pair_n": len(query_overlaps),
        "unique_exact_query_overlap_n": len(unique_query_overlaps),
        "image_md5_overlap_n": len(image_md5_overlaps),
        "dhash_near_threshold": args.near_threshold,
        "dhash_near_overlap_n": len(dhash_near),
        "outputs": {
            "query_overlap": str(prefix.with_suffix(".query_overlap.csv")),
            "image_md5_overlap": str(prefix.with_suffix(".image_md5_overlap.csv")),
            "dhash_near": str(prefix.with_suffix(".dhash_near.csv")),
        },
    }
    prefix.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
