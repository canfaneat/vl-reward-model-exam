from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No rows.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for key in headers:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--top-k", type=int, default=0, help="Keep all rows when <=0.")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    overview = [
        {"item": "n", "value": summary.get("n")},
        {"item": "parsed", "value": summary.get("parsed", "")},
        {"item": "parse_rate", "value": summary.get("parse_rate", "")},
        {"item": "accuracy", "value": float(summary.get("accuracy", 0.0))},
        {"item": "accuracy_percent", "value": float(summary.get("accuracy", 0.0)) * 100},
        {"item": "elapsed_sec", "value": float(summary.get("elapsed_sec", 0.0))},
        {"item": "cuda_max_mem_gb", "value": float(summary.get("cuda_max_mem_gb", 0.0))},
        {"item": "model_path", "value": summary.get("model_path", "")},
        {"item": "short_output", "value": summary.get("short_output", "")},
        {"item": "max_new_tokens", "value": summary.get("max_new_tokens", "")},
        {"item": "score_head", "value": summary.get("score_head", "")},
        {"item": "lora_adapter", "value": summary.get("lora_adapter", "")},
    ]
    by_source = []
    for source, item in summary.get("by_query_source", {}).items():
        n = int(item["n"])
        acc = float(item["accuracy"])
        by_source.append(
            {
                "query_source": source or "(empty)",
                "n": n,
                "accuracy": acc,
                "accuracy_percent": acc * 100,
            }
        )
    by_source.sort(key=lambda row: (-row["n"], row["query_source"]))
    if args.top_k > 0:
        by_source = by_source[: args.top_k]

    prefix = Path(args.output_prefix)
    write_csv(prefix.with_suffix(".overview.csv"), overview)
    write_md(prefix.with_suffix(".overview.md"), overview)
    write_csv(prefix.with_suffix(".by_source.csv"), by_source)
    write_md(prefix.with_suffix(".by_source.md"), by_source)
    print(json.dumps({"summary": str(summary_path), "output_prefix": str(prefix)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
