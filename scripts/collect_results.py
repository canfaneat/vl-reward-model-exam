from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_setting(path: Path) -> str:
    stem = path.stem
    for suffix in ["_summary", "_vlrb_full_summary", "_vlrb50_summary", "_vlrb10_summary"]:
        stem = stem.replace(suffix, "")
    return stem


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
        path.write_text("No results.\n", encoding="utf-8")
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
    parser.add_argument("--eval-dir", default="outputs/eval")
    parser.add_argument("--output-prefix", default="artifacts/report_assets/results")
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(Path(args.eval_dir).glob("*summary.json")):
        summary = read_json(summary_path)
        rows.append(
            {
                "setting": infer_setting(summary_path),
                "n": int(summary.get("n", 0)),
                "parsed": summary.get("parsed", ""),
                "parse_rate": summary.get("parse_rate", ""),
                "accuracy": float(summary.get("accuracy", 0.0)),
                "accuracy_percent": float(summary.get("accuracy", 0.0)) * 100,
                "max_tiles": summary.get("max_tiles", ""),
                "model_path": summary.get("model_path", ""),
                "short_output": summary.get("short_output", ""),
                "score_head": summary.get("score_head", ""),
                "lora_adapter": summary.get("lora_adapter", ""),
                "summary_path": str(summary_path),
            }
        )
    rows.sort(key=lambda row: (row["n"], row["accuracy"]), reverse=True)
    prefix = Path(args.output_prefix)
    write_csv(prefix.with_suffix(".csv"), rows)
    write_md(prefix.with_suffix(".md"), rows)
    print(json.dumps({"n": len(rows), "output_prefix": str(prefix)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
