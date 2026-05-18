from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_window(rows: list[dict], start: int, end: int) -> dict:
    chunk = rows[start:end]
    if not chunk:
        return {}
    gaps = [float(r["score_gap"]) for r in chunk]
    losses = [float(r["loss"]) for r in chunk]
    return {
        "start_step": int(chunk[0]["step"]),
        "end_step": int(chunk[-1]["step"]),
        "n": len(chunk),
        "loss_mean": mean(losses),
        "score_gap_mean": mean(gaps),
        "gap_positive_rate": sum(g > 0 for g in gaps) / len(gaps),
        "score_chosen_mean": mean(float(r["score_chosen"]) for r in chunk),
        "score_rejected_mean": mean(float(r["score_rejected"]) for r in chunk),
        "chosen_len_mean": mean(float(r["chosen_len"]) for r in chunk),
        "rejected_len_mean": mean(float(r["rejected_len"]) for r in chunk),
    }


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
    parser.add_argument("--log", required=True, help="Training JSONL log path.")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    rows = read_jsonl(log_path)
    if not rows:
        raise SystemExit(f"No rows found in {log_path}")

    summaries = []
    for start in range(0, len(rows), args.window):
        item = summarize_window(rows, start, min(start + args.window, len(rows)))
        if item:
            summaries.append(item)

    first = summarize_window(rows, 0, min(args.window, len(rows)))
    last = summarize_window(rows, max(0, len(rows) - args.window), len(rows))
    overall = summarize_window(rows, 0, len(rows))
    compact = []
    for label, item in [("first_window", first), ("last_window", last), ("overall", overall)]:
        row = {"window": label}
        row.update(item)
        compact.append(row)

    prefix = Path(args.output_prefix)
    write_csv(prefix.with_suffix(".windows.csv"), summaries)
    write_md(prefix.with_suffix(".windows.md"), summaries)
    write_csv(prefix.with_suffix(".summary.csv"), compact)
    write_md(prefix.with_suffix(".summary.md"), compact)
    print(json.dumps({"n": len(rows), "window": args.window, "output_prefix": str(prefix)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
