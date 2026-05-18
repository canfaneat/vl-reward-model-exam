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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md_table(path: Path, rows: list[dict]) -> None:
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
                text = str(value).replace("\n", " ")
                values.append(text[:240])
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_by_source(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get("query_source") or "(empty)", []).append(row)
    out = []
    for source, chunk in groups.items():
        acc = mean(bool(r["correct"]) for r in chunk)
        out.append({"query_source": source, "n": len(chunk), "accuracy": acc, "accuracy_percent": acc * 100})
    out.sort(key=lambda row: (-row["n"], row["query_source"]))
    return out


def add_length_features(rows: list[dict]) -> None:
    for row in rows:
        response0 = row.get("response0", "")
        response1 = row.get("response1", "")
        row["response0_len"] = len(response0)
        row["response1_len"] = len(response1)
        row["selected_len"] = row["response0_len"] if int(row["pred"]) == 0 else row["response1_len"]
        row["target_len"] = row["response0_len"] if int(row["target"]) == 0 else row["response1_len"]
        row["selected_longer"] = row["selected_len"] > (
            row["response1_len"] if int(row["pred"]) == 0 else row["response0_len"]
        )
        row["target_longer"] = row["target_len"] > (
            row["response1_len"] if int(row["target"]) == 0 else row["response0_len"]
        )


def length_bias_summary(rows: list[dict]) -> list[dict]:
    add_length_features(rows)
    correct = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    groups = [("all", rows), ("correct", correct), ("wrong", wrong)]
    out = []
    for name, chunk in groups:
        if not chunk:
            continue
        out.append(
            {
                "group": name,
                "n": len(chunk),
                "selected_longer_rate": mean(bool(r["selected_longer"]) for r in chunk),
                "target_longer_rate": mean(bool(r["target_longer"]) for r in chunk),
                "selected_len_mean": mean(float(r["selected_len"]) for r in chunk),
                "target_len_mean": mean(float(r["target_len"]) for r in chunk),
                "score_gap_abs_mean": mean(abs(float(r.get("score_gap_0_minus_1", 0.0))) for r in chunk),
            }
        )
    return out


def wrong_examples(rows: list[dict], limit: int) -> list[dict]:
    wrong = [r for r in rows if not r["correct"]]
    wrong.sort(key=lambda r: abs(float(r.get("score_gap_0_minus_1", 0.0))), reverse=True)
    out = []
    for row in wrong[:limit]:
        pred = int(row["pred"])
        target = int(row["target"])
        out.append(
            {
                "id": row.get("id", ""),
                "query_source": row.get("query_source") or "(empty)",
                "target": target,
                "pred": pred,
                "score0": float(row.get("score0", 0.0)),
                "score1": float(row.get("score1", 0.0)),
                "score_gap_0_minus_1": float(row.get("score_gap_0_minus_1", 0.0)),
                "query": row.get("query", ""),
                "target_response": row.get(f"response{target}", ""),
                "pred_response": row.get(f"response{pred}", ""),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--wrong-limit", type=int, default=12)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.eval_jsonl))
    if not rows:
        raise SystemExit(f"No rows found in {args.eval_jsonl}")

    prefix = Path(args.output_prefix)
    by_source = summarize_by_source(rows)
    length_summary = length_bias_summary(rows)
    wrong = wrong_examples(rows, args.wrong_limit)

    write_csv(prefix.with_suffix(".by_source.csv"), by_source)
    write_md_table(prefix.with_suffix(".by_source.md"), by_source)
    write_csv(prefix.with_suffix(".length_bias.csv"), length_summary)
    write_md_table(prefix.with_suffix(".length_bias.md"), length_summary)
    write_csv(prefix.with_suffix(".wrong_examples.csv"), wrong)
    write_md_table(prefix.with_suffix(".wrong_examples.md"), wrong)
    print(json.dumps({"n": len(rows), "output_prefix": str(prefix)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
