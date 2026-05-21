from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def adjusted_pred(row: dict, alpha: float) -> int:
    len0 = max(1, len(row["response0"]))
    len1 = max(1, len(row["response1"]))
    score0 = float(row["score0"]) - alpha * math.log(len0)
    score1 = float(row["score1"]) - alpha * math.log(len1)
    return 0 if score0 >= score1 else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--alphas", default="-0.20,-0.10,-0.05,-0.02,0,0.02,0.05,0.10,0.20")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.eval_jsonl))
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    out_rows = []
    for alpha in alphas:
        correct = 0
        pred_longer = 0
        for row in rows:
            pred = adjusted_pred(row, alpha)
            correct += int(pred == int(row["target"]))
            len_pred = len(row["response0"]) if pred == 0 else len(row["response1"])
            len_other = len(row["response1"]) if pred == 0 else len(row["response0"])
            pred_longer += int(len_pred > len_other)
        out_rows.append(
            {
                "alpha": alpha,
                "n": len(rows),
                "accuracy": correct / len(rows),
                "accuracy_percent": 100 * correct / len(rows),
                "pred_longer_rate": pred_longer / len(rows),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(json.dumps(out_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
