from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_txt(path: Path, indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(i) for i in indices) + ("\n" if indices else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-overlap", required=True)
    parser.add_argument("--dhash-near", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--train-limit", type=int, required=True)
    parser.add_argument("--min-query-words", type=int, default=16)
    parser.add_argument("--max-query-train-frequency", type=int, default=3)
    args = parser.parse_args()

    query_rows = read_rows(Path(args.query_overlap))
    dhash_rows = read_rows(Path(args.dhash_near))

    query_train_freq = Counter(row.get("query", "") for row in query_rows)
    query_indices = set()
    ignored_generic = set()
    for row in query_rows:
        query = row.get("query", "")
        train_idx = int(row["train_idx"])
        word_count = len(query.split())
        train_frequency = query_train_freq[query]
        if word_count >= args.min_query_words and train_frequency <= args.max_query_train_frequency:
            query_indices.add(train_idx)
        else:
            ignored_generic.add(train_idx)

    dhash_indices = {int(row["train_idx"]) for row in dhash_rows}
    exclude = sorted(query_indices | dhash_indices)

    prefix = Path(args.output_prefix)
    txt_path = prefix.with_suffix(".refined.exclude_indices.txt")
    json_path = prefix.with_suffix(".refined.summary.json")
    write_txt(txt_path, exclude)
    summary = {
        "mode": "refined_query_plus_dhash",
        "train_limit": args.train_limit,
        "min_query_words": args.min_query_words,
        "max_query_train_frequency": args.max_query_train_frequency,
        "raw_query_overlap_train_idx": len({int(row["train_idx"]) for row in query_rows}),
        "kept_generic_query_overlap_train_idx": len(ignored_generic - query_indices),
        "refined_query_exclude_train_idx": len(query_indices),
        "dhash_near_unique_train_idx": len(dhash_indices),
        "exclude_indices_n": len(exclude),
        "kept_after_exclude": args.train_limit - len([i for i in exclude if 0 <= i < args.train_limit]),
        "exclude_indices_path": str(txt_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
