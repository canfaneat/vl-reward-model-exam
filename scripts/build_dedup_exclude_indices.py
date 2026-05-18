from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_indices(path: Path) -> set[int]:
    indices: set[int] = set()
    if not path.exists() or path.stat().st_size == 0:
        return indices
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "train_idx" not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain a train_idx column")
        for row in reader:
            indices.add(int(row["train_idx"]))
    return indices


def write_txt(path: Path, indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(i) for i in indices) + ("\n" if indices else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-overlap", default="artifacts/report_assets/overlap_rlaifv1024_vlrb.query_overlap.csv")
    parser.add_argument("--dhash-near", default="artifacts/report_assets/overlap_rlaifv1024_vlrb.dhash_near.csv")
    parser.add_argument("--mode", choices=["query", "dhash", "union"], default="union")
    parser.add_argument("--output-prefix", default="artifacts/report_assets/dedup_rlaifv1024_vlrb")
    args = parser.parse_args()

    query_indices = read_indices(Path(args.query_overlap))
    dhash_indices = read_indices(Path(args.dhash_near))
    if args.mode == "query":
        exclude = query_indices
    elif args.mode == "dhash":
        exclude = dhash_indices
    else:
        exclude = query_indices | dhash_indices

    sorted_indices = sorted(exclude)
    prefix = Path(args.output_prefix)
    txt_path = prefix.with_suffix(f".{args.mode}.exclude_indices.txt")
    json_path = prefix.with_suffix(f".{args.mode}.summary.json")
    write_txt(txt_path, sorted_indices)
    summary = {
        "mode": args.mode,
        "query_overlap_unique_train_idx": len(query_indices),
        "dhash_near_unique_train_idx": len(dhash_indices),
        "exclude_indices_n": len(sorted_indices),
        "kept_in_first_1024": 1024 - len([i for i in sorted_indices if 0 <= i < 1024]),
        "exclude_indices_path": str(txt_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
