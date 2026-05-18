from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODEL_REPO = "OpenGVLab/InternVL2_5-2B"
BENCH_REPO = "MMInstruction/VL-RewardBench"
TRAINING_REPOS = [
    ("MMInstruction/VLFeedback", "dataset"),
    ("trl-lib/rlaif-v", "dataset"),
    ("openbmb/RLAIF-V-Dataset", "dataset"),
    ("openbmb/RLHF-V-Dataset", "dataset"),
]


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{stamp()}] {message}", flush=True)


def run_df(path: Path) -> None:
    try:
        out = subprocess.check_output(["df", "-h", str(path)], text=True)
        print(out.strip(), flush=True)
    except Exception as exc:
        log(f"df failed: {exc}")


def repo_tree_summary(repo_id: str, repo_type: str) -> dict:
    api = HfApi()
    total_size = 0
    files = []
    for item in api.list_repo_tree(repo_id=repo_id, repo_type=repo_type, recursive=True):
        if getattr(item, "type", None) != "file":
            continue
        size = getattr(item, "size", None) or 0
        total_size += size
        files.append({"path": item.path, "size": size})
    files.sort(key=lambda x: x["size"], reverse=True)
    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "file_count": len(files),
        "total_size_bytes": total_size,
        "total_size_gb": round(total_size / (1024**3), 3),
        "largest_files": files[:20],
    }


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {path}")


def download_snapshot(repo_id: str, repo_type: str, local_dir: Path, max_workers: int) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    log(f"start snapshot_download repo={repo_id} type={repo_type}")
    log(f"local_dir={local_dir}")
    start = time.time()
    result = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        max_workers=max_workers,
    )
    elapsed = time.time() - start
    log(f"finished repo={repo_id} elapsed={elapsed / 60:.1f} min result={result}")
    return Path(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["model", "benchmark", "training_repo_summaries"],
        choices=["model", "benchmark", "training_repo_summaries"],
    )
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    models_dir = Path(os.environ.get("REWARD_MODELS_DIR", "/root/private_data/models/reward-model-exam"))
    datasets_dir = Path(os.environ.get("REWARD_DATASETS_DIR", "/root/private_data/datasets/reward-model-exam"))
    outputs_dir = Path(os.environ.get("REWARD_OUTPUTS_DIR", "/root/private_data/projects/reward model/outputs"))
    logs_dir = outputs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log("download assets started")
    run_df(Path("/root/private_data"))

    if "training_repo_summaries" in args.assets:
        summaries = []
        for repo_id, repo_type in TRAINING_REPOS:
            try:
                log(f"inspect repo tree: {repo_id}")
                summaries.append(repo_tree_summary(repo_id, repo_type))
            except Exception as exc:
                summaries.append({"repo_id": repo_id, "repo_type": repo_type, "error": repr(exc)})
                log(f"inspect failed for {repo_id}: {exc}")
        save_json(logs_dir / "training_repo_summaries.json", summaries)

    if "benchmark" in args.assets:
        download_snapshot(
            repo_id=BENCH_REPO,
            repo_type="dataset",
            local_dir=datasets_dir / "VL-RewardBench",
            max_workers=args.max_workers,
        )

    if "model" in args.assets:
        download_snapshot(
            repo_id=MODEL_REPO,
            repo_type="model",
            local_dir=models_dir / "OpenGVLab" / "InternVL2_5-2B",
            max_workers=args.max_workers,
        )

    log("download assets finished")
    run_df(Path("/root/private_data"))
    for path in [models_dir, datasets_dir]:
        if path.exists():
            usage = shutil.disk_usage(path)
            log(f"disk usage for {path}: free={usage.free / (1024**3):.1f} GiB")


if __name__ == "__main__":
    main()

