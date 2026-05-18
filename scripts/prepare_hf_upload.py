from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints/lora_v1_1k")
    parser.add_argument("--model-card", default="docs/HF_MODEL_CARD_DRAFT.md")
    parser.add_argument("--output-dir", default="outputs/hf_upload/internvl2-5-2b-vl-reward-model")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_tree(checkpoint_dir / "lora_final", output_dir / "lora_final")
    copy_file(checkpoint_dir / "score_head_final.pt", output_dir / "score_head_final.pt")
    copy_file(checkpoint_dir / "training_meta.json", output_dir / "training_meta.json")
    copy_file(Path(args.model_card), output_dir / "README.md")

    config = {
        "base_model": "OpenGVLab/InternVL2_5-2B",
        "reward_model_type": "internvl2_5_2b_lora_score_head",
        "score_head": "score_head_final.pt",
        "lora_adapter": "lora_final",
        "pooling": "final_valid_token",
        "max_tiles": 2,
        "training_data": "trl-lib/rlaif-v first 1024 preference pairs",
        "benchmark": "MMInstruction/VL-RewardBench",
        "vlrewardbench_full_accuracy": 0.7465918203688853,
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
