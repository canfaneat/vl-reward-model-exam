from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE_COLOR = "#7A7F87"
HEAD_COLOR = "#B7A57A"
LORA_COLOR = "#2F80ED"
ACCENT_COLOR = "#D95F02"
GRID_COLOR = "#E6E8EC"
TEXT_COLOR = "#222222"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9CDD3")
    ax.spines["bottom"].set_color("#C9CDD3")
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_overall(results_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(results_csv)
    lookup = {row["setting"]: float(row["accuracy_percent"]) for _, row in df.iterrows()}
    rows = [
        ("Base judge", lookup["base_internvl2_5_2b_vlrb_full_short"], BASE_COLOR),
        ("Head-only v0", lookup["reward_head_v0_128_vlrb_full"], HEAD_COLOR),
        ("LoRA v1", lookup["lora_v1_1k_vlrb_full"], LORA_COLOR),
    ]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.barh(labels, values, color=colors, height=0.56)
    ax.axvline(58.0, color=ACCENT_COLOR, linestyle="--", linewidth=1.4, label="Target 58.0")
    ax.axvline(64.8, color="#6A3D9A", linestyle=":", linewidth=1.6, label="Reference 64.8")
    for bar, value in zip(bars, values):
        ax.text(value + 1.0, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontsize=10)
    ax.set_xlim(0, 85)
    ax.set_xlabel("VLRewardBench accuracy (%)")
    ax.set_title("Overall Accuracy")
    style_axes(ax)
    ax.legend(loc="lower right", frameon=False)
    savefig(output_dir / "overall_accuracy.png")


def plot_source_grouped(delta_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(delta_csv)
    keep = [
        "POVID_preference_data_for_VLLMs",
        "(empty)",
        "wildvision-battle",
        "COCO",
        "LCS-558K",
        "coco",
        "VQAv2",
        "OK-VQA",
        "GQA",
        "vqav2",
    ]
    df = df[df["query_source"].isin(keep)].copy()
    df["order"] = df["query_source"].map({name: i for i, name in enumerate(keep)})
    df = df.sort_values("order", ascending=False)

    labels = df["query_source"].replace({"POVID_preference_data_for_VLLMs": "POVID"}).tolist()
    y = range(len(df))
    height = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.barh([i - height / 2 for i in y], df["v0_acc"] * 100, height=height, color=HEAD_COLOR, label="Head-only v0")
    ax.barh([i + height / 2 for i in y], df["lora_acc"] * 100, height=height, color=LORA_COLOR, label="LoRA v1")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Source-level Accuracy")
    style_axes(ax)
    ax.legend(loc="lower right", frameon=False)
    savefig(output_dir / "source_grouped_accuracy.png")


def plot_weighted_contribution(contribution_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(contribution_csv).head(12).copy()
    df = df.sort_values("overall_contribution_pp")
    labels = df["query_source"].replace({"POVID_preference_data_for_VLLMs": "POVID"}).tolist()
    values = df["overall_contribution_pp"].tolist()
    colors = [LORA_COLOR if value >= 0 else ACCENT_COLOR for value in values]

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    bars = ax.barh(labels, values, color=colors, height=0.55)
    for bar, value in zip(bars, values):
        x = value + (0.15 if value >= 0 else -0.15)
        ha = "left" if value >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.2f} pp", va="center", ha=ha, fontsize=9)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Contribution to overall gain (percentage points)")
    ax.set_title("Weighted Contribution by Source")
    style_axes(ax)
    savefig(output_dir / "weighted_contribution.png")


def plot_training_curve(windows_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(windows_csv)
    x = (df["start_step"] + df["end_step"]) / 2
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.8), sharex=True)

    axes[0].plot(x, df["loss_mean"], marker="o", color=ACCENT_COLOR, linewidth=1.8)
    axes[0].set_ylabel("Loss mean")
    axes[0].set_title("LoRA v1 Training Dynamics")
    axes[0].grid(axis="both", color=GRID_COLOR, linewidth=0.8)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(x, df["score_gap_mean"], marker="o", color=LORA_COLOR, linewidth=1.8)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Chosen - rejected")
    axes[1].set_xlabel("Training step")
    axes[1].grid(axis="both", color=GRID_COLOR, linewidth=0.8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    savefig(output_dir / "training_loss_gap.png")


def plot_length_bias(length_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(length_csv)
    labels = df["group"].tolist()
    x = range(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar([i - width / 2 for i in x], df["selected_longer_rate"] * 100, width=width, color=LORA_COLOR, label="Selected longer")
    ax.bar([i + width / 2 for i in x], df["target_longer_rate"] * 100, width=width, color=HEAD_COLOR, label="Target longer")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 65)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Length Bias Check")
    style_axes(ax)
    ax.legend(frameon=False)
    savefig(output_dir / "length_bias.png")


def write_manifest(output_dir: Path) -> None:
    files = sorted(str(path.relative_to(output_dir.parent)) for path in output_dir.glob("*.png"))
    manifest = {
        "output_dir": str(output_dir),
        "figures": files,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", default="artifacts/report_assets")
    parser.add_argument("--output-dir", default="artifacts/figures/report_v1")
    args = parser.parse_args()

    assets = Path(args.assets_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    plot_overall(assets / "results.csv", output_dir)
    plot_source_grouped(assets / "lora_v1_vs_v0_by_source.csv", output_dir)
    plot_weighted_contribution(assets / "lora_v1_vs_v0_weighted_contribution.csv", output_dir)
    plot_training_curve(assets / "lora_v1_1k_train.windows.csv", output_dir)
    plot_length_bias(assets / "lora_v1_1k_vlrb_full_analysis.length_bias.csv", output_dir)
    write_manifest(output_dir)
    print(json.dumps({"output_dir": str(output_dir), "figures": len(list(output_dir.glob('*.png')))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
