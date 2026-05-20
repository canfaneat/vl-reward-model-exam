from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SKY = "#2DA8FF"
CYAN = "#21D4E8"
MINT = "#36D39A"
LILAC = "#A78BFA"
CORAL = "#FF7A90"
AMBER = "#FFD76A"
PALE_YELLOW = "#FFE89A"
LIME = "#A6E22E"
ROSE = "#F65AAD"
INDIGO = "#5E7CFF"
TEAL = "#16C7B7"
INK = "#27313F"
GRID = "#EAF0F6"
PANEL = "#F7FBFF"
TEXT = "#1F2A37"
MUTED = "#617086"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11.5,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.7,
        "figure.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
    }
)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.55)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7E3EF")
    ax.spines["bottom"].set_color("#D7E3EF")
    ax.tick_params(colors=TEXT)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor("#FFFFFF")


def annotate_barh(ax, bars, fmt="{:.2f}%", dx=0.65, fontsize=8.5) -> None:
    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + dx,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            ha="left",
            va="center",
            fontsize=fontsize,
            color=TEXT,
        )


def soften_axis(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color("#D7E3EF")
    ax.tick_params(colors=TEXT)


def metric_legend(ax, *args, loc="lower right", **kwargs) -> None:
    default = dict(
        frameon=True,
        framealpha=0.96,
        facecolor="#FFFFFF",
        edgecolor="#DCEBF8",
        handlelength=1.35,
        handletextpad=0.45,
        borderpad=0.45,
        labelspacing=0.35,
        columnspacing=0.8,
    )
    default.update(kwargs)
    ax.legend(*args, loc=loc, **default)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["accuracy_percent"] = pd.to_numeric(df["accuracy_percent"], errors="coerce")
    return df


def plot_accuracy_vs_prompt_mass(df: pd.DataFrame, output_dir: Path) -> None:
    keep = df.dropna(subset=["accuracy_percent"]).copy()
    label_map = {
        "raw_first_1024": "Raw 1k",
        "strict_query_image_4096": "Strict query+image",
        "image_only_refined_4096": "Image-only",
        "promptcap20_imageonly_4096": "Cap20 + image",
        "promptcap50_nobench_4096": "PromptCap50",
        "promptcap20_nobench_4096": "PromptCap20",
        "promptcap10_nobench_4096": "PromptCap10",
    }
    colors = {
        "raw_first_1024": CORAL,
        "strict_query_image_4096": LILAC,
        "image_only_refined_4096": INDIGO,
        "promptcap20_imageonly_4096": AMBER,
        "promptcap50_nobench_4096": SKY,
        "promptcap20_nobench_4096": ROSE,
        "promptcap10_nobench_4096": CYAN,
    }
    fig, ax = plt.subplots(figsize=(7.15, 3.85))
    ax.set_facecolor("#FFFFFF")
    for _, row in keep.iterrows():
        name = row["selection"]
        is_main = name == "promptcap50_nobench_4096"
        point = ax.scatter(
            row["top20_prompt_mass"] * 100,
            row["accuracy_percent"],
            s=152 if is_main else 102,
            color=colors.get(name, SKY),
            edgecolor="white",
            linewidth=1.7,
            alpha=0.96,
            label=label_map.get(name, name),
            zorder=4 if is_main else 3,
        )
        ax.text(
            row["top20_prompt_mass"] * 100 + 0.45,
            row["accuracy_percent"] + 0.32,
            f"{row['accuracy_percent']:.1f}",
            fontsize=8.6,
            color=INK if is_main else TEXT,
            weight="bold" if is_main else "normal",
        )
    ax.axhspan(70, 75.8, color="#FFF8DA", alpha=0.72, zorder=0)
    ax.axvspan(0, 25, color="#F0FFF9", alpha=0.5, zorder=0)
    ax.set_xlabel("Top-20 prompt mass (%)")
    ax.set_ylabel("VLRewardBench acc. (%)")
    ax.set_title("Prompt concentration and reward accuracy", color=INK, pad=6)
    ax.set_ylim(48, 77)
    ax.set_xlim(-1, 49)
    style(ax)
    metric_legend(
        ax,
        loc="lower right",
        ncol=2,
    )
    savefig(output_dir / "prompt_concentration_vs_accuracy.png")


def plot_promptcap_curve(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for cap, name in [
        (50, "promptcap50_nobench_4096"),
        (20, "promptcap20_nobench_4096"),
        (10, "promptcap10_nobench_4096"),
    ]:
        match = df[df["selection"] == name]
        if not match.empty and pd.notna(match.iloc[0]["accuracy_percent"]):
            row = match.iloc[0].copy()
            row["cap"] = cap
            rows.append(row)
    cap_df = pd.DataFrame(rows).sort_values("cap", ascending=False)
    fig, ax1 = plt.subplots(figsize=(6.85, 3.75))
    ax1.set_facecolor("#FFFFFF")
    ax1.axhspan(70, 73.5, color="#FFF8DA", alpha=0.76, zorder=0)
    line1 = ax1.plot(
        cap_df["cap"],
        cap_df["accuracy_percent"],
        marker="o",
        color=SKY,
        linewidth=2.4,
        markersize=7,
        label="Accuracy",
    )
    for _, row in cap_df.iterrows():
        ax1.text(
            row["cap"],
            row["accuracy_percent"] + 0.48,
            f"{row['accuracy_percent']:.2f}%",
            ha="center",
            fontsize=8.7,
            color=INK,
            weight="bold" if row["cap"] == 50 else "normal",
        )
    ax1.set_xlabel("Prompt cap")
    ax1.set_ylabel("Accuracy (%)", color=SKY)
    ax1.tick_params(axis="y", labelcolor=SKY)
    ax1.set_xticks(cap_df["cap"].tolist())
    ax1.set_ylim(55, 75)
    ax1.grid(axis="y", color=GRID, linewidth=0.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        cap_df["cap"],
        cap_df["effective_prompt_n"],
        marker="s",
        color=TEAL,
        linewidth=2.0,
        markersize=6,
        label="Effective prompts",
    )
    for _, row in cap_df.iterrows():
        ax2.text(
            row["cap"],
            row["effective_prompt_n"] + 92,
            f"{row['effective_prompt_n']:.0f}",
            ha="center",
            fontsize=8.1,
            color=TEAL,
        )
    ax2.set_ylabel("Effective prompt count", color=TEAL)
    ax2.tick_params(axis="y", labelcolor=TEAL)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color("#D7E3EF")
    ax1.set_title("PromptCap strength curve", color=INK, pad=6)
    lines = line1 + line2
    metric_legend(
        ax1,
        lines,
        [line.get_label() for line in lines],
        loc="lower right",
    )
    savefig(output_dir / "promptcap_strength_curve.png")


def plot_source_accuracy(output_dir: Path) -> None:
    sources = ["POVID_preference_data_for_VLLMs", "", "wildvision-battle", "COCO", "OK-VQA", "VQAv2"]
    labels = ["POVID", "empty", "wildvision", "COCO", "OK-VQA", "VQAv2"]
    summaries = {
        "Raw 1k": "outputs/eval/lora_v1_1k_vlrb_full_summary.json",
        "PromptCap50": "outputs/eval/D_PromptCap50NoBench_4k_Linear_vlrb_full_summary.json",
        "PromptCap10": "outputs/eval/D_PromptCap10NoBench_4k_Linear_vlrb_full_summary.json",
        "PromptCap20": "outputs/eval/D_PromptCap20NoBench_4k_Linear_vlrb_full_summary.json",
    }
    import json

    rows = []
    for label, path_text in summaries.items():
        path = Path(path_text)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))["by_query_source"]
        for src, src_label in zip(sources, labels):
            item = data.get(src)
            if item:
                rows.append({"model": label, "source": src_label, "accuracy": item["accuracy"] * 100})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="source", columns="model", values="accuracy").loc[labels]

    fig, ax = plt.subplots(figsize=(8.15, 4.05))
    x = range(len(pivot))
    width = 0.17
    model_order = [col for col in ["Raw 1k", "PromptCap50", "PromptCap10", "PromptCap20"] if col in pivot.columns]
    colors = [CORAL, SKY, AMBER, ROSE]
    for offset, (model, color) in enumerate(zip(model_order, colors)):
        xs = [i + (offset - (len(model_order) - 1) / 2) * width for i in x]
        bars = ax.bar(xs, pivot[model], width=width, label=model, color=color, alpha=0.94)
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.25,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=6.75,
                color=INK if model == "PromptCap50" else TEXT,
                weight="bold" if model == "PromptCap50" else "normal",
                rotation=0,
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Source-level accuracy under data selections", color=INK, pad=6)
    ax.set_ylim(0, 105)
    style(ax)
    metric_legend(
        ax,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    savefig(output_dir / "data_selection_source_accuracy.png")


def plot_benchmark_style_table(output_dir: Path) -> None:
    rows = [
        ("Base InternVL2.5-2B", "generative judge", 46.51, "Score_base"),
        ("Head-only v0", "128 pairs", 47.79, "sanity"),
        ("Reference in task doc", "InternVL2.5-2B-Reward", 64.80, "reference"),
        ("Strict query+image", "4k pairs", 70.17, "audit"),
        ("PromptCap50", "4k pairs", 71.69, "main"),
        ("Raw RLAIF-V 1k", "1k pairs", 74.66, "in-domain"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 3.05))
    ax.axis("off")
    ax.set_facecolor("#FFFFFF")

    columns = ["Setting", "Data / role", "Acc.", "Note"]
    table_data = [[name, role, f"{score:.2f}%", note] for name, role, score, note in rows]
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.35, 0.27, 0.13, 0.17],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 1.14)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CFE7F8")
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor("#E5F7FF")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 else "#F8FBFE")
            if rows[r - 1][3] == "main":
                cell.set_facecolor("#DFFBF3")
                cell.set_text_props(weight="bold", color=INK)
            elif rows[r - 1][3] == "reference":
                cell.set_facecolor("#FFF3BE")
            elif rows[r - 1][3] == "Score_base":
                cell.set_facecolor("#EDF7FF")
    ax.set_title("VLRewardBench result table", fontsize=11.5, color=INK, pad=5)
    savefig(output_dir / "benchmark_style_table.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="artifacts/report_assets/data_similarity_diversity.summary.csv")
    parser.add_argument("--output-dir", default="artifacts/figures/data_centric")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    df = load_data(Path(args.summary))
    plot_accuracy_vs_prompt_mass(df, output_dir)
    plot_promptcap_curve(df, output_dir)
    plot_source_accuracy(output_dir)
    plot_benchmark_style_table(output_dir)
    print({"output_dir": str(output_dir), "figures": len(list(output_dir.glob("*.png")))})


if __name__ == "__main__":
    main()
