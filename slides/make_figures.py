"""Draw the architecture diagram (R6) and, if results exist, the results chart.

    python slides/make_figures.py   ->  slides/figures/architecture.png, results_chart.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "slides" / "figures"

C_UI, C_PRE, C_MODEL, C_POST, C_OFF, C_CMP = "#DCE8F7", "#E3F1E3", "#FCE3C9", "#EDE3F5", "#EFEFEF", "#FFF6D6"
EDGE = "#3A3A3A"


def box(ax, x, y, w, h, title, body="", color="#FFF", fs=9.5, bold_color="#1B1B1B"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=EDGE, lw=1.1))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=fs + 0.5,
            fontweight="bold", color=bold_color)
    if body:
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=fs - 1.3,
                color="#333", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, text="", style="-|>", ls="-", text_dy=0.12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                 lw=1.2, color=EDGE, linestyle=ls))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + text_dy, text, ha="center", va="bottom",
                fontsize=7.5, color="#555", style="italic")


def architecture():
    fig, ax = plt.subplots(figsize=(13.3, 6.6), dpi=200)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # run-time region
    ax.add_patch(FancyBboxPatch((0.15, 3.05), 19.7, 6.8, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="none", ec="#7A9CC6", lw=1.2, ls="--"))
    ax.text(0.4, 3.45, "RUN TIME · local laptop / server, CPU only  (python app.py)", fontsize=8.5,
            color="#3D6496", fontweight="bold", va="top")

    box(ax, 0.4, 6.0, 2.7, 2.6, "User", "types a message\n(Turkish / English)\nin the browser", C_UI)
    box(ax, 3.6, 6.0, 2.9, 2.6, "Gradio UI", "app.py\ntext box, routed team,\ntop-5 intents,\nmodel comparison table", C_UI)
    box(ax, 7.0, 5.7, 3.6, 3.2, "Preprocessing", "src/text_utils.py\n• regex normalization\n(URL, e-mail, repeats)\n"
        "• model-code regex DW-6534\n• error-code regex E21\n• edit-distance spelling\ncorrection (product lexicon)",
        C_PRE)
    box(ax, 11.1, 7.2, 3.3, 1.9, "ELIZA rules", "src/eliza.py\ngreeting / small talk\n→ rule-based reply", C_PRE)
    box(ax, 11.1, 3.5, 3.3, 3.3, "Primary model", "DistilmBERT student\n(distilled from mBERT)\n"
        "INT8 dynamic quantization\nPyTorch, CPU\nWordPiece tokenizer\n→ softmax over 157 classes", C_MODEL)
    box(ax, 15.0, 5.5, 4.6, 3.6, "Post-processing", "src/predictors.py · src/routing.py\n"
        "• confidence < τ or 'oos'\n  → Human agent\n• intent → domain → team\n• small-talk domain → ELIZA\n"
        "τ tuned on validation set", C_POST)
    box(ax, 15.0, 3.5, 4.6, 1.7, "Comparison (optional)", "TF-IDF+LR (FP64) · fastText+FFN (FP32)\n"
        "mBERT teacher (FP32) · student (FP32)", C_CMP, fs=9)

    arrow(ax, 3.1, 7.3, 3.6, 7.3)
    arrow(ax, 6.5, 7.3, 7.0, 7.3)
    arrow(ax, 10.6, 8.1, 11.1, 8.1)
    arrow(ax, 10.6, 6.2, 11.1, 5.6)
    arrow(ax, 14.4, 5.5, 15.0, 6.3)
    arrow(ax, 14.4, 8.1, 15.0, 8.1)
    arrow(ax, 12.75, 3.5, 15.0, 4.3, style="-", ls=":")
    # output back to UI (drawn above the boxes)
    ax.plot([17.3, 17.3, 5.05, 5.05], [9.1, 9.45, 9.45, 8.75], color=EDGE, lw=1.2)
    arrow(ax, 5.05, 8.8, 5.05, 8.6)
    ax.text(11.2, 9.5, "routed team · intent · confidence · top-5 · bot reply  → shown in the UI",
            fontsize=8, color="#555", style="italic", ha="center", va="bottom")
    # offline training region
    ax.add_patch(FancyBboxPatch((0.15, 0.15), 19.7, 2.55, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="none", ec="#999", lw=1.2, ls="--"))
    ax.text(0.4, 2.55, "OFFLINE TRAINING · Google Colab (T4 GPU)", fontsize=8.5, color="#666",
            fontweight="bold", va="top")
    box(ax, 0.5, 0.35, 3.6, 1.8, "Data", "CLINC150 oos_plus (EN)\n+ 96 hand-written TR/EN\nappliance messages", C_OFF, fs=9)
    box(ax, 4.7, 0.35, 4.6, 1.8, "Training (src/*.py)", "TF-IDF+LR · fastText (skip-gram) + FFN\n"
        "mBERT fine-tuning → KD → student", C_OFF, fs=9)
    box(ax, 9.9, 0.35, 3.6, 1.8, "Evaluation", "same held-out test set\naccuracy, OOS recall,\nsize, latency",
        C_OFF, fs=9)
    box(ax, 14.1, 0.35, 5.4, 1.8, "Hugging Face Hub", "weights (FP32 student, teacher, FFN,\nbaseline)"
        " → downloaded into models/\nat start-up; INT8 made on load", C_OFF, fs=9)
    arrow(ax, 4.1, 1.25, 4.7, 1.25)
    arrow(ax, 9.3, 1.25, 9.9, 1.25)
    arrow(ax, 13.5, 1.25, 14.1, 1.25)
    arrow(ax, 16.8, 2.15, 16.8, 3.5, "models/", ls="--")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "architecture.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / "architecture.png")


def results_chart():
    p = ROOT / "results" / "results.json"
    if not p.exists():
        print("no results/results.json yet - skipping results chart")
        return
    rows = json.loads(p.read_text())["models"]
    names = [r["model"].replace(" (", "\n(").replace(" + INT8", "\n+ INT8") for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=200)
    for ax, key, title in zip(axes, ["accuracy", "size_mb", "latency_ms_mean"],
                              ["Accuracy (test)", "Size (MB)", "Latency (ms, CPU)"]):
        vals = [r[key] for r in rows]
        bars = ax.bar(range(len(rows)), vals, color=["#9BB7D4", "#9BB7D4", "#E8A86B", "#E8A86B", "#C0504D"])
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, fontsize=6.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}" if key == "accuracy" else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "results_chart.png", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / "results_chart.png")


if __name__ == "__main__":
    architecture()
    results_chart()
