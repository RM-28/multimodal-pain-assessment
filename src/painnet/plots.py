"""Figures for the report. Matplotlib, styled once so everything matches.

Design rules we're following (so the report looks like one document, not five):
  * confusion matrix  -> SINGLE-hue sequential ramp, light->dark. Not viridis,
    not jet -- cell counts are a magnitude, and a multi-hue map invents
    category boundaries that aren't in the data.
  * learning curves   -> 2 series (train/val), fixed blue/orange, 2px lines.
  * per-class F1      -> ONE series across 4 categories. The class names are
    already on the axis; colouring each bar differently would encode identity
    twice and buy nothing.
  * grid/axes recessive, values labelled directly, text in ink not series colour.

Palette is colourblind-checked (protan/deutan/tritan separation verified).
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import config

# --- palette ------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# single-hue blue ramp, light -> dark (steps 100..700)
_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
BLUES = LinearSegmentedColormap.from_list("painnet_blues", _BLUE_STEPS)


def use_style() -> None:
    """Call once at the top of a notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        # NB: "medium" isn't a real weight in DejaVu and makes matplotlib warn
        "axes.titleweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 2.0,      # 2px marks
        "lines.markersize": 8,       # >=8px markers
        "figure.dpi": 110,
    })


# --- confusion matrix ---------------------------------------------------------
def confusion_matrix(cm, title="Confusion matrix", normalize=False, ax=None):
    """cm: DataFrame from evaluate.confusion(), or a raw square array.

    Rows = true class, columns = predicted. Every cell is labelled, because a
    colour alone shouldn't be the only way to read a number.
    """
    labels = list(getattr(cm, "index", config.PAIN_CLASSES))
    vals = np.asarray(cm)
    if normalize:
        vals = vals / np.clip(vals.sum(axis=1, keepdims=True), 1, None)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.6))

    im = ax.imshow(vals, cmap=BLUES, vmin=0, vmax=vals.max() or 1)

    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    ax.grid(False)

    # direct labels; flip ink to white once the cell gets dark enough to need it
    thresh = (vals.max() or 1) * 0.55
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(
                j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                ha="center", va="center", fontsize=9,
                color="#ffffff" if v > thresh else INK,
            )

    cb = ax.figure.colorbar(im, ax=ax, fraction=0.045)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=GRID, labelcolor=INK_MUTED)
    return ax


# --- learning curves ----------------------------------------------------------
def learning_curves(history, metric="loss", title=None, ax=None):
    """history: the keras History object (or its .history dict)."""
    h = getattr(history, "history", history)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.6))

    epochs = range(1, len(h[metric]) + 1)
    ax.plot(epochs, h[metric], color=SERIES[0], label=f"train {metric}")
    if f"val_{metric}" in h:
        ax.plot(epochs, h[f"val_{metric}"], color=SERIES[1], label=f"val {metric}")

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Training vs validation {metric}")
    ax.grid(True, axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=9)   # 2 series -> legend always present
    return ax


# --- per-class F1 -------------------------------------------------------------
def per_class_f1(scores: dict, title="Per-class F1", ax=None):
    """scores: {'no_pain': 0.42, ...} or the dict from evaluate.per_class_f1().

    Horizontal bars, one colour. Sorted by value so the weak class is obvious --
    which, with 10 menstrual_pain subjects, it will be.
    """
    clean = {k.replace("f1_", ""): v for k, v in scores.items()}
    names = sorted(clean, key=clean.get)
    vals = [clean[n] for n in names]

    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 0.5 * len(names) + 1.6))

    ax.barh(names, vals, color=SERIES[0], height=0.62)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title(title)      # single series -> title names it, no legend
    ax.grid(True, axis="x", alpha=0.7)
    ax.set_axisbelow(True)

    for y, v in enumerate(vals):
        ax.text(v + 0.015, y, f"{v:.3f}", va="center", fontsize=9, color=INK_MUTED)
    return ax


# --- class balance ------------------------------------------------------------
def subject_balance(counts: dict | None = None, ax=None):
    """Subjects per class -- the plot that explains why macro-F1 is the metric.

    Deliberately counts SUBJECTS, not rows. Rows would suggest ~100k samples
    when the real sample size is 83.
    """
    counts = counts or config.SUBJECT_COUNTS
    names = sorted(counts, key=counts.get, reverse=True)
    vals = [counts[n] for n in names]

    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.2))

    ax.bar(names, vals, color=SERIES[0], width=0.62)
    ax.set_ylabel("subjects")
    ax.set_title(f"Subjects per class (n={sum(vals)})")
    ax.grid(True, axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    for x, v in enumerate(vals):
        ax.text(x, v + 0.6, str(v), ha="center", fontsize=9, color=INK_MUTED)

    # the number every result has to beat
    ax.axhline(0, color=GRID)
    return ax


def save(fig_or_ax, name: str) -> str:
    """Write a figure into results/ and return the path (for the report)."""
    config.ensure_dirs()
    fig = getattr(fig_or_ax, "figure", fig_or_ax)
    path = config.RESULTS_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE, dpi=160)
    return str(path)
