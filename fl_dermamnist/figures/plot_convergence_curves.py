"""Convergence-curve plots - multi-panel-style multi-panel figures.

Produces 4 figures using a clean multi-panel layout suitable for
federated-learning result reporting:

    F1: Headline convergence (3-panel horizontal)
        – val_macro_f1, val_loss, best-vs-final bar chart
        – Source: results/headline/ (pure-PyTorch headline)

    F2: Per-class convergence grid (7-panel grid, one panel per class)
        – val_f1_class_<c> over rounds for FedAvg vs FedProx
        – Source: results/headline/

    F3: Cross-runtime convergence (2-panel horizontal)
        – Pure-PyTorch vs Flower replication side-by-side
        – Source: results/headline/ + results/flower_C0_baseline/

    F4: 4-partition Flower convergence (2x2 grid)
        – IID, Dirichlet, Specialist, Balanced paired on the Flower runtime
        – Source: results/iid/, dirichlet_a01/, specialist_partition/,
                  flower_C0_baseline/

Outputs to:
    results/thesis_ready/figures/F1_headline_convergence.{pdf,png}
    results/thesis_ready/figures/F2_per_class_convergence.{pdf,png}
    results/thesis_ready/figures/F3_cross_runtime_convergence.{pdf,png}
    results/thesis_ready/figures/F4_partition_convergence.{pdf,png}

Visual style:
    comparable-thesis-inspired palette - soft mint-green for FedAvg, dark blue for
    FedProx, warm brown for FedNova, deep slate for centralised reference.
    Mean ± SEM bands across 10 paired seeds (more rigorous than the
    single-seed plots typical in MPhil-scale FL theses, which lack
    error envelopes).

Usage:
    python fl_dermamnist/figures/plot_convergence_curves.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----- Paths --------------------------------------------------------------

THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
RESULTS = REPO_ROOT / "fl_dermamnist" / "results"
OUT_FIG = RESULTS / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)


# ----- Configuration -------------------------------------------------------

PAIRED_SEEDS = [42, 123, 456, 789, 999, 2024, 31337, 161803, 271828, 8675309]
NUM_ROUNDS = 150

# comparable-thesis-inspired palette
COL_FEDAVG    = "#7FBF94"   # soft mint green
COL_FEDPROX   = "#3D5A80"   # dark slate blue
COL_FEDNOVA   = "#9C6644"   # warm brown
COL_CENTRAL   = "#293241"   # deep navy reference
COL_BAND_FA   = "#7FBF94"
COL_BAND_FP   = "#3D5A80"

CLASS_DISPLAY = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]

# Matplotlib global style
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.6,
    "lines.linewidth":   1.6,
    "figure.dpi":       110,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


# ----- Loaders -------------------------------------------------------------

def load_history(sweep_dir: Path, algo: str, mu: str) -> pd.DataFrame:
    """Stack per-seed history CSVs into a long DataFrame.

    Returns a DataFrame with columns
        seed, round, val_loss, val_accuracy, val_balanced_accuracy,
        val_macro_f1, val_f1_class_0..6, train_loss
    """
    frames = []
    for seed in PAIRED_SEEDS:
        p = sweep_dir / f"history_{algo}_mu{mu}_E20_s{seed}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if "seed" not in df.columns:
            df["seed"] = seed
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No history files in {sweep_dir} for {algo} mu={mu}")
    return pd.concat(frames, ignore_index=True)


def mean_sem_curve(df: pd.DataFrame, ycol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rounds, mean, sem) for ycol grouped by round across seeds."""
    g = df.groupby("round")[ycol]
    rounds = np.array(sorted(df["round"].unique()))
    m = g.mean().reindex(rounds).to_numpy()
    n = g.count().reindex(rounds).to_numpy()
    sd = g.std().reindex(rounds).to_numpy()
    sem = sd / np.sqrt(np.clip(n, 1, None))
    return rounds, m, sem


def test_best_macro_f1(sweep_dir: Path, algo: str, mu: str) -> np.ndarray:
    """Read test_at_best_*.json files and return array of macro_f1."""
    vals = []
    for seed in PAIRED_SEEDS:
        p = sweep_dir / f"test_at_best_{algo}_mu{mu}_E20_s{seed}.json"
        if p.is_file():
            vals.append(float(json.load(open(p))["macro_f1"]))
    return np.array(vals)


def final_round_macro_f1(df: pd.DataFrame) -> np.ndarray:
    """Per-seed val_macro_f1 at the final round."""
    return df[df["round"] == NUM_ROUNDS].groupby("seed")["val_macro_f1"].first().to_numpy()


# ----- Plotting helpers ----------------------------------------------------

def plot_curve(ax, rounds, mean, sem, *, color, label):
    """Plot mean line with ±SEM shaded band."""
    ax.plot(rounds, mean, color=color, label=label)
    ax.fill_between(rounds, mean - sem, mean + sem, color=color, alpha=0.15, linewidth=0)


def annotate_panel(ax, title):
    """multi-panel-style: title placed inside panel at top."""
    ax.set_title(title, loc="left", fontweight="bold", pad=6)


# ----- F1 : Headline convergence (3-panel horizontal layout) --------------------

def fig_headline_convergence():
    """3-panel horizontal: val_macro_f1, val_loss, best/final bars."""
    head_dir = RESULTS / "headline"
    fa_df = load_history(head_dir, "fedavg",  "0.0")
    fp_df = load_history(head_dir, "fedprox", "0.01")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: val_macro_f1
    r1, fa_m, fa_s = mean_sem_curve(fa_df, "val_macro_f1")
    r2, fp_m, fp_s = mean_sem_curve(fp_df, "val_macro_f1")
    ax = axes[0]
    plot_curve(ax, r1, fa_m, fa_s, color=COL_FEDAVG,  label="FedAvg")
    plot_curve(ax, r2, fp_m, fp_s, color=COL_FEDPROX, label="FedProx ($\\mu=0.01$)")
    annotate_panel(ax, "Global Model — Val. macro-F1")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation macro-F1")
    ax.set_xlim(0, NUM_ROUNDS)
    ax.set_ylim(0.05, 0.6)
    ax.legend(loc="lower right", frameon=False)

    # Panel 2: val_loss
    r1, fa_m, fa_s = mean_sem_curve(fa_df, "val_loss")
    r2, fp_m, fp_s = mean_sem_curve(fp_df, "val_loss")
    ax = axes[1]
    plot_curve(ax, r1, fa_m, fa_s, color=COL_FEDAVG,  label="FedAvg")
    plot_curve(ax, r2, fp_m, fp_s, color=COL_FEDPROX, label="FedProx")
    annotate_panel(ax, "Global Model — Val. loss")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation cross-entropy loss")
    ax.set_xlim(0, NUM_ROUNDS)

    # Panel 3: Best vs Final bar chart
    ax = axes[2]
    fa_test = test_best_macro_f1(head_dir, "fedavg",  "0.0")
    fp_test = test_best_macro_f1(head_dir, "fedprox", "0.01")
    fa_final = final_round_macro_f1(fa_df)
    fp_final = final_round_macro_f1(fp_df)

    # Centralised reference
    central_dir = RESULTS / "centralised"
    cent_vals = []
    for seed in PAIRED_SEEDS:
        p = central_dir / f"centralised_seed{seed}.json"
        if p.is_file():
            cent_vals.append(float(json.load(open(p))["macro_f1"]))
    cent_mean = float(np.mean(cent_vals)) if cent_vals else None

    labels = ["FedAvg", "FedProx", "Cent."]
    means_best  = [fa_test.mean(),  fp_test.mean(),  cent_mean]
    means_final = [fa_final.mean(), fp_final.mean(), cent_mean]
    colors      = [COL_FEDAVG, COL_FEDPROX, COL_CENTRAL]

    x = np.arange(len(labels))
    w = 0.34
    bars_best  = ax.bar(x - w/2, means_best,  width=w, color=colors,
                        edgecolor="black", linewidth=0.6, label="Best @ val")
    bars_final = ax.bar(x + w/2, means_final, width=w, color=colors,
                        edgecolor="black", linewidth=0.6, hatch="///",
                        alpha=0.85, label="Final round")
    # Hatch fill colour fix
    for bf in bars_final:
        bf.set_facecolor("none")
        bf.set_hatch("///")
        bf.set_edgecolor(colors[bars_final.index(bf)])
        bf.set_linewidth(0.8)
    # Annotate values on bars
    for i, (b, f) in enumerate(zip(means_best, means_final)):
        if b is not None:
            ax.text(x[i] - w/2, b + 0.012, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
        if f is not None:
            ax.text(x[i] + w/2, f + 0.012, f"{f:.3f}", ha="center", va="bottom", fontsize=8)
    annotate_panel(ax, "Best vs Final test macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0, 0.7)
    # Legend with hatched-bar swatch
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="grey", edgecolor="black", label="Best @ val"),
        Patch(facecolor="none", edgecolor="grey", hatch="///", label="Final round"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)

    fig.suptitle("Headline convergence on engineered partition (pure-PyTorch, $n=10$ paired seeds, mean $\\pm$ SEM)",
                 fontsize=11, y=1.02, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F1_headline_convergence.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


# ----- F2 : Per-class convergence grid -------------------------------------

def fig_per_class_convergence():
    """7-panel grid: per-class val_f1_class_<c> over rounds (FedAvg vs FedProx)."""
    head_dir = RESULTS / "headline"
    fa_df = load_history(head_dir, "fedavg",  "0.0")
    fp_df = load_history(head_dir, "fedprox", "0.01")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()

    for c in range(7):
        ax = axes[c]
        ycol = f"val_f1_class_{c}"
        r1, fa_m, fa_s = mean_sem_curve(fa_df, ycol)
        r2, fp_m, fp_s = mean_sem_curve(fp_df, ycol)
        plot_curve(ax, r1, fa_m, fa_s, color=COL_FEDAVG,  label="FedAvg")
        plot_curve(ax, r2, fp_m, fp_s, color=COL_FEDPROX, label="FedProx")
        annotate_panel(ax, CLASS_DISPLAY[c])
        ax.set_xlabel("Round")
        ax.set_ylabel("Val. F1")
        ax.set_xlim(0, NUM_ROUNDS)
        ax.set_ylim(0, 1.0)
        if c == 0:
            ax.legend(loc="upper left", frameon=False)

    # 8th cell becomes a legend / summary panel
    ax = axes[7]
    ax.axis("off")
    ax.text(0.05, 0.95, "Notes:", fontweight="bold", transform=ax.transAxes, va="top")
    notes = [
        r"Mean $\pm$ SEM across 10",
        r"paired seeds, pure-PyTorch,",
        r"engineered partition.",
        "",
        r"Validation set only;",
        r"test-set per-class $\Delta$",
        r"and Holm-corrected $p$-values",
        r"are reported in Table T02.",
        "",
        r"Melanocytic nevi (majority class)",
        r"is held by every client in the",
        r"engineered partition.",
    ]
    for i, t in enumerate(notes):
        ax.text(0.05, 0.88 - 0.06 * i, t, transform=ax.transAxes, va="top", fontsize=9)

    fig.suptitle("Per-class validation F1 over rounds (pure-PyTorch headline, $n=10$ paired seeds, mean $\\pm$ SEM)",
                 fontsize=11, y=1.01, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F2_per_class_convergence.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


# ----- F3 : Cross-runtime convergence (3-panel horizontal layout) ---------------

def fig_cross_runtime_convergence():
    """2-panel horizontal: PT vs Flower side-by-side on engineered partition."""
    pt_fa = load_history(RESULTS / "headline",            "fedavg",  "0.0")
    pt_fp = load_history(RESULTS / "headline",            "fedprox", "0.01")
    fl_fa = load_history(RESULTS / "flower_C0_baseline",  "fedavg",  "0.0")
    fl_fp = load_history(RESULTS / "flower_C0_baseline",  "fedprox", "0.01")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for ax, fa_df, fp_df, runtime_name in [
        (axes[0], pt_fa, pt_fp, "Pure-PyTorch (primary)"),
        (axes[1], fl_fa, fl_fp, "Flower simulation (replication)"),
    ]:
        r1, fa_m, fa_s = mean_sem_curve(fa_df, "val_macro_f1")
        r2, fp_m, fp_s = mean_sem_curve(fp_df, "val_macro_f1")
        plot_curve(ax, r1, fa_m, fa_s, color=COL_FEDAVG,  label="FedAvg")
        plot_curve(ax, r2, fp_m, fp_s, color=COL_FEDPROX, label="FedProx")
        annotate_panel(ax, runtime_name)
        ax.set_xlabel("Communication round")
        ax.set_xlim(0, NUM_ROUNDS)
        ax.set_ylim(0.05, 0.6)
    axes[0].set_ylabel("Validation macro-F1")
    axes[0].legend(loc="lower right", frameon=False)

    fig.suptitle("Cross-runtime convergence on engineered partition ($n=10$ paired seeds)",
                 fontsize=11, y=1.02, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F3_cross_runtime_convergence.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


# ----- F4 : 4-partition Flower convergence (2x2 grid) ----------------------

def fig_partition_convergence():
    """2x2 grid: IID, Dirichlet α=0.1, specialist, engineered (Flower runtime)."""
    cells = [
        ("iid",                  "IID (falsification)"),
        ("dirichlet_a01",        r"Dirichlet $\alpha=0.1$"),
        ("specialist_partition", "Specialist (1-of-7)"),
        ("flower_C0_baseline",   "Balanced paired (engineered)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    axes = axes.flatten()
    for ax, (sweep_name, display) in zip(axes, cells):
        sweep_dir = RESULTS / sweep_name
        fa_df = load_history(sweep_dir, "fedavg",  "0.0")
        fp_df = load_history(sweep_dir, "fedprox", "0.01")
        r1, fa_m, fa_s = mean_sem_curve(fa_df, "val_macro_f1")
        r2, fp_m, fp_s = mean_sem_curve(fp_df, "val_macro_f1")
        plot_curve(ax, r1, fa_m, fa_s, color=COL_FEDAVG,  label="FedAvg")
        plot_curve(ax, r2, fp_m, fp_s, color=COL_FEDPROX, label="FedProx")
        annotate_panel(ax, display)
        ax.set_xlabel("Communication round")
        ax.set_xlim(0, NUM_ROUNDS)
        ax.set_ylim(0.0, 0.7)
    axes[0].set_ylabel("Validation macro-F1")
    axes[2].set_ylabel("Validation macro-F1")
    axes[0].legend(loc="lower right", frameon=False)

    fig.suptitle("Partition robustness — Flower runtime, $n=10$ paired seeds (mean $\\pm$ SEM)",
                 fontsize=11, y=1.01, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F4_partition_convergence.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


# ----- Main ----------------------------------------------------------------

def main():
    print("=" * 80)
    print(" Convergence-curve plotting (multi-panel-style)")
    print(f" Output directory: {OUT_FIG}")
    print("=" * 80)
    fig_headline_convergence()
    fig_per_class_convergence()
    fig_cross_runtime_convergence()
    fig_partition_convergence()
    print("\nDone.")


if __name__ == "__main__":
    main()
