"""Generate all thesis-ready figures from the per-seed and per-class data.

Style note: panels designed to mirror the conventions of FL papers
- McMahan et al. 2017 (FedAvg): bar charts with explicit numerical annotations
- Li et al. 2020 (FedProx): per-class breakdowns under non-IID, paired comparisons
- Hsu, Qi & Brown 2019: Dirichlet partition visualisations (we use balanced_paired instead)
- Bouthillier et al. 2021: forest/strip plots showing per-seed variance

All figures saved as PNG (300 dpi) and PDF (vector) for thesis use.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent colours across all figures (colour-blind safe)
C_FA   = "#2b6cb0"   # blue   — FedAvg
C_FP   = "#dd6b20"   # orange — FedProx
C_GAIN = "#38a169"   # green  — positive delta
C_LOSS = "#e53e3e"   # red    — negative delta
C_NULL = "#a0aec0"   # grey   — not significant

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def savefig(fig, name: str):
    p_png = FIG_DIR / f"{name}.png"
    p_pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(p_png, dpi=300, bbox_inches="tight")
    fig.savefig(p_pdf,           bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p_png.name}  (+ pdf)")


# -----------------------------------------------------------------------------
# Figure 1 — Headline summary
# Bar chart of mean macro-F1, FedAvg vs FedProx, with error bars and significance.
# Style: McMahan 2017 Figure 2 (final accuracy bars).
# -----------------------------------------------------------------------------
def fig_01_headline_summary():
    stats = json.load(open(DATA_DIR / "summary_statistics.json"))
    h = stats["headline"]
    means = [h["fedavg_mean"], h["fedprox_mean"]]
    stds  = [h["fedavg_std"],  h["fedprox_std"]]
    p_value = stats["statistical_tests"]["primary"]["p_value"]

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    x = np.arange(2)
    bars = ax.bar(x, means, yerr=stds, capsize=8,
                  color=[C_FA, C_FP], edgecolor="black", linewidth=1.2,
                  error_kw=dict(ecolor="black", elinewidth=1.5))
    for bar, m, sd in zip(bars, means, stds):
        ax.annotate(f"{m:.4f}\n± {sd:.4f}",
                    (bar.get_x() + bar.get_width()/2, m + sd + 0.005),
                    ha="center", fontsize=11, fontweight="bold")
    # Significance bracket
    y_top = max(m + sd for m, sd in zip(means, stds)) + 0.04
    ax.plot([0, 0, 1, 1], [y_top, y_top + 0.005, y_top + 0.005, y_top],
            color="black", linewidth=1.2)
    sig = "*" if p_value < 0.05 else "ns"
    ax.text(0.5, y_top + 0.008, f"$p = {p_value:.4f}$ ({sig})",
            ha="center", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(["FedAvg\n($\\mu = 0$)", "FedProx\n($\\mu = 0.01$)"], fontsize=11)
    ax.set_ylabel("Test macro-F1 (mean ± SD across 10 paired seeds)")
    ax.set_ylim(0, max(means) + max(stds) + 0.1)
    ax.set_title("FedAvg vs FedProx on DermaMNIST — headline result")
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "01_headline_summary")


# -----------------------------------------------------------------------------
# Figure 2 — Paired per-seed forest plot
# Each seed: horizontal line spanning FedAvg → FedProx, coloured by win/loss.
# Style: forest plots used in meta-analyses and paired clinical trials.
# -----------------------------------------------------------------------------
def fig_02_paired_forest():
    df = pd.read_csv(DATA_DIR / "per_seed_results.csv")
    df = df.sort_values("delta_macro_f1", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    y = np.arange(len(df))

    for i, row in df.iterrows():
        colour = C_GAIN if row["delta_macro_f1"] > 0 else C_LOSS
        # Line from FedAvg to FedProx
        ax.plot([row["fedavg_test_macro_f1"], row["fedprox_test_macro_f1"]],
                [i, i], color=colour, linewidth=2.5, alpha=0.7, zorder=1)
        # Dots
        ax.scatter(row["fedavg_test_macro_f1"], i, s=100, color=C_FA,
                   edgecolor="black", linewidth=1, zorder=2, label="FedAvg" if i == 0 else None)
        ax.scatter(row["fedprox_test_macro_f1"], i, s=100, color=C_FP,
                   edgecolor="black", linewidth=1, zorder=2, label="FedProx" if i == 0 else None)
        # Delta annotation on the right
        sign = "+" if row["delta_macro_f1"] >= 0 else ""
        ax.annotate(f"{sign}{row['delta_macro_f1']:.4f}",
                    (max(row["fedavg_test_macro_f1"], row["fedprox_test_macro_f1"]) + 0.015, i),
                    va="center", fontsize=10,
                    color=colour, fontweight="bold")

    # Mean Δ marker line (vertical)
    stats = json.load(open(DATA_DIR / "summary_statistics.json"))
    mean_fa = stats["headline"]["fedavg_mean"]
    mean_fp = stats["headline"]["fedprox_mean"]
    ax.axvline(mean_fa, color=C_FA, linestyle="--", alpha=0.6, label=f"FedAvg mean = {mean_fa:.4f}")
    ax.axvline(mean_fp, color=C_FP, linestyle="--", alpha=0.6, label=f"FedProx mean = {mean_fp:.4f}")

    ax.set_yticks(y)
    ax.set_yticklabels([f"seed {int(s)}" for s in df["seed"]])
    ax.set_xlabel("Test macro-F1")
    ax.set_title(f"Per-seed paired comparison (n = {len(df)})\n"
                 f"FedProx wins {(df['delta_macro_f1'] > 0).sum()}/{len(df)}")
    ax.grid(alpha=0.25, axis="x")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "02_paired_forest")


# -----------------------------------------------------------------------------
# Figure 3 — Per-seed Δ strip plot
# Distribution of within-pair improvements. Style: Bouthillier 2021.
# -----------------------------------------------------------------------------
def fig_03_delta_strip():
    df = pd.read_csv(DATA_DIR / "per_seed_results.csv")
    deltas = df["delta_macro_f1"].values

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    y = np.full_like(deltas, 1.0, dtype=float)
    # Jitter
    rng = np.random.default_rng(0)
    y += rng.uniform(-0.08, 0.08, size=len(y))
    colours = [C_GAIN if d > 0 else C_LOSS for d in deltas]
    ax.scatter(deltas, y, s=180, c=colours, edgecolor="black", linewidth=1.5, zorder=3)

    for d, yi, s in zip(deltas, y, df["seed"]):
        ax.annotate(f"{int(s)}", (d, yi + 0.12), ha="center", fontsize=8, color="dimgray")

    # Mean and CI
    mean_d = deltas.mean()
    boot = [np.random.default_rng(i).choice(deltas, len(deltas), replace=True).mean()
            for i in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ax.errorbar([mean_d], [0.5], xerr=[[mean_d - lo], [hi - mean_d]],
                fmt="D", color="black", capsize=10, capthick=2, markersize=10,
                label=f"Mean Δ = {mean_d:+.4f}\n95% CI = [{lo:+.4f}, {hi:+.4f}]")
    ax.axvline(0, color="black", linewidth=1, linestyle="-")
    ax.text(0, 1.6, "no\neffect", ha="center", fontsize=9, color="dimgray", style="italic")

    ax.set_yticks([])
    ax.set_ylim(-0.2, 1.7)
    ax.set_xlabel("Δ Test macro-F1 (FedProx − FedAvg)")
    ax.set_title("Per-seed paired differences with bootstrap 95% CI")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    savefig(fig, "03_delta_strip")


# -----------------------------------------------------------------------------
# Figure 4 — Per-class comparison grouped bars
# Style: Li 2020 per-task breakdown table → bar chart form.
# -----------------------------------------------------------------------------
def fig_04_per_class_bars():
    df = pd.read_csv(DATA_DIR / "per_class_results.csv")
    short = ["actinic", "basal", "benign\nkerat", "dermato",
             "melanoma", "mel_nevi", "vascular"]

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w/2, df["fedavg_mean_f1"],  w, color=C_FA,
           edgecolor="black", linewidth=0.8, label="FedAvg")
    ax.bar(x + w/2, df["fedprox_mean_f1"], w, color=C_FP,
           edgecolor="black", linewidth=0.8, label="FedProx")
    for i, row in df.iterrows():
        ax.annotate(f"{row['fedavg_mean_f1']:.2f}",  (i - w/2, row["fedavg_mean_f1"]),
                    xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
        ax.annotate(f"{row['fedprox_mean_f1']:.2f}", (i + w/2, row["fedprox_mean_f1"]),
                    xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
        if row["significant_at_05"]:
            ax.annotate("*", (i, max(row["fedavg_mean_f1"], row["fedprox_mean_f1"]) + 0.05),
                        ha="center", fontsize=18, color=C_GAIN, fontweight="bold")

    # Prevalence labels on x-axis (twin)
    prevalences = df["prevalence_train_pct"].values

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({p:.1f}%)" for n, p in zip(short, prevalences)], fontsize=9)
    ax.set_ylabel("Test F1 (mean across 10 seeds)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-class test F1 — FedAvg vs FedProx\n* = significant at p < 0.05 (paired Wilcoxon)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "04_per_class_bars")


# -----------------------------------------------------------------------------
# Figure 5 — Per-class Δ with significance + tolerance line
# Style: Li 2020 ablation tables; clinical literature uses horizontal Δ plots.
# -----------------------------------------------------------------------------
def fig_05_per_class_delta():
    df = pd.read_csv(DATA_DIR / "per_class_results.csv")
    short = ["actinic", "basal", "benign\nkerat", "dermato",
             "melanoma", "mel_nevi", "vascular"]
    df = df.assign(class_short=short).sort_values("delta_mean_f1", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    y = np.arange(len(df))
    deltas = df["delta_mean_f1"].values
    colours = [C_GAIN if (d > 0 and s) else C_LOSS if (d < 0 and s) else C_NULL
               for d, s in zip(deltas, df["significant_at_05"])]

    bars = ax.barh(y, deltas, color=colours, edgecolor="black", linewidth=0.8)

    # Annotations: delta + p-value
    for i, row in df.iterrows():
        d = row["delta_mean_f1"]; p = row["wilcoxon_p"]
        sig_marker = " *" if row["significant_at_05"] else ""
        offset = 0.005 if d >= 0 else -0.005
        ha = "left" if d >= 0 else "right"
        ax.annotate(f"{d:+.4f}  (p={p:.3f}){sig_marker}",
                    (d + offset, i), va="center", ha=ha, fontsize=9, fontweight="bold")

    # Tolerance line at -0.05
    ax.axvline(-0.05, color="red", linestyle=":", alpha=0.6, label="±0.05 design tolerance")
    ax.axvline(0.05,  color="red", linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["class_short"].values)
    ax.set_xlabel("Δ F1 (FedProx − FedAvg), mean across 10 seeds")
    ax.set_title("Per-class FedProx advantage with significance\n"
                 "Coloured bars = significant at p < 0.05; grey = not significant")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.08, 0.16)
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "05_per_class_delta")


# -----------------------------------------------------------------------------
# Figure 6 — Distribution comparison (box + strip)
# Shows that FedProx not only has higher mean, but lower variance.
# Style: Bouthillier 2021 (ML variance benchmarks).
# -----------------------------------------------------------------------------
def fig_06_distribution():
    df = pd.read_csv(DATA_DIR / "per_seed_results.csv")

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    data = [df["fedavg_test_macro_f1"].values, df["fedprox_test_macro_f1"].values]
    bp = ax.boxplot(data, widths=0.4, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=2),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5))
    for patch, c in zip(bp["boxes"], [C_FA, C_FP]):
        patch.set_facecolor(c); patch.set_alpha(0.4)

    rng = np.random.default_rng(0)
    for i, (algo_data, c) in enumerate(zip(data, [C_FA, C_FP]), start=1):
        x = i + rng.uniform(-0.08, 0.08, size=len(algo_data))
        ax.scatter(x, algo_data, s=80, color=c, edgecolor="black",
                   linewidth=1.2, zorder=3, alpha=0.9)

    # Paired connector lines (showing within-seed pairing)
    for _, row in df.iterrows():
        ax.plot([1, 2], [row["fedavg_test_macro_f1"], row["fedprox_test_macro_f1"]],
                color="gray", alpha=0.35, linewidth=1, zorder=1)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["FedAvg", "FedProx"], fontsize=12)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Distribution across 10 paired seeds\n"
                 "(grey lines connect paired runs by seed)")
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "06_distribution")


# -----------------------------------------------------------------------------
# Figure 7 — Effect size and variance comparison
# A small "thesis-summary" panel with the most-cited statistics.
# -----------------------------------------------------------------------------
def fig_07_summary_panel():
    stats = json.load(open(DATA_DIR / "summary_statistics.json"))
    h = stats["headline"]; t = stats["statistical_tests"]
    v = stats["variance_comparison"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)

    # Panel A — means with paired Δ
    ax = axes[0]
    means = [h["fedavg_mean"], h["fedprox_mean"]]
    stds  = [h["fedavg_std"], h["fedprox_std"]]
    ax.bar(["FedAvg", "FedProx"], means, yerr=stds, capsize=8,
           color=[C_FA, C_FP], edgecolor="black", linewidth=1.2,
           error_kw=dict(ecolor="black", elinewidth=1.5))
    ax.set_ylabel("Test macro-F1"); ax.set_ylim(0, max(means) + 0.1)
    ax.set_title(f"(A) Mean macro-F1\nΔ = {h['mean_delta']:+.4f}")
    ax.grid(alpha=0.25, axis="y"); ax.spines[["top","right"]].set_visible(False)

    # Panel B — Effect size visual (a thermometer / horizontal bar of r_rb)
    ax = axes[1]
    r = t["effect_size"]["value"]
    ax.barh([0], [r], color=C_GAIN if r > 0 else C_LOSS, edgecolor="black", linewidth=1.5)
    for line, name in [(0.1, "small"), (0.3, "medium"), (0.5, "large"), (0.7, "very large")]:
        ax.axvline(line, color="gray", linestyle=":", alpha=0.5)
        ax.text(line, 0.45, name, fontsize=8, ha="left", color="gray")
    ax.scatter([r], [0], s=200, color="white", edgecolor="black", linewidth=2, zorder=5)
    ax.text(r, 0, f"{r:+.3f}", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(-1.0, 1.0); ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Rank-biserial r")
    ax.set_title(f"(B) Effect size\nr = {r:+.3f} ({t['effect_size']['magnitude']})")
    ax.axvline(0, color="black", linewidth=1)
    ax.spines[["top","right","left"]].set_visible(False)

    # Panel C — variance reduction
    ax = axes[2]
    std_fa = v["fedavg_macroF1_std"]; std_fp = v["fedprox_macroF1_std"]
    bars = ax.bar(["FedAvg", "FedProx"], [std_fa, std_fp],
                   color=[C_FA, C_FP], edgecolor="black", linewidth=1.2)
    for bar, val in zip(bars, [std_fa, std_fp]):
        ax.annotate(f"{val:.4f}", (bar.get_x()+bar.get_width()/2, val),
                    xytext=(0,3), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Across-seed std of test macro-F1")
    ax.set_title(f"(C) Variance\n{v['ratio']*100:.0f}% lower with FedProx")
    ax.grid(alpha=0.25, axis="y"); ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("FedProx vs FedAvg — summary statistics (10 paired seeds)",
                 fontsize=13, fontweight="bold")
    savefig(fig, "07_summary_panel")


if __name__ == "__main__":
    print("Generating thesis figures...")
    print(f"  Output dir: {FIG_DIR}")
    print()
    fig_01_headline_summary()
    fig_02_paired_forest()
    fig_03_delta_strip()
    fig_04_per_class_bars()
    fig_05_per_class_delta()
    fig_06_distribution()
    fig_07_summary_panel()
    print("\nAll figures written. PNG @ 300 dpi + PDF (vector) for each.")
