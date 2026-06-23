"""Mechanism-evidence figures and tables for the §Statistical Heterogeneity
section.

Produces:
  F10 -- Update-norm trajectory: mean per-client per-round ||w_i^(t+1) - w^t||_2
         for FedAvg vs FedProx, +/- SEM across seeds.  Direct empirical evidence
         for the proximal-anchor drift-control mechanism.  Read from the Flower
         runtime sweep (results/flower_C0_baseline/) because only that sweep
         logged update norms; the mechanism is algorithm-level and transfers to
         the pure-PyTorch primary.

  F11 -- Forest plot of per-seed paired Delta (FedProx - FedAvg) for the
         pure-PyTorch headline.  One row per seed with per-seed bootstrap CI;
         headline mean Delta line.

  T11 -- Best-vs-final-round table from the pure-PyTorch headline.  Peak val
         macro-F1, final-round val macro-F1, paired drop (peak - final) for
         FedAvg and FedProx.

  Trajectory-volatility statistic (printed; insert into prose as a callout):
         mean |val_macro_F1(r) - val_macro_F1(r-1)| across rounds, averaged
         over 10 seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
RESULTS = REPO_ROOT / "fl_dermamnist" / "results"
HEADLINE  = RESULTS / "headline"                # pure-PyTorch primary
FLOWER    = RESULTS / "flower_C0_baseline"      # Flower runtime, has update norms
OUT_FIG   = RESULTS / "thesis_ready" / "figures"
OUT_TBL   = RESULTS / "thesis_ready" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TBL.mkdir(parents=True, exist_ok=True)

NUM_ROUNDS = 150
PAIRED_SEEDS = [42, 123, 456, 789, 999, 2024, 31337, 161803, 271828, 8675309]

COL_FEDAVG  = "#7FBF94"
COL_FEDPROX = "#3D5A80"


plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":   11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


# ---------------------------------------------------------------------------
# F10 -- Update-norm trajectory
# ---------------------------------------------------------------------------

def fig_update_norms():
    """Mean ||Delta w|| per round, FedAvg vs FedProx, +/- SEM across seeds."""
    print("\n=== F10: Update-norm trajectory ===")
    fa_per_seed: list[pd.Series] = []
    fp_per_seed: list[pd.Series] = []
    for s in PAIRED_SEEDS:
        fa_p = FLOWER / f"client_update_norms_fedavg_mu0.0_E20_s{s}.csv"
        fp_p = FLOWER / f"client_update_norms_fedprox_mu0.01_E20_s{s}.csv"
        if fa_p.is_file():
            df = pd.read_csv(fa_p)
            # mean across clients per round
            fa_per_seed.append(df.groupby("round")["update_norm"].mean())
        if fp_p.is_file():
            df = pd.read_csv(fp_p)
            fp_per_seed.append(df.groupby("round")["update_norm"].mean())
    print(f"  FedAvg seeds with update_norms: {len(fa_per_seed)}")
    print(f"  FedProx seeds with update_norms: {len(fp_per_seed)}")

    rounds = np.arange(1, NUM_ROUNDS + 1)
    fa_mat = np.full((len(fa_per_seed), NUM_ROUNDS), np.nan)
    fp_mat = np.full((len(fp_per_seed), NUM_ROUNDS), np.nan)
    for i, sr in enumerate(fa_per_seed):
        fa_mat[i, : len(sr)] = sr.values[:NUM_ROUNDS]
    for i, sr in enumerate(fp_per_seed):
        fp_mat[i, : len(sr)] = sr.values[:NUM_ROUNDS]
    fa_mean = np.nanmean(fa_mat, axis=0)
    fa_sem  = np.nanstd(fa_mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(fa_mat), axis=0))
    fp_mean = np.nanmean(fp_mat, axis=0)
    fp_sem  = np.nanstd(fp_mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(fp_mat), axis=0))

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(rounds, fa_mean, color=COL_FEDAVG, linewidth=1.6,
            label="FedAvg")
    ax.fill_between(rounds, fa_mean - fa_sem, fa_mean + fa_sem,
                    color=COL_FEDAVG, alpha=0.18, linewidth=0)
    ax.plot(rounds, fp_mean, color=COL_FEDPROX, linewidth=1.6,
            label="FedProx ($\\mu = 0.01$)")
    ax.fill_between(rounds, fp_mean - fp_sem, fp_mean + fp_sem,
                    color=COL_FEDPROX, alpha=0.18, linewidth=0)

    ratio = float(np.nanmean(fa_mat)) / float(np.nanmean(fp_mat))
    ax.text(0.97, 0.55,
            f"Across-all-rounds mean ratio:\n"
            f"$\\|\\Delta w\\|_{{\\mathrm{{FedAvg}}}} / "
            f"\\|\\Delta w\\|_{{\\mathrm{{FedProx}}}}$ = {ratio:.2f}$\\times$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(facecolor="white", edgecolor="grey",
                                    boxstyle="round,pad=0.4"))

    ax.set_xlabel("Communication round")
    ax.set_ylabel(r"Mean per-client update norm $\|w_i^{t+1} - w^t\|_2$")
    ax.set_xlim(0, NUM_ROUNDS)
    ax.set_title("Direct mechanism evidence: per-client update norms across rounds\n"
                 "(Flower runtime, engineered partition, mean $\\pm$ SEM across seeds)",
                 loc="left", fontweight="bold", pad=6)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F10_update_norms.{ext}"
        fig.savefig(out)
        print(f"  Wrote: {out}")
    plt.close(fig)
    print(f"  Headline finding: FedAvg ||Delta w|| / FedProx ||Delta w|| = {ratio:.2f}x")


# ---------------------------------------------------------------------------
# F11 -- Forest plot of per-seed Delta
# ---------------------------------------------------------------------------

def fig_forest_per_seed():
    """Forest plot: one row per seed, paired Delta with bootstrap CI."""
    print("\n=== F11: Forest plot of per-seed Delta ===")
    fa, fp = {}, {}
    for s in PAIRED_SEEDS:
        fa_p = HEADLINE / f"test_at_best_fedavg_mu0.0_E20_s{s}.json"
        fp_p = HEADLINE / f"test_at_best_fedprox_mu0.01_E20_s{s}.json"
        if fa_p.is_file() and fp_p.is_file():
            fa[s] = json.load(open(fa_p))["macro_f1"]
            fp[s] = json.load(open(fp_p))["macro_f1"]

    seeds = sorted(set(fa) & set(fp))
    deltas = np.array([fp[s] - fa[s] for s in seeds])
    mean_delta = float(np.mean(deltas))

    # Per-seed values: just plot the single point per seed (no per-seed CI
    # because a single seed has n=1 in the paired analysis; we instead show
    # the headline mean + paired CI as a vertical reference band)
    # Walsh 95% CI from analyse_statistical_heterogeneity output:
    walsh_ci = (0.0021, 0.0528)  # exact Walsh 95% CI for n=10
    boot_ci  = (0.0073, 0.0484)  # bootstrap 95% CI on the mean

    fig, ax = plt.subplots(1, 1, figsize=(9, 6.5))
    y_positions = np.arange(len(seeds))[::-1]  # newest seed at top

    # Reference band: Walsh CI on the headline mean
    ax.axvspan(walsh_ci[0], walsh_ci[1],
               color=COL_FEDPROX, alpha=0.10, linewidth=0,
               label=f"Walsh 95\\% CI: [{walsh_ci[0]:+.3f}, {walsh_ci[1]:+.3f}]")
    ax.axvline(mean_delta, color=COL_FEDPROX, linewidth=1.6, linestyle="-",
               label=f"Mean $\\Delta$ = {mean_delta:+.4f}")
    ax.axvline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    # Plot per-seed points
    for y, s, d in zip(y_positions, seeds, deltas):
        color = COL_FEDPROX if d > 0 else COL_FEDAVG
        marker = "o" if d > 0 else "s"
        ax.scatter(d, y, s=80, color=color, edgecolor="black", linewidth=0.6,
                   marker=marker, zorder=3)
        ax.text(d, y + 0.18, f"{d:+.3f}", ha="center", va="bottom",
                fontsize=8, color="black", zorder=4)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"seed {s}" for s in seeds])
    ax.set_ylim(-0.7, len(seeds) - 0.3)
    ax.set_xlabel(r"Within-pair $\Delta$ test macro-F1 (FedProx $-$ FedAvg)")
    ax.set_xlim(-0.05, 0.12)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5, axis="x")

    # Vertical annotations
    ax.text(0.02, 0.98, f"FedProx wins: {int(np.sum(deltas > 0))}/{len(deltas)}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="grey",
                      boxstyle="round,pad=0.3"))

    ax.set_title("Per-seed paired $\\Delta$ (FedProx $-$ FedAvg) with headline mean and CI\n"
                 "(Pure-PyTorch primary, $n=10$ paired seeds, engineered partition)",
                 loc="left", fontweight="bold", pad=6)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F11_forest_per_seed.{ext}"
        fig.savefig(out)
        print(f"  Wrote: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# T11 -- Best-vs-final round
# ---------------------------------------------------------------------------

def table_best_vs_final():
    """Peak val_macro_F1 vs final-round val_macro_F1 for FA and FP across seeds."""
    print("\n=== T11: Best-vs-final-round table ===")
    fa_peak, fp_peak = [], []
    fa_final, fp_final = [], []
    for s in PAIRED_SEEDS:
        fa_h = pd.read_csv(HEADLINE / f"history_fedavg_mu0.0_E20_s{s}.csv")
        fp_h = pd.read_csv(HEADLINE / f"history_fedprox_mu0.01_E20_s{s}.csv")
        fa_peak.append(fa_h["val_macro_f1"].max())
        fp_peak.append(fp_h["val_macro_f1"].max())
        fa_final.append(fa_h[fa_h["round"] == NUM_ROUNDS]["val_macro_f1"].values[0])
        fp_final.append(fp_h[fp_h["round"] == NUM_ROUNDS]["val_macro_f1"].values[0])

    fa_peak = np.array(fa_peak); fp_peak = np.array(fp_peak)
    fa_final = np.array(fa_final); fp_final = np.array(fp_final)
    fa_drop = fa_peak - fa_final
    fp_drop = fp_peak - fp_final

    # Paired Wilcoxon on the drop
    drop_diff = fa_drop - fp_drop  # positive = FA drops more
    try:
        p_drop = stats.wilcoxon(drop_diff)[1]
    except Exception:
        p_drop = float("nan")

    print(f"  FA  peak  mean: {fa_peak.mean():.4f} +/- {fa_peak.std(ddof=1):.4f}")
    print(f"  FA  final mean: {fa_final.mean():.4f} +/- {fa_final.std(ddof=1):.4f}")
    print(f"  FA  drop  mean: {fa_drop.mean():+.4f} +/- {fa_drop.std(ddof=1):.4f}")
    print()
    print(f"  FP  peak  mean: {fp_peak.mean():.4f} +/- {fp_peak.std(ddof=1):.4f}")
    print(f"  FP  final mean: {fp_final.mean():.4f} +/- {fp_final.std(ddof=1):.4f}")
    print(f"  FP  drop  mean: {fp_drop.mean():+.4f} +/- {fp_drop.std(ddof=1):.4f}")
    print()
    print(f"  Paired Wilcoxon on (FA_drop - FP_drop): p = {p_drop:.4f}")

    # LaTeX table
    p_drop_fmt = (r"\textbf{<0.01}" if p_drop < 0.01 else
                  fr"\textbf{{{p_drop:.3f}}}" if p_drop < 0.05 else
                  f"{p_drop:.3f}")
    lines = [
        r"% T11 -- Best-vs-final-round (auto-generated)",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Peak vs final-round validation macro-F1 over the 150 "
        r"communication rounds (pure-PyTorch headline, $n=10$ paired "
        r"seeds, engineered partition). The post-peak drop "
        r"quantifies how much each algorithm overfits between its best "
        r"round and the final round.}",
        r"\label{tab:best-vs-final}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Algorithm & Peak val F1 (mean$\pm$SD) & Final val F1 (mean$\pm$SD) & Drop (peak$-$final) \\",
        r"\midrule",
        fr"FedAvg  & ${fa_peak.mean():.4f} \pm {fa_peak.std(ddof=1):.4f}$ & "
        fr"${fa_final.mean():.4f} \pm {fa_final.std(ddof=1):.4f}$ & "
        fr"${fa_drop.mean():+.4f} \pm {fa_drop.std(ddof=1):.4f}$ \\",
        fr"FedProx & ${fp_peak.mean():.4f} \pm {fp_peak.std(ddof=1):.4f}$ & "
        fr"${fp_final.mean():.4f} \pm {fp_final.std(ddof=1):.4f}$ & "
        fr"${fp_drop.mean():+.4f} \pm {fp_drop.std(ddof=1):.4f}$ \\",
        r"\midrule",
        fr"Paired Wilcoxon on (FA drop $-$ FP drop): $p$ = {p_drop_fmt} & & & \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT_TBL / "T11_best_vs_final.tex").write_text("\n".join(lines) + "\n")
    print(f"  Wrote: {OUT_TBL / 'T11_best_vs_final.tex'}")


# ---------------------------------------------------------------------------
# Trajectory volatility callout (statistic only; for prose insertion)
# ---------------------------------------------------------------------------

def trajectory_volatility_callout():
    print("\n=== Trajectory volatility (callout) ===")
    fa_vols, fp_vols = [], []
    for s in PAIRED_SEEDS:
        fa_h = pd.read_csv(HEADLINE / f"history_fedavg_mu0.0_E20_s{s}.csv")
        fp_h = pd.read_csv(HEADLINE / f"history_fedprox_mu0.01_E20_s{s}.csv")
        fa_vols.append(fa_h["val_macro_f1"].diff().abs().mean())
        fp_vols.append(fp_h["val_macro_f1"].diff().abs().mean())
    ratio = float(np.mean(fa_vols)) / float(np.mean(fp_vols))
    print(f"  Pure-PyTorch headline (n=10):")
    print(f"    FedAvg  mean |F1(r)-F1(r-1)| = {np.mean(fa_vols):.4f}")
    print(f"    FedProx mean |F1(r)-F1(r-1)| = {np.mean(fp_vols):.4f}")
    print(f"    Ratio FA / FP                = {ratio:.2f}x")
    print(f"  Suggested prose: 'Across the 10 seeds, FedAvg's per-round "
          f"validation macro-F1 volatility "
          f"(mean |F1(r) -- F1(r-1)|) is {ratio:.2f}x higher than "
          f"FedProx's ({np.mean(fa_vols):.3f} vs {np.mean(fp_vols):.3f}), "
          f"consistent with the proximal anchor's drift-control mechanism.'")


def main():
    fig_update_norms()
    fig_forest_per_seed()
    table_best_vs_final()
    trajectory_volatility_callout()


if __name__ == "__main__":
    main()
