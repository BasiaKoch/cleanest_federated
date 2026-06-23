"""Aggregate the clean mu-sensitivity Flower sweep.

Reads:
  results/mu_sensitivity_flower/history_<algo>_mu<mu>_E20_s<seed>.csv
  results/mu_sensitivity_flower/test_at_best_<algo>_mu<mu>_E20_s<seed>.json
  results/mu_sensitivity_flower/client_update_norms_<algo>_mu<mu>_E20_s<seed>.csv

Produces (only what existing logging supports; see notes in module
docstring for what is NOT measured):

  Outcome:
    figures/F_mu_sensitivity_outcome.pdf     - mu vs test macro-F1 + FedAvg ref
    figures/F_mu_sensitivity_pairedΔ.pdf     - paired Δ(mu) - Δ(0) per seed,
                                               bootstrap 95% CI

  Mechanism (from update-norm CSV only):
    figures/F_mu_sensitivity_mechanism.pdf   - mean per-client update norm
                                               vs mu, and inter-client
                                               dispersion (std across clients
                                               within a round, then averaged)

  Convergence:
    figures/F_mu_sensitivity_convergence.pdf - validation macro-F1 trajectories
                                               per mu, mean +/- SEM

  Tables (printed; the user pastes the LaTeX into the thesis):
    Decoupling table: per mu, test macro-F1, paired Δ vs mu=0, mean update
    norm, inter-client dispersion, rounds-to-95%-of-best-FedAvg.

NOT measured by this script (the existing logging does not capture them;
add to client/server code separately if needed):
  - cosine similarity between client update and aggregated update
  - global step norm (would need post-aggregation global state per round)
  - proximal-loss magnitude vs CE-loss magnitude

The script is descriptive only: no p-values are emphasised; bootstrap CIs
are reported for paired differences.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
SWEEP_DIR = REPO_ROOT / "fl_dermamnist" / "results" / "mu_sensitivity_flower"
OUT_FIG = REPO_ROOT / "fl_dermamnist" / "results" / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

NUM_ROUNDS = 150
MU_VALUES = [0.0, 0.001, 0.01, 0.1, 1.0]
SEEDS_CANONICAL = [42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828]

COL_FEDAVG  = "#7FBF94"
COL_FEDPROX = "#3D5A80"
PALETTE = {
    0.0:   "#7FBF94",
    0.001: "#A8DADC",
    0.01:  "#3D5A80",
    0.1:   "#E9C46A",
    1.0:   "#E76F51",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


def _algo_for(mu: float) -> str:
    return "fedavg" if mu == 0.0 else "fedprox"


def _mu_str(mu: float) -> str:
    return "0.0" if mu == 0.0 else str(mu)


def load_test_jsons(mu: float) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    pat = re.compile(rf"test_at_best_{_algo_for(mu)}_mu{re.escape(_mu_str(mu))}_E20_s(\d+)\.json")
    for f in sorted(SWEEP_DIR.glob(f"test_at_best_{_algo_for(mu)}_*.json")):
        m = pat.match(f.name)
        if not m: continue
        out[int(m.group(1))] = json.load(open(f))
    return out


def load_history_csvs(mu: float) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    pat = re.compile(rf"history_{_algo_for(mu)}_mu{re.escape(_mu_str(mu))}_E20_s(\d+)\.csv")
    for f in sorted(SWEEP_DIR.glob(f"history_{_algo_for(mu)}_*.csv")):
        m = pat.match(f.name)
        if not m: continue
        out[int(m.group(1))] = pd.read_csv(f)
    return out


def load_update_norms(mu: float) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    pat = re.compile(rf"client_update_norms_{_algo_for(mu)}_mu{re.escape(_mu_str(mu))}_E20_s(\d+)\.csv")
    for f in sorted(SWEEP_DIR.glob(f"client_update_norms_{_algo_for(mu)}_*.csv")):
        m = pat.match(f.name)
        if not m: continue
        out[int(m.group(1))] = pd.read_csv(f)
    return out


def bootstrap_ci(xs: np.ndarray, n_boot: int = 10_000, ci: float = 0.95,
                 rng: Optional[np.random.Generator] = None) -> Tuple[float, float, float]:
    """Returns (mean, lo, hi) percentile bootstrap CI."""
    if rng is None:
        rng = np.random.default_rng(0)
    xs = np.asarray(xs, dtype=float)
    xs = xs[~np.isnan(xs)]
    if len(xs) < 2:
        return float(np.mean(xs)) if len(xs) else np.nan, np.nan, np.nan
    boots = rng.choice(xs, size=(n_boot, len(xs)), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return float(xs.mean()), lo, hi


def summarise() -> pd.DataFrame:
    rows = []
    fedavg_jsons = load_test_jsons(0.0)
    fedavg_macro_by_seed = {s: float(d["macro_f1"]) for s, d in fedavg_jsons.items()}

    fedavg_hist = load_history_csvs(0.0)
    fedavg_best_val = {}
    for s, df in fedavg_hist.items():
        if "val_macro_f1" in df.columns and len(df):
            fedavg_best_val[s] = float(df["val_macro_f1"].max())
    fedavg_best_val_mean = (np.mean(list(fedavg_best_val.values()))
                            if fedavg_best_val else np.nan)
    threshold = 0.95 * fedavg_best_val_mean if not np.isnan(fedavg_best_val_mean) else np.nan

    for mu in MU_VALUES:
        tj = load_test_jsons(mu)
        hist = load_history_csvs(mu)
        un = load_update_norms(mu)

        macros = np.array([tj[s]["macro_f1"] for s in tj]) if tj else np.array([])
        seeds_here = sorted(tj.keys())

        # Paired differences vs mu=0 (only for seeds present in both arms).
        if mu != 0.0:
            paired = np.array([
                tj[s]["macro_f1"] - fedavg_macro_by_seed[s]
                for s in seeds_here if s in fedavg_macro_by_seed
            ])
        else:
            paired = np.array([])

        # Mean per-client update norm (across all (round, client, seed)).
        mean_norm = np.nan
        client_dispersion = np.nan
        if un:
            all_norms = []
            disps = []
            for s, df in un.items():
                all_norms.extend(df["update_norm"].tolist())
                # Inter-client dispersion: std across clients within each round, then mean.
                per_round_std = df.groupby("round")["update_norm"].std()
                disps.append(per_round_std.mean())
            if all_norms:
                mean_norm = float(np.mean(all_norms))
            if disps:
                client_dispersion = float(np.nanmean(disps))

        # Rounds-to-threshold per seed (smallest round with val_macro_f1 >= threshold).
        r2t = []
        if not np.isnan(threshold):
            for s, df in hist.items():
                if "val_macro_f1" not in df.columns: continue
                m = df[df["val_macro_f1"] >= threshold]
                r2t.append(int(m["round"].iloc[0]) if len(m) else np.nan)
        r2t_arr = np.array(r2t, dtype=float) if r2t else np.array([])

        macro_mean, macro_lo, macro_hi = (
            bootstrap_ci(macros) if len(macros) else (np.nan, np.nan, np.nan)
        )
        paired_mean, paired_lo, paired_hi = (
            bootstrap_ci(paired) if len(paired) else (np.nan, np.nan, np.nan)
        )

        rows.append(dict(
            mu=mu,
            n_seeds=len(seeds_here),
            macro_f1_mean=macro_mean,
            macro_f1_lo=macro_lo,
            macro_f1_hi=macro_hi,
            macro_f1_sd=float(np.std(macros, ddof=1)) if len(macros) > 1 else np.nan,
            paired_delta_mean=paired_mean,
            paired_delta_lo=paired_lo,
            paired_delta_hi=paired_hi,
            paired_delta_median=float(np.median(paired)) if len(paired) else np.nan,
            mean_update_norm=mean_norm,
            interclient_dispersion=client_dispersion,
            rounds_to_threshold_median=(float(np.nanmedian(r2t_arr))
                                        if len(r2t_arr) else np.nan),
        ))
    df = pd.DataFrame(rows)
    return df


def plot_outcome(summary: pd.DataFrame):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    pos = np.arange(len(summary))   # categorical x to handle mu=0 cleanly
    ax.errorbar(
        pos, summary["macro_f1_mean"],
        yerr=[
            summary["macro_f1_mean"] - summary["macro_f1_lo"],
            summary["macro_f1_hi"] - summary["macro_f1_mean"],
        ],
        fmt="o-", color="#3D5A80", linewidth=1.6, capsize=4,
    )
    fa_mean = summary.loc[summary["mu"] == 0.0, "macro_f1_mean"].values
    if len(fa_mean):
        ax.axhline(fa_mean[0], color="#7FBF94", linestyle="--",
                   linewidth=1.2, label=f"FedAvg ($\\mu=0$) = {fa_mean[0]:.3f}")
        ax.legend(frameon=False, loc="upper left")
    # Headroom so the upper-left legend does not overlap with the curve
    # or the FedAvg reference line at y = fa_mean.
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi + 0.025)
    ax.set_xticks(pos)
    ax.set_xticklabels([
        f"$\\mu = {mu:g}$" + ("\n(FedAvg)" if mu == 0.0 else "")
        for mu in summary["mu"]
    ])
    ax.set_ylabel("Test macro-F1 (mean, 95\\% bootstrap CI)")
    ax.set_xlabel("FedProx proximal coefficient")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"F_mu_sensitivity_outcome.{ext}")
    plt.close(fig)


def plot_paired_delta(summary: pd.DataFrame):
    fp_rows = summary[summary["mu"] > 0.0].reset_index(drop=True)
    if not len(fp_rows):
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    pos = np.arange(len(fp_rows))
    ax.errorbar(
        pos, fp_rows["paired_delta_mean"],
        yerr=[
            fp_rows["paired_delta_mean"] - fp_rows["paired_delta_lo"],
            fp_rows["paired_delta_hi"] - fp_rows["paired_delta_mean"],
        ],
        fmt="s-", color="#3D5A80", linewidth=1.6, capsize=4,
    )
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"$\\mu = {mu:g}$" for mu in fp_rows["mu"]])
    ax.set_ylabel("Paired $\\Delta$ macro-F1 vs FedAvg\\\\(95\\% bootstrap CI)")
    ax.set_xlabel("FedProx proximal coefficient")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"F_mu_sensitivity_paired_delta.{ext}")
    plt.close(fig)


def plot_mechanism(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"wspace": 0.30})
    pos = np.arange(len(summary))
    axes[0].plot(pos, summary["mean_update_norm"], "o-",
                 color="#3D5A80", linewidth=1.6)
    axes[0].set_xticks(pos)
    axes[0].set_xticklabels([f"$\\mu = {mu:g}$" for mu in summary["mu"]])
    axes[0].set_ylabel("Mean per-client update norm $\\|\\Delta w\\|_2$")
    axes[0].set_xlabel("FedProx proximal coefficient")
    axes[0].grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].set_title("(a) Drift magnitude", loc="left", fontweight="bold")

    axes[1].plot(pos, summary["interclient_dispersion"], "s-",
                 color="#E76F51", linewidth=1.6)
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels([f"$\\mu = {mu:g}$" for mu in summary["mu"]])
    axes[1].set_ylabel("Inter-client update-norm dispersion (mean round-wise SD)")
    axes[1].set_xlabel("FedProx proximal coefficient")
    axes[1].grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].set_title("(b) Inter-client dispersion", loc="left", fontweight="bold")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"F_mu_sensitivity_mechanism.{ext}")
    plt.close(fig)


def plot_convergence():
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8))
    for mu in MU_VALUES:
        hist = load_history_csvs(mu)
        if not hist: continue
        mat = np.full((len(hist), NUM_ROUNDS), np.nan)
        for i, (_, df) in enumerate(hist.items()):
            for _, row in df.iterrows():
                r = int(row["round"])
                if 1 <= r <= NUM_ROUNDS:
                    mat[i, r - 1] = float(row.get("val_macro_f1", np.nan))
        rounds = np.arange(1, NUM_ROUNDS + 1)
        mean = np.nanmean(mat, axis=0)
        sem  = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
        label = f"FedAvg ($\\mu=0$)" if mu == 0.0 else f"$\\mu = {mu:g}$"
        ax.plot(rounds, mean, color=PALETTE[mu], linewidth=1.5, label=label)
        ax.fill_between(rounds, mean - sem, mean + sem,
                        color=PALETTE[mu], alpha=0.15, linewidth=0)
    ax.set_xlim(1, NUM_ROUNDS)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation macro-F1 (mean $\\pm$ SEM)")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"F_mu_sensitivity_convergence.{ext}")
    plt.close(fig)


def plot_outcome_and_convergence(summary: pd.DataFrame):
    """Combined 2-panel figure: (a) endpoint vs mu, (b) trajectory by mu."""
    fig, (ax_o, ax_c) = plt.subplots(
        1, 2, figsize=(13.5, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.22},
    )

    # --- (a) Endpoint vs mu ---------------------------------------------
    pos = np.arange(len(summary))
    ax_o.errorbar(
        pos, summary["macro_f1_mean"],
        yerr=[
            summary["macro_f1_mean"] - summary["macro_f1_lo"],
            summary["macro_f1_hi"] - summary["macro_f1_mean"],
        ],
        fmt="o-", color="#3D5A80", linewidth=1.6, capsize=4,
    )
    fa_mean = summary.loc[summary["mu"] == 0.0, "macro_f1_mean"].values
    if len(fa_mean):
        ax_o.axhline(fa_mean[0], color="#7FBF94", linestyle="--",
                     linewidth=1.2,
                     label=f"FedAvg ($\\mu=0$) = {fa_mean[0]:.3f}")
        ax_o.legend(frameon=False, loc="upper left")
    y_lo, y_hi = ax_o.get_ylim()
    ax_o.set_ylim(y_lo, y_hi + 0.025)
    ax_o.set_xticks(pos)
    ax_o.set_xticklabels([
        f"$\\mu = {mu:g}$" + ("\n(FedAvg)" if mu == 0.0 else "")
        for mu in summary["mu"]
    ])
    ax_o.set_ylabel("Test macro-F1 (mean, 95\\% bootstrap CI)")
    ax_o.set_xlabel("FedProx proximal coefficient")
    ax_o.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax_o.spines["top"].set_visible(False)
    ax_o.spines["right"].set_visible(False)
    ax_o.set_title("(a) Final test macro-F1", loc="left", fontweight="bold")

    # --- (b) Trajectory by mu -------------------------------------------
    for mu in MU_VALUES:
        hist = load_history_csvs(mu)
        if not hist:
            continue
        mat = np.full((len(hist), NUM_ROUNDS), np.nan)
        for i, (_, df) in enumerate(hist.items()):
            for _, row in df.iterrows():
                r = int(row["round"])
                if 1 <= r <= NUM_ROUNDS:
                    mat[i, r - 1] = float(row.get("val_macro_f1", np.nan))
        rounds = np.arange(1, NUM_ROUNDS + 1)
        mean = np.nanmean(mat, axis=0)
        sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(
            np.sum(~np.isnan(mat), axis=0)
        )
        label = "FedAvg ($\\mu=0$)" if mu == 0.0 else f"$\\mu = {mu:g}$"
        ax_c.plot(rounds, mean, color=PALETTE[mu], linewidth=1.5, label=label)
        ax_c.fill_between(rounds, mean - sem, mean + sem,
                          color=PALETTE[mu], alpha=0.15, linewidth=0)
    ax_c.set_xlim(1, NUM_ROUNDS)
    ax_c.set_xlabel("Communication round")
    ax_c.set_ylabel("Validation macro-F1 (mean $\\pm$ SEM)")
    ax_c.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.legend(frameon=False, fontsize=9, loc="lower right")
    ax_c.set_title("(b) Validation trajectory by $\\mu$",
                   loc="left", fontweight="bold")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"F_mu_sensitivity_outcome_and_convergence.{ext}")
    plt.close(fig)


def print_decoupling_table(summary: pd.DataFrame):
    print()
    print("Decoupling table (paste into thesis as Table mu-decoupling):")
    print(f"{'mu':>8} {'n':>3} {'macroF1':>14} {'pairedΔvsFA':>18} "
          f"{'mean‖Δw‖':>10} {'dispersion':>11} {'r→95%FA':>9}")
    print("-" * 78)
    for _, row in summary.iterrows():
        mu = row["mu"]
        n = int(row["n_seeds"])
        m = row["macro_f1_mean"]; lo = row["macro_f1_lo"]; hi = row["macro_f1_hi"]
        p = row["paired_delta_mean"]; plo = row["paired_delta_lo"]; phi = row["paired_delta_hi"]
        un = row["mean_update_norm"]
        disp = row["interclient_dispersion"]
        r2 = row["rounds_to_threshold_median"]
        macro_str = (f"{m:.3f} [{lo:.3f},{hi:.3f}]"
                     if not np.isnan(m) else "  --  ")
        paired_str = (f"{p:+.3f} [{plo:+.3f},{phi:+.3f}]"
                      if not np.isnan(p) else "  --  ")
        print(f"{mu:>8} {n:>3} {macro_str:>14} {paired_str:>18} "
              f"{un:>10.3f} {disp:>11.3f} {r2:>9.0f}")


def main():
    if not SWEEP_DIR.exists():
        print(f"Sweep directory does not exist yet: {SWEEP_DIR}")
        print("Run the smoke test first, then submit the full sweep, then re-run this script.")
        return

    n_jsons = len(list(SWEEP_DIR.glob("test_at_best_*.json")))
    n_norms = len(list(SWEEP_DIR.glob("client_update_norms_*.csv")))
    print(f"Sweep dir: {SWEEP_DIR}")
    print(f"  test JSONs found:   {n_jsons}")
    print(f"  update-norm CSVs:   {n_norms}")
    if n_jsons == 0:
        print("No runs found. Nothing to aggregate.")
        return

    summary = summarise()
    print()
    print(summary.to_string(index=False))

    plot_outcome(summary)
    plot_paired_delta(summary)
    plot_mechanism(summary)
    plot_convergence()
    plot_outcome_and_convergence(summary)
    print_decoupling_table(summary)

    print()
    print("Wrote figures to:", OUT_FIG)


if __name__ == "__main__":
    main()
