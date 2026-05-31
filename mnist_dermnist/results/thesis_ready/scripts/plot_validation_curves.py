"""Plot validation-macro-F1 curves for all 7 new experiments.

Reads history_*.csv files from each experiment directory and produces
one figure per experiment showing val_macro_f1 vs round, with one curve
per (condition × seed). Selected-round markers are overlaid.

Outputs:
  mnist_dermnist/results/thesis_ready/figures/
    F_val_curves_perfect_storm.{pdf,png}
    F_val_curves_li2020_asymmetric.{pdf,png}
    F_val_curves_extended_rounds_L3.{pdf,png}
    F_val_curves_node_pinned_L4.{pdf,png}
    F_val_curves_mu_sweep_ladder.{pdf,png}
    F_val_curves_asymmetric_mu.{pdf,png}
    F_val_curves_fednova_unequal_E.{pdf,png}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
RESULTS = REPO_ROOT / "mnist_dermnist/results"
OUT_FIG = REPO_ROOT / "mnist_dermnist/results/thesis_ready/figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def _selected_round(hist_file: Path) -> int | None:
    """Find the matching test_at_best JSON for this history CSV and return its selected_round."""
    json_path = hist_file.parent / hist_file.name.replace("history_", "test_at_best_").replace(".csv", ".json")
    if not json_path.exists():
        return None
    try:
        return int(json.load(open(json_path)).get("selected_round", -1))
    except Exception:
        return None


def plot_curves(curves: list[dict], title: str, out_stem: str,
                ylim: tuple[float, float] = (0.0, 0.7),
                xlim: tuple[float, float] | None = None):
    """curves: list of dicts with keys 'df', 'label', 'color', 'linestyle', 'alpha', 'sel_round'."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    legend_seen = set()
    for c in curves:
        df = c["df"]
        if df is None or len(df) == 0 or "val_macro_f1" not in df.columns:
            continue
        label = c.get("label") if c.get("label") not in legend_seen else None
        if label:
            legend_seen.add(label)
        ax.plot(df["round"], df["val_macro_f1"],
                color=c.get("color", "#444"),
                linestyle=c.get("linestyle", "-"),
                linewidth=c.get("linewidth", 1.2),
                alpha=c.get("alpha", 0.85),
                label=label)
        sr = c.get("sel_round")
        if sr is not None and sr > 0:
            # Mark selected-round with a small marker
            row = df[df["round"] == sr]
            if len(row):
                ax.scatter(sr, float(row["val_macro_f1"].iloc[0]),
                           color=c.get("color", "#444"),
                           edgecolor="white", linewidth=0.8, s=45, zorder=4)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)
    if len(legend_seen) > 0:
        ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"{out_stem}.{ext}")
    plt.close(fig)
    print(f"  Wrote {OUT_FIG / out_stem}.pdf")


# === 1. PERFECT-STORM L4 ===
print("=== Perfect-storm L4 ===")
PS_DIR = RESULTS / "fedprox_perfect_storm_L4"
ps_stems = [
    ("FedAvg + drop (Li §5.2 FA)",     "fedavg_mu0.0_E20_sh-random_stragglers_drop",  "#C03A2B"),
    ("FedProx μ=1.0 + γ-inexact",      "fedprox_mu1.0_E20_sh-random_stragglers",      "#3D5A80"),
    ("FedProx μ=0.01 + γ-inexact",     "fedprox_mu0.01_E20_sh-random_stragglers",     "#7FBF94"),
]
curves = []
for label, stem, color in ps_stems:
    for seed in (42, 123, 456):
        f = PS_DIR / f"history_{stem}_s{seed}.csv"
        if not f.exists():
            continue
        curves.append(dict(df=pd.read_csv(f), label=label, color=color,
                           linewidth=1.3, alpha=0.7,
                           sel_round=_selected_round(f)))
plot_curves(curves, "Perfect-storm L4 — validation macro-F1 trajectories (3 seeds)",
            "F_val_curves_perfect_storm",
            ylim=(0.0, 0.60))


# === 2. LI 2020 §5.2 ASYMMETRIC PROTOCOL ===
print("=== Li 2020 §5.2 ===")
L2_DIR = RESULTS / "li2020_asymmetric_L4"
li_stems = [
    ("1. FedAvg baseline (no straggler)",       "fedavg_mu0.0_E20",                              "#7FBF94"),
    ("2. FedAvg + drop + straggler",            "fedavg_mu0.0_E20_sh-fixed_stragglers_drop",     "#C03A2B"),
    ("3. FedProx + γ-inexact + straggler ⭐",    "fedprox_mu0.01_E20_sh-fixed_stragglers",        "#3D5A80"),
    ("4. FedProx + drop control",               "fedprox_mu0.01_E20_sh-fixed_stragglers_drop",   "#C9A227"),
]
curves = []
for label, stem, color in li_stems:
    for seed in (42, 123, 456):
        f = L2_DIR / f"history_{stem}_s{seed}.csv"
        if not f.exists():
            continue
        curves.append(dict(df=pd.read_csv(f), label=label, color=color,
                           linewidth=1.3, alpha=0.7,
                           sel_round=_selected_round(f)))
plot_curves(curves, "Li 2020 §5.2 asymmetric protocol on L4 — validation macro-F1 (3 seeds)",
            "F_val_curves_li2020_asymmetric", ylim=(0.0, 0.65))


# === 3. EXTENDED-ROUNDS L3 ===
print("=== Extended-rounds L3 ===")
ER_DIR = RESULTS / "extended_rounds_L3"
curves = []
for seed in (42, 123, 456):
    for algo, mu, color, label in [
        ("fedavg", "0.0", "#7FBF94", "FedAvg"),
        ("fedprox", "0.01", "#3D5A80", "FedProx (μ=0.01)"),
    ]:
        f = ER_DIR / f"history_{algo}_mu{mu}_E20_s{seed}.csv"
        if not f.exists():
            continue
        curves.append(dict(df=pd.read_csv(f), label=label, color=color,
                           linewidth=1.3, alpha=0.7,
                           sel_round=_selected_round(f)))
plot_curves(curves, "Extended-rounds L3 — validation macro-F1 trajectories (250 rounds, 3 seeds)",
            "F_val_curves_extended_rounds_L3",
            ylim=(0.0, 0.70), xlim=(0, 250))


# === 4. NODE-PINNED L4 ===
print("=== Node-pinned L4 ===")
NP_DIR = RESULTS / "node_pinned_L4"
curves = []
for seed in (42, 123, 456):
    for algo, mu, color, label in [
        ("fedavg", "0.0", "#7FBF94", "FedAvg"),
        ("fedprox", "0.01", "#3D5A80", "FedProx (μ=0.01)"),
    ]:
        f = NP_DIR / f"history_{algo}_mu{mu}_E20_s{seed}.csv"
        if not f.exists():
            continue
        curves.append(dict(df=pd.read_csv(f), label=label, color=color,
                           linewidth=1.3, alpha=0.7,
                           sel_round=_selected_round(f)))
plot_curves(curves, "Node-pinned L4 — validation macro-F1 trajectories (variance isolation, 3 seeds)",
            "F_val_curves_node_pinned_L4", ylim=(0.0, 0.65))


# === 5. μ-SWEEP LADDER ===
print("=== μ-sweep ladder ===")
MS_ROOT = RESULTS / "mu_sweep_ladder"
LADDER_LEVELS = {
    "L0": ("two_client_50_50_stratified_iid", "IID 50/50"),
    "L2": ("two_client_50_50_label_skew_only", "Label-skew 50/50"),
    "L4": ("two_client_90_10_rare_stress",     "Severe 90/10"),
}
MU_COLORS = {"0.001": "#7FBF94", "0.01": "#3D5A80", "0.1": "#C9A227", "1.0": "#C03A2B"}

# One figure per level
for level, (part, lbl) in LADDER_LEVELS.items():
    d = MS_ROOT / f"{level}_{part}"
    curves = []
    # Add FedAvg baseline (from heterogeneity_ladder/ originally)
    fa_dir = RESULTS / f"heterogeneity_ladder/{level}_{part}"
    fa_hist = fa_dir / "history_fedavg_mu0.0_E20_s42.csv"
    if fa_hist.exists():
        curves.append(dict(df=pd.read_csv(fa_hist), label="FedAvg baseline",
                           color="#666666", linewidth=2.0, linestyle="--", alpha=0.9,
                           sel_round=_selected_round(fa_hist)))
    for mu in ("0.001", "0.01", "0.1", "1.0"):
        f = d / f"history_fedprox_mu{mu}_E20_s42.csv"
        if not f.exists():
            continue
        curves.append(dict(df=pd.read_csv(f), label=f"FedProx μ={mu}",
                           color=MU_COLORS[mu], linewidth=1.5, alpha=0.85,
                           sel_round=_selected_round(f)))
    plot_curves(curves,
                f"μ-sweep at {level} ({lbl}, JS = {0.0 if level=='L0' else 0.104 if level=='L2' else 0.385:.3f}) — single seed (42)",
                f"F_val_curves_mu_sweep_{level}", ylim=(0.0, 0.70))


# === 6. ASYMMETRIC PER-CLIENT μ ===
print("=== Asymmetric μ L4 ===")
AS_DIR = RESULTS / "asymmetric_mu_L4"
asym_stems = [
    ("FedAvg",                          "fedavg_mu0.0_E20",                              "#7FBF94"),
    ("FedProx symmetric (μ=0.01)",      "fedprox_mu0.01_E20",                            "#3D5A80"),
    ("⭐ Anchor-large (μ_0=0.01, μ_1=0)", "fedprox_mu0.01_E20_muPC-c0m0.01-c1m0.0",        "#C03A2B"),
    ("Anchor-small CTRL (μ_0=0, μ_1=0.01)", "fedprox_mu0.01_E20_muPC-c0m0.0-c1m0.01",     "#C9A227"),
]
curves = []
for label, stem, color in asym_stems:
    for seed in (42, 123, 456):
        f = AS_DIR / f"history_{stem}_s{seed}.csv"
        if not f.exists():
            continue
        curves.append(dict(df=pd.read_csv(f), label=label, color=color,
                           linewidth=1.3, alpha=0.7,
                           sel_round=_selected_round(f)))
plot_curves(curves, "Asymmetric per-client μ on L4 — validation macro-F1 (3 seeds, Yao 2024 ablation)",
            "F_val_curves_asymmetric_mu", ylim=(0.0, 0.65))


# === 7. FEDNOVA × EQUAL/UNEQUAL-E ===
print("=== FedNova × E ===")
FN_ROOT = RESULTS / "fednova_unequal_E"
fn_levels = [("L3", "two_client_70_30_rare_enriched"), ("L4", "two_client_90_10_rare_stress")]
fn_stems = [
    ("FedAvg (equal-E)",    "fedavg_mu0.0_E20",                      "#7FBF94", "-"),
    ("FedAvg (unequal-E)",  "fedavg_mu0.0_E20_sh-fixed_stragglers",  "#7FBF94", "--"),
    ("FedProx (equal-E)",   "fedprox_mu0.01_E20",                     "#3D5A80", "-"),
    ("FedProx (unequal-E)", "fedprox_mu0.01_E20_sh-fixed_stragglers", "#3D5A80", "--"),
    ("FedNova (equal-E)",   "fednova_mu0.0_E20",                      "#C9A227", "-"),
    ("FedNova (unequal-E)", "fednova_mu0.0_E20_sh-fixed_stragglers",  "#C9A227", "--"),
]
for level, part in fn_levels:
    d = FN_ROOT / f"{level}_{part}"
    curves = []
    for label, stem, color, ls in fn_stems:
        f = d / f"history_{stem}_s42.csv"
        if not f.exists():
            continue
        curves.append(dict(df=pd.read_csv(f), label=label, color=color,
                           linestyle=ls, linewidth=1.5, alpha=0.85,
                           sel_round=_selected_round(f)))
    plot_curves(curves,
                f"FedNova × equal/unequal-E at {level} — validation macro-F1 (seed 42)",
                f"F_val_curves_fednova_unequal_E_{level}", ylim=(0.0, 0.65))

print()
print("Done.  All validation-curve figures in:", OUT_FIG)
