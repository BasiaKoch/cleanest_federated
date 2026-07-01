"""Build thesis-ready figures from the engineered 2-client pilot and the
balanced_paired E-sweep pilot.

Three multi-panel figures, each grouping closely related findings:

  thesis_fig_A_mechanism.png
    A1) Per-client drift (update-norm) trajectories - engineered 2-client,
        E=20, seed=42. Smoothed (rolling mean). Shows the proximal term
        restraining the dominant client more than the specialist.
    A2) Mechanism dormancy at E=1 (balanced_paired, seed=42): per-client
        mean drift FedAvg vs FedProx bars - they are essentially equal,
        confirming μ has nothing to do at E=1.

  thesis_fig_B_per_class.png
    B1) Per-class F1 delta (FedProx − FedAvg) for the engineered 2-client
        pilot, colour-coded by client ownership.
    B2) Δ recall vs Δ precision for the three critical classes; point size
        encodes test-set support. Shows the precision-not-recall pattern.

  thesis_fig_C_honesty.png
    C1) Per-seed Δ macro-F1 at E=20 on balanced_paired (n=3): individual
        points + mean - shows the high seed-to-seed variance.
    C2) Engineered 2-client (single seed) vs balanced_paired (mean of 3
        seeds) Δ macro-F1 at E=20, μ=0.01 - illustrates that the engineered
        partition shows a larger directional signal than the headline.

All figures saved under results/thesis_figures/. Read-only on results/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fl_dermamnist.data.partition import CLASS_NAMES


REPO_ROOT = Path(__file__).resolve().parents[2]
ENG_DIR = REPO_ROOT / "fl_dermamnist" / "results" / "two_client_90_10_rare_stress"
ESWEEP_DIR = REPO_ROOT / "fl_dermamnist" / "results" / "e_sweep"
HEADLINE_DIR = REPO_ROOT / "fl_dermamnist" / "results" / "headline"
OUT_DIR = REPO_ROOT / "fl_dermamnist" / "results" / "thesis_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Critical/rare classes engineered onto Client 1 in the 2-client partition.
CRITICAL = {3, 4, 6}

# Class-name labels for plotting (short, readable)
CLASS_LABEL = {
    0: "actinic", 1: "basal", 2: "benign_kerat", 3: "dermato",
    4: "melanoma", 5: "nevi", 6: "vascular",
}


def _load_eng_pilot():
    """Single-seed engineered pilot - returns (preds, history, norms) per algo."""
    out = {}
    for algo in ("fedavg", "fedprox"):
        suffix = "mu0.0_E20_s42" if algo == "fedavg" else "mu0.01_E20_s42"
        out[algo] = {
            "meta": json.load(open(ENG_DIR / f"test_at_best_{algo}_{suffix}.json")),
            "history": pd.read_csv(ENG_DIR / f"history_{algo}_{suffix}.csv"),
            "norms": pd.read_csv(ENG_DIR / f"client_update_norms_{algo}_{suffix}.csv"),
            "preds": np.load(ENG_DIR / f"test_predictions_{algo}_{suffix}.npz"),
        }
    return out


def _load_esweep_e1():
    """E=1 balanced_paired pilot - returns history + norms per algo."""
    out = {}
    for algo in ("fedavg", "fedprox"):
        suffix = "mu0.0_E1_s42" if algo == "fedavg" else "mu0.01_E1_s42"
        out[algo] = {
            "meta": json.load(open(ESWEEP_DIR / f"test_at_best_{algo}_{suffix}.json")),
            "norms": pd.read_csv(ESWEEP_DIR / f"client_update_norms_{algo}_{suffix}.csv"),
        }
    return out


def _load_headline_E20():
    """Headline balanced_paired E=20 - seeds 42/123/456 paired."""
    pairs = []
    for seed in (42, 123, 456):
        try:
            a = json.load(open(HEADLINE_DIR / f"test_at_best_fedavg_mu0.0_E20_s{seed}.json"))
            p = json.load(open(HEADLINE_DIR / f"test_at_best_fedprox_mu0.01_E20_s{seed}.json"))
            pairs.append({
                "seed": seed,
                "fedavg_macro_f1": float(a["macro_f1"]),
                "fedprox_macro_f1": float(p["macro_f1"]),
                "delta": float(p["macro_f1"]) - float(a["macro_f1"]),
            })
        except FileNotFoundError:
            pass
    return pd.DataFrame(pairs)


# ============================================================================
# FIGURE A - MECHANISM (drift trajectories + E=1 dormancy)
# ============================================================================

def fig_A_mechanism(eng, e1):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6),
                                   gridspec_kw={"width_ratios": [2.0, 1.0]})

    # ----- A1: engineered drift trajectories, smoothed -----
    palette = {("fedavg", 0): "#1f77b4", ("fedavg", 1): "#aec7e8",
               ("fedprox", 0): "#d62728", ("fedprox", 1): "#ff9896"}
    window = 5
    for algo in ("fedavg", "fedprox"):
        un = eng[algo]["norms"]
        for cid in sorted(un["client_id"].unique()):
            sub = un[un["client_id"] == cid].sort_values("round")
            smoothed = sub["update_norm"].rolling(window, min_periods=1).mean()
            ls = "--" if algo == "fedavg" else "-"
            lbl = f"{algo}  client {cid} " + ("(86% dominant)" if cid == 0 else "(14% specialist)")
            axL.plot(sub["round"], smoothed, ls=ls, lw=2.0,
                     color=palette[(algo, cid)], label=lbl, alpha=0.95)

    # Annotate the mean-drift reduction per client.
    avg_means = eng["fedavg"]["norms"].groupby("client_id")["update_norm"].mean()
    prox_means = eng["fedprox"]["norms"].groupby("client_id")["update_norm"].mean()
    for cid in (0, 1):
        red = 1.0 - prox_means[cid] / avg_means[cid]
        axL.text(0.99, 0.95 - 0.07 * cid,
                 f"client {cid}:  drift  FedAvg={avg_means[cid]:.2f}  →  "
                 f"FedProx={prox_means[cid]:.2f}    (−{red*100:.0f}%)",
                 transform=axL.transAxes, ha="right", va="top", fontsize=9,
                 family="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray"))

    axL.set_xlabel("communication round")
    axL.set_ylabel("‖w_k − w_global‖₂  (5-round moving average)")
    axL.set_title("A1.  Engineered 2-client (E=20, seed=42):\n"
                  "FedProx restrains drift on both clients, more so on the dominant",
                  fontsize=11)
    axL.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axL.grid(alpha=0.3)

    # ----- A2: mechanism dormancy at E=1 - bar chart by client -----
    e1_means = {algo: e1[algo]["norms"].groupby("client_id")["update_norm"].mean()
                for algo in ("fedavg", "fedprox")}
    n_clients = max(len(e1_means["fedavg"]), len(e1_means["fedprox"]))
    xs = np.arange(n_clients)
    w = 0.38
    axR.bar(xs - w / 2, e1_means["fedavg"].values, width=w,
            label="FedAvg", color="#1f77b4", alpha=0.85)
    axR.bar(xs + w / 2, e1_means["fedprox"].values, width=w,
            label="FedProx (μ=0.01)", color="#d62728", alpha=0.85)
    axR.set_xticks(xs)
    axR.set_xticklabels([f"c{c}" for c in range(n_clients)], fontsize=9)
    axR.set_xlabel("client id")
    axR.set_ylabel("mean drift over 150 rounds")
    overall_fa = e1["fedavg"]["norms"]["update_norm"].mean()
    overall_fp = e1["fedprox"]["norms"]["update_norm"].mean()
    pct = (1 - overall_fp / overall_fa) * 100
    axR.set_title(f"A2.  E=1 dormancy (balanced_paired, seed=42):\n"
                  f"overall mean  FA={overall_fa:.2f}  vs  FP={overall_fp:.2f}  "
                  f"(−{pct:.1f}%, dormant)",
                  fontsize=11)
    axR.legend(loc="upper right", fontsize=9)
    axR.grid(axis="y", alpha=0.3)

    fig.suptitle("Figure A — The FedProx mechanism: drift correction is partition- and E-dependent",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "thesis_fig_A_mechanism.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIGURE B - PER-CLASS STORY (Δ F1 + recall vs precision)
# ============================================================================

def fig_B_per_class(eng):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6),
                                   gridspec_kw={"width_ratios": [1.3, 1.0]})

    fa_f1 = eng["fedavg"]["meta"]["per_class_f1"]
    fp_f1 = eng["fedprox"]["meta"]["per_class_f1"]
    deltas = [fp_f1[c] - fa_f1[c] for c in range(7)]
    on_client_1 = [c in CRITICAL for c in range(7)]
    colors = ["#d62728" if x else "#1f77b4" for x in on_client_1]

    # ----- B1: per-class Δ F1 bar chart -----
    xs = np.arange(7)
    axL.bar(xs, deltas, color=colors, alpha=0.88,
            edgecolor="black", linewidth=0.6)
    for i, d in enumerate(deltas):
        axL.text(i, d + (0.005 if d > 0 else -0.012),
                 f"{d:+.3f}", ha="center", va="bottom" if d > 0 else "top",
                 fontsize=8.5)
    axL.axhline(0, color="black", lw=0.7)
    axL.set_xticks(xs)
    axL.set_xticklabels([f"{c}\n{CLASS_LABEL[c]}" for c in xs], fontsize=8.5)
    axL.set_ylabel("Δ F1   =   FedProx − FedAvg")
    axL.set_title("B1.  Per-class F1 delta (engineered 2-client, seed=42)\n"
                  "red = Client 1 (specialist, 14%);  blue = Client 0 (dominant, 86%)",
                  fontsize=11)
    axL.grid(axis="y", alpha=0.3)

    # ----- B2: Δ recall vs Δ precision scatter for critical classes -----
    preds_fa = eng["fedavg"]["preds"]
    preds_fp = eng["fedprox"]["preds"]
    targets = preds_fa["targets"]
    deltas_recall_precision = []
    for c in range(7):
        tp_a = ((preds_fa["predictions"] == c) & (targets == c)).sum()
        fp_a = ((preds_fa["predictions"] == c) & (targets != c)).sum()
        fn_a = ((preds_fa["predictions"] != c) & (targets == c)).sum()
        rec_a = tp_a / max(tp_a + fn_a, 1)
        prec_a = tp_a / max(tp_a + fp_a, 1)

        tp_p = ((preds_fp["predictions"] == c) & (targets == c)).sum()
        fp_p = ((preds_fp["predictions"] == c) & (targets != c)).sum()
        fn_p = ((preds_fp["predictions"] != c) & (targets == c)).sum()
        rec_p = tp_p / max(tp_p + fn_p, 1)
        prec_p = tp_p / max(tp_p + fp_p, 1)

        deltas_recall_precision.append({
            "class": c, "name": CLASS_LABEL[c],
            "support": int((targets == c).sum()),
            "d_recall": rec_p - rec_a,
            "d_precision": prec_p - prec_a,
        })
    df = pd.DataFrame(deltas_recall_precision)

    # All classes plotted; critical classes highlighted.
    for _, r in df.iterrows():
        is_crit = int(r["class"]) in CRITICAL
        color = "#d62728" if is_crit else "#9aa0a6"
        size = max(50, r["support"] / 5)
        axR.scatter(r["d_recall"], r["d_precision"], s=size, color=color,
                    alpha=0.85, edgecolors="black", linewidth=0.7,
                    zorder=3 if is_crit else 2)
        if is_crit:
            axR.annotate(r["name"], (r["d_recall"], r["d_precision"]),
                         xytext=(7, 4), textcoords="offset points", fontsize=9)
    axR.axhline(0, color="black", lw=0.5, ls="--")
    axR.axvline(0, color="black", lw=0.5, ls="--")
    axR.set_xlabel("Δ recall   =   FedProx − FedAvg")
    axR.set_ylabel("Δ precision   =   FedProx − FedAvg")
    axR.set_title("B2.  On rare classes, FedProx trades recall for precision\n"
                  "(point size ∝ test support; red = critical classes)",
                  fontsize=11)
    axR.grid(alpha=0.3)
    axR.text(0.02, 0.98, "upper-left = precision↑ recall↓\n(more selective)",
             transform=axR.transAxes, va="top", fontsize=8.5,
             style="italic", color="#555")

    fig.suptitle("Figure B — The per-class story: Δ F1 lands on Client 1's classes; "
                 "rare classes gain precision more than recall",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "thesis_fig_B_per_class.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIGURE C - EFFECT-SIZE HONESTY (seed variance + partition comparison)
# ============================================================================

def fig_C_honesty(eng, headline):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6),
                                   gridspec_kw={"width_ratios": [1.0, 1.0]})

    # ----- C1: seed-to-seed variance at E=20 on balanced_paired -----
    if not headline.empty:
        ys = headline["delta"].to_numpy()
        xs = np.zeros_like(ys)
        axL.scatter(xs, ys, s=120, color="#1f77b4", alpha=0.85,
                    edgecolors="black", linewidth=0.8, zorder=3)
        for s, d in zip(headline["seed"], ys):
            axL.annotate(f"seed {int(s)}", (0, d),
                         xytext=(15, 0), textcoords="offset points",
                         va="center", fontsize=9)
        m = float(np.mean(ys))
        axL.axhline(m, color="#d62728", lw=2, label=f"mean = {m:+.4f}")
        axL.axhline(0, color="black", lw=0.5, ls="--")
        axL.set_xlim(-0.4, 0.6)
        axL.set_xticks([])
        axL.set_ylabel("Δ macro-F1   =   FedProx − FedAvg")
        axL.set_title(f"C1.  Per-seed Δ at E=20 (balanced_paired, n={len(ys)}):\n"
                      "one seed dominates; two are ≈ 0",
                      fontsize=11)
        axL.legend(loc="upper right", fontsize=9)
        axL.grid(axis="y", alpha=0.3)

    # ----- C2: engineered (n=1) vs balanced_paired (n=3 mean) -----
    eng_delta = (float(eng["fedprox"]["meta"]["macro_f1"]) -
                 float(eng["fedavg"]["meta"]["macro_f1"]))
    headline_mean = float(headline["delta"].mean()) if not headline.empty else 0.0

    labels = [f"balanced_paired\n(headline)\nn={len(headline)} seeds, mean",
              "engineered 2-client\n(86/14 stress)\nn=1 seed"]
    values = [headline_mean, eng_delta]
    colors = ["#1f77b4", "#d62728"]
    bars = axR.bar(labels, values, color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.7, width=0.55)
    for b, v in zip(bars, values):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.001,
                 f"{v:+.4f}", ha="center", va="bottom", fontsize=11,
                 fontweight="bold")
    axR.axhline(0, color="black", lw=0.5, ls="--")
    axR.set_ylabel("Δ macro-F1   =   FedProx − FedAvg")
    axR.set_title("C2.  Effect-size by partition (E=20, μ=0.01):\n"
                  "engineered partition shows the larger directional signal",
                  fontsize=11)
    axR.grid(axis="y", alpha=0.3)

    fig.suptitle("Figure C — Effect-size honesty: high seed variance on headline; "
                 "engineered partition exceeds headline mean",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "thesis_fig_C_honesty.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Loading data …")
    eng = _load_eng_pilot()
    e1 = _load_esweep_e1()
    headline = _load_headline_E20()
    print(f"  engineered 2-client: seed 42 paired (FedAvg, FedProx)")
    print(f"  E=1 dormancy:        seed 42 paired (FedAvg, FedProx)")
    print(f"  headline E=20:       {len(headline)} paired seeds")

    print("\nBuilding figures …")
    fig_A_mechanism(eng, e1)
    print("  ✓ thesis_fig_A_mechanism.png")
    fig_B_per_class(eng)
    print("  ✓ thesis_fig_B_per_class.png")
    fig_C_honesty(eng, headline)
    print("  ✓ thesis_fig_C_honesty.png")

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
