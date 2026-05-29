"""Analyse the local-epoch (E) sweep on Dirichlet α=0.1 — FedProx mechanism test.

Direct replication-style analysis of the canonical FedProx-paper §5.2
mechanism test (Li et al. 2020 MLSys, Fig. 4):

  Hypothesis: Δ(FedProx − FedAvg) grows monotonically with E because
  client drift accumulates approximately linearly with local SGD steps
  (Karimireddy et al. 2020 SCAFFOLD §3). At E=1 the proximal term has
  nothing to restrain (Δ ≈ 0); at high E FedAvg may oscillate while
  FedProx stays stable.

Data sources:
  - mnist_dermnist/results/e_sweep_dirichlet_a01/  (this sweep)
  - mnist_dermnist/results/dirichlet_a01/          (existing E=20 anchor;
                                                    reused for free)

Outputs (all under e_sweep_dirichlet_a01/analysis/):
  1. headline_e_sweep.csv             — per (E, seed, algo) global metrics
  2. delta_vs_E.csv                   — paired Δ per (E, seed) + means + SEM
  3. delta_vs_E_curve.png             — THE figure: Δ macro-F1 vs E, CI band
  4. loss_curves_by_E.png             — train/val loss faceted by E
                                        (FedAvg instability check at high E)
  5. update_norm_by_E.png             — per-client drift trajectories, faceted by E
  6. per_class_delta_at_E_max.png     — where the advantage concentrates at max E
  7. summary.json                     — rolled-up numbers + Wilcoxon p per E
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mnist_dermnist.data.partition import CLASS_NAMES, NUM_CLASSES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "mnist_dermnist" / "results" / "e_sweep_dirichlet_a01"
DEFAULT_ANCHOR_DIR = REPO_ROOT / "mnist_dermnist" / "results" / "dirichlet_a01"

_FILE_PAT = re.compile(
    r"test_at_best_(?P<algo>fedavg|fedprox)_mu(?P<mu>[0-9.]+)"
    r"_E(?P<E>\d+)(?:_sh-[a-z_]+)?(?:_C[0-9.]+)?"
    r"(?:_arch-[a-z]+)?(?:_drop)?_s(?P<seed>\d+)\.json"
)


def _load_dir(d: Path,
              wanted_mu: float = 0.01,
              wanted_seeds: Optional[List[int]] = None,
              wanted_es: Optional[List[int]] = None) -> List[Dict]:
    """Load every test_at_best_*.json + companion artifacts from a directory."""
    runs: List[Dict] = []
    if not d.is_dir():
        return runs
    for j in sorted(d.glob("test_at_best_*.json")):
        m = _FILE_PAT.match(j.name)
        if not m:
            continue
        algo = m.group("algo")
        mu = float(m.group("mu"))
        E = int(m.group("E"))
        seed = int(m.group("seed"))
        if algo == "fedprox" and abs(mu - wanted_mu) > 1e-9:
            continue
        if wanted_seeds is not None and seed not in wanted_seeds:
            continue
        if wanted_es is not None and E not in wanted_es:
            continue
        with open(j) as f:
            doc = json.load(f)
        stem = j.stem.replace("test_at_best_", "")
        runs.append({
            "algo": algo, "mu": mu, "E": E, "seed": seed,
            "source_dir": d.name,
            "json_path": j,
            "history_path": d / f"history_{stem}.csv",
            "preds_path": d / f"test_predictions_{stem}.npz",
            "norms_path": d / f"client_update_norms_{stem}.csv",
            "doc": doc,
        })
    return runs


def _build_headline_table(runs: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        d = r["doc"]
        row = {
            "E": r["E"], "seed": r["seed"], "algorithm": r["algo"], "mu": r["mu"],
            "source": r["source_dir"],
            "macro_f1": d.get("macro_f1"),
            "balanced_accuracy": d.get("balanced_accuracy"),
            "accuracy": d.get("accuracy"),
            "loss": d.get("loss"),
            "selected_round": d.get("selected_round"),
        }
        for c, f1c in enumerate(d.get("per_class_f1", [])):
            row[f"f1_class_{c}"] = float(f1c)
        rows.append(row)
    return (pd.DataFrame(rows)
              .sort_values(["E", "seed", "algorithm"])
              .reset_index(drop=True))


def _paired_delta_per_E(headline: pd.DataFrame) -> pd.DataFrame:
    """Compute Δ = FedProx − FedAvg per (E, seed) where both algorithms ran."""
    rows = []
    for (E, seed), sub in headline.groupby(["E", "seed"]):
        algos = set(sub["algorithm"])
        if not {"fedavg", "fedprox"}.issubset(algos):
            continue
        a = sub[sub["algorithm"] == "fedavg"].iloc[0]
        p = sub[sub["algorithm"] == "fedprox"].iloc[0]
        row = {
            "E": E, "seed": seed,
            "delta_macro_f1": float(p["macro_f1"]) - float(a["macro_f1"]),
            "delta_balanced_acc": float(p["balanced_accuracy"]) - float(a["balanced_accuracy"]),
            "delta_accuracy": float(p["accuracy"]) - float(a["accuracy"]),
            "fedavg_macro_f1": float(a["macro_f1"]),
            "fedprox_macro_f1": float(p["macro_f1"]),
        }
        for c in range(NUM_CLASSES):
            col = f"f1_class_{c}"
            if col in a and col in p and pd.notna(a[col]) and pd.notna(p[col]):
                row[f"delta_f1_class_{c}"] = float(p[col]) - float(a[col])
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_per_E(deltas: pd.DataFrame) -> pd.DataFrame:
    """Mean / SEM / median / sign of Δ per E, plus Wilcoxon when n ≥ 4."""
    try:
        from scipy.stats import wilcoxon
        have_scipy = True
    except ImportError:
        have_scipy = False

    rows = []
    for E, sub in deltas.groupby("E"):
        vals = sub["delta_macro_f1"].to_numpy()
        n = len(vals)
        row = {
            "E": int(E),
            "n_seeds": n,
            "mean_delta": float(np.mean(vals)) if n else float("nan"),
            "median_delta": float(np.median(vals)) if n else float("nan"),
            "sem_delta": float(np.std(vals, ddof=0) / np.sqrt(n)) if n > 1 else float("nan"),
            "fedprox_wins": int(np.sum(vals > 0)),
        }
        if have_scipy and n >= 4 and not np.allclose(vals, 0):
            try:
                w = wilcoxon(vals, alternative="two-sided", zero_method="pratt")
                row["wilcoxon_p"] = float(w.pvalue)
            except ValueError:
                row["wilcoxon_p"] = float("nan")
        else:
            row["wilcoxon_p"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("E").reset_index(drop=True)


def _plot_delta_vs_E(deltas: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    """The canonical figure: Δ macro-F1 vs E with CI band."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if not deltas.empty:
        for _, row in deltas.iterrows():
            ax.scatter(row["E"], row["delta_macro_f1"], color="#1f77b4",
                       alpha=0.45, s=22, zorder=2)
    s = summary.sort_values("E")
    if not s.empty:
        ax.plot(s["E"], s["mean_delta"], color="#1f77b4", marker="o", lw=2,
                label="mean Δ macro-F1", zorder=3)
        if s["sem_delta"].notna().any():
            ax.fill_between(s["E"],
                            s["mean_delta"] - 1.96 * s["sem_delta"].fillna(0),
                            s["mean_delta"] + 1.96 * s["sem_delta"].fillna(0),
                            color="#1f77b4", alpha=0.18,
                            label="±1.96 SEM (3 seeds)", zorder=1)
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("local epochs per round  (E)")
    ax.set_ylabel("Δ macro-F1   =   FedProx (μ=0.01)  −  FedAvg")
    ax.set_title("FedProx mechanism: drift correction grows with local-epoch count\n"
                 "dirichlet α=0.1 partition; replication of Li et al. 2020 MLSys §5.2")
    ax.grid(alpha=0.3, which="both")
    if not s.empty:
        ax.set_xticks(s["E"].tolist())
        ax.set_xticklabels([str(int(e)) for e in s["E"].tolist()])
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_loss_curves_by_E(runs: List[Dict], out: Path) -> None:
    """Validation-loss curves faceted by E — diagnoses FedAvg instability."""
    es = sorted({r["E"] for r in runs})
    if not es:
        return
    ncols = min(len(es), 3)
    nrows = (len(es) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows),
                             sharey=True, squeeze=False)
    for ax, E in zip(axes.flatten(), es):
        any_data = False
        for r in [x for x in runs if x["E"] == E]:
            if not r["history_path"].exists():
                continue
            h = pd.read_csv(r["history_path"])
            ycol = "val_loss" if "val_loss" in h.columns else None
            if ycol is None:
                continue
            ls = "--" if r["algo"] == "fedavg" else "-"
            color = "#d62728" if r["algo"] == "fedavg" else "#2ca02c"
            ax.plot(h["round"], h[ycol], ls=ls, color=color, alpha=0.85,
                    label=f"{r['algo']} s={r['seed']}")
            any_data = True
        ax.set_title(f"E = {E}")
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
        if any_data:
            ax.legend(fontsize=7, ncol=2, loc="upper right")
    axes[0, 0].set_ylabel("val cross-entropy loss")
    for ax in axes.flatten()[len(es):]:
        ax.set_visible(False)
    fig.suptitle("Validation loss vs round, by E — FedAvg (red dashed) vs FedProx (green solid)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_update_norms_by_E(runs: List[Dict], out: Path) -> None:
    """Per-client update-norm trajectories, faceted by E."""
    es = sorted({r["E"] for r in runs if r["norms_path"].exists()})
    if not es:
        return
    ncols = min(len(es), 3)
    nrows = (len(es) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows),
                             sharey=False, squeeze=False)
    for ax, E in zip(axes.flatten(), es):
        any_data = False
        for r in [x for x in runs if x["E"] == E and x["norms_path"].exists()]:
            un = pd.read_csv(r["norms_path"])
            mean_per_round = un.groupby("round")["update_norm"].mean()
            ls = "--" if r["algo"] == "fedavg" else "-"
            color = "#d62728" if r["algo"] == "fedavg" else "#2ca02c"
            ax.plot(mean_per_round.index, mean_per_round.values, ls=ls,
                    color=color, alpha=0.85, label=f"{r['algo']} s={r['seed']}")
            any_data = True
        ax.set_title(f"E = {E}")
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
        if any_data:
            ax.legend(fontsize=7, ncol=2, loc="upper right")
    axes[0, 0].set_ylabel("mean per-client ‖w_k − w_global‖₂")
    for ax in axes.flatten()[len(es):]:
        ax.set_visible(False)
    fig.suptitle("Client drift vs round, by E (Karimireddy et al. 2020 framework)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_per_class_delta_at_E_max(deltas: pd.DataFrame, out: Path) -> None:
    if deltas.empty:
        return
    E_max = int(deltas["E"].max())
    sub = deltas[deltas["E"] == E_max]
    if sub.empty:
        return
    means = []
    sems = []
    for c in range(NUM_CLASSES):
        col = f"delta_f1_class_{c}"
        if col not in sub.columns:
            means.append(0.0)
            sems.append(0.0)
            continue
        vals = sub[col].dropna().to_numpy()
        means.append(float(np.mean(vals)) if len(vals) else 0.0)
        sems.append(float(np.std(vals, ddof=0) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(NUM_CLASSES)
    ax.bar(xs, means, yerr=sems, capsize=4, color="#2ca02c", alpha=0.8)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}\n{CLASS_NAMES[c][:10]}" for c in xs], fontsize=8)
    ax.set_ylabel("Δ F1   =   FedProx − FedAvg")
    ax.set_title(f"Per-class delta at E = {E_max}  (where drift is largest)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--anchor-dir", default=str(DEFAULT_ANCHOR_DIR),
                    help="Directory holding existing E=20 dirichlet_a01 results "
                         "to reuse as the E=20 anchor (set to '' to disable).")
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--seeds", default="42,123,456",
                    help="Comma-separated seeds to include from the anchor dir.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted_seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print(f"Loading sweep runs from {results_dir}")
    sweep_runs = _load_dir(results_dir, wanted_mu=args.mu, wanted_seeds=wanted_seeds)
    print(f"  → {len(sweep_runs)} runs found.")

    anchor_runs: List[Dict] = []
    if args.anchor_dir:
        anchor_dir = Path(args.anchor_dir)
        print(f"Loading E=20 anchor from {anchor_dir} (seeds {wanted_seeds})")
        anchor_runs = _load_dir(anchor_dir, wanted_mu=args.mu,
                                wanted_seeds=wanted_seeds, wanted_es=[20])
        print(f"  → {len(anchor_runs)} anchor runs found.")

    runs = sweep_runs + anchor_runs
    if not runs:
        print("\nNo runs found yet. Run the submit script first:")
        print("  MODE=pilot bash mnist_dermnist/scripts/submit_e_sweep_dirichlet_a01.sh")
        return

    for r in runs:
        print(f"  - E={r['E']:>3}  {r['algo']:7s}  μ={r['mu']:>5}  seed={r['seed']}  "
              f"macro_f1={r['doc'].get('macro_f1', float('nan')):.4f}  ({r['source_dir']})")

    print("\n[1] headline_e_sweep.csv")
    headline = _build_headline_table(runs)
    headline.to_csv(out_dir / "headline_e_sweep.csv", index=False)

    print("[2] delta_vs_E.csv")
    deltas = _paired_delta_per_E(headline)
    deltas.to_csv(out_dir / "delta_vs_E.csv", index=False)

    summary = _summary_per_E(deltas)
    summary.to_csv(out_dir / "summary_per_E.csv", index=False)

    print("[3] delta_vs_E_curve.png")
    _plot_delta_vs_E(deltas, summary, out_dir / "delta_vs_E_curve.png")

    print("[4] loss_curves_by_E.png")
    _plot_loss_curves_by_E(runs, out_dir / "loss_curves_by_E.png")

    print("[5] update_norm_by_E.png")
    _plot_update_norms_by_E(runs, out_dir / "update_norm_by_E.png")

    print("[6] per_class_delta_at_E_max.png")
    _plot_per_class_delta_at_E_max(deltas, out_dir / "per_class_delta_at_E_max.png")

    print("[7] summary.json")
    rolled = {
        "n_runs": int(len(headline)),
        "seeds_included": sorted(set(int(s) for s in headline["seed"].unique())),
        "E_values": sorted(set(int(e) for e in headline["E"].unique())),
        "per_E": summary.to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(rolled, f, indent=2)

    print(f"\nAll outputs in: {out_dir}")
    if not summary.empty:
        print("\n=== Δ macro-F1 vs E (paired FedProx − FedAvg) ===")
        for _, r in summary.iterrows():
            line = (f"  E={int(r['E']):>3}  n={int(r['n_seeds'])}  "
                    f"mean Δ = {r['mean_delta']:+.4f}")
            if not np.isnan(r["sem_delta"]):
                line += f"  SEM = {r['sem_delta']:.4f}"
            if not np.isnan(r["wilcoxon_p"]):
                line += f"  Wilcoxon p = {r['wilcoxon_p']:.3g}"
            line += f"   wins = {int(r['fedprox_wins'])}/{int(r['n_seeds'])}"
            print(line)


if __name__ == "__main__":
    main()
