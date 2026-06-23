"""Analysis for the engineered 2-client 90/10 rare-class stress test.

Reads test_at_best_*.json + history_*.csv + test_predictions_*.npz +
(optionally) client_update_norms_*.csv from
``fl_dermamnist/results/two_client_90_10_rare_stress/`` and writes:

  1. partition_class_counts.csv       — clients × classes table
  2. partition_heatmap.png            — same, as a normalised heatmap
  3. headline_metrics.csv             — per-seed FedAvg vs FedProx global metrics
  4. per_class_f1.csv                 — per-seed, per-class F1 for both algos
  5. per_class_delta.csv              — FedProx − FedAvg per class (per seed + mean)
  6. per_class_recall_precision.csv   — derived from saved predictions
  7. per_class_delta_bar.png          — bar plot of FedProx − FedAvg per class
  8. learning_curves.png              — val_macro_f1 vs round, paired
  9. confusion_matrix_fedavg_s<seed>.png   — confusion matrix per seed × algo
     confusion_matrix_fedprox_s<seed>.png
 10. update_norm_per_client.png       — if --log-update-norms was set
 11. summary.json                     — single rolled-up summary

Works with as few as one (FedAvg, FedProx) paired seed (pilot) and scales
to multi-seed runs. The script is read-only on results/ and the dataset.

Usage:
    PYTHONPATH=. python fl_dermamnist/analysis/analyse_two_client_90_10_rare_stress.py
    PYTHONPATH=. python fl_dermamnist/analysis/analyse_two_client_90_10_rare_stress.py \
        --results-dir fl_dermamnist/results/two_client_90_10_rare_stress \
        --mu 0.01
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Use a headless backend so the script runs without a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fl_dermamnist.data.partition import (
    CLASS_NAMES,
    NUM_CLASSES,
    class_count_table,
    two_client_90_10_rare_stress,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "fl_dermamnist" / "results" / "two_client_90_10_rare_stress"
DEFAULT_NPZ = REPO_ROOT / "dermamnist_64.npz"

# DermMNIST training-set class counts — used both as a sanity check and to
# label the heatmap.
TRAIN_CLASS_COUNTS = {0: 228, 1: 359, 2: 769, 3: 80, 4: 779, 5: 4693, 6: 99}

# The three classes that distinguish the engineered partition: 100% on Client 1.
CRITICAL_CLASSES = (3, 4, 6)
MELANOMA_CLASS = 4

_FILE_PAT = re.compile(
    r"test_at_best_(?P<algo>fedavg|fedprox)_mu(?P<mu>[0-9.]+)_E\d+"
    r"(?:_sh-[a-z_]+)?(?:_C[0-9.]+)?(?:_arch-[a-z]+)?(?:_drop)?_s(?P<seed>\d+)\.json"
)


def _load_runs(results_dir: Path, only_mu: Optional[float] = None) -> List[Dict]:
    """Load every test_at_best_*.json plus its history_*.csv + predictions npz."""
    runs: List[Dict] = []
    for j in sorted(results_dir.glob("test_at_best_*.json")):
        m = _FILE_PAT.match(j.name)
        if not m:
            continue
        algo = m.group("algo")
        mu = float(m.group("mu"))
        seed = int(m.group("seed"))
        if only_mu is not None and algo == "fedprox" and abs(mu - only_mu) > 1e-9:
            continue
        with open(j) as f:
            doc = json.load(f)
        stem = j.stem.replace("test_at_best_", "")
        history_csv = results_dir / f"history_{stem}.csv"
        preds_npz = results_dir / f"test_predictions_{stem}.npz"
        norms_csv = results_dir / f"client_update_norms_{stem}.csv"
        runs.append({
            "algo": algo, "mu": mu, "seed": seed, "stem": stem,
            "json_path": j, "history_path": history_csv,
            "preds_path": preds_npz, "norms_path": norms_csv,
            "doc": doc,
        })
    return runs


def _save_partition_artifacts(out_dir: Path, npz_path: Path) -> None:
    """Recompute the canonical partition (seed=42) and save count table + heatmap."""
    data = np.load(npz_path)
    labels = np.asarray(data["train_labels"]).reshape(-1).astype(np.int64)
    clients, _ = two_client_90_10_rare_stress(labels, seed=42)
    counts = class_count_table(clients, labels)
    counts.to_csv(out_dir / "partition_class_counts.csv")

    mat = counts.iloc[:, :NUM_CLASSES].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 2.5))
    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:d}", ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black",
                    fontsize=9)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"{c}\n{CLASS_NAMES[c][:10]}" for c in range(NUM_CLASSES)],
                       rotation=0, fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f"client {i}" for i in range(mat.shape[0])])
    ax.set_title("two_client_90_10_rare_stress: per-client class counts")
    fig.colorbar(im, ax=ax, label="# samples")
    fig.tight_layout()
    fig.savefig(out_dir / "partition_heatmap.png", dpi=120)
    plt.close(fig)


def _build_global_metrics_table(runs: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        d = r["doc"]
        row = {
            "seed": r["seed"], "algorithm": r["algo"], "mu": r["mu"],
            "accuracy": d.get("accuracy"),
            "balanced_accuracy": d.get("balanced_accuracy"),
            "macro_f1": d.get("macro_f1"),
            "loss": d.get("loss"),
            "selected_round": d.get("selected_round"),
        }
        for c, f1c in enumerate(d.get("per_class_f1", [])):
            row[f"f1_class_{c}"] = float(f1c)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["seed", "algorithm"]).reset_index(drop=True)


def _per_class_recall_precision_from_preds(runs: List[Dict]) -> pd.DataFrame:
    """Compute per-class recall + precision from saved test_predictions_*.npz."""
    rows = []
    for r in runs:
        if not r["preds_path"].exists():
            continue
        z = np.load(r["preds_path"])
        preds = np.asarray(z["predictions"])
        targets = np.asarray(z["targets"])
        for c in range(NUM_CLASSES):
            tp = int(((preds == c) & (targets == c)).sum())
            fp = int(((preds == c) & (targets != c)).sum())
            fn = int(((preds != c) & (targets == c)).sum())
            support = int((targets == c).sum())
            recall = tp / support if support > 0 else float("nan")
            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            f1 = (2 * recall * precision / (recall + precision)
                  if (recall + precision) > 0 else float("nan"))
            rows.append({
                "seed": r["seed"], "algorithm": r["algo"], "mu": r["mu"],
                "class": c, "class_name": CLASS_NAMES[c],
                "support": support, "tp": tp, "fp": fp, "fn": fn,
                "recall": recall, "precision": precision, "f1": f1,
            })
    return pd.DataFrame(rows)


def _per_class_delta(global_metrics: pd.DataFrame,
                     recall_prec: pd.DataFrame) -> pd.DataFrame:
    """For each seed where BOTH algos ran, compute FedProx − FedAvg per class."""
    deltas = []
    for seed, sub in global_metrics.groupby("seed"):
        if not {"fedavg", "fedprox"}.issubset(set(sub["algorithm"])):
            continue
        avg = sub[sub["algorithm"] == "fedavg"].iloc[0]
        prox = sub[sub["algorithm"] == "fedprox"].iloc[0]
        for c in range(NUM_CLASSES):
            deltas.append({
                "seed": seed,
                "class": c,
                "class_name": CLASS_NAMES[c],
                "delta_f1": float(prox[f"f1_class_{c}"]) - float(avg[f"f1_class_{c}"]),
            })
    df_d = pd.DataFrame(deltas)

    # Augment with recall/precision deltas when predictions are available.
    if not recall_prec.empty:
        rp = recall_prec.pivot_table(
            index=["seed", "class"], columns="algorithm",
            values=["recall", "precision", "f1"],
        )
        rp.columns = [f"{m}_{a}" for m, a in rp.columns]
        rp = rp.reset_index()
        if {"recall_fedavg", "recall_fedprox"}.issubset(rp.columns):
            rp["delta_recall"] = rp["recall_fedprox"] - rp["recall_fedavg"]
            rp["delta_precision"] = rp["precision_fedprox"] - rp["precision_fedavg"]
            df_d = df_d.merge(
                rp[["seed", "class", "delta_recall", "delta_precision"]],
                on=["seed", "class"], how="left",
            )
    return df_d


def _plot_per_class_delta_bar(per_class_delta: pd.DataFrame, out_path: Path) -> None:
    """Bar plot of mean per-class delta (FedProx − FedAvg)."""
    if per_class_delta.empty:
        return
    grp = per_class_delta.groupby("class")
    mean_d = grp["delta_f1"].mean()
    n_seeds = grp["seed"].nunique().max()
    sem = grp["delta_f1"].std(ddof=0) / max(np.sqrt(n_seeds), 1.0) if n_seeds > 1 else None

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(NUM_CLASSES)
    colors = ["#1f77b4" if c not in CRITICAL_CLASSES else "#d62728" for c in xs]
    ax.bar(xs, [mean_d.get(c, 0.0) for c in xs], color=colors,
           yerr=([sem.get(c, 0.0) for c in xs] if sem is not None else None),
           capsize=4)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}\n{CLASS_NAMES[c][:10]}" for c in xs], fontsize=8)
    ax.set_ylabel("FedProx − FedAvg  (per-class test F1)")
    title = "Per-class F1 delta (red = critical/rare class on Client 1)"
    if n_seeds is not None and n_seeds > 1:
        title += f"  —  n = {int(n_seeds)} paired seeds"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_learning_curves(runs: List[Dict], out_path: Path) -> None:
    """val_macro_f1 across rounds for every (algo, seed)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_data = False
    for r in runs:
        if not r["history_path"].exists():
            continue
        h = pd.read_csv(r["history_path"])
        if "val_macro_f1" not in h.columns:
            continue
        ls = "--" if r["algo"] == "fedavg" else "-"
        col = "#d62728" if r["algo"] == "fedavg" else "#2ca02c"
        ax.plot(h["round"], h["val_macro_f1"], ls=ls, color=col, alpha=0.85,
                label=f"{r['algo']} seed={r['seed']}")
        any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel("communication round")
    ax.set_ylabel("validation macro-F1")
    ax.set_title("two_client_90_10_rare_stress: validation macro-F1 over rounds")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_confusion_matrices(runs: List[Dict], out_dir: Path) -> None:
    for r in runs:
        if not r["preds_path"].exists():
            continue
        z = np.load(r["preds_path"])
        preds = np.asarray(z["predictions"])
        targets = np.asarray(z["targets"])
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for t, p in zip(targets, preds):
            cm[int(t), int(p)] += 1
        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{cm[i, j]:d}",
                        ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels([f"{c}\n{CLASS_NAMES[c][:8]}" for c in range(NUM_CLASSES)],
                           fontsize=7, rotation=0)
        ax.set_yticklabels([f"{c}: {CLASS_NAMES[c][:14]}" for c in range(NUM_CLASSES)],
                           fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Confusion matrix — {r['algo']} (μ={r['mu']}, seed={r['seed']})")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        out_path = out_dir / f"confusion_matrix_{r['algo']}_mu{r['mu']}_s{r['seed']}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)


def _plot_update_norms(runs: List[Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_data = False
    for r in runs:
        if not r["norms_path"].exists():
            continue
        un = pd.read_csv(r["norms_path"])
        for cid, sub in un.groupby("client_id"):
            ls = "--" if r["algo"] == "fedavg" else "-"
            col = "#1f77b4" if cid == 0 else "#d62728"
            ax.plot(sub["round"], sub["update_norm"], ls=ls, color=col, alpha=0.85,
                    label=f"{r['algo']} client {cid} seed={r['seed']}")
            any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel("communication round")
    ax.set_ylabel("‖w_k − w_global‖₂  (post-local-train, pre-aggregation)")
    ax.set_title("Per-client update norms — dominant (blue) vs specialist (red)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _summarise(global_metrics: pd.DataFrame,
               per_class_delta: pd.DataFrame) -> Dict:
    """Roll up the headline numbers into a single JSON summary."""
    out: Dict = {"n_runs": len(global_metrics),
                 "paired_seeds": [],
                 "global": {},
                 "critical_classes": {}}
    paired_seeds = sorted(
        int(s) for s in global_metrics["seed"].unique()
        if {"fedavg", "fedprox"}.issubset(
            set(global_metrics[global_metrics["seed"] == s]["algorithm"]))
    )
    out["paired_seeds"] = paired_seeds
    if not paired_seeds:
        return out

    rows = []
    for s in paired_seeds:
        sub = global_metrics[global_metrics["seed"] == s]
        a = sub[sub["algorithm"] == "fedavg"].iloc[0]
        p = sub[sub["algorithm"] == "fedprox"].iloc[0]
        rows.append({
            "seed": int(s),
            "delta_macro_f1": float(p["macro_f1"]) - float(a["macro_f1"]),
            "delta_balanced_acc": float(p["balanced_accuracy"]) - float(a["balanced_accuracy"]),
            "delta_accuracy": float(p["accuracy"]) - float(a["accuracy"]),
        })
    df = pd.DataFrame(rows)
    out["global"] = {
        "mean_delta_macro_f1": float(df["delta_macro_f1"].mean()),
        "mean_delta_balanced_acc": float(df["delta_balanced_acc"].mean()),
        "mean_delta_accuracy": float(df["delta_accuracy"].mean()),
        "per_seed": df.to_dict(orient="records"),
    }
    if not per_class_delta.empty:
        for c in CRITICAL_CLASSES:
            sub = per_class_delta[per_class_delta["class"] == c]
            entry: Dict = {
                "class": int(c),
                "class_name": CLASS_NAMES[c],
                "mean_delta_f1": float(sub["delta_f1"].mean()) if not sub.empty else float("nan"),
                "n_seeds": int(sub["seed"].nunique()),
            }
            if "delta_recall" in sub.columns:
                entry["mean_delta_recall"] = float(sub["delta_recall"].mean())
                entry["mean_delta_precision"] = float(sub["delta_precision"].mean())
            out["critical_classes"][CLASS_NAMES[c]] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--npz-path", default=str(DEFAULT_NPZ))
    ap.add_argument("--mu", type=float, default=None,
                    help="If set, restrict FedProx to runs with this μ value.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from {results_dir}")
    runs = _load_runs(results_dir, only_mu=args.mu)
    print(f"Found {len(runs)} runs.")
    for r in runs:
        print(f"  - {r['algo']:7s}  μ={r['mu']:>5}  seed={r['seed']}  "
              f"macro_f1={r['doc'].get('macro_f1', float('nan')):.4f}")

    print(f"\n[1] partition_class_counts.csv + partition_heatmap.png")
    _save_partition_artifacts(out_dir, Path(args.npz_path))

    if not runs:
        print("\nNo runs found — partition artifacts written. Run the submit "
              "script first, then re-run this analyser.")
        return

    print("\n[2] headline_metrics.csv")
    global_metrics = _build_global_metrics_table(runs)
    global_metrics.to_csv(out_dir / "headline_metrics.csv", index=False)

    print("[3] per_class_f1.csv (from JSON metrics)")
    per_class_cols = ["seed", "algorithm", "mu"] + [f"f1_class_{c}" for c in range(NUM_CLASSES)]
    global_metrics[per_class_cols].to_csv(out_dir / "per_class_f1.csv", index=False)

    print("[4] per_class_recall_precision.csv (from saved predictions)")
    recall_prec = _per_class_recall_precision_from_preds(runs)
    recall_prec.to_csv(out_dir / "per_class_recall_precision.csv", index=False)

    print("[5] per_class_delta.csv  +  per_class_delta_bar.png")
    delta = _per_class_delta(global_metrics, recall_prec)
    delta.to_csv(out_dir / "per_class_delta.csv", index=False)
    _plot_per_class_delta_bar(delta, out_dir / "per_class_delta_bar.png")

    print("[6] learning_curves.png")
    _plot_learning_curves(runs, out_dir / "learning_curves.png")

    print("[7] confusion_matrix_<algo>_mu<mu>_s<seed>.png")
    _plot_confusion_matrices(runs, out_dir)

    print("[8] update_norm_per_client.png (if --log-update-norms was set)")
    _plot_update_norms(runs, out_dir / "update_norm_per_client.png")

    print("[9] summary.json")
    summary = _summarise(global_metrics, delta)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs in: {out_dir}")
    if summary.get("paired_seeds"):
        print("\n=== Headline (paired seeds only) ===")
        print(f"  paired seeds:               {summary['paired_seeds']}")
        g = summary["global"]
        print(f"  mean Δ macro-F1:            {g['mean_delta_macro_f1']:+.4f}")
        print(f"  mean Δ balanced accuracy:   {g['mean_delta_balanced_acc']:+.4f}")
        print(f"  mean Δ accuracy:            {g['mean_delta_accuracy']:+.4f}")
        if summary.get("critical_classes"):
            print("  per-critical-class deltas (FedProx − FedAvg):")
            for name, ent in summary["critical_classes"].items():
                line = f"    {name:<25s}  Δ F1 = {ent['mean_delta_f1']:+.4f}"
                if "mean_delta_recall" in ent:
                    line += f"  Δ recall = {ent['mean_delta_recall']:+.4f}"
                line += f"   (n={ent['n_seeds']})"
                print(line)


if __name__ == "__main__":
    main()
