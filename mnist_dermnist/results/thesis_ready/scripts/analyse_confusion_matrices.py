"""Per-seed confusion matrices and per-class precision / recall (audit P2.6).

Reads `test_predictions_<algo>_mu*_E*_s<seed>.npz` sibling files from the
headline (or any other) results directory and produces:

  1. A 2-panel confusion-matrix figure for FedAvg vs FedProx at a chosen
     seed (default 42). Counts and row-normalised heatmaps are saved as
     `confusion_seed<seed>.png` + `.pdf`.
  2. A per-class precision/recall/F1 table aggregated across all
     paired seeds (CSV at `data/per_class_precision_recall.csv`).

These outputs substitute for the qualitative segmentation panels in
Marija's thesis: classification doesn't naturally produce visualisable
predictions per sample, but the confusion matrix is the closest
classification analogue — it shows where each algorithm misclassifies
into which other classes.

If no `.npz` prediction files exist (e.g., because the headline runs
were produced before the prediction-saving infrastructure landed), the
script exits cleanly with an actionable message rather than crashing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = (
    "actinic", "basal", "benign_kerat", "dermato",
    "melanoma", "mel_nevi", "vascular",
)
NUM_CLASSES = len(CLASS_NAMES)


def _confusion(targets: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Row-by-row confusion counts: cm[true, pred]."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    return cm


def _per_class_pr(cm: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in range(NUM_CLASSES):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else 0.0)
        rows.append({
            "class_id": c, "class_name": CLASS_NAMES[c],
            "support": int(cm[c, :].sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        })
    return pd.DataFrame(rows)


def _load_npz(p: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(p)
    return d["targets"], d["predictions"]


def _find_pred_files(results_dir: Path) -> dict[tuple[str, int], Path]:
    """Return {(algo, seed): path} from test_predictions_*.npz files."""
    pat = re.compile(
        r"test_predictions_(?P<algo>fedavg|fedprox|fednova)"
        r"_mu[0-9.]+_E\d+(?:_sh-[a-z_]+)?(?:_C[0-9.]+)?_s(?P<seed>\d+)\.npz"
    )
    out: dict[tuple[str, int], Path] = {}
    for f in sorted(results_dir.glob("test_predictions_*.npz")):
        m = pat.match(f.name)
        if not m:
            continue
        out[(m["algo"], int(m["seed"]))] = f
    return out


def _plot_pair(cm_fa: np.ndarray, cm_fp: np.ndarray, seed: int, out_stem: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for col, (cm, title) in enumerate([(cm_fa, f"FedAvg (seed {seed})"),
                                       (cm_fp, f"FedProx (seed {seed})")]):
        # Top row: raw counts
        ax = axes[0, col]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{title} — counts")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, int(cm[i, j]),
                        ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() * 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)

        # Bottom row: row-normalised (recall per class)
        ax = axes[1, col]
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm / row_sums
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{title} — row-normalised")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Test-set confusion matrices at best-val checkpoint",
                 fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default="mnist_dermnist/results/headline",
                    help="Directory containing test_predictions_*.npz files.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed to render confusion-matrix figure for.")
    ap.add_argument("--out-stem", default=None,
                    help="Path stem for the figure (default: "
                         "thesis_ready/figures/confusion_seed<seed>).")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: results-dir not found: {results_dir}", file=sys.stderr)
        return 1

    pred_files = _find_pred_files(results_dir)
    if not pred_files:
        print(f"INFO: no test_predictions_*.npz files in {results_dir}.")
        print( "      The current headline data predates the prediction-saving")
        print( "      infrastructure (run_one*.py now writes them automatically).")
        print( "      To produce confusion matrices, re-run AT LEAST seed 42 for")
        print( "      both FedAvg and FedProx with the updated runners; the .npz")
        print( "      sibling files will then appear here.")
        return 0

    # ---- Figure: FedAvg vs FedProx at chosen seed ----
    pair = [pred_files.get(("fedavg", args.seed)),
            pred_files.get(("fedprox", args.seed))]
    if all(pair):
        t_fa, p_fa = _load_npz(pair[0])
        t_fp, p_fp = _load_npz(pair[1])
        cm_fa = _confusion(t_fa, p_fa)
        cm_fp = _confusion(t_fp, p_fp)
        out_stem = (Path(args.out_stem) if args.out_stem
                    else FIG_DIR / f"confusion_seed{args.seed}")
        _plot_pair(cm_fa, cm_fp, args.seed, out_stem)
        print(f"Wrote confusion figure: {out_stem}.png / .pdf")
    else:
        missing = [("fedavg", args.seed) if not pair[0] else None,
                   ("fedprox", args.seed) if not pair[1] else None]
        missing = [m for m in missing if m is not None]
        print(f"WARN: missing predictions for {missing}; skipping figure.")

    # ---- Aggregate per-class precision/recall across all available seeds ----
    rows = []
    for (algo, seed), p in sorted(pred_files.items()):
        t, pr = _load_npz(p)
        cm = _confusion(t, pr)
        df = _per_class_pr(cm)
        df.insert(0, "seed", seed)
        df.insert(0, "algorithm", algo)
        rows.append(df)
    if rows:
        agg = pd.concat(rows, ignore_index=True)
        out_csv = DATA_DIR / "per_class_precision_recall.csv"
        agg.to_csv(out_csv, index=False)
        print(f"Wrote per-class precision/recall: {out_csv}")
        print("\nMean per-class metrics (across seeds, per algorithm):")
        print(agg.groupby(["algorithm", "class_name"])[["precision", "recall", "f1"]]
                .mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
