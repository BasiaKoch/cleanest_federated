"""Analysis script for the heterogeneity-ladder experiments.

Reads results from ``fl_dermamnist/results/heterogeneity_ladder/`` and
produces the CSV/figure/LaTeX outputs specified in PART 7 of the thesis
experimental plan. Designed to be run AFTER Stage A pilot (or Stage B
3-seed run) returns data --- the script tolerates missing levels and
methods and only emits outputs for what's actually on disk.

The script is intentionally idempotent and side-effect-free: it reads
the JSON / NPZ / CSV files produced by the Flower runner(s) and writes
to ``results/heterogeneity_ladder/analysis/``. It does not touch the
existing result files.

Expected layout
---------------
  fl_dermamnist/results/heterogeneity_ladder/
    L0_two_client_50_50_stratified_iid/
        test_at_best_fedavg_mu0.0_E20_s42.json
        test_at_best_fedprox_mu0.01_E20_s42.json
        client_update_norms_*.csv  (optional)
        test_predictions_*.npz     (used for confusion matrices)
    L1_two_client_86_14_quantity_only_stratified/
        ... (+ fednova results)
    L2_two_client_50_50_label_skew_only/
        ...
    L3_two_client_70_30_rare_enriched/
        ... (+ fednova results)
    L4_two_client_90_10_rare_stress/
        ... (+ fednova results)

Outputs (PART 7 of the plan)
-----------------------------
  analysis/
    ladder_summary.csv          one row per (level, method, seed)
    ladder_deltas.csv           per-method delta vs FedAvg
    ladder_mean_std.csv         mean ± std across seeds
    partition_heatmaps/         per-level class-distribution heatmaps
    figure_ladder_macro_f1_delta.{pdf,png}
    figure_ladder_rare_f1_delta.{pdf,png}
    figure_ladder_melanoma_recall_delta.{pdf,png}
    figure_update_norms_level4.{pdf,png}
    figure_confusion_level4.{pdf,png}
    thesis_table_ladder.tex
    thesis_table_deltas.tex

Run
---
PYTHONPATH=. python3 -m fl_dermamnist.analysis.analyse_heterogeneity_ladder
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
LADDER_ROOT = REPO_ROOT / "fl_dermamnist" / "results" / "heterogeneity_ladder"
OUT_DIR = LADDER_ROOT / "analysis"

CLASS_NAMES = [
    "actinic_keratoses",
    "basal_cell_carcinoma",
    "benign_keratosis",
    "dermatofibroma",
    "melanoma",
    "melanocytic_nevi",
    "vascular_lesions",
]
RARE_CLASS_INDICES = (3, 4, 6)  # dermato, melanoma, vascular
MELANOMA_IDX = 4

# Ladder spec — kept in sync with submit_ladder_pilot.sh.
LADDER = [
    ("L0", "two_client_50_50_stratified_iid",          "IID control",       0.0,  0.0),
    ("L1", "two_client_86_14_quantity_only_stratified","Quantity-only",     1.0,  0.0),
    ("L2", "two_client_50_50_label_skew_only",         "Label-skew only",   0.0,  1.0),
    ("L3", "two_client_70_30_rare_enriched",           "Mixed (mild)",      0.5,  1.0),
    ("L4", "two_client_90_10_rare_stress",             "Mixed (severe)",    1.0,  1.0),
]


# ---------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------

JSON_PAT = re.compile(
    r"test_at_best_(fedavg|fedprox|fednova)_mu(?P<mu>[\d.]+)_"
    r"E(?P<E>\d+)(?P<tags>_[^_]*)*_s(?P<seed>\d+)\.json"
)


def parse_filename(name: str) -> Dict[str, str] | None:
    """Extract (algorithm, mu, local_epochs, seed) from a result filename.

    Tolerates the various tags (system-het, drop-stragglers, etc.) in
    the middle of the stem; only the leading algo+mu+E and the trailing
    _s<seed> are required.
    """
    m = re.match(
        r"test_at_best_(?P<algo>fedavg|fedprox|fednova)_"
        r"mu(?P<mu>[\d.]+)_E(?P<E>\d+).*?_s(?P<seed>\d+)\.json",
        name,
    )
    if not m:
        return None
    return {
        "algorithm": m.group("algo"),
        "mu": float(m.group("mu")),
        "local_epochs": int(m.group("E")),
        "seed": int(m.group("seed")),
    }


def load_runs() -> pd.DataFrame:
    """Walk LADDER_ROOT, parse every test_at_best JSON, return long-form."""
    if not LADDER_ROOT.exists():
        print(f"WARNING: ladder root does not exist yet: {LADDER_ROOT}")
        return pd.DataFrame()

    rows: List[Dict] = []
    for level, partition, label, qs, ls in LADDER:
        run_dir = LADDER_ROOT / f"{level}_{partition}"
        if not run_dir.exists():
            print(f"  [{level}] no directory yet ({run_dir.name}); skip")
            continue
        for f in sorted(run_dir.glob("test_at_best_*.json")):
            meta = parse_filename(f.name)
            if meta is None:
                continue
            with open(f) as fp:
                d = json.load(fp)
            per_class = d.get("per_class_f1") or []
            rare_f1 = (
                float(np.mean([per_class[i] for i in RARE_CLASS_INDICES]))
                if len(per_class) >= 7 else float("nan")
            )
            rows.append({
                "level": level,
                "partition": partition,
                "label": label,
                "quantity_skew": qs,
                "label_skew": ls,
                **meta,
                "macro_f1": float(d.get("macro_f1", float("nan"))),
                "balanced_accuracy": float(d.get("balanced_accuracy", float("nan"))),
                "accuracy": float(d.get("accuracy", float("nan"))),
                "loss": float(d.get("loss", float("nan"))),
                "melanoma_f1": (per_class[MELANOMA_IDX]
                                if len(per_class) >= 7 else float("nan")),
                "rare_avg_f1": rare_f1,
                "per_class_f1": per_class,
                "selected_round": d.get("selected_round"),
                "json_path": str(f),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Per-class precision / recall from saved predictions
# ---------------------------------------------------------------------

def per_class_pr_from_predictions(npz_path: Path) -> Dict[str, List[float]]:
    """Reconstruct per-class precision/recall from saved test predictions."""
    from sklearn.metrics import precision_score, recall_score
    d = np.load(npz_path)
    y_true = d["targets"].astype(int)
    y_pred = d["predictions"].astype(int)
    labels = list(range(7))
    return {
        "per_class_precision": precision_score(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        ).tolist(),
    }


def enrich_with_precision_recall(df: pd.DataFrame) -> pd.DataFrame:
    """Attach melanoma_recall + rare_avg_recall columns by reading .npz."""
    if df.empty:
        return df
    mel_recall = []
    rare_recall = []
    for _, row in df.iterrows():
        json_path = Path(row["json_path"])
        npz_candidates = list(json_path.parent.glob(
            json_path.stem.replace("test_at_best_", "test_predictions_") + ".npz"
        ))
        if not npz_candidates:
            mel_recall.append(float("nan"))
            rare_recall.append(float("nan"))
            continue
        pr = per_class_pr_from_predictions(npz_candidates[0])
        mel_recall.append(pr["per_class_recall"][MELANOMA_IDX])
        rare_recall.append(float(np.mean(
            [pr["per_class_recall"][i] for i in RARE_CLASS_INDICES]
        )))
    df = df.copy()
    df["melanoma_recall"] = mel_recall
    df["rare_avg_recall"] = rare_recall
    return df


# ---------------------------------------------------------------------
# Update-norm summaries
# ---------------------------------------------------------------------

def load_update_norm_means(run_dir: Path) -> Dict[str, Dict[str, float]]:
    """Per-(algorithm, client) mean update norm across the run."""
    out: Dict[str, Dict[str, float]] = {}
    for f in sorted(run_dir.glob("client_update_norms_*.csv")):
        # Strip prefix/suffix to recover algo+seed.
        m = re.match(
            r"client_update_norms_(?P<algo>fedavg|fedprox|fednova)_"
            r"mu(?P<mu>[\d.]+)_E\d+.*?_s(?P<seed>\d+)\.csv", f.name)
        if not m:
            continue
        key = f"{m.group('algo')}_s{m.group('seed')}"
        df = pd.read_csv(f)
        means = df.groupby("client_id")["update_norm"].mean().to_dict()
        out[key] = {
            "client0": float(means.get(0, float("nan"))),
            "client1": float(means.get(1, float("nan"))),
        }
        if out[key]["client1"]:
            out[key]["ratio_c0_over_c1"] = (
                out[key]["client0"] / out[key]["client1"]
            )
    return out


# ---------------------------------------------------------------------
# Aggregate tables
# ---------------------------------------------------------------------

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """ladder_summary.csv: one row per (level, method, seed) with metrics."""
    if df.empty:
        return df
    cols = [
        "level", "partition", "label", "quantity_skew", "label_skew",
        "algorithm", "mu", "seed", "local_epochs",
        "macro_f1", "balanced_accuracy", "accuracy", "loss",
        "melanoma_f1", "rare_avg_f1",
    ]
    if "melanoma_recall" in df.columns:
        cols += ["melanoma_recall", "rare_avg_recall"]
    return df[cols].sort_values(["level", "algorithm", "seed"]).reset_index(drop=True)


def build_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """ladder_deltas.csv: per-seed (FedProx - FedAvg) and (FedNova - FedAvg)."""
    if summary.empty:
        return summary
    delta_rows: List[Dict] = []
    metric_cols = [
        "macro_f1", "balanced_accuracy", "melanoma_f1", "rare_avg_f1",
    ]
    if "melanoma_recall" in summary.columns:
        metric_cols += ["melanoma_recall", "rare_avg_recall"]

    for (level, seed), grp in summary.groupby(["level", "seed"]):
        fa_rows = grp[grp["algorithm"] == "fedavg"]
        if fa_rows.empty:
            continue
        fa = fa_rows.iloc[0]
        for algo in ("fedprox", "fednova"):
            other_rows = grp[grp["algorithm"] == algo]
            if other_rows.empty:
                continue
            other = other_rows.iloc[0]
            row = {
                "level": level,
                "partition": fa["partition"],
                "method": algo,
                "seed": int(seed),
            }
            for m in metric_cols:
                row[f"delta_{m}"] = float(other[m]) - float(fa[m])
            delta_rows.append(row)
    return pd.DataFrame(delta_rows)


def build_mean_std(summary: pd.DataFrame) -> pd.DataFrame:
    """ladder_mean_std.csv: mean ± std across seeds per (level, method)."""
    if summary.empty:
        return summary
    metric_cols = [
        "macro_f1", "balanced_accuracy", "melanoma_f1", "rare_avg_f1",
    ]
    if "melanoma_recall" in summary.columns:
        metric_cols += ["melanoma_recall", "rare_avg_recall"]

    agg = (summary
           .groupby(["level", "partition", "label", "algorithm"], as_index=False)
           [metric_cols].agg(["mean", "std"]))
    agg.columns = [
        "_".join([c for c in col if c]) if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    return agg.reset_index(drop=True)


# ---------------------------------------------------------------------
# Figures (to be filled in once data exists)
# ---------------------------------------------------------------------

def make_figures(summary: pd.DataFrame, deltas: pd.DataFrame, out_dir: Path) -> None:
    """Stub. Implement once data exists; documented in PART 7."""
    if summary.empty:
        print("  (no data — skipping figures)")
        return
    print("  Ladder figures are not generated for the single-seed pilot.")


def make_latex_tables(summary: pd.DataFrame, deltas: pd.DataFrame,
                      out_dir: Path) -> None:
    """Stub. Implement once data exists."""
    print("  Ladder LaTeX tables are not generated for the single-seed pilot.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading ladder results from {LADDER_ROOT}")

    df = load_runs()
    if df.empty:
        print("No runs found. Run submit_ladder_pilot.sh and try again.")
        return

    print(f"  found {len(df)} runs across "
          f"{df['level'].nunique()} levels and {df['algorithm'].nunique()} methods")
    df = enrich_with_precision_recall(df)

    summary = build_summary(df)
    deltas = build_deltas(summary)
    mean_std = build_mean_std(summary)

    summary.to_csv(OUT_DIR / "ladder_summary.csv", index=False)
    deltas.to_csv(OUT_DIR / "ladder_deltas.csv", index=False)
    mean_std.to_csv(OUT_DIR / "ladder_mean_std.csv", index=False)
    print(f"  wrote {OUT_DIR / 'ladder_summary.csv'}")
    print(f"  wrote {OUT_DIR / 'ladder_deltas.csv'}")
    print(f"  wrote {OUT_DIR / 'ladder_mean_std.csv'}")

    make_figures(summary, deltas, OUT_DIR)
    make_latex_tables(summary, deltas, OUT_DIR)

    # Per-level update-norm summary (for the mechanism column).
    for level, partition, *_ in LADDER:
        run_dir = LADDER_ROOT / f"{level}_{partition}"
        if not run_dir.exists():
            continue
        norms = load_update_norm_means(run_dir)
        if norms:
            print(f"  {level} update norms: {norms}")


if __name__ == "__main__":
    main()
