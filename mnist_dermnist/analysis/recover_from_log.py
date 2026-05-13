"""Reconstruct history CSV + comparison JSON from a Flower run's stdout log.

Used when the original output files were lost (e.g. Colab session died) but
the training log was captured. Parses the Flower [SUMMARY] block and the
final "Head-to-head" report.

Usage:
    python -m mnist_dermnist.analysis.recover_from_log \\
        --log mnist_dermnist/results/colab_recovered/colab_log.txt \\
        --algorithm fedprox --mu 0.01 --seed 42 --local-epochs 20 \\
        --out-dir mnist_dermnist/results/colab_recovered
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


METRIC_NAMES = [
    "val_loss",
    "val_accuracy",
    "val_balanced_accuracy",
    "val_macro_f1",
    "train_loss_weighted",
]


def strip_info_prefix(text: str) -> str:
    """Strip the 'INFO :     ' line prefix Flower uses."""
    return re.sub(r"^\s*INFO\s*:\s*", "", text, flags=re.MULTILINE)


def parse_metric(text: str, metric: str) -> dict[int, float]:
    """Extract all (round, value) tuples for a named metric."""
    pat = re.compile(rf"'{re.escape(metric)}'\s*:\s*\[(.*?)\]", re.DOTALL)
    m = pat.search(text)
    if not m:
        return {}
    body = m.group(1)
    tuples = re.findall(r"\(\s*(\d+)\s*,\s*([0-9.eE+\-]+)\s*\)", body)
    return {int(r): float(v) for r, v in tuples}


def parse_head_to_head(text: str) -> dict | None:
    """Pull out the 'Head-to-head' summary block at the end of a paired run."""
    m = re.search(r"Head-to-head.*?(?=\n\S|\Z)", text, re.DOTALL)
    if not m:
        return None
    block = m.group(0)

    def first_float(pat: str) -> float | None:
        m = re.search(pat, block)
        return float(m.group(1)) if m else None

    fa_f1 = first_float(r"FedAvg\s+([\d.]+)\s+vs\s+FedProx")
    fp_f1 = first_float(r"vs\s+FedProx\s+([\d.]+)")
    delta_f1 = first_float(r"Δ\s*=\s*([+\-]?[\d.]+).*macro")

    fa_bacc = first_float(r"FedAvg\s+([\d.]+)\s+vs\s+FedProx\s+[\d.]+\s+Δ.*\n.*balanced")
    # Re-parse balanced acc line more robustly
    bal_line = re.search(
        r"test balanced acc.*?FedAvg\s+([\d.]+)\s+vs\s+FedProx\s+([\d.]+).*Δ\s*=\s*([+\-]?[\d.]+)",
        block,
    )
    if bal_line:
        fa_bacc, fp_bacc, delta_bacc = map(float, bal_line.groups())
    else:
        fp_bacc = None; delta_bacc = None

    acc_line = re.search(
        r"test accuracy.*?FedAvg\s+([\d.]+)\s+vs\s+FedProx\s+([\d.]+)", block
    )
    fa_acc, fp_acc = (map(float, acc_line.groups()) if acc_line else (None, None))

    # per-class
    classes = ["actinic", "basal", "benign_k", "dermato", "melanoma", "mel_nevi", "vascular"]
    pc_fa, pc_fp = [], []
    for c in classes:
        m = re.search(rf"{c}\s*:\s*([\d.]+)\s*→\s*([\d.]+)", block)
        if m:
            pc_fa.append(float(m.group(1)))
            pc_fp.append(float(m.group(2)))

    return {
        "fedavg":  {"test_macro_f1": fa_f1, "test_balanced_accuracy": fa_bacc,
                    "test_accuracy": fa_acc, "test_per_class_f1": pc_fa},
        "fedprox": {"test_macro_f1": fp_f1, "test_balanced_accuracy": fp_bacc,
                    "test_accuracy": fp_acc, "test_per_class_f1": pc_fp},
        "delta_macro_f1": delta_f1,
        "delta_balanced_accuracy": delta_bacc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--algorithm", choices=["fedavg", "fedprox"], required=True)
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-epochs", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    raw = Path(args.log).read_text()
    text = strip_info_prefix(raw)

    # ---- history CSV ----
    metrics = {name: parse_metric(text, name) for name in METRIC_NAMES}
    n_per_metric = {k: len(v) for k, v in metrics.items()}
    print(f"Parsed entries per metric: {n_per_metric}")

    all_rounds = sorted({r for d in metrics.values() for r in d})
    df = pd.DataFrame({"round": all_rounds})
    for name, d in metrics.items():
        df[name] = df["round"].map(d)

    mu_str = "0.0" if args.algorithm == "fedavg" else f"{args.mu}"
    csv_name = f"history_{args.algorithm}_mu{mu_str}_E{args.local_epochs}_s{args.seed}.csv"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / csv_name
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}  ({len(df)} rows)")
    print(df.head())
    print("...")
    print(df.tail())

    # ---- head-to-head JSON ----
    h2h = parse_head_to_head(text)
    if h2h:
        h2h.update({
            "framework": "flower-simulation (recovered from log)",
            "seed": args.seed, "num_rounds": len(df) - (1 if 0 in df["round"].values else 0),
            "local_epochs": args.local_epochs, "mu": args.mu,
        })
        json_path = out_dir / f"flower_comparison_seed{args.seed}_mu{args.mu}.json"
        with open(json_path, "w") as f:
            json.dump(h2h, f, indent=2)
        print(f"\nWrote {json_path}")
        print(f"  FedAvg  macro-F1 = {h2h['fedavg']['test_macro_f1']}")
        print(f"  FedProx macro-F1 = {h2h['fedprox']['test_macro_f1']}")
        print(f"  Δ macro-F1       = {h2h['delta_macro_f1']:+.4f}")
    else:
        print("\nNo Head-to-head block found — only history CSV written.")


if __name__ == "__main__":
    main()
