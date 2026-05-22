"""Architecture ablation comparison: BatchNorm (bn) vs GroupNorm (gn).

Compares ``results/arch_ablation_bn/`` (BN variant, 3 seeds) against the
matching subset of ``results/flower_C0_baseline/`` (GN headline) on the
same engineered partition and same paired seeds (42, 123, 456).

Reports for each variant:
  - Per-seed macro_f1 for FedAvg and FedProx
  - Within-pair delta (FedProx - FedAvg)
  - Across-seed mean delta and SD

Then reports the headline cross-variant contrast:
  - delta_BN - delta_GN per seed
  - Mean cross-variant contrast

Hypothesis (Li et al. 2021 FedBN; Hsieh et al. 2020 quagmire):
  Mean delta_BN > Mean delta_GN, because BN's running-stats drift compounds
  with parameter drift under non-IID data, and FedProx's parameter-side
  anchor partially absorbs both.

Idempotent. Re-run any time:
    python mnist_dermnist/scripts/check_arch_ablation.py
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from statistics import mean, stdev


BN_DIR = "mnist_dermnist/results/arch_ablation_bn"
GN_DIR = "mnist_dermnist/results/flower_C0_baseline"
ABLATION_SEEDS = [42, 123, 456]


_BN_PAT = re.compile(
    r"test_at_best_(?P<algo>fedavg|fedprox)_mu[0-9.]+_E20_arch-bn_s(?P<seed>\d+)\.json"
)
_GN_PAT = re.compile(
    r"test_at_best_(?P<algo>fedavg|fedprox)_mu[0-9.]+_E20_s(?P<seed>\d+)\.json"
)


def load_variant(dir_path, pattern, seeds):
    """Return {algo: {seed: macro_f1}} for the requested seeds only."""
    out = defaultdict(dict)
    if not os.path.isdir(dir_path):
        return out
    for fname in sorted(os.listdir(dir_path)):
        m = pattern.match(fname)
        if not m:
            continue
        seed = int(m.group("seed"))
        if seed not in seeds:
            continue
        algo = m.group("algo")
        with open(os.path.join(dir_path, fname)) as f:
            doc = json.load(f)
        out[algo][seed] = float(doc["macro_f1"])
    return out


def paired_deltas(variant):
    """Return ordered (seed -> delta) for seeds present in both algos."""
    common = sorted(set(variant.get("fedavg", {})) & set(variant.get("fedprox", {})))
    return {s: variant["fedprox"][s] - variant["fedavg"][s] for s in common}


def print_variant_table(label, variant):
    print(f"\n=== {label} ===")
    fa = variant.get("fedavg", {})
    fp = variant.get("fedprox", {})
    seeds = sorted(set(fa) | set(fp))
    if not seeds:
        print("  (no files yet)")
        return {}
    print(f"  {'seed':>8s}  {'FedAvg':>8s}  {'FedProx':>8s}  {'delta':>9s}")
    print("  " + "-" * 42)
    deltas = {}
    for s in seeds:
        fa_v = fa.get(s)
        fp_v = fp.get(s)
        if fa_v is not None and fp_v is not None:
            d = fp_v - fa_v
            deltas[s] = d
            print(f"  {s:>8d}  {fa_v:>8.4f}  {fp_v:>8.4f}  {d:>+9.4f}")
        else:
            mark_fa = f"{fa_v:>8.4f}" if fa_v is not None else "     ---"
            mark_fp = f"{fp_v:>8.4f}" if fp_v is not None else "     ---"
            print(f"  {s:>8d}  {mark_fa}  {mark_fp}  (partial)")
    if deltas:
        vals = list(deltas.values())
        print(f"\n  n={len(vals)}  mean delta = {mean(vals):+.4f}", end="")
        if len(vals) > 1:
            print(f"  SD = {stdev(vals):.4f}")
        else:
            print()
    return deltas


def main():
    print("=" * 70)
    print(" ARCHITECTURE ABLATION: BatchNorm (bn) vs GroupNorm (gn)")
    print("=" * 70)
    print(f"\n Seeds compared: {ABLATION_SEEDS}")
    print(f" GN dir: {GN_DIR}")
    print(f" BN dir: {BN_DIR}")

    bn_variant = load_variant(BN_DIR, _BN_PAT, set(ABLATION_SEEDS))
    gn_variant = load_variant(GN_DIR, _GN_PAT, set(ABLATION_SEEDS))

    gn_deltas = print_variant_table("GroupNorm headline (gn)", gn_variant)
    bn_deltas = print_variant_table("BatchNorm ablation (bn)", bn_variant)

    # Cross-variant contrast
    print("\n" + "=" * 70)
    print(" CROSS-VARIANT CONTRAST: delta_BN - delta_GN")
    print("=" * 70)
    common = sorted(set(bn_deltas) & set(gn_deltas))
    if not common:
        print("\n  (no seeds present in both variants yet)")
        print("\n  Hint: run the BN ablation first via")
        print("    bash mnist_dermnist/scripts/runpod_arch_ablation_bn.sh")
        return

    print(f"\n  {'seed':>8s}  {'delta_GN':>9s}  {'delta_BN':>9s}  {'BN - GN':>9s}")
    print("  " + "-" * 44)
    cross = []
    for s in common:
        gn_d = gn_deltas[s]
        bn_d = bn_deltas[s]
        contrast = bn_d - gn_d
        cross.append(contrast)
        print(f"  {s:>8d}  {gn_d:>+9.4f}  {bn_d:>+9.4f}  {contrast:>+9.4f}")

    n = len(cross)
    print(f"\n  n={n}  mean(BN - GN) = {mean(cross):+.4f}", end="")
    if n > 1:
        print(f"  SD = {stdev(cross):.4f}")
    else:
        print()

    print("\n  Interpretation:")
    print("    Positive contrast (BN - GN > 0)  =>  FedProx helps MORE on BN")
    print("                                          (consistent with FedBN hypothesis)")
    print("    Near zero                        =>  FedProx is normalization-insensitive")
    print("    Negative contrast                =>  FedProx helps LESS on BN")
    print("                                          (unexpected; would indicate")
    print("                                          parameter-side anchor is")
    print("                                          dominated by BN-stats drift)")


if __name__ == "__main__":
    main()
