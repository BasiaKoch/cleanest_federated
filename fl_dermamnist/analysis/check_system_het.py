"""Quick audit of system-heterogeneity sweeps (C0 / C1 / C2).

Idempotent. Re-run any time to refresh:
    python fl_dermamnist/analysis/check_system_het.py

Reports:
  1. File counts per directory vs expected
  2. Per-seed FedAvg vs FedProx delta for C1 (fixed) and C2 (random)
  3. FedNova C2 status (documented failure mode)
  4. H2 between-condition contrast: delta_C1 - delta_C0, delta_C2 - delta_C0
"""
import json
import glob
import os
from collections import defaultdict
from statistics import mean, median, stdev


def load_dir(d):
    """Load {algo: {seed: macro_f1}} from a results directory."""
    results = defaultdict(dict)
    for f in sorted(glob.glob(f"{d}/test_at_best_*.json")):
        name = os.path.basename(f).replace("test_at_best_", "").replace(".json", "")
        # Stem may include sh- tag in the middle; algo is parts[0], seed is parts[-1]
        parts = name.split("_")
        algo = parts[0]
        seed = int(parts[-1].lstrip("s"))
        doc = json.load(open(f))
        results[algo][seed] = float(doc["macro_f1"])
    return results


def paired_deltas(results):
    """Return list of (FedProx - FedAvg) for seeds present in both algorithms."""
    seeds = sorted(set(results["fedavg"]) & set(results["fedprox"]))
    return seeds, [results["fedprox"][s] - results["fedavg"][s] for s in seeds]


def report_condition(label, dir_path, expected_files=20):
    print(f"\n=== {label} ({dir_path}) ===")
    results = load_dir(dir_path)
    total = sum(len(v) for v in results.values())
    print(f"  Files: {total} / {expected_files} expected")
    for algo in ("fedavg", "fedprox", "fednova"):
        if algo in results:
            n = len(results[algo])
            vals = list(results[algo].values())
            bad = sum(1 for v in vals if v < 0.20)
            line = f"    {algo:8s}: {n:2d} files"
            if vals:
                line += f"  mean = {mean(vals):.4f}"
                if n > 1:
                    line += f"  SD = {stdev(vals):.4f}"
                if bad:
                    line += f"  BAD = {bad} (macro_f1 < 0.20)"
            print(line)
    return results


def main():
    print("=" * 70)
    print(" SYSTEM HETEROGENEITY AUDIT")
    print("=" * 70)

    # C0 = flower_C0_baseline (the uniform-compute reference)
    c0 = report_condition("C0 (uniform compute)",
                          "fl_dermamnist/results/flower_C0_baseline",
                          expected_files=30)

    # C1 = system_het_fixed
    c1 = report_condition("C1 (fixed stragglers, C5+C6 at E=5)",
                          "fl_dermamnist/results/system_het_fixed",
                          expected_files=20)

    # C2 = system_het_random
    c2 = report_condition("C2 (random stragglers, primary inferential)",
                          "fl_dermamnist/results/system_het_random",
                          expected_files=20)

    # FedNova C2 — documented failure mode
    fednova_c2 = report_condition("C2 FedNova (KNOWN FAILURE MODE — Issue 2)",
                                   "fl_dermamnist/results/system_het_random_fednova",
                                   expected_files=10)

    # Per-condition paired delta
    print("\n" + "=" * 70)
    print(" Per-condition paired delta (FedProx - FedAvg)")
    print("=" * 70)

    deltas_by_cond = {}
    for name, results in [("C0", c0), ("C1", c1), ("C2", c2)]:
        seeds, deltas = paired_deltas(results)
        deltas_by_cond[name] = dict(zip(seeds, deltas))
        if deltas:
            n = len(deltas)
            wins = sum(1 for d in deltas if d > 0)
            mu = mean(deltas)
            med = median(deltas)
            sd = stdev(deltas) if n > 1 else 0.0
            print(f"\n  {name}: n={n}  mean={mu:+.4f}  median={med:+.4f}  SD={sd:.4f}  FedProx wins={wins}/{n}")
            try:
                from scipy.stats import wilcoxon
                if any(d != 0 for d in deltas):
                    w = wilcoxon(deltas, alternative="two-sided")
                    print(f"        Paired Wilcoxon two-sided p = {w.pvalue:.4f}")
            except ImportError:
                pass
        else:
            print(f"\n  {name}: no paired data yet")

    # H2 contrast: delta_C2 - delta_C0 (primary), delta_C1 - delta_C0 (descriptive)
    print("\n" + "=" * 70)
    print(" H2 between-condition contrast (does FedProx help MORE under het?)")
    print("=" * 70)

    for cond in ("C1", "C2"):
        common = sorted(set(deltas_by_cond[cond]) & set(deltas_by_cond["C0"]))
        if not common:
            print(f"\n  {cond} vs C0: no common seeds yet")
            continue
        contrast = [deltas_by_cond[cond][s] - deltas_by_cond["C0"][s] for s in common]
        n = len(contrast)
        mu = mean(contrast)
        print(f"\n  delta_{cond} - delta_C0  (n={n}):  mean = {mu:+.4f}")
        if n >= 3:
            try:
                from scipy.stats import wilcoxon
                if any(c != 0 for c in contrast):
                    w = wilcoxon(contrast, alternative="two-sided")
                    note = "" if cond == "C2" else "  (descriptive only — C1 has straggler-identity confound)"
                    print(f"        Paired Wilcoxon two-sided p = {w.pvalue:.4f}{note}")
            except ImportError:
                pass


if __name__ == "__main__":
    main()
