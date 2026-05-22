"""Quick verification + comparison of flower_C0_baseline (engineered paired).

Idempotent audit. Re-run anytime to refresh:
    python mnist_dermnist/scripts/check_engineered.py

Reports:
  1. File counts vs expected 30/30 (10 seeds * 3 algos)
  2. Provenance completeness
  3. Per-seed FedAvg vs FedProx macro_f1 + within-pair delta
  4. Mean/median/SD delta, win rate, paired Wilcoxon p
  5. FedNova standalone macro_f1 (C0)
  6. Cross-runtime comparison vs pure-PyTorch headline
"""
import json
import glob
import os
from collections import defaultdict
from statistics import mean, median, stdev


def load_dir(d):
    """Load {algo: {seed: macro_f1}} from a results directory."""
    results = defaultdict(dict)
    provenance_ok = True
    for f in sorted(glob.glob(f"{d}/test_at_best_*.json")):
        name = os.path.basename(f).replace("test_at_best_", "").replace(".json", "")
        parts = name.split("_")
        algo = parts[0]
        seed = int(parts[-1].lstrip("s"))
        doc = json.load(open(f))
        results[algo][seed] = float(doc["macro_f1"])
        for k in ("framework", "runner_script", "loss_type"):
            if k not in doc:
                provenance_ok = False
    return results, provenance_ok


def paired_deltas(results):
    """Return list of (FedProx - FedAvg) for seeds present in both algorithms."""
    paired = []
    seeds = sorted(set(results["fedavg"]) & set(results["fedprox"]))
    return seeds, [results["fedprox"][s] - results["fedavg"][s] for s in seeds]


def main():
    print("=== flower_C0_baseline (engineered paired_7_clients) audit ===\n")
    D = "mnist_dermnist/results/flower_C0_baseline"
    results, prov_ok = load_dir(D)

    total = sum(len(v) for v in results.values())
    print(f"Files on disk: {total} / 30 expected")
    print(f"  FedAvg : {len(results['fedavg'])} / 10")
    print(f"  FedProx: {len(results['fedprox'])} / 10")
    print(f"  FedNova: {len(results['fednova'])} / 10")
    print(f"  Provenance complete: {prov_ok}\n")

    print("=== Per-seed macro_f1 (FedAvg vs FedProx) ===\n")
    seeds, paired = paired_deltas(results)
    print(f"  {'seed':>8s}  {'FedAvg':>8s}  {'FedProx':>8s}  {'delta':>9s}  {'sign':>5s}")
    print("  " + "-" * 54)
    for s, d in zip(seeds, paired):
        fa = results["fedavg"][s]
        fp = results["fedprox"][s]
        sign = "FP" if d > 0 else "FA"
        print(f"  {s:>8d}  {fa:>8.4f}  {fp:>8.4f}  {d:>+9.4f}  {sign:>5s}")

    if paired:
        n = len(paired)
        wins = sum(1 for d in paired if d > 0)
        print(f"\n  n = {n} paired seeds")
        print(f"  Mean   delta = {mean(paired):+.4f}")
        print(f"  Median delta = {median(paired):+.4f}")
        if n > 1:
            print(f"  SD     delta = {stdev(paired):.4f}")
        print(f"  FedProx wins: {wins}/{n}")

        try:
            from scipy.stats import wilcoxon
            if any(d != 0 for d in paired):
                w = wilcoxon(paired, alternative="two-sided")
                print(f"  Paired Wilcoxon two-sided p = {w.pvalue:.4f}")
        except ImportError:
            print("  (scipy not available; skip Wilcoxon)")

    print("\n=== FedNova standalone macro_f1 (no FedAvg pair) ===")
    if results["fednova"]:
        for s in sorted(results["fednova"]):
            print(f"  seed {s:>8d}: {results['fednova'][s]:.4f}")
        fn_vals = list(results["fednova"].values())
        print(f"  Mean = {mean(fn_vals):.4f}")
    else:
        print("  (no FedNova files yet)")

    print("\n=== Cross-runtime comparison vs pure-PyTorch headline ===")
    HD = "mnist_dermnist/results/headline"
    if os.path.isdir(HD):
        pt_results, _ = load_dir(HD)
        _, pt_paired = paired_deltas(pt_results)
        if pt_paired:
            print(f"  Pure-PyTorch (n={len(pt_paired)}):  mean delta = {mean(pt_paired):+.4f}")
        if paired:
            print(f"  Flower       (n={len(paired)}):  mean delta = {mean(paired):+.4f}")
            if pt_paired and len(pt_paired) > 0 and len(paired) > 0:
                gap = mean(pt_paired) - mean(paired)
                print(f"  Cross-runtime gap (PT - Flower): {gap:+.4f}")
    else:
        print("  (no pure-PyTorch headline directory)")


if __name__ == "__main__":
    main()
