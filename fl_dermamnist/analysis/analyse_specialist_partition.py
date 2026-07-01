"""Analyse the specialist-partition defensive sweep.

Reads the 20 test_at_best_*.json files in
``fl_dermamnist/results/specialist_partition/`` (produced by
``submit_specialist_partition.sh``) and computes:

  1. Paired Wilcoxon signed-rank test on Δ = FedProx − FedAvg
     (two-sided, n=10).
  2. Hodges–Lehmann point estimate of median(Δ), via the median of
     all (n choose 2) + n Walsh averages.
  3. Exact Wilcoxon distribution-free confidence interval for
     median(Δ), via Walsh-average inversion. NOT a bootstrap CI:
     the bounds are deterministic functions of the data and depend
     only on n, not on a resampling RNG.
  4. Sign test on Pr(Δ > 0).
  5. The four-point dose-response table comparing
       Δ_specialist  vs  Δ_dirichlet  vs  Δ_paired  (vs  Δ_IID anchor)
     so the reader can see the monotone (or non-monotone) relationship
     between amount of label skew and FedProx advantage.

Outputs:
  - results/thesis_ready/data/specialist_partition_results.json
  - prints a summary table to stdout

The script refuses to interpret the result direction; it just reports
the numbers. Which of the four pre-registered scenarios it implicates
is decided in writing/specialist_partition_scenarios.tex.
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import combinations_with_replacement
from pathlib import Path
from fl_dermamnist.common.paths import repo_root, package_root, results_root, thesis_ready_root, thesis_data_dir, thesis_figures_dir  # noqa: E402

import numpy as np
from scipy import stats


# Wilcoxon signed-rank critical values for two-sided α = 0.05.
# Smallest rank-sum w such that P(W ≤ w) > α/2 under the null.
# Source: standard Wilcoxon tables (or scipy.stats.wilcoxon's exact
# distribution). For n = 10, the exact lower-tail probability is:
#   P(W ≤ 5) = 0.00977  (too strict)
#   P(W ≤ 8) = 0.02441  (just under α/2 = 0.025)
#   P(W ≤ 9) = 0.03223  (over α/2)
# Therefore the inversion uses w = 8 + 1 = 9th-smallest Walsh average
# as the lower bound and the 9th-largest as the upper bound.
# (Equivalently: positions 9 and 47 = 55 − 9 + 1 in the sorted Walsh
# averages, 1-indexed.)
WILCOXON_CI_POSITION_N10_ALPHA05 = 9


# parents[2] is fl_dermamnist/results/ (script is at fl_dermamnist/analysis/).
RESULTS_DIR    = results_root()
SPECIALIST_DIR = RESULTS_DIR / "specialist_partition"
# Default: compare against the headline directory (currently the legacy
# pure-PyTorch data - see results/headline/README_PROVENANCE.md). To
# compare against the new Flower-runtime headline once it lands, pass:
#   --paired-dir fl_dermamnist/results/flower_C0_baseline
PAIRED_DIR     = RESULTS_DIR / "headline"
DIRICHLET_DIR  = RESULTS_DIR / "dirichlet_a01"
IID_DIR        = RESULTS_DIR / "iid"
OUT_DIR        = RESULTS_DIR / "thesis_ready" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


_PAT = re.compile(
    r"test_at_best_(?P<algo>fedavg|fedprox)_mu[0-9.]+_E20"
    r"(?:_sh-[a-z_]+)?(?:_C[0-9.]+)?_s(?P<seed>\d+)\.json"
)


def _load_pairs(results_dir: Path) -> tuple[dict[int, float], dict[int, float]]:
    """Load {seed: macro_f1} dicts for FedAvg and FedProx from a results dir."""
    fa: dict[int, float] = {}
    fp: dict[int, float] = {}
    if not results_dir.is_dir():
        return fa, fp
    for p in sorted(results_dir.glob("test_at_best_*.json")):
        m = _PAT.match(p.name)
        if not m:
            continue
        algo = m.group("algo")
        seed = int(m.group("seed"))
        doc = json.load(open(p))
        macro = float(doc["macro_f1"])
        (fa if algo == "fedavg" else fp)[seed] = macro
    return fa, fp


def _paired_deltas(fa: dict[int, float], fp: dict[int, float]) -> np.ndarray:
    """Return paired Δ array sorted by seed; only seeds present in both."""
    paired_seeds = sorted(set(fa) & set(fp))
    return np.array([fp[s] - fa[s] for s in paired_seeds], dtype=float)


def walsh_averages(x: np.ndarray) -> np.ndarray:
    """All Walsh averages {(x_i + x_j) / 2 : i ≤ j}. Length n(n+1)/2."""
    pairs = list(combinations_with_replacement(range(len(x)), 2))
    return np.array([(x[i] + x[j]) / 2.0 for i, j in pairs], dtype=float)


def hodges_lehmann(x: np.ndarray) -> float:
    """Hodges–Lehmann estimator = median of Walsh averages."""
    return float(np.median(walsh_averages(x)))


def exact_wilcoxon_ci(x: np.ndarray, k: int = WILCOXON_CI_POSITION_N10_ALPHA05) -> tuple[float, float]:
    """Distribution-free Wilcoxon CI for median(x) via Walsh-average inversion.

    For n = 10 paired observations, two-sided α = 0.05, the CI bounds are
    the ``k``-th smallest and ``k``-th largest Walsh averages (1-indexed).
    For other n this function still applies - pass the appropriate ``k``
    from the Wilcoxon critical-value table.
    """
    wa = np.sort(walsh_averages(x))
    return float(wa[k - 1]), float(wa[-k])


def sign_test_two_sided(x: np.ndarray) -> tuple[int, int, float]:
    """Two-sided sign test on Pr(x > 0). Returns (n_pos, n_nonzero, p)."""
    nonzero = x[x != 0]
    n_pos = int((nonzero > 0).sum())
    n = len(nonzero)
    # Two-sided binomial p under H0: P(X > 0) = 0.5.
    p = 2 * min(
        stats.binom.cdf(n_pos, n, 0.5),
        stats.binom.sf(n_pos - 1, n, 0.5),
    )
    return n_pos, n, float(min(p, 1.0))


def summarise(name: str, deltas: np.ndarray, *, alpha: float = 0.05) -> dict:
    """Run all paired-tests on one Δ array; return a dict-of-everything."""
    if len(deltas) == 0:
        return {"name": name, "n": 0, "note": "no data"}
    summary: dict = {"name": name, "n": int(len(deltas)),
                     "deltas": deltas.tolist(),
                     "mean": float(np.mean(deltas)),
                     "std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                     "median": float(np.median(deltas))}
    if len(deltas) >= 2 and np.any(deltas != 0):
        try:
            res = stats.wilcoxon(deltas, alternative="two-sided",
                                 zero_method="wilcox", method="exact")
            summary["wilcoxon_p_two_sided"] = float(res.pvalue)
            summary["wilcoxon_significant_at_05"] = bool(res.pvalue < alpha)
        except Exception as e:
            summary["wilcoxon_error"] = repr(e)
    summary["hodges_lehmann"] = hodges_lehmann(deltas)
    if len(deltas) == 10:
        lo, hi = exact_wilcoxon_ci(deltas, k=WILCOXON_CI_POSITION_N10_ALPHA05)
        summary["hl_ci_lo_exact_n10_alpha05"] = lo
        summary["hl_ci_hi_exact_n10_alpha05"] = hi
        summary["ci_method"] = "exact Walsh-average inversion (n=10, α=0.05)"
    n_pos, n_nonzero, p_sign = sign_test_two_sided(deltas)
    summary["sign_test_n_pos"] = n_pos
    summary["sign_test_n"] = n_nonzero
    summary["sign_test_p_two_sided"] = p_sign
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--specialist-dir", default=str(SPECIALIST_DIR))
    ap.add_argument("--paired-dir",     default=str(PAIRED_DIR))
    ap.add_argument("--dirichlet-dir",  default=str(DIRICHLET_DIR))
    ap.add_argument("--iid-dir",        default=str(IID_DIR))
    args = ap.parse_args()

    specialist = _paired_deltas(*_load_pairs(Path(args.specialist_dir)))
    paired     = _paired_deltas(*_load_pairs(Path(args.paired_dir)))
    dirichlet  = _paired_deltas(*_load_pairs(Path(args.dirichlet_dir)))
    iid        = _paired_deltas(*_load_pairs(Path(args.iid_dir)))

    summaries = {
        "specialist_7_clients":         summarise("specialist",   specialist),
        "balanced_paired_7_clients":    summarise("paired",       paired),
        "dirichlet_alpha01_7_clients":  summarise("dirichlet_α=0.1", dirichlet),
        "iid_7_clients":                summarise("IID anchor",   iid),
    }

    # --- Stdout report -----------------------------------------------------
    print("=" * 92)
    print("  SPECIALIST-PARTITION DEFENSIVE SWEEP — paired-Δ inference")
    print("=" * 92)
    fmt = "  {name:24s}  n={n:2d}  mean Δ = {mean:+.4f}  HL = {hl:+.4f}  Wilcoxon p = {p:>7s}  sign p = {sp:>7s}"
    for key, s in summaries.items():
        if s.get("n", 0) == 0:
            print(f"  {key:24s}  (no data on disk yet)")
            continue
        print(fmt.format(
            name=s["name"],
            n=s["n"], mean=s["mean"],
            hl=s.get("hodges_lehmann", float("nan")),
            p=f"{s.get('wilcoxon_p_two_sided', float('nan')):.4f}"
               if "wilcoxon_p_two_sided" in s else "—",
            sp=f"{s.get('sign_test_p_two_sided', float('nan')):.4f}"
               if "sign_test_p_two_sided" in s else "—",
        ))

    # --- 4-point dose-response ---------------------------------------------
    if specialist.size and paired.size:
        print()
        print("=" * 92)
        print("  4-POINT DOSE-RESPONSE")
        print("=" * 92)
        rows = []
        for label, deltas in [
            ("IID (null mechanism)",     iid),
            ("specialist (1 client / class)", specialist),
            ("Dirichlet α=0.1",          dirichlet),
            ("paired (2 clients / class)",   paired),
        ]:
            if deltas.size == 0:
                rows.append((label, "—", "—"))
                continue
            rows.append((label, f"{deltas.mean():+.4f}", f"{hodges_lehmann(deltas):+.4f}"))
        print(f"  {'condition':35s}  mean Δ      HL Δ")
        print("  " + "-" * 60)
        for label, m, hl in rows:
            print(f"  {label:35s}  {m:>9s}   {hl:>9s}")
        print()
        print("  Pre-registered prediction (commit 5aba7ad, 2026-05-21):")
        print("    Δ_specialist > 0  AND  |Δ_specialist| < |Δ_paired|")
        if specialist.size and paired.size:
            obs_dir  = float(np.median(specialist)) > 0
            obs_mag  = abs(float(np.median(specialist))) < abs(float(np.median(paired)))
            print(f"  Direction (specialist median > 0)      observed: {obs_dir}")
            print(f"  Magnitude (|specialist| < |paired|)    observed: {obs_mag}")

    # --- Write JSON --------------------------------------------------------
    out_path = OUT_DIR / "specialist_partition_results.json"
    out_path.write_text(json.dumps(summaries, indent=2))
    print()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
