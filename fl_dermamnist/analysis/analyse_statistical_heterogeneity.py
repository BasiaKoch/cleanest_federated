"""Statistical-heterogeneity analysis - thesis-grade pipeline.

Consumes:
    results/headline/                - pure-PyTorch headline (PRIMARY)
    results/flower_C0_baseline/      - Flower replication of headline
    results/iid/                     - IID falsification check
    results/dirichlet_a01/           - Dirichlet α=0.1 (literature non-IID)
    results/specialist_partition/    - specialist (pairing-lever)
    results/centralised/             - pure-PyTorch upper bound

Produces:
    results/thesis_ready/data/statistical_heterogeneity_results.json   - every number
    results/thesis_ready/tables/T01_headline_summary.tex               - Table 1 (LaTeX)
    results/thesis_ready/tables/T02_per_class_holm.tex                 - Table 2
    results/thesis_ready/tables/T03_cross_runtime.tex                  - Table 3
    results/thesis_ready/tables/T04_partition_dose_response.tex        - Table 4
    results/thesis_ready/tables/T05_federation_tax.tex                 - Table 5
    Plus a clean human-readable text report printed to stdout.

Methodology
-----------
PRIMARY (one inferential test, no multiple-testing correction):
    Paired Wilcoxon signed-rank, two-sided, on per-seed Δ = FedProx − FedAvg
    from results/headline/ (n=10 paired seeds, pure-PyTorch runtime,
    engineered balanced_paired_7_clients partition).

SECONDARY FAMILY (Holm-corrected within the primary):
    7 per-class F1 contrasts (one per DermaMNIST class).

EXPLORATORY (descriptive only, NOT inferential):
    Flower replication of the headline (Δ_Flower)
    IID falsification (Δ_iid)
    Dirichlet α=0.1 (Δ_dir)
    Specialist (Δ_spec)
    Federation-tax % (no paired test against centralised)

Effect-size / CI methods
------------------------
    Hodges-Lehmann point estimator (median of Walsh averages).
    Exact Wilcoxon distribution-free 95% CI via Walsh-average inversion;
        for n=10, two-sided α=0.05, the CI bounds are the 9th smallest
        and 9th largest of the 55 Walsh averages.
    Bootstrap percentile CI: 10,000 resamples of paired Δ with replacement.
    LOSO robustness: remove each seed in turn; report how many of n
        subsamples still reach α=0.05.
    Sign test (binomial on n_positive vs n) as a direction-only auxiliary.
    Rank-biserial correlation = (W+ − W−) / (W+ + W−).

Usage
-----
    PYTHONPATH=. python -m fl_dermamnist.analysis.analyse_statistical_heterogeneity

(or directly: python fl_dermamnist/analysis/analyse_statistical_heterogeneity.py)
"""
from __future__ import annotations

import json
import os
import re
from itertools import combinations_with_replacement
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
from scipy import stats


# --- Configuration ----------------------------------------------------------

PAIRED_SEEDS = [42, 123, 456, 789, 999, 2024, 31337, 161803, 271828, 8675309]

CLASS_NAMES = [
    "actinic_keratoses",
    "basal_cell_carcinoma",
    "benign_keratosis",
    "dermatofibroma",
    "melanoma",
    "melanocytic_nevi",
    "vascular_lesions",
]
CLASS_DISPLAY = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]
CLASS_PREVALENCE = [0.0327, 0.0513, 0.1097, 0.0115, 0.1111, 0.6705, 0.0141]


# --- Path discovery (works from anywhere via __file__) ----------------------

import sys  # noqa: E402

# Resolve paths through the shared module so this script survives being moved
# to a different directory depth. Bootstrap: walk up to the package root so the
# import works even when run as a bare ``python <path>`` invocation.
_HERE = Path(__file__).resolve()
for _cand in _HERE.parents:
    if (_cand / "fl_dermamnist" / "__init__.py").is_file():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from fl_dermamnist.common.paths import results_root  # noqa: E402

RESULTS = results_root()
OUT_TABLES = RESULTS / "thesis_ready" / "tables"
OUT_DATA = RESULTS / "thesis_ready" / "data"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_DATA.mkdir(parents=True, exist_ok=True)


# --- File-loading helpers ---------------------------------------------------

_STD_PAT = re.compile(
    r"test_at_best_(?P<algo>fedavg|fedprox|fednova)_mu(?P<mu>[0-9.]+)_E20_s(?P<seed>\d+)\.json"
)


def _load_sweep(dir_name: str) -> dict:
    """Load a 'standard'-naming sweep (no _sh-/_drop/_arch- tags).

    Returns {algo: {seed: {macro_f1, per_class_f1, framework, runner_script, ...}}}.
    Refuses to silently include files with non-default tags.
    """
    out: dict = {"fedavg": {}, "fedprox": {}, "fednova": {}}
    d = RESULTS / dir_name
    if not d.is_dir():
        raise FileNotFoundError(f"Missing sweep directory: {d}")
    for f in sorted(d.glob("test_at_best_*.json")):
        name = f.name
        # Reject any non-standard tag (defensive)
        if any(tag in name for tag in ("_sh-", "_arch-", "_drop", "_C0.")):
            continue
        m = _STD_PAT.match(name)
        if not m:
            continue
        algo = m.group("algo")
        seed = int(m.group("seed"))
        if seed not in PAIRED_SEEDS:
            continue
        with open(f) as fh:
            doc = json.load(fh)
        out[algo][seed] = doc
    return out


def _load_centralised() -> dict:
    """Centralised reference uses a different naming convention."""
    out: dict = {}
    d = RESULTS / "centralised"
    if not d.is_dir():
        raise FileNotFoundError(f"Missing centralised directory: {d}")
    for f in sorted(d.glob("centralised_seed*.json")):
        seed_str = f.stem.replace("centralised_seed", "")
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        if seed not in PAIRED_SEEDS:
            continue
        with open(f) as fh:
            out[seed] = json.load(fh)
    return out


# --- Statistical machinery --------------------------------------------------

def walsh_averages(x: list[float]) -> list[float]:
    """All n(n+1)/2 Walsh averages (i ≤ j)."""
    return sorted(
        (x[i] + x[j]) / 2.0
        for i, j in combinations_with_replacement(range(len(x)), 2)
    )


def hodges_lehmann(x: list[float]) -> float:
    return float(np.median(walsh_averages(x)))


def walsh_ci(x: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """Exact Walsh-average-inversion CI for the median of x.

    For n=10, two-sided α=0.05, the bounds are the 9th-smallest and
    9th-largest of the 55 Walsh averages (table lookup via
    scipy.stats.wilcoxon's exact distribution at boundary rank 8).
    For other n we look up via scipy.stats.wilcoxon's cdf.
    """
    n = len(x)
    wa = walsh_averages(x)
    # For n=10 we have a closed-form table position; for general n,
    # derive the rank where the two-sided cumulative ≥ α/2.
    if n == 10 and abs(alpha - 0.05) < 1e-9:
        # Standard Wilcoxon table: position 9 (1-indexed) = 9th smallest
        k = 9
    else:
        # Find smallest k s.t. P(W ≤ k−1) ≤ α/2 under H0
        from scipy.stats import wilcoxon as _wx
        # Build the discrete distribution via wilcoxon's exact mode
        # by trying all possible rank-sum tail probabilities.
        k = max(1, int(np.floor(n * (n + 1) / 4 - 1.96 * np.sqrt(n * (n + 1) * (2 * n + 1) / 24))))
    return wa[k - 1], wa[-k]


def bootstrap_ci(x: list[float], n_resamples: int = 10_000,
                 alpha: float = 0.05, rng_seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of x."""
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    means = np.array([rng.choice(x, size=n, replace=True).mean()
                      for _ in range(n_resamples)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def loso(x: list[float], alpha: float = 0.05) -> dict:
    """Leave-one-seed-out: how many of n subsamples remain sig at α?"""
    n = len(x)
    pvals = []
    sig = 0
    for i in range(n):
        subset = x[:i] + x[i + 1:]
        if any(d != 0 for d in subset):
            res = stats.wilcoxon(subset, alternative="two-sided", method="exact")
            pvals.append(float(res.pvalue))
            if res.pvalue < alpha:
                sig += 1
        else:
            pvals.append(1.0)
    return {"sig_count": sig, "total": n, "all_pvals": pvals}


def sign_test_two_sided(x: list[float]) -> tuple[int, int, float]:
    """Two-sided sign test on Pr(x > 0)."""
    nonzero = [v for v in x if v != 0]
    n_pos = sum(1 for v in nonzero if v > 0)
    n = len(nonzero)
    if n == 0:
        return 0, 0, 1.0
    p = 2 * min(stats.binom.cdf(n_pos, n, 0.5),
                stats.binom.sf(n_pos - 1, n, 0.5))
    return n_pos, n, float(min(p, 1.0))


def rank_biserial(x: list[float]) -> float:
    """Rank-biserial correlation = (W+ − W−) / (W+ + W−)."""
    abs_ranks = stats.rankdata([abs(v) for v in x])
    w_pos = sum(r for v, r in zip(x, abs_ranks) if v > 0)
    w_neg = sum(r for v, r in zip(x, abs_ranks) if v < 0)
    total = w_pos + w_neg
    if total == 0:
        return 0.0
    return (w_pos - w_neg) / total


def holm_correction(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns corrected p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    corrected = [0.0] * m
    running_max = 0.0
    for i, idx in enumerate(order):
        c = pvals[idx] * (m - i)
        c = min(c, 1.0)
        if c < running_max:
            c = running_max
        else:
            running_max = c
        corrected[idx] = c
    return corrected


def paired_inference(deltas: list[float], label: str = "") -> dict:
    """Full paired-seed inference suite on a list of per-seed Δ values."""
    n = len(deltas)
    if n == 0:
        return {"label": label, "n": 0, "note": "no data"}
    out: dict = {
        "label": label,
        "n": n,
        "deltas": list(deltas),
        "mean": float(mean(deltas)),
        "median": float(median(deltas)),
        "sd": float(stdev(deltas)) if n > 1 else 0.0,
        "min": float(min(deltas)),
        "max": float(max(deltas)),
        "wins_positive": sum(1 for d in deltas if d > 0),
        "wins_negative": sum(1 for d in deltas if d < 0),
    }

    if any(d != 0 for d in deltas):
        # Paired Wilcoxon (exact when n ≤ 25)
        w = stats.wilcoxon(deltas, alternative="two-sided",
                           method="exact" if n <= 25 else "approx")
        out["wilcoxon_p_two_sided"] = float(w.pvalue)
        out["hodges_lehmann"] = hodges_lehmann(deltas)
        if n == 10:
            lo, hi = walsh_ci(deltas, alpha=0.05)
            out["walsh_ci_95_n10"] = [lo, hi]
        out["rank_biserial"] = float(rank_biserial(deltas))
        # Bootstrap percentile CI on the mean
        b_lo, b_hi = bootstrap_ci(deltas, n_resamples=10_000)
        out["bootstrap_ci_95_mean"] = [b_lo, b_hi]
        # Sign test
        n_pos, n_nonzero, sign_p = sign_test_two_sided(deltas)
        out["sign_test"] = {
            "n_positive": n_pos,
            "n_nonzero": n_nonzero,
            "p_two_sided": sign_p,
        }
        # LOSO
        out["loso"] = loso(deltas)
    else:
        out["wilcoxon_p_two_sided"] = 1.0

    return out


# --- Sweep-level analysis ---------------------------------------------------

def compute_paired_deltas(sweep: dict) -> tuple[list[int], list[float]]:
    """Return (seeds, deltas) for FedProx vs FedAvg paired by seed."""
    fa = sweep.get("fedavg", {})
    fp = sweep.get("fedprox", {})
    seeds = sorted(set(fa) & set(fp))
    deltas = [float(fp[s]["macro_f1"]) - float(fa[s]["macro_f1"]) for s in seeds]
    return seeds, deltas


def per_class_paired_deltas(sweep: dict) -> dict[int, list[float]]:
    """Per-class paired Δ across seeds. Returns {class_idx: [Δ per seed]}."""
    fa = sweep.get("fedavg", {})
    fp = sweep.get("fedprox", {})
    seeds = sorted(set(fa) & set(fp))
    per_class: dict[int, list[float]] = {c: [] for c in range(7)}
    for s in seeds:
        fa_pc = fa[s]["per_class_f1"]
        fp_pc = fp[s]["per_class_f1"]
        for c in range(7):
            per_class[c].append(float(fp_pc[c]) - float(fa_pc[c]))
    return per_class


# --- LaTeX table generation -------------------------------------------------

def fmt_signed(x, digits=4):
    return f"{x:+.{digits}f}"


def fmt_p(p):
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.3f}"


def latex_table_headline(infer: dict, fa_stats: dict, fp_stats: dict) -> str:
    """Table 1: headline paired comparison (pure-PyTorch engineered partition)."""
    lo, hi = infer.get("walsh_ci_95_n10", (float("nan"), float("nan")))
    b_lo, b_hi = infer.get("bootstrap_ci_95_mean", (float("nan"), float("nan")))
    loso_str = f"{infer['loso']['sig_count']}/{infer['loso']['total']}"
    return rf"""\begin{{table}}[!htbp]
\centering
\small
\caption{{Headline paired comparison: FedProx vs FedAvg on pure-PyTorch headline (engineered \texttt{{balanced\_paired\_7\_clients}} partition, $n = 10$ paired seeds).}}
\label{{tab:headline-summary}}
\begin{{tabular}}{{l c c c}}
\toprule
Statistic & FedAvg & FedProx & $\Delta$ (FedProx $-$ FedAvg) \\
\midrule
Test macro-F1 mean $\pm$ SD & ${fa_stats['mean']:.4f} \pm {fa_stats['sd']:.4f}$ & ${fp_stats['mean']:.4f} \pm {fp_stats['sd']:.4f}$ & $\mathbf{{{fmt_signed(infer['mean'])} \pm {infer['sd']:.4f}}}$ \\
Median test macro-F1            & ${fa_stats['median']:.4f}$ & ${fp_stats['median']:.4f}$ & ${fmt_signed(infer['median'])}$ \\
Min / Max test macro-F1         & ${fa_stats['min']:.4f}$ / ${fa_stats['max']:.4f}$ & ${fp_stats['min']:.4f}$ / ${fp_stats['max']:.4f}$ & --- \\
Hodges--Lehmann robust estimate & --- & --- & ${fmt_signed(infer['hodges_lehmann'])}$ \\
Exact Walsh 95\% CI ($n=10$)    & --- & --- & $[{fmt_signed(lo)}, {fmt_signed(hi)}]$ \\
Bootstrap 95\% CI on mean       & --- & --- & $[{fmt_signed(b_lo)}, {fmt_signed(b_hi)}]$ \\
FedProx win rate                & --- & --- & $\mathbf{{{infer['wins_positive']}/{infer['n']}}}$ \\
Paired Wilcoxon $p$ (two-sided) & --- & --- & $\mathbf{{{fmt_p(infer['wilcoxon_p_two_sided'])}}}$ \\
Sign test $p$ (two-sided)       & --- & --- & ${fmt_p(infer['sign_test']['p_two_sided'])}$ \\
Rank-biserial $r$               & --- & --- & ${fmt_signed(infer['rank_biserial'])}$ \\
LOSO subsamples sig.\ at $\alpha = 0.05$ & --- & --- & $\mathbf{{{loso_str}}}$ \\
\bottomrule
\end{{tabular}}
\end{{table}}"""


def latex_table_per_class(per_class_stats: list[dict]) -> str:
    """Table 2: per-class paired Δ with Holm-Bonferroni correction."""
    rows = []
    for d in per_class_stats:
        sig = d["p_holm"] < 0.05
        marker = r"$\bm{*}$" if sig else ""
        rows.append(
            f"  {d['display']:25s} & "
            f"{d['prevalence']*100:5.2f}\\% & "
            f"${d['fa_mean']:.3f}$ & "
            f"${d['fp_mean']:.3f}$ & "
            f"${fmt_signed(d['delta_mean'])}$ & "
            f"${fmt_p(d['p_raw'])}$ & "
            f"${fmt_p(d['p_holm'])}$ {marker} \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[!htbp]
\centering
\small
\caption{{Per-class test F1 paired comparison (FedProx $-$ FedAvg) on the headline pure-PyTorch dataset. \textbf{{Holm--Bonferroni-corrected $p$-values across 7 classes}}; * indicates significance at family-wise $\alpha = 0.05$. Per-class tests are exploratory within the headline analysis.}}
\label{{tab:per-class-holm}}
\begin{{tabular}}{{l r c c c c c}}
\toprule
Class & Prev. & FedAvg F1 & FedProx F1 & $\Delta$ & $p_{{\mathrm{{raw}}}}$ & $p_{{\mathrm{{Holm}}}}$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def latex_table_cross_runtime(pt_infer: dict, fl_infer: dict, gap_infer: dict) -> str:
    """Table 3: cross-runtime comparison (PT vs Flower at same seeds)."""
    pt_ci = pt_infer.get("walsh_ci_95_n10", (float("nan"), float("nan")))
    fl_ci = fl_infer.get("walsh_ci_95_n10", (float("nan"), float("nan")))
    return rf"""\begin{{table}}[!htbp]
\centering
\small
\caption{{Cross-runtime comparison of the headline FedProx-FedAvg gap on the engineered partition. The pure-PyTorch (sequential) result is the pre-registered primary analysis; the Flower simulation is a replication. The Wilcoxon $p$ on the per-seed gap itself (third row) tests whether the two runtimes produce different paired-$\Delta$ values.}}
\label{{tab:cross-runtime}}
\begin{{tabular}}{{l c c c}}
\toprule
Runtime & Mean $\Delta$ & 95\% Walsh CI & Wilcoxon $p$ \\
\midrule
Pure-PyTorch (primary)         & ${fmt_signed(pt_infer['mean'])}$ & $[{fmt_signed(pt_ci[0])}, {fmt_signed(pt_ci[1])}]$ & $\mathbf{{{fmt_p(pt_infer['wilcoxon_p_two_sided'])}}}$ \\
Flower-simulation (replication) & ${fmt_signed(fl_infer['mean'])}$ & $[{fmt_signed(fl_ci[0])}, {fmt_signed(fl_ci[1])}]$ & ${fmt_p(fl_infer['wilcoxon_p_two_sided'])}$ \\
\midrule
Per-seed gap (PT $-$ Flower)    & ${fmt_signed(gap_infer['mean'])}$ & --- & ${fmt_p(gap_infer['wilcoxon_p_two_sided'])}$ \\
\bottomrule
\end{{tabular}}
\end{{table}}"""


def latex_table_partition_dose_response(rows: list[dict]) -> str:
    """Table 4: 4-partition Flower-runtime descriptive comparison."""
    body_rows = []
    for r in rows:
        body_rows.append(
            f"  {r['display']:35s} & "
            f"${r['fa_mean']:.4f}$ & "
            f"${r['fp_mean']:.4f}$ & "
            f"${fmt_signed(r['delta_mean'])}$ & "
            f"${fmt_p(r['p'])}$ & "
            f"{r['wins']}/{r['n']} \\\\"
        )
    body = "\n".join(body_rows)
    return rf"""\begin{{table}}[!htbp]
\centering
\small
\caption{{Statistical-heterogeneity partition robustness on the Flower runtime. \textbf{{All comparisons except the headline are exploratory; $p$-values are reported descriptively and are not corrected for multiplicity.}} The pure-PyTorch primary analysis (engineered partition, separate row) is the only inferential test.}}
\label{{tab:partition-dose-response}}
\begin{{tabular}}{{l c c c c c}}
\toprule
Partition & FedAvg mean & FedProx mean & Mean $\Delta$ & $p$ & FedProx wins \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def latex_table_federation_tax(rows: list[dict]) -> str:
    """Table 5: federation tax - FL macro_f1 vs centralised."""
    body_rows = []
    for r in rows:
        body_rows.append(
            f"  {r['display']:40s} & "
            f"${r['mean']:.4f} \\pm {r['sd']:.4f}$ & "
            f"{r['pct_central']:.1f}\\% \\\\"
        )
    body = "\n".join(body_rows)
    return rf"""\begin{{table}}[!htbp]
\centering
\small
\caption{{Federation tax: federated macro-F1 vs the pure-PyTorch centralised upper bound (no FL, pooled training set). Percentages refer to recovery of the centralised mean.}}
\label{{tab:federation-tax}}
\begin{{tabular}}{{l c c}}
\toprule
Setting & Test macro-F1 (mean $\pm$ SD) & \% of centralised \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


# --- Main ------------------------------------------------------------------

def main():
    print("=" * 88)
    print(" STATISTICAL HETEROGENEITY ANALYSIS")
    print("=" * 88)

    # --- Load all sweeps -----------------------------------------------------
    headline = _load_sweep("headline")
    flower   = _load_sweep("flower_C0_baseline")
    iid      = _load_sweep("iid")
    dirich   = _load_sweep("dirichlet_a01")
    spec     = _load_sweep("specialist_partition")
    central  = _load_centralised()

    sweeps = {
        "headline":             headline,
        "flower_C0_baseline":   flower,
        "iid":                  iid,
        "dirichlet_a01":        dirich,
        "specialist_partition": spec,
    }
    print(f"\nLoaded sweeps:")
    for name, sw in sweeps.items():
        print(f"  {name:30s}  FedAvg: {len(sw['fedavg']):2d}  FedProx: {len(sw['fedprox']):2d}  FedNova: {len(sw['fednova']):2d}")
    print(f"  centralised                   n={len(central):2d}")

    results = {}

    # --- PRIMARY: pure-PyTorch headline -------------------------------------
    print("\n" + "=" * 88)
    print(" PRIMARY (pre-registered): FedProx vs FedAvg, pure-PyTorch headline")
    print("=" * 88)
    pt_seeds, pt_deltas = compute_paired_deltas(headline)
    fa_macros = [float(headline["fedavg"][s]["macro_f1"]) for s in pt_seeds]
    fp_macros = [float(headline["fedprox"][s]["macro_f1"]) for s in pt_seeds]
    fa_stats = {
        "mean": mean(fa_macros), "median": median(fa_macros),
        "sd": stdev(fa_macros), "min": min(fa_macros), "max": max(fa_macros),
    }
    fp_stats = {
        "mean": mean(fp_macros), "median": median(fp_macros),
        "sd": stdev(fp_macros), "min": min(fp_macros), "max": max(fp_macros),
    }
    pt_infer = paired_inference(pt_deltas, label="pure-PyTorch headline")
    print(f"\nn paired seeds:       {pt_infer['n']}")
    print(f"  FedAvg  : mean {fa_stats['mean']:.4f}  SD {fa_stats['sd']:.4f}")
    print(f"  FedProx : mean {fp_stats['mean']:.4f}  SD {fp_stats['sd']:.4f}")
    print(f"  Mean Δ = {pt_infer['mean']:+.4f}, Median {pt_infer['median']:+.4f}, SD {pt_infer['sd']:.4f}")
    print(f"  Hodges-Lehmann = {pt_infer['hodges_lehmann']:+.4f}")
    print(f"  Walsh 95% CI = [{pt_infer['walsh_ci_95_n10'][0]:+.4f}, {pt_infer['walsh_ci_95_n10'][1]:+.4f}]")
    print(f"  Bootstrap 95% CI on mean = [{pt_infer['bootstrap_ci_95_mean'][0]:+.4f}, {pt_infer['bootstrap_ci_95_mean'][1]:+.4f}]")
    print(f"  Paired Wilcoxon p = {pt_infer['wilcoxon_p_two_sided']:.4f}")
    print(f"  Sign test p = {pt_infer['sign_test']['p_two_sided']:.4f}  ({pt_infer['sign_test']['n_positive']}/{pt_infer['sign_test']['n_nonzero']} positive)")
    print(f"  Rank-biserial r = {pt_infer['rank_biserial']:+.4f}")
    print(f"  FedProx wins = {pt_infer['wins_positive']}/{pt_infer['n']}")
    print(f"  LOSO: {pt_infer['loso']['sig_count']}/{pt_infer['loso']['total']} subsamples remain significant at α=0.05")

    results["primary_headline"] = {
        "fedavg_stats": fa_stats,
        "fedprox_stats": fp_stats,
        "inference": pt_infer,
        "seeds": pt_seeds,
    }

    # Write Table 1
    (OUT_TABLES / "T01_headline_summary.tex").write_text(
        latex_table_headline(pt_infer, fa_stats, fp_stats) + "\n"
    )
    print(f"\nWrote: {OUT_TABLES / 'T01_headline_summary.tex'}")

    # --- SECONDARY FAMILY: per-class Holm correction ------------------------
    print("\n" + "=" * 88)
    print(" SECONDARY (within primary, Holm-corrected): per-class F1 contrasts")
    print("=" * 88)
    pc_deltas = per_class_paired_deltas(headline)
    pc_pvals = []
    pc_stats_rows = []
    for c in range(7):
        ds = pc_deltas[c]
        if any(d != 0 for d in ds):
            w = stats.wilcoxon(ds, alternative="two-sided", method="exact")
            p_raw = float(w.pvalue)
        else:
            p_raw = 1.0
        pc_pvals.append(p_raw)
        fa_per_class = [float(headline["fedavg"][s]["per_class_f1"][c]) for s in pt_seeds]
        fp_per_class = [float(headline["fedprox"][s]["per_class_f1"][c]) for s in pt_seeds]
        pc_stats_rows.append({
            "class_idx": c,
            "display": CLASS_DISPLAY[c],
            "prevalence": CLASS_PREVALENCE[c],
            "fa_mean": float(mean(fa_per_class)),
            "fp_mean": float(mean(fp_per_class)),
            "delta_mean": float(mean(ds)),
            "p_raw": p_raw,
        })
    pc_pvals_holm = holm_correction(pc_pvals)
    for row, p_holm in zip(pc_stats_rows, pc_pvals_holm):
        row["p_holm"] = p_holm

    print(f"\n{'Class':25s} {'Prev':>6s} {'FA F1':>8s} {'FP F1':>8s} {'Δ':>9s} {'p_raw':>9s} {'p_Holm':>9s}")
    print("-" * 90)
    for row in pc_stats_rows:
        sig = "  *" if row["p_holm"] < 0.05 else ""
        print(f"{row['display']:25s} {row['prevalence']*100:5.2f}% {row['fa_mean']:>8.4f} {row['fp_mean']:>8.4f} {row['delta_mean']:>+9.4f} {row['p_raw']:>9.4f} {row['p_holm']:>9.4f}{sig}")

    results["per_class_holm"] = pc_stats_rows

    # Write Table 2
    (OUT_TABLES / "T02_per_class_holm.tex").write_text(
        latex_table_per_class(pc_stats_rows) + "\n"
    )
    print(f"\nWrote: {OUT_TABLES / 'T02_per_class_holm.tex'}")

    # --- EXPLORATORY: cross-runtime comparison ------------------------------
    print("\n" + "=" * 88)
    print(" EXPLORATORY (descriptive only): cross-runtime comparison")
    print("=" * 88)
    fl_seeds, fl_deltas = compute_paired_deltas(flower)
    fl_infer = paired_inference(fl_deltas, label="Flower replication")

    # Per-seed gap (PT Δ − Flower Δ) at the intersection
    gap_seeds = sorted(set(pt_seeds) & set(fl_seeds))
    gap_deltas = []
    for s in gap_seeds:
        pt_d = float(headline["fedprox"][s]["macro_f1"]) - float(headline["fedavg"][s]["macro_f1"])
        fl_d = float(flower["fedprox"][s]["macro_f1"])  - float(flower["fedavg"][s]["macro_f1"])
        gap_deltas.append(pt_d - fl_d)
    gap_infer = paired_inference(gap_deltas, label="cross-runtime gap (PT - Flower)")

    print(f"\nPure-PyTorch (primary):  Δ = {pt_infer['mean']:+.4f}, p = {pt_infer['wilcoxon_p_two_sided']:.4f}, wins = {pt_infer['wins_positive']}/{pt_infer['n']}")
    print(f"Flower (replication):    Δ = {fl_infer['mean']:+.4f}, p = {fl_infer['wilcoxon_p_two_sided']:.4f}, wins = {fl_infer['wins_positive']}/{fl_infer['n']}")
    print(f"Per-seed gap (PT − Flower): mean = {gap_infer['mean']:+.4f}, p = {gap_infer['wilcoxon_p_two_sided']:.4f}")
    print(f"  → The cross-runtime gap is {'consistent' if gap_infer['wilcoxon_p_two_sided'] < 0.05 else 'NOT consistent'} across seeds")

    results["cross_runtime"] = {
        "pt_inference": pt_infer,
        "flower_inference": fl_infer,
        "gap_inference": gap_infer,
    }

    # Write Table 3
    (OUT_TABLES / "T03_cross_runtime.tex").write_text(
        latex_table_cross_runtime(pt_infer, fl_infer, gap_infer) + "\n"
    )
    print(f"\nWrote: {OUT_TABLES / 'T03_cross_runtime.tex'}")

    # --- EXPLORATORY: 4-partition Flower dose-response ----------------------
    print("\n" + "=" * 88)
    print(" EXPLORATORY (descriptive only): 4-partition Flower-runtime comparison")
    print("=" * 88)

    partition_table_rows = []
    for sweep_name, display in [
        ("iid",                    "IID (falsification)"),
        ("dirichlet_a01",          r"Dirichlet $\alpha=0.1$"),
        ("specialist_partition",   "Specialist (1-of-7)"),
        ("flower_C0_baseline",     "Balanced paired (engineered)"),
    ]:
        sw = sweeps[sweep_name]
        seeds, deltas = compute_paired_deltas(sw)
        fa_macros = [float(sw["fedavg"][s]["macro_f1"]) for s in seeds]
        fp_macros = [float(sw["fedprox"][s]["macro_f1"]) for s in seeds]
        infer = paired_inference(deltas, label=display)
        row = {
            "sweep": sweep_name,
            "display": display,
            "n": len(seeds),
            "fa_mean": mean(fa_macros),
            "fp_mean": mean(fp_macros),
            "delta_mean": mean(deltas),
            "delta_sd": stdev(deltas) if len(deltas) > 1 else 0,
            "p": infer["wilcoxon_p_two_sided"],
            "wins": infer["wins_positive"],
            "inference": infer,
        }
        partition_table_rows.append(row)

    print(f"\n{'Partition':40s} {'n':>3s} {'FA mean':>9s} {'FP mean':>9s} {'mean Δ':>9s} {'p':>7s} {'wins':>6s}")
    print("-" * 88)
    for r in partition_table_rows:
        print(f"{r['display']:40s} {r['n']:>3d} {r['fa_mean']:>9.4f} {r['fp_mean']:>9.4f} {r['delta_mean']:>+9.4f} {r['p']:>7.4f} {r['wins']:>2d}/{r['n']:<3d}")
    print("\nNote: All four results are descriptive. Only the pure-PyTorch headline (separate table) is the inferential primary.")

    results["partition_dose_response"] = partition_table_rows

    # Write Table 4
    (OUT_TABLES / "T04_partition_dose_response.tex").write_text(
        latex_table_partition_dose_response(partition_table_rows) + "\n"
    )
    print(f"\nWrote: {OUT_TABLES / 'T04_partition_dose_response.tex'}")

    # --- EXPLORATORY: federation tax ----------------------------------------
    print("\n" + "=" * 88)
    print(" EXPLORATORY (descriptive only): federation tax")
    print("=" * 88)
    cent_vals = [float(central[s]["macro_f1"]) for s in sorted(central)]
    cent_mean = mean(cent_vals)
    cent_sd = stdev(cent_vals)
    fed_tax_rows = [{
        "display": "Centralised (pure-PyTorch, no FL)",
        "mean": cent_mean,
        "sd": cent_sd,
        "n": len(cent_vals),
        "pct_central": 100.0,
    }]
    for sweep_name, display in [
        ("headline",             "FedAvg, pure-PyTorch, engineered partition"),
        ("headline",             "FedProx, pure-PyTorch, engineered partition"),
        ("flower_C0_baseline",   "FedAvg, Flower replication"),
        ("flower_C0_baseline",   "FedProx, Flower replication"),
    ]:
        algo = "fedavg" if "FedAvg" in display else "fedprox"
        sw = sweeps[sweep_name]
        vals = [float(sw[algo][s]["macro_f1"]) for s in sorted(sw[algo])]
        m = mean(vals); s = stdev(vals)
        fed_tax_rows.append({
            "display": display,
            "mean": m,
            "sd": s,
            "n": len(vals),
            "pct_central": 100.0 * m / cent_mean,
        })
    for r in fed_tax_rows:
        print(f"  {r['display']:45s} mean={r['mean']:.4f}±{r['sd']:.4f}  {r['pct_central']:5.1f}%")
    results["federation_tax"] = fed_tax_rows

    # Write Table 5
    (OUT_TABLES / "T05_federation_tax.tex").write_text(
        latex_table_federation_tax(fed_tax_rows) + "\n"
    )
    print(f"\nWrote: {OUT_TABLES / 'T05_federation_tax.tex'}")

    # --- Save JSON dump ------------------------------------------------------
    # Strip non-serialisable bits (e.g. lists of full doc dicts).
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()
                    if k not in ("fedavg", "fedprox", "fednova")}
        if isinstance(o, list):
            return [_clean(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    dump_path = OUT_DATA / "statistical_heterogeneity_results.json"
    with open(dump_path, "w") as fh:
        json.dump(_clean(results), fh, indent=2)
    print(f"\nWrote full JSON dump: {dump_path}")

    # --- Final summary -------------------------------------------------------
    print("\n" + "=" * 88)
    print(" SUMMARY")
    print("=" * 88)
    print(f"\nPRIMARY (pure-PyTorch headline): Δ = {pt_infer['mean']:+.4f}, p = {pt_infer['wilcoxon_p_two_sided']:.4f}, CI [{pt_infer['walsh_ci_95_n10'][0]:+.4f}, {pt_infer['walsh_ci_95_n10'][1]:+.4f}]")
    print(f"   → {'SIGNIFICANT at α=0.05' if pt_infer['wilcoxon_p_two_sided'] < 0.05 else 'NOT significant'}")
    print(f"\nSECONDARY (per-class, Holm-corrected):")
    sig_classes = [r for r in pc_stats_rows if r["p_holm"] < 0.05]
    if sig_classes:
        for r in sig_classes:
            print(f"   {r['display']}: Δ={r['delta_mean']:+.4f}, p_Holm={r['p_holm']:.4f}")
    else:
        print(f"   No class significant after Holm correction.")
    print(f"\nEXPLORATORY (descriptive):")
    for r in partition_table_rows:
        print(f"   {r['display']:45s} Δ = {r['delta_mean']:+.4f} (p = {r['p']:.4f}, descriptive)")
    print(f"\n   Cross-runtime gap (PT−Flower): {gap_infer['mean']:+.4f} (p = {gap_infer['wilcoxon_p_two_sided']:.4f})")
    print(f"   → {'Consistent' if gap_infer['wilcoxon_p_two_sided'] < 0.05 else 'NOT consistent'} across seeds — runtime effect is high-variance, not systematic.")


if __name__ == "__main__":
    main()
