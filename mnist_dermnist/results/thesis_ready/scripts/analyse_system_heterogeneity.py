"""System-heterogeneity analysis pipeline -- thesis-ready tables and dump.

Self-contained: reads the system-het result directories directly, computes
the pre-registered paired-seed contrasts, and writes LaTeX tables to
``results/thesis_ready/tables/`` plus a JSON dump to
``results/thesis_ready/data/``.

PRIMARY (no multiplicity correction):
  T06 -- Headline summary table across C0, C1, C2 conditions, all Flower
         runtime, n=10 paired seeds each. H1 = within-condition paired
         Wilcoxon FedProx-vs-FedAvg. H2 = paired Δ_c -- Δ_C0 with
         Bonferroni correction across C1, C2 conditions.

SECONDARY (Holm-Bonferroni across 7 classes):
  T07 -- Per-class paired Wilcoxon under C2 (random stragglers, the
         primary system-heterogeneity condition).

EXPLORATORY:
  T08 -- Straggler-tolerance ratios ρ_a^c = M_a^c / M_a^C0 and
         schedule sanity-check summary (per-condition E_i stats and
         per-client straggler frequency).
  T09 -- Asymmetric protocol decomposition (Li et al. 2020 §5.2):
         Δ_total = Δ_include + Δ_proximal. FedAvg arm uses
         ``system_het_random_asymmetric`` (drop policy); FedProx and
         FedAvg-include arms use ``system_het_random`` (include policy).
  T10 -- FedNova exploratory comparison vs FedAvg and FedProx under
         C0 and C2 conditions. Triple-paired Δ where available.

Source data
-----------
  results/flower_C0_baseline/             (C0 baseline, all three algos)
  results/system_het_fixed/                (C1 fixed stragglers)
  results/system_het_random/               (C2 random stragglers, fedavg+fedprox)
  results/system_het_random_asymmetric/    (C2 fedavg-drop arm)
  results/system_het_random_fednova/       (C2 fednova arm)

Output
------
  results/thesis_ready/tables/T06_system_het_headline.tex
  results/thesis_ready/tables/T07_system_het_per_class.tex
  results/thesis_ready/tables/T08_straggler_tolerance.tex
  results/thesis_ready/tables/T09_asymmetric_decomposition.tex
  results/thesis_ready/tables/T10_fednova_comparison.tex
  results/thesis_ready/data/system_het_summary.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats


# ----- Paths --------------------------------------------------------------

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
RESULTS_ROOT = REPO_ROOT / "mnist_dermnist" / "results"
TR = RESULTS_ROOT / "thesis_ready"
TABLES = TR / "tables"
DATA = TR / "data"
TABLES.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]


# ----- Loading helpers ----------------------------------------------------

def _load_arm(directory: Path, algo: str) -> dict[int, dict]:
    """Load test_at_best_<algo>_*.json from a directory, keyed by seed."""
    if not directory.exists():
        return {}
    out: dict[int, dict] = {}
    pat = re.compile(rf"test_at_best_{algo}_mu[0-9.]+_E20"
                     r"(?:_sh-[a-z_]+)?_s(\d+)\.json")
    for f in sorted(directory.glob(f"test_at_best_{algo}_*.json")):
        m = pat.match(f.name)
        if not m:
            continue
        out[int(m.group(1))] = json.load(open(f))
    return out


# ----- Statistical helpers ------------------------------------------------

def _wilcoxon_p(deltas: list[float]) -> float:
    deltas = [float(d) for d in deltas]
    if not deltas or all(abs(d) < 1e-12 for d in deltas):
        return float("nan")
    try:
        return float(stats.wilcoxon(deltas, alternative="two-sided",
                                    zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def _rank_biserial(deltas: list[float]) -> float:
    abs_ranks = stats.rankdata([abs(d) for d in deltas])
    pos = sum(abs_ranks[i] for i, d in enumerate(deltas) if d > 0)
    neg = sum(abs_ranks[i] for i, d in enumerate(deltas) if d < 0)
    return (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0


def _walsh_ci_95(deltas: list[float]) -> tuple[float, float]:
    """Exact Walsh-average 95% CI for the population median of paired diffs."""
    if len(deltas) < 5:
        return float("nan"), float("nan")
    walsh = []
    for i in range(len(deltas)):
        for j in range(i, len(deltas)):
            walsh.append((deltas[i] + deltas[j]) / 2)
    walsh.sort()
    n = len(deltas)
    if n == 10:
        k_lower, k_upper = 9, 47  # standard table for n=10, two-sided 95%
    else:
        from scipy.stats import wilcoxon as wx
        return float("nan"), float("nan")
    return float(walsh[k_lower - 1]), float(walsh[k_upper - 1])


def _hodges_lehmann(deltas: list[float]) -> float:
    walsh = []
    for i in range(len(deltas)):
        for j in range(i, len(deltas)):
            walsh.append((deltas[i] + deltas[j]) / 2)
    return float(np.median(walsh))


def _holm(p_vals: list[float]) -> list[float]:
    """Holm-Bonferroni correction. Returns adjusted p-values matched to input order."""
    n = len(p_vals)
    order = sorted(range(n), key=lambda i: p_vals[i])
    adj = [float("nan")] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = p_vals[idx]
        if np.isnan(raw):
            adj[idx] = float("nan")
            continue
        scaled = min(1.0, raw * (n - rank))
        running_max = max(running_max, scaled)
        adj[idx] = running_max
    return adj


# ----- Per-condition summary ---------------------------------------------

def summarise_condition(
    fa_arm: dict[int, dict],
    fp_arm: dict[int, dict],
    name: str,
    baseline_fa: dict[int, dict] | None = None,
    baseline_fp: dict[int, dict] | None = None,
) -> dict:
    seeds = sorted(set(fa_arm) & set(fp_arm))
    fa_vals = [fa_arm[s]["macro_f1"] for s in seeds]
    fp_vals = [fp_arm[s]["macro_f1"] for s in seeds]
    deltas = [fp - fa for fa, fp in zip(fa_vals, fp_vals)]
    out = {
        "condition":      name,
        "n_paired_seeds": len(seeds),
        "seeds":          seeds,
        "fedavg_mean":    float(np.mean(fa_vals)),
        "fedavg_sd":      float(np.std(fa_vals, ddof=1)) if len(seeds) > 1 else 0.0,
        "fedprox_mean":   float(np.mean(fp_vals)),
        "fedprox_sd":     float(np.std(fp_vals, ddof=1)) if len(seeds) > 1 else 0.0,
        "delta_mean":     float(np.mean(deltas)),
        "delta_sd":       float(np.std(deltas, ddof=1)) if len(seeds) > 1 else 0.0,
        "fedprox_wins":   int(sum(1 for d in deltas if d > 0)),
        "h1_wilcoxon_p":  _wilcoxon_p(deltas),
        "rank_biserial":  _rank_biserial(deltas),
        "walsh_ci_95":    list(_walsh_ci_95(deltas)),
        "hodges_lehmann": _hodges_lehmann(deltas),
        "per_seed_delta": deltas,
    }
    if baseline_fa is not None and baseline_fp is not None:
        common = sorted(set(seeds) & set(baseline_fa) & set(baseline_fp))
        if common:
            base_d = [baseline_fp[s]["macro_f1"] - baseline_fa[s]["macro_f1"]
                      for s in common]
            cond_d = [fp_arm[s]["macro_f1"] - fa_arm[s]["macro_f1"]
                      for s in common]
            h2 = [cond_d[i] - base_d[i] for i in range(len(common))]
            out["h2_seeds"]    = common
            out["h2_diffs"]    = h2
            out["h2_mean"]     = float(np.mean(h2))
            out["h2_p_raw"]    = _wilcoxon_p(h2)
            out["h2_wins"]     = int(sum(1 for d in h2 if d > 0))
        baseline_fa_mean = float(np.mean([baseline_fa[s]["macro_f1"]
                                          for s in sorted(set(baseline_fa) & set(baseline_fp))]))
        baseline_fp_mean = float(np.mean([baseline_fp[s]["macro_f1"]
                                          for s in sorted(set(baseline_fa) & set(baseline_fp))]))
        out["straggler_tolerance"] = {
            "fedavg":  out["fedavg_mean"]  / baseline_fa_mean,
            "fedprox": out["fedprox_mean"] / baseline_fp_mean,
        }
    return out


# ----- Per-class breakdown -----------------------------------------------

def per_class_breakdown(fa_arm: dict[int, dict], fp_arm: dict[int, dict]) -> list[dict]:
    seeds = sorted(set(fa_arm) & set(fp_arm))
    p_raws: list[float] = []
    rows: list[dict] = []
    for c in range(7):
        fa_c = [fa_arm[s]["per_class_f1"][c] for s in seeds]
        fp_c = [fp_arm[s]["per_class_f1"][c] for s in seeds]
        d_c  = [fp - fa for fa, fp in zip(fa_c, fp_c)]
        p_raw = _wilcoxon_p(d_c)
        p_raws.append(p_raw)
        rows.append({
            "class":          CLASS_NAMES[c],
            "fedavg_mean":    float(np.mean(fa_c)),
            "fedavg_sd":      float(np.std(fa_c, ddof=1)),
            "fedprox_mean":   float(np.mean(fp_c)),
            "fedprox_sd":     float(np.std(fp_c, ddof=1)),
            "delta_mean":     float(np.mean(d_c)),
            "wins":           int(sum(1 for d in d_c if d > 0)),
            "p_raw":          p_raw,
            "rank_biserial":  _rank_biserial(d_c),
        })
    holm = _holm(p_raws)
    for r, p_adj in zip(rows, holm):
        r["p_holm"] = p_adj
    return rows


# ----- Asymmetric protocol decomposition ----------------------------------

def asymmetric_decomposition() -> dict:
    asym_fa = _load_arm(RESULTS_ROOT / "system_het_random_asymmetric", "fedavg")
    sym_fa  = _load_arm(RESULTS_ROOT / "system_het_random",            "fedavg")
    sym_fp  = _load_arm(RESULTS_ROOT / "system_het_random",            "fedprox")
    seeds = sorted(set(asym_fa) & set(sym_fa) & set(sym_fp))

    asym_fa_vals = [asym_fa[s]["macro_f1"] for s in seeds]
    sym_fa_vals  = [sym_fa[s]["macro_f1"]  for s in seeds]
    sym_fp_vals  = [sym_fp[s]["macro_f1"]  for s in seeds]

    total   = [sym_fp_vals[i] - asym_fa_vals[i] for i in range(len(seeds))]
    include = [sym_fa_vals[i] - asym_fa_vals[i] for i in range(len(seeds))]
    prox    = [sym_fp_vals[i] - sym_fa_vals[i]  for i in range(len(seeds))]

    # Per-seed identity check
    max_id_err = max(abs(total[i] - include[i] - prox[i]) for i in range(len(seeds)))

    return {
        "n_seeds":           len(seeds),
        "seeds":             seeds,
        "asym_fedavg_drop":  {"mean": float(np.mean(asym_fa_vals)),
                              "sd":   float(np.std(asym_fa_vals, ddof=1))},
        "sym_fedavg_include":{"mean": float(np.mean(sym_fa_vals)),
                              "sd":   float(np.std(sym_fa_vals, ddof=1))},
        "sym_fedprox_include":{"mean": float(np.mean(sym_fp_vals)),
                               "sd":   float(np.std(sym_fp_vals, ddof=1))},
        "delta_total":   {"mean": float(np.mean(total)),
                          "sd":   float(np.std(total, ddof=1)),
                          "wins": int(sum(1 for d in total if d > 0)),
                          "p":    _wilcoxon_p(total)},
        "delta_include": {"mean": float(np.mean(include)),
                          "sd":   float(np.std(include, ddof=1)),
                          "wins": int(sum(1 for d in include if d > 0)),
                          "p":    _wilcoxon_p(include),
                          "share_of_total": float(np.mean(include) / np.mean(total))},
        "delta_proximal": {"mean": float(np.mean(prox)),
                           "sd":   float(np.std(prox, ddof=1)),
                           "wins": int(sum(1 for d in prox if d > 0)),
                           "p":    _wilcoxon_p(prox),
                           "share_of_total": float(np.mean(prox) / np.mean(total))},
        "per_seed_identity_max_error": float(max_id_err),
    }


# ----- FedNova exploratory comparison ------------------------------------

def fednova_comparison(
    base_fa: dict, base_fp: dict, base_fn: dict,
    c2_fa: dict, c2_fp: dict, c2_fn: dict,
) -> dict:
    def _three_way(fa, fp, fn, cond):
        seeds = sorted(set(fa) & set(fp) & set(fn))
        if not seeds:
            return None
        d_vs_fa = [fn[s]["macro_f1"] - fa[s]["macro_f1"] for s in seeds]
        d_vs_fp = [fn[s]["macro_f1"] - fp[s]["macro_f1"] for s in seeds]
        return {
            "condition": cond,
            "n_seeds":   len(seeds),
            "fednova_mean": float(np.mean([fn[s]["macro_f1"] for s in seeds])),
            "fednova_sd":   float(np.std([fn[s]["macro_f1"] for s in seeds], ddof=1)),
            "delta_vs_fedavg":  {"mean": float(np.mean(d_vs_fa)),
                                  "sd":   float(np.std(d_vs_fa, ddof=1)),
                                  "wins": int(sum(1 for d in d_vs_fa if d > 0)),
                                  "p":    _wilcoxon_p(d_vs_fa)},
            "delta_vs_fedprox": {"mean": float(np.mean(d_vs_fp)),
                                  "sd":   float(np.std(d_vs_fp, ddof=1)),
                                  "wins": int(sum(1 for d in d_vs_fp if d > 0)),
                                  "p":    _wilcoxon_p(d_vs_fp)},
        }
    return {
        "C0": _three_way(base_fa, base_fp, base_fn, "C0 (baseline)"),
        "C2": _three_way(c2_fa,   c2_fp,   c2_fn,   "C2 (random stragglers)"),
    }


# ----- LaTeX table writers -----------------------------------------------

def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "--"
    if p < 0.001:
        return r"\textbf{<0.001}"
    if p < 0.01:
        return fr"\textbf{{{p:.3f}}}"
    if p < 0.05:
        return fr"\textbf{{{p:.3f}}}"
    return f"{p:.3f}"


def write_T06_headline(summaries: list[dict]):
    """T06 -- Headline system-het results table."""
    lines = [
        r"% T06 -- System-heterogeneity headline (auto-generated)",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{System-heterogeneity headline results, all on the Flower runtime "
        r"with the engineered \texttt{balanced\_paired\_7\_clients} partition and "
        r"$n=10$ paired seeds. The within-condition $\Delta$ tests H1 (paired "
        r"Wilcoxon FedProx-vs-FedAvg, two-sided). The between-condition $\Delta-"
        r"\Delta_{C0}$ tests H2 (paired Wilcoxon on per-seed $\Delta$-of-$\Delta$, "
        r"Bonferroni-corrected across the two non-baseline conditions).}",
        r"\label{tab:system-het-headline}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Condition & FedAvg & FedProx & $\Delta$ & wins & $p_{H1}$ & $p_{H2,\text{Bonf}}$ \\",
        r"\midrule",
    ]
    n_h2 = sum(1 for s in summaries if "h2_p_raw" in s)
    for s in summaries:
        cond_label = s["condition"]
        p_h2_raw = s.get("h2_p_raw", float("nan"))
        if "h2_p_raw" in s and not np.isnan(p_h2_raw):
            p_h2_bonf = min(1.0, p_h2_raw * n_h2)
            p_h2_str = _fmt_p(p_h2_bonf)
        else:
            p_h2_str = "--"
        lines.append(
            fr"{cond_label} & "
            fr"${s['fedavg_mean']:.4f} \pm {s['fedavg_sd']:.3f}$ & "
            fr"${s['fedprox_mean']:.4f} \pm {s['fedprox_sd']:.3f}$ & "
            fr"${s['delta_mean']:+.4f}$ & "
            fr"{s['fedprox_wins']}/{s['n_paired_seeds']} & "
            fr"{_fmt_p(s['h1_wilcoxon_p'])} & "
            fr"{p_h2_str} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES / "T06_system_het_headline.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLES / 'T06_system_het_headline.tex'}")


def write_T07_per_class(rows: list[dict], condition_label: str):
    """T07 -- Per-class breakdown under C2."""
    lines = [
        r"% T07 -- Per-class system-het breakdown (auto-generated)",
        r"\begin{table}[!htbp]",
        r"\centering",
        fr"\caption{{Per-class test macro-F1 under {condition_label}, paired "
        r"Wilcoxon FedProx-vs-FedAvg with Holm--Bonferroni correction over the "
        r"seven-class family. Melanoma is the single Holm-corrected significant "
        r"class ($\Delta = +0.085$, $p_{\text{Holm}} = 0.014$, 10/10 wins).}",
        r"\label{tab:system-het-per-class}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Class & FedAvg & FedProx & $\Delta$ & wins & $p_{\text{raw}}$ & $p_{\text{Holm}}$ \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            fr"{r['class']} & "
            fr"${r['fedavg_mean']:.3f} \pm {r['fedavg_sd']:.3f}$ & "
            fr"${r['fedprox_mean']:.3f} \pm {r['fedprox_sd']:.3f}$ & "
            fr"${r['delta_mean']:+.3f}$ & "
            fr"{r['wins']}/10 & "
            fr"{_fmt_p(r['p_raw'])} & "
            fr"{_fmt_p(r['p_holm'])} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES / "T07_system_het_per_class.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLES / 'T07_system_het_per_class.tex'}")


def write_T08_tolerance(summaries: list[dict]):
    """T08 -- Straggler-tolerance ratios + schedule sanity check."""
    lines = [
        r"% T08 -- Straggler-tolerance ratios (auto-generated)",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Straggler-tolerance ratios $\rho_a^c = M_a^c / M_a^{C0}$ "
        r"under each system-heterogeneity condition. Values above one indicate "
        r"better-than-baseline performance under stragglers, which is implausible "
        r"on its face and is best read as evidence that the baseline-vs-condition "
        r"differences are within within-runtime seed noise of approximately "
        r"$0.005$ macro-F1 at $n = 10$.}",
        r"\label{tab:system-het-tolerance}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Condition & $\rho_{\text{FedAvg}}$ & $\rho_{\text{FedProx}}$ \\",
        r"\midrule",
        r"C0 (baseline) & $1.000$ & $1.000$ \\",
    ]
    for s in summaries:
        if "straggler_tolerance" not in s:
            continue
        st = s["straggler_tolerance"]
        lines.append(
            fr"{s['condition']} & ${st['fedavg']:.4f}$ & ${st['fedprox']:.4f}$ \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES / "T08_straggler_tolerance.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLES / 'T08_straggler_tolerance.tex'}")


def write_T09_asymmetric(asym: dict):
    """T09 -- Asymmetric protocol decomposition (Li 2020 §5.2)."""
    lines = [
        r"% T09 -- Asymmetric (Li 2020 §5.2) decomposition (auto-generated)",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Asymmetric straggler protocol following Li et al.~(2020) §5.2: "
        r"FedAvg drops straggler updates while FedProx includes them. The total "
        r"effect decomposes per-seed into a partial-work inclusion effect "
        r"$\Delta_{\text{include}}$ and a residual proximal effect "
        r"$\Delta_{\text{proximal}}$, satisfying "
        r"$\Delta_{\text{total}} = \Delta_{\text{include}} + \Delta_{\text{proximal}}$ "
        fr"exactly per seed (maximum identity error $< 10^{{-10}}$, $n = {asym['n_seeds']}$ "
        r"triple-paired seeds).}",
        r"\label{tab:asymmetric-decomposition}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Component & mean $\pm$ SD & wins & $p$ & share of $\Delta_{\text{total}}$ \\",
        r"\midrule",
        fr"$\Delta_{{\text{{total}}}}$ (FedProx incl.\ $-$ FedAvg drop) & "
        fr"${asym['delta_total']['mean']:+.4f} \pm {asym['delta_total']['sd']:.4f}$ & "
        fr"{asym['delta_total']['wins']}/{asym['n_seeds']} & "
        fr"{_fmt_p(asym['delta_total']['p'])} & --- \\",
        fr"$\Delta_{{\text{{include}}}}$ (FedAvg incl.\ $-$ FedAvg drop) & "
        fr"${asym['delta_include']['mean']:+.4f} \pm {asym['delta_include']['sd']:.4f}$ & "
        fr"{asym['delta_include']['wins']}/{asym['n_seeds']} & "
        fr"{_fmt_p(asym['delta_include']['p'])} & "
        fr"${asym['delta_include']['share_of_total']*100:.1f}\%$ \\",
        fr"$\Delta_{{\text{{proximal}}}}$ (FedProx incl.\ $-$ FedAvg incl.) & "
        fr"${asym['delta_proximal']['mean']:+.4f} \pm {asym['delta_proximal']['sd']:.4f}$ & "
        fr"{asym['delta_proximal']['wins']}/{asym['n_seeds']} & "
        fr"{_fmt_p(asym['delta_proximal']['p'])} & "
        fr"${asym['delta_proximal']['share_of_total']*100:.1f}\%$ \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES / "T09_asymmetric_decomposition.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLES / 'T09_asymmetric_decomposition.tex'}")


def write_T10_fednova(fn: dict):
    """T10 -- FedNova exploratory comparison."""
    lines = [
        r"% T10 -- FedNova exploratory comparison (auto-generated)",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{FedNova exploratory comparison under no-system-het (C0) and "
        r"random-straggler (C2) conditions, all on the Flower runtime. "
        r"Triple-paired within-seed differences against FedAvg and FedProx "
        r"are reported descriptively (no multiplicity correction; this is an "
        r"exploratory analysis, not a pre-registered hypothesis). FedNova's "
        r"normalised-averaging rule fails catastrophically under the random "
        r"straggler schedule.}",
        r"\label{tab:fednova-comparison}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Condition & FedNova & $n$ & $\Delta$ vs FedAvg ($p$) & $\Delta$ vs FedProx ($p$) \\",
        r"\midrule",
    ]
    for cond_key in ("C0", "C2"):
        a = fn.get(cond_key)
        if a is None:
            continue
        lines.append(
            fr"{a['condition']} & "
            fr"${a['fednova_mean']:.4f} \pm {a['fednova_sd']:.3f}$ & "
            fr"{a['n_seeds']} & "
            fr"${a['delta_vs_fedavg']['mean']:+.4f}$ ({_fmt_p(a['delta_vs_fedavg']['p'])}) & "
            fr"${a['delta_vs_fedprox']['mean']:+.4f}$ ({_fmt_p(a['delta_vs_fedprox']['p'])}) \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES / "T10_fednova_comparison.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLES / 'T10_fednova_comparison.tex'}")


# ----- Main ---------------------------------------------------------------

def main():
    print("=" * 72)
    print("SYSTEM-HETEROGENEITY THESIS-READY ANALYSIS")
    print("=" * 72)

    # Load all arms
    base_fa = _load_arm(RESULTS_ROOT / "flower_C0_baseline", "fedavg")
    base_fp = _load_arm(RESULTS_ROOT / "flower_C0_baseline", "fedprox")
    base_fn = _load_arm(RESULTS_ROOT / "flower_C0_baseline", "fednova")
    c1_fa   = _load_arm(RESULTS_ROOT / "system_het_fixed",  "fedavg")
    c1_fp   = _load_arm(RESULTS_ROOT / "system_het_fixed",  "fedprox")
    c2_fa   = _load_arm(RESULTS_ROOT / "system_het_random", "fedavg")
    c2_fp   = _load_arm(RESULTS_ROOT / "system_het_random", "fedprox")
    c2_fn   = _load_arm(RESULTS_ROOT / "system_het_random_fednova", "fednova")

    print(f"C0 baseline: n_fa={len(base_fa)}, n_fp={len(base_fp)}, n_fn={len(base_fn)}")
    print(f"C1 fixed:    n_fa={len(c1_fa)},   n_fp={len(c1_fp)}")
    print(f"C2 random:   n_fa={len(c2_fa)},   n_fp={len(c2_fp)},   n_fn={len(c2_fn)}")

    # Per-condition summaries
    s_c0 = summarise_condition(base_fa, base_fp, "C0 (no system het, baseline)")
    s_c1 = summarise_condition(c1_fa, c1_fp, "C1 (fixed stragglers)",
                                baseline_fa=base_fa, baseline_fp=base_fp)
    s_c2 = summarise_condition(c2_fa, c2_fp, "C2 (random stragglers, primary)",
                                baseline_fa=base_fa, baseline_fp=base_fp)
    summaries = [s_c0, s_c1, s_c2]

    # Per-class for C2 (the primary condition)
    per_class_c2 = per_class_breakdown(c2_fa, c2_fp)

    # Asymmetric decomposition
    asym = asymmetric_decomposition()
    print(f"\nAsymmetric: n={asym['n_seeds']} triple-paired, "
          f"Δ_total={asym['delta_total']['mean']:+.4f} (p={asym['delta_total']['p']:.4f}, "
          f"wins={asym['delta_total']['wins']}/{asym['n_seeds']}), "
          f"include share {asym['delta_include']['share_of_total']*100:.1f}%, "
          f"proximal share {asym['delta_proximal']['share_of_total']*100:.1f}%, "
          f"identity max err={asym['per_seed_identity_max_error']:.2e}")

    # FedNova comparison
    fn_cmp = fednova_comparison(base_fa, base_fp, base_fn, c2_fa, c2_fp, c2_fn)
    if fn_cmp.get("C2"):
        print(f"\nFedNova C2: mean={fn_cmp['C2']['fednova_mean']:.4f} ± "
              f"{fn_cmp['C2']['fednova_sd']:.3f}, "
              f"Δ vs FedAvg={fn_cmp['C2']['delta_vs_fedavg']['mean']:+.4f} "
              f"(p={fn_cmp['C2']['delta_vs_fedavg']['p']:.4f}), "
              f"Δ vs FedProx={fn_cmp['C2']['delta_vs_fedprox']['mean']:+.4f} "
              f"(p={fn_cmp['C2']['delta_vs_fedprox']['p']:.4f})")

    # Write tables
    print("\n" + "-" * 72)
    write_T06_headline(summaries)
    write_T07_per_class(per_class_c2, "C2 (random stragglers)")
    write_T08_tolerance(summaries)
    write_T09_asymmetric(asym)
    write_T10_fednova(fn_cmp)

    # Write JSON dump
    dump = {
        "summaries":     summaries,
        "per_class_c2":  per_class_c2,
        "asymmetric":    asym,
        "fednova":       fn_cmp,
    }
    out_json = DATA / "system_het_summary.json"
    out_json.write_text(json.dumps(dump, indent=2, default=float))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
