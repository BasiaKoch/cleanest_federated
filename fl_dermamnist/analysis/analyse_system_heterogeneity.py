"""System-heterogeneity analysis pipeline -- thesis-ready tables and dump.

Scope (after 2026-05-24 simplification):
  - FedAvg vs FedProx only (FedNova excluded from thesis scope)
  - C0 / C1 / C2 on BOTH the engineered (balanced_paired) and IID partitions
  - Asymmetric Li 2020 §5.2 replication reports only Δ_total (no decomposition)

Outputs (field-standard descriptive reporting):
  T06 -- Headline system-het summary (engineered partition, C0+C1+C2)
  T07 -- Per-class breakdown under C2 engineered partition
  T08 -- Straggler-tolerance ratios (engineered partition)
  T09 -- Li 2020 §5.2 asymmetric protocol replication (Δ_total only)
  T12 -- System-het summary on IID partition (C0+C1+C2)
  T13 -- Cross-partition summary (engineered vs IID side-by-side)
  data/system_het_summary.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
RESULTS_ROOT = REPO_ROOT / "fl_dermamnist" / "results"
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
CLASS_PREV = [3.27, 5.13, 10.97, 1.15, 11.11, 67.05, 1.41]


def _load_arm(directory: Path, algo: str) -> dict[int, dict]:
    if not directory.exists():
        return {}
    out: dict[int, dict] = {}
    # Filename pattern accommodates optional sh-mode and C-fraction suffixes:
    #   test_at_best_{algo}_mu{X}_E20[_sh-{mode}][_C{frac}]_s{seed}.json
    pat = re.compile(rf"test_at_best_{algo}_mu[0-9.]+_E20"
                     r"(?:_sh-[a-z_]+)?(?:_C[0-9.]+)?_s(\d+)\.json")
    for f in sorted(directory.glob(f"test_at_best_{algo}_*.json")):
        m = pat.match(f.name)
        if not m:
            continue
        out[int(m.group(1))] = json.load(open(f))
    return out


def mean_sd(vals: list[float]) -> tuple[float, float]:
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def bold(s: str) -> str:
    return r"\textbf{" + s + r"}"


# ----- Per-condition summary ---------------------------------------------

def summarise_condition(
    fa_arm: dict[int, dict],
    fp_arm: dict[int, dict],
    name: str,
) -> dict:
    seeds = sorted(set(fa_arm) & set(fp_arm))
    fa_vals = [fa_arm[s]["macro_f1"] for s in seeds]
    fp_vals = [fp_arm[s]["macro_f1"] for s in seeds]
    deltas = [fp - fa for fa, fp in zip(fa_vals, fp_vals)]
    fa_m, fa_sd = mean_sd(fa_vals)
    fp_m, fp_sd = mean_sd(fp_vals)
    d_m, d_sd   = mean_sd(deltas)
    return {
        "condition":      name,
        "n_paired_seeds": len(seeds),
        "seeds":          seeds,
        "fedavg_mean":    fa_m,
        "fedavg_sd":      fa_sd,
        "fedprox_mean":   fp_m,
        "fedprox_sd":     fp_sd,
        "delta_mean":     d_m,
        "delta_sd":       d_sd,
        "fedprox_wins":   int(sum(1 for d in deltas if d > 0)),
        "per_seed_delta": deltas,
    }


def per_class_breakdown(fa_arm: dict[int, dict], fp_arm: dict[int, dict]) -> list[dict]:
    seeds = sorted(set(fa_arm) & set(fp_arm))
    rows: list[dict] = []
    for c in range(7):
        fa_c = [fa_arm[s]["per_class_f1"][c] for s in seeds]
        fp_c = [fp_arm[s]["per_class_f1"][c] for s in seeds]
        d_c  = [fp - fa for fa, fp in zip(fa_c, fp_c)]
        rows.append({
            "class":          CLASS_NAMES[c],
            "prevalence":     CLASS_PREV[c],
            "fedavg_mean":    float(np.mean(fa_c)),
            "fedavg_sd":      float(np.std(fa_c, ddof=1)),
            "fedprox_mean":   float(np.mean(fp_c)),
            "fedprox_sd":     float(np.std(fp_c, ddof=1)),
            "delta_mean":     float(np.mean(d_c)),
            "wins":           int(sum(1 for d in d_c if d > 0)),
        })
    return rows


# ----- Asymmetric replication (Δ_total only) -----------------------------

def asymmetric_total() -> dict:
    asym_fa = _load_arm(RESULTS_ROOT / "system_het_random_asymmetric", "fedavg")
    sym_fp  = _load_arm(RESULTS_ROOT / "system_het_random",            "fedprox")
    seeds = sorted(set(asym_fa) & set(sym_fp))
    asym_fa_vals = [asym_fa[s]["macro_f1"] for s in seeds]
    sym_fp_vals  = [sym_fp[s]["macro_f1"]  for s in seeds]
    total = [sym_fp_vals[i] - asym_fa_vals[i] for i in range(len(seeds))]
    return {
        "n_seeds":             len(seeds),
        "seeds":               seeds,
        "fedavg_drop_mean":    float(np.mean(asym_fa_vals)),
        "fedavg_drop_sd":      float(np.std(asym_fa_vals, ddof=1)),
        "fedprox_include_mean":float(np.mean(sym_fp_vals)),
        "fedprox_include_sd":  float(np.std(sym_fp_vals,  ddof=1)),
        "delta_total_mean":    float(np.mean(total)),
        "delta_total_sd":      float(np.std(total, ddof=1)),
        "fedprox_wins":        int(sum(1 for d in total if d > 0)),
    }


# ----- LaTeX writers (field-standard, no p-values) -----------------------

def _row_with_bold_winner(label: str, fa_m: float, fa_sd: float,
                          fp_m: float, fp_sd: float, d: float, wins: int, n: int) -> str:
    fa_cell = f"${fa_m:.4f} \\pm {fa_sd:.4f}$"
    fp_cell = f"${fp_m:.4f} \\pm {fp_sd:.4f}$"
    if fp_m > fa_m:
        fp_cell = bold(fp_cell)
    else:
        fa_cell = bold(fa_cell)
    return fr"{label} & {fa_cell} & {fp_cell} & ${d:+.4f}$ & {wins}/{n} \\"


def write_T06(summaries: list[dict]):
    lines = [
        r"% T06 -- System-heterogeneity headline, engineered partition (auto-generated)",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{System-heterogeneity headline results on the engineered "
        r"\texttt{balanced\_paired\_7\_clients} partition under the Flower "
        r"runtime with $n = 10$ paired seeds. Values are test macro-F1 "
        r"mean $\pm$ standard deviation across seeds; the winning algorithm "
        r"per row is shown in bold.}",
        r"\label{tab:system-het-headline}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Condition & FedAvg & FedProx & $\Delta$ & Wins \\",
        r"\midrule",
    ]
    for s in summaries:
        lines.append(_row_with_bold_winner(
            s["condition"], s["fedavg_mean"], s["fedavg_sd"],
            s["fedprox_mean"], s["fedprox_sd"], s["delta_mean"],
            s["fedprox_wins"], s["n_paired_seeds"],
        ))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "T06_system_het_headline.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLES / 'T06_system_het_headline.tex'}")


def write_T07(rows: list[dict]):
    lines = [
        r"% T07 -- Per-class system-het breakdown, engineered C2 (auto-generated)",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Per-class test F1 under C2 (random stragglers) on the "
        r"engineered partition, reported as mean $\pm$ standard deviation "
        r"across $n = 10$ paired seeds. Class prevalence in the training "
        r"set is shown in parentheses. The winning algorithm per row is "
        r"shown in bold; $\Delta$ is the mean within-pair difference "
        r"(FedProx $-$ FedAvg).}",
        r"\label{tab:system-het-per-class}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Class (prevalence) & FedAvg & FedProx & $\Delta$ \\",
        r"\midrule",
    ]
    for r in rows:
        fa_cell = f"${r['fedavg_mean']:.3f} \\pm {r['fedavg_sd']:.3f}$"
        fp_cell = f"${r['fedprox_mean']:.3f} \\pm {r['fedprox_sd']:.3f}$"
        if r["fedprox_mean"] > r["fedavg_mean"]:
            fp_cell = bold(fp_cell)
        else:
            fa_cell = bold(fa_cell)
        lines.append(fr"{r['class']} ({r['prevalence']:.2f}\%) & "
                     fr"{fa_cell} & {fp_cell} & ${r['delta_mean']:+.3f}$ \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "T07_system_het_per_class.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLES / 'T07_system_het_per_class.tex'}")


def write_T08(summaries: list[dict], baseline: dict):
    lines = [
        r"% T08 -- Straggler-tolerance ratios (auto-generated)",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Straggler-tolerance ratios "
        r"$\rho_a^c = M_a^c / M_a^{C0}$ on the engineered partition. Values "
        r"close to one indicate the algorithm preserves its no-system-het "
        r"performance under the straggler schedule. Values slightly above "
        r"one are within the across-seed noise floor of approximately "
        r"$0.005$ macro-F1.}",
        r"\label{tab:system-het-tolerance}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Condition & $\rho_{\text{FedAvg}}$ & $\rho_{\text{FedProx}}$ \\",
        r"\midrule",
        r"C0 (baseline) & $1.000$ & $1.000$ \\",
    ]
    fa_base = baseline["fedavg_mean"]
    fp_base = baseline["fedprox_mean"]
    for s in summaries:
        if "C0" in s["condition"]:
            continue
        rho_fa = s["fedavg_mean"]  / fa_base
        rho_fp = s["fedprox_mean"] / fp_base
        lines.append(fr"{s['condition']} & ${rho_fa:.4f}$ & ${rho_fp:.4f}$ \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "T08_straggler_tolerance.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLES / 'T08_straggler_tolerance.tex'}")


def write_T09(asym: dict):
    """Li 2020 §5.2 asymmetric replication: ONLY Δ_total, no decomposition."""
    fa_cell = f"${asym['fedavg_drop_mean']:.4f} \\pm {asym['fedavg_drop_sd']:.4f}$"
    fp_cell = bold(f"${asym['fedprox_include_mean']:.4f} \\pm {asym['fedprox_include_sd']:.4f}$")
    lines = [
        r"% T09 -- Li 2020 §5.2 asymmetric protocol replication (auto-generated)",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Replication of the asymmetric straggler protocol of "
        r"\citet{li2020fedprox} §5.2, in which FedAvg drops straggler "
        r"updates (its canonical behaviour) while FedProx includes them "
        r"as $\gamma$-inexact partial work. Values are mean $\pm$ standard "
        r"deviation of test macro-F1 across $n = 10$ paired seeds on the "
        r"engineered partition with random stragglers (4 of 7 clients per "
        r"round).}",
        r"\label{tab:asymmetric-replication}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Algorithm (aggregation policy) & Test macro-F1 & $\Delta$ & Wins \\",
        r"\midrule",
        fr"FedAvg (drops stragglers) & {fa_cell} & --- & --- \\",
        fr"FedProx (includes stragglers) & {fp_cell} & "
        fr"${asym['delta_total_mean']:+.4f} \pm {asym['delta_total_sd']:.4f}$ & "
        fr"{asym['fedprox_wins']}/{asym['n_seeds']} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES / "T09_asymmetric_decomposition.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLES / 'T09_asymmetric_decomposition.tex'}")


def write_T12(summaries: list[dict]):
    """IID-partition system-het summary."""
    lines = [
        r"% T12 -- System-het on IID partition (auto-generated)",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{System-heterogeneity results on the IID partition "
        r"(\texttt{iid\_7\_clients}) under the Flower runtime with $n = "
        r"10$ paired seeds. Values are test macro-F1 mean $\pm$ standard "
        r"deviation across seeds; the winning algorithm per row is shown "
        r"in bold. The IID partition serves as a mechanism-null control: "
        r"with no inter-client class skew, the proximal anchor is "
        r"theoretically inert.}",
        r"\label{tab:system-het-iid}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Condition & FedAvg & FedProx & $\Delta$ & Wins \\",
        r"\midrule",
    ]
    for s in summaries:
        lines.append(_row_with_bold_winner(
            s["condition"], s["fedavg_mean"], s["fedavg_sd"],
            s["fedprox_mean"], s["fedprox_sd"], s["delta_mean"],
            s["fedprox_wins"], s["n_paired_seeds"],
        ))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "T12_system_het_iid.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLES / 'T12_system_het_iid.tex'}")


def write_T14(partial_summary: dict, baseline_summary: dict):
    """Partial-participation (C=0.5) summary, paired against the C=1.0
    engineered baseline. Reported descriptively as one new condition."""
    fa_cell = f"${baseline_summary['fedavg_mean']:.4f} \\pm {baseline_summary['fedavg_sd']:.4f}$"
    fp_cell = f"${baseline_summary['fedprox_mean']:.4f} \\pm {baseline_summary['fedprox_sd']:.4f}$"
    if baseline_summary["fedprox_mean"] > baseline_summary["fedavg_mean"]:
        fp_cell = bold(fp_cell)
    else:
        fa_cell = bold(fa_cell)
    pp_fa_cell = f"${partial_summary['fedavg_mean']:.4f} \\pm {partial_summary['fedavg_sd']:.4f}$"
    pp_fp_cell = f"${partial_summary['fedprox_mean']:.4f} \\pm {partial_summary['fedprox_sd']:.4f}$"
    if partial_summary["fedprox_mean"] > partial_summary["fedavg_mean"]:
        pp_fp_cell = bold(pp_fp_cell)
    else:
        pp_fa_cell = bold(pp_fa_cell)
    lines = [
        r"% T14 -- Partial participation (C=0.5) summary (auto-generated)",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Partial-participation result on the engineered "
        r"partition with uniform $E = 20$, contrasted with the full-"
        r"participation engineered C0 baseline. Both rows use the same "
        r"ten paired seeds, identical hyperparameters, and the same "
        r"Flower runtime; the only difference is the per-round client "
        r"participation fraction $C$. The winning algorithm per row is "
        r"shown in bold.}",
        r"\label{tab:partial-participation}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Setting & FedAvg & FedProx & $\Delta$ & Wins \\",
        r"\midrule",
        fr"$C = 1.0$ (full, baseline)   & {fa_cell} & {fp_cell} & "
        fr"${baseline_summary['delta_mean']:+.4f}$ & "
        fr"{baseline_summary['fedprox_wins']}/{baseline_summary['n_paired_seeds']} \\",
        fr"$C = 0.5$ (partial, 4/7 clients per round)   & {pp_fa_cell} & {pp_fp_cell} & "
        fr"${partial_summary['delta_mean']:+.4f}$ & "
        fr"{partial_summary['fedprox_wins']}/{partial_summary['n_paired_seeds']} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES / "T14_partial_participation.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLES / 'T14_partial_participation.tex'}")


def write_T13(eng_summaries: list[dict], iid_summaries: list[dict]):
    """Cross-partition summary: engineered vs IID side-by-side."""
    def fmt_delta(s):
        return f"${s['delta_mean']:+.4f}$"
    def fmt_wins(s):
        return f"{s['fedprox_wins']}/{s['n_paired_seeds']}"
    lines = [
        r"% T13 -- Cross-partition system-het summary (auto-generated)",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Cross-partition comparison of the FedProx-vs-FedAvg "
        r"within-pair difference under each system-heterogeneity condition. "
        r"The engineered partition contains substantial class skew across "
        r"clients (every minority class held by two clients); the IID "
        r"partition contains no inter-client class skew. The proximal "
        r"anchor's mechanism predicts a positive $\Delta$ on the "
        r"engineered partition and $\Delta \approx 0$ on IID.}",
        r"\label{tab:system-het-cross-partition}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Engineered partition} & \multicolumn{2}{c}{IID partition} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Condition & $\Delta$ & Wins & $\Delta$ & Wins \\",
        r"\midrule",
    ]
    cond_labels = ["C0 (no system het)", "C1 (fixed stragglers)", "C2 (random stragglers)"]
    for i, lbl in enumerate(cond_labels):
        lines.append(fr"{lbl} & {fmt_delta(eng_summaries[i])} & {fmt_wins(eng_summaries[i])} & "
                     fr"{fmt_delta(iid_summaries[i])} & {fmt_wins(iid_summaries[i])} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "T13_cross_partition.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLES / 'T13_cross_partition.tex'}")


# ----- Main ---------------------------------------------------------------

def main():
    print("=" * 72)
    print("SYSTEM-HETEROGENEITY ANALYSIS — FedAvg vs FedProx, engineered + IID")
    print("=" * 72)

    # Engineered partition arms
    base_fa = _load_arm(RESULTS_ROOT / "flower_C0_baseline", "fedavg")
    base_fp = _load_arm(RESULTS_ROOT / "flower_C0_baseline", "fedprox")
    c1_fa   = _load_arm(RESULTS_ROOT / "system_het_fixed",  "fedavg")
    c1_fp   = _load_arm(RESULTS_ROOT / "system_het_fixed",  "fedprox")
    c2_fa   = _load_arm(RESULTS_ROOT / "system_het_random", "fedavg")
    c2_fp   = _load_arm(RESULTS_ROOT / "system_het_random", "fedprox")

    # Partial-participation arm (engineered partition, C=0.5)
    pp_fa = _load_arm(RESULTS_ROOT / "system_het_partial_C0.5", "fedavg")
    pp_fp = _load_arm(RESULTS_ROOT / "system_het_partial_C0.5", "fedprox")

    # IID partition arms
    iid_c0_fa = _load_arm(RESULTS_ROOT / "flower_C0_iid_baseline", "fedavg")
    iid_c0_fp = _load_arm(RESULTS_ROOT / "flower_C0_iid_baseline", "fedprox")
    iid_c1_fa = _load_arm(RESULTS_ROOT / "system_het_iid_fixed",   "fedavg")
    iid_c1_fp = _load_arm(RESULTS_ROOT / "system_het_iid_fixed",   "fedprox")
    iid_c2_fa = _load_arm(RESULTS_ROOT / "system_het_iid_random",  "fedavg")
    iid_c2_fp = _load_arm(RESULTS_ROOT / "system_het_iid_random",  "fedprox")

    print(f"Engineered: C0={len(base_fa)}+{len(base_fp)}, "
          f"C1={len(c1_fa)}+{len(c1_fp)}, C2={len(c2_fa)}+{len(c2_fp)}")
    print(f"IID:        C0={len(iid_c0_fa)}+{len(iid_c0_fp)}, "
          f"C1={len(iid_c1_fa)}+{len(iid_c1_fp)}, C2={len(iid_c2_fa)}+{len(iid_c2_fp)}")

    eng_summaries = [
        summarise_condition(base_fa, base_fp, "C0 (no system het)"),
        summarise_condition(c1_fa, c1_fp,     "C1 (fixed stragglers)"),
        summarise_condition(c2_fa, c2_fp,     "C2 (random stragglers)"),
    ]
    iid_summaries = [
        summarise_condition(iid_c0_fa, iid_c0_fp, "C0 (no system het)"),
        summarise_condition(iid_c1_fa, iid_c1_fp, "C1 (fixed stragglers)"),
        summarise_condition(iid_c2_fa, iid_c2_fp, "C2 (random stragglers)"),
    ]

    print("\n--- Engineered partition ---")
    for s in eng_summaries:
        print(f"  {s['condition']:<28}: Δ = {s['delta_mean']:+.4f}, wins = "
              f"{s['fedprox_wins']}/{s['n_paired_seeds']}")
    print("\n--- IID partition ---")
    for s in iid_summaries:
        print(f"  {s['condition']:<28}: Δ = {s['delta_mean']:+.4f}, wins = "
              f"{s['fedprox_wins']}/{s['n_paired_seeds']}")

    per_class_c2 = per_class_breakdown(c2_fa, c2_fp)
    asym = asymmetric_total()

    print(f"\nAsymmetric replication (Li 2020 §5.2): Δ_total = "
          f"{asym['delta_total_mean']:+.4f} ± {asym['delta_total_sd']:.4f}, "
          f"wins = {asym['fedprox_wins']}/{asym['n_seeds']}")

    # Write tables
    print()
    write_T06(eng_summaries)
    write_T07(per_class_c2)
    write_T08(eng_summaries, eng_summaries[0])
    write_T09(asym)
    write_T12(iid_summaries)
    write_T13(eng_summaries, iid_summaries)

    # T14: partial participation (only if the sweep has landed)
    if pp_fa and pp_fp:
        pp_summary = summarise_condition(pp_fa, pp_fp, "C=0.5 partial participation")
        print(f"\n--- Partial participation (engineered, C=0.5) ---")
        print(f"  {pp_summary['condition']:<35}: Δ = {pp_summary['delta_mean']:+.4f}, "
              f"wins = {pp_summary['fedprox_wins']}/{pp_summary['n_paired_seeds']}")
        write_T14(pp_summary, eng_summaries[0])
    else:
        print(f"\n--- Partial participation sweep not yet present "
              f"(submit_partial_participation.sh) ---")

    # Delete T10 (FedNova) if present
    t10 = TABLES / "T10_fednova_comparison.tex"
    if t10.exists():
        t10.unlink()
        print(f"Removed: {t10}")

    # Dump JSON
    dump = {
        "engineered":   eng_summaries,
        "iid":          iid_summaries,
        "per_class_c2": per_class_c2,
        "asymmetric":   asym,
    }
    (DATA / "system_het_summary.json").write_text(json.dumps(dump, indent=2, default=float))
    print(f"Wrote {DATA / 'system_het_summary.json'}")


if __name__ == "__main__":
    main()
