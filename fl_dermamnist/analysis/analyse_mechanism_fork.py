"""Stage-2 mechanism-fork aggregator (FEDNOVA_RANDOM_TAU_THESIS_PLAN.md §8–§9).

Reads the 5 pilot arms' result JSONs and emits the §9 statistics table plus the
§8 decision-rule verdict (magnitude vs direction vs jitter vs deeper).

Per arm it reports: n, mean±std FINAL-round macro-F1, min, mean±std BEST-VAL
macro-F1, collapse rate (#seeds < 0.40) for both, mean per-class F1, and a
paired (same-seed) comparison vs the baseline arm (wins/losses + Wilcoxon).

Arm name -> directory: results/system_het_random_fednova_<arm>/
Each arm dir holds one test_at_final_*_s<seed>.json and test_at_best_*_s<seed>.json
per seed (older pre-instrumentation runs have only test_at_best — handled).

Usage:
    python -m fl_dermamnist.scripts.analyse_mechanism_fork \
        --results-root fl_dermamnist/results \
        --out fl_dermamnist/results/_mechanism_fork_summary
    # defaults cover the 5 pilot arms x 5 pilot seeds.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from typing import Dict, List, Optional

try:
    from scipy.stats import wilcoxon  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

COLLAPSE_BAR = 0.40          # macro-F1 below this = collapsed (plan §9)
RESCUE_MEAN = 0.45           # a fix is "strong" only if final mean >= this ...
RESCUE_COLLAPSE_RATE = 0     # ... AND collapse rate == 0 (plan §8)

DEFAULT_ARMS = ["baseline", "mom0", "tauclip320", "servmom", "serverlr03"]
DEFAULT_SEEDS = [42, 123, 8675309, 31337, 271828]
ARM_LABEL = {
    "baseline": "FedNova baseline (m=0.9)",
    "mom0": "client momentum = 0.0",
    "tauclip320": "tau-clip-min 320 (direction / Hyp B)",
    "servmom": "server momentum 0.9 (jitter / Hyp C)",
    "serverlr03": "server-lr 0.3 (magnitude / Hyp A)",
}


def _load(p: str) -> Optional[dict]:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _find_for_seed(dirpath: str, prefix: str, seed: int) -> Optional[dict]:
    """Find the single <prefix>_*_s<seed>.json in dirpath."""
    if not os.path.isdir(dirpath):
        return None
    hits = [p for p in glob.glob(os.path.join(dirpath, f"{prefix}_*.json"))
            if re.search(rf"_s{seed}\.json$", os.path.basename(p))]
    return _load(hits[0]) if hits else None


def collect_arm(results_root: str, arm: str, seeds: List[int]) -> Dict:
    dirpath = os.path.join(results_root, f"system_het_random_fednova_{arm}")
    rows = {}
    for s in seeds:
        final = _find_for_seed(dirpath, "test_at_final", s)
        best = _find_for_seed(dirpath, "test_at_best", s)
        if final is None and best is None:
            rows[s] = None              # not run yet / missing
            continue
        rows[s] = {
            "final_macro_f1": (float(final["final_macro_f1"]) if final and "final_macro_f1" in final else None),
            "final_per_class_f1": (final.get("final_per_class_f1") if final else None),
            "collapse_round": (final.get("collapse_round") if final else None),
            "best_macro_f1": (float(best["macro_f1"]) if best and "macro_f1" in best else
                              (float(final.get("best_macro_f1")) if final and "best_macro_f1" in final else None)),
            "best_per_class_f1": (best.get("per_class_f1") if best else None),
        }
    return {"dir": dirpath, "exists": os.path.isdir(dirpath), "rows": rows}


def _stats(vals: List[float]) -> Dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None,
                "collapse_rate": None}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "collapse_rate": sum(1 for v in vals if v < COLLAPSE_BAR),
    }


def _paired_vs_baseline(arm_rows, base_rows, key: str):
    """Paired (same-seed) deltas of `key` for arm vs baseline."""
    a, b, seeds = [], [], []
    for s in arm_rows:
        av = arm_rows[s]["final_macro_f1"] if (arm_rows[s] and key == "final") else \
             (arm_rows[s]["best_macro_f1"] if arm_rows[s] else None)
        bv = base_rows.get(s)
        bv = (bv["final_macro_f1"] if (bv and key == "final") else (bv["best_macro_f1"] if bv else None))
        if av is not None and bv is not None:
            a.append(av); b.append(bv); seeds.append(s)
    if not a:
        return None
    deltas = [x - y for x, y in zip(a, b)]
    wins = sum(1 for d in deltas if d > 1e-9)
    losses = sum(1 for d in deltas if d < -1e-9)
    p = None
    if _HAVE_SCIPY and len(deltas) >= 1 and any(abs(d) > 1e-12 for d in deltas):
        try:
            p = float(wilcoxon(a, b).pvalue)
        except Exception:
            p = None
    return {"n": len(a), "mean_delta": statistics.mean(deltas),
            "wins": wins, "losses": losses, "ties": len(deltas) - wins - losses,
            "wilcoxon_p": p}


def _is_rescue(st: Dict) -> bool:
    return (st["n"] > 0 and st["mean"] is not None
            and st["mean"] >= RESCUE_MEAN
            and st["collapse_rate"] == RESCUE_COLLAPSE_RATE)


def main():
    ap = argparse.ArgumentParser(description="FedNova mechanism-fork aggregator.")
    ap.add_argument("--results-root", default="fl_dermamnist/results")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--baseline-arm", default="baseline")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--out", default=None, help="optional dir for a CSV summary")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    data = {arm: collect_arm(args.results_root, arm, seeds) for arm in arms}

    print(f"\n=== FedNova mechanism-fork pilot — {len(seeds)} seeds {seeds} ===")
    print(f"collapse bar = {COLLAPSE_BAR}; rescue = mean>={RESCUE_MEAN} AND collapse-rate=={RESCUE_COLLAPSE_RATE}\n")
    print(f"{'arm':<12} {'n':>2} | {'FINAL macroF1':>20} {'min':>6} {'coll':>5} | "
          f"{'BEST-val macroF1':>20} {'coll':>5} | vs baseline (final)")
    print("-" * 110)

    fstats, bstats = {}, {}
    base_rows = data[args.baseline_arm]["rows"] if args.baseline_arm in data else {}
    csv_rows = []
    for arm in arms:
        rows = data[arm]["rows"]
        present = [s for s in seeds if rows.get(s)]
        fst = _stats([rows[s]["final_macro_f1"] for s in present])
        bst = _stats([rows[s]["best_macro_f1"] for s in present])
        fstats[arm], bstats[arm] = fst, bst
        if not data[arm]["exists"]:
            print(f"{arm:<12}  - | (dir not found: {data[arm]['dir']})")
            continue
        fmean = f"{fst['mean']:.3f}±{fst['std']:.3f}" if fst['mean'] is not None else "  NA  "
        bmean = f"{bst['mean']:.3f}±{bst['std']:.3f}" if bst['mean'] is not None else "  NA  "
        fmin = f"{fst['min']:.3f}" if fst['min'] is not None else " NA"
        cmp = ""
        if arm != args.baseline_arm and base_rows:
            pc = _paired_vs_baseline(rows, base_rows, "final")
            if pc:
                pstr = f" p={pc['wilcoxon_p']:.3f}" if pc['wilcoxon_p'] is not None else ""
                cmp = f"Δ={pc['mean_delta']:+.3f} {pc['wins']}W/{pc['losses']}L/{pc['ties']}T{pstr}"
        rescue = " ✅RESCUE" if (arm != args.baseline_arm and _is_rescue(fst)) else ""
        print(f"{arm:<12} {fst['n']:>2} | {fmean:>20} {fmin:>6} {str(fst['collapse_rate']):>5} | "
              f"{bmean:>20} {str(bst['collapse_rate']):>5} | {cmp}{rescue}")
        csv_rows.append({
            "arm": arm, "label": ARM_LABEL.get(arm, arm), "n": fst["n"],
            "final_mean": fst["mean"], "final_std": fst["std"], "final_min": fst["min"],
            "final_collapse_rate": fst["collapse_rate"],
            "best_mean": bst["mean"], "best_collapse_rate": bst["collapse_rate"],
        })

    # Per-arm missing seeds
    print("\nMissing / not-yet-complete (arm: seeds):")
    any_missing = False
    for arm in arms:
        miss = [s for s in seeds if not data[arm]["rows"].get(s)]
        if miss:
            any_missing = True
            print(f"  {arm}: {miss}")
    if not any_missing:
        print("  (none — all arms × seeds present)")

    # ---- §8 decision rule ----
    print("\n=== DECISION (plan §8) ===")
    tau = fstats.get("tauclip320", {})
    slr = fstats.get("serverlr03", {})
    smom = fstats.get("servmom", {})
    mom0 = fstats.get("mom0", {})
    base = fstats.get(args.baseline_arm, {})

    def ready(st):
        return st.get("n", 0) == len(seeds) and st.get("mean") is not None

    if not (ready(tau) and ready(slr) and ready(base)):
        print("  ⏳ Incomplete: need full tauclip320, serverlr03 and baseline arms before the")
        print("     magnitude-vs-direction verdict is valid. Re-run when the missing seeds finish.")
    else:
        tau_fix = _is_rescue(tau)
        slr_fix = _is_rescue(slr)
        if slr_fix and not tau_fix:
            verdict = "MAGNITUDE / implicit server-LR (Hyp A): server-lr 0.3 rescues, tau-clip does not."
        elif tau_fix and not slr_fix:
            verdict = "DIRECTION domination (Hyp B): tau-clip rescues, server-lr does not."
        elif tau_fix and slr_fix:
            verdict = "MIXED: both server-lr and tau-clip rescue."
        else:
            extra = []
            if _is_rescue(smom): extra.append("server-momentum DOES rescue → jitter (Hyp C)")
            if _is_rescue(mom0): extra.append("m=0 DOES rescue → client-momentum-driven")
            verdict = ("NEITHER server-lr nor tau-clip rescues → cumulative random-τ instability "
                       "(Hyp C) or deeper FedNova issue." + (" [" + "; ".join(extra) + "]" if extra else ""))
        print("  " + verdict)
        print(f"  (baseline final {base['mean']:.3f}; tauclip320 {tau['mean']:.3f}; "
              f"serverlr03 {slr['mean']:.3f}; "
              f"servmom {smom.get('mean') if smom.get('mean') is None else round(smom['mean'],3)}; "
              f"mom0 {mom0.get('mean') if mom0.get('mean') is None else round(mom0['mean'],3)})")

    if args.out and csv_rows:
        os.makedirs(args.out, exist_ok=True)
        import csv as _csv
        path = os.path.join(args.out, "mechanism_fork_summary.csv")
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader(); w.writerows(csv_rows)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
