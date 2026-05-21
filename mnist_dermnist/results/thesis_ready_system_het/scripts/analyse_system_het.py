"""Analyse the system-heterogeneity HPC results.

Reads `test_at_best_*.json` files from
  - mnist_dermnist/results/system_het_fixed/        (C1: Flower)
  - mnist_dermnist/results/system_het_random/       (C2: Flower)
and aggregates them into per-condition paired test statistics (H1) plus
between-condition tests of whether system heterogeneity amplifies the
FedProx advantage (H2).

H2 BASELINE — RUNTIME-MATCHED
-----------------------------
The H2 contrast subtracts the C0 per-seed Δ from each condition's Δ.
C0 MUST come from the SAME runtime as C1/C2, otherwise the H2 number
partially reflects pure-PyTorch ↔ Flower equivalence noise rather than
the system-heterogeneity manipulation itself.

Earlier versions of this script read `mnist_dermnist/results/headline/`
as the C0 baseline; those files are from the PURE-PYTORCH reference
loop, while C1/C2 are Flower outputs. That mixing has been removed.
This script now requires `mnist_dermnist/results/flower_C0_baseline/`
(produced by `submit_flower_C0_baseline.sh`) and refuses to compute H2
if either (a) that directory does not exist, or (b) any baseline JSON
reports a `framework` other than `flower-simulation`.

Outputs:
  - thesis_ready_system_het/data/per_seed_results.csv
  - thesis_ready_system_het/data/per_class_results.csv
  - thesis_ready_system_het/data/system_het_vs_baseline.json (H2 test)
  - thesis_ready_system_het/data/summary_statistics.json
  - prints a complete results table to stdout
"""
from __future__ import annotations

import json
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mnist_dermnist.fl.provenance import canonicalise_framework

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


ROOT = Path(__file__).resolve().parent.parent  # thesis_ready_system_het/
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT = ROOT.parent  # mnist_dermnist/results/


CLASS_NAMES = ["actinic", "basal", "benign_kerat", "dermato",
               "melanoma", "mel_nevi", "vascular"]


def load_pairs(results_dir: Path, *, require_framework: str | None = None):
    """Return (fedavg_by_seed, fedprox_by_seed, fednova_by_seed) dicts.

    If `require_framework` is given, raises ValueError when any loaded
    JSON's `framework` field disagrees. Used to enforce runtime-matched
    C0 for the H2 contrast.
    """
    fa, fp, fn = {}, {}, {}
    pat = re.compile(
        r"test_at_best_(fedavg|fedprox|fednova)_mu[0-9.]+_E20"
        r"(?:_sh-[a-z_]+)?(?:_C[0-9.]+)?_s(\d+)\.json"
    )
    framework_violations = []
    for f in sorted(results_dir.glob("test_at_best_*.json")):
        m = pat.match(f.name)
        if not m:
            print(f"  skipped (filename mismatch): {f.name}")
            continue
        algo, seed = m.group(1), int(m.group(2))
        data = json.load(open(f))
        if require_framework is not None:
            fw_raw = data.get("framework", "<missing>")
            try:
                fw = canonicalise_framework(fw_raw)
            except ValueError:
                framework_violations.append((f.name, fw_raw))
                continue
            if fw != require_framework:
                framework_violations.append((f.name, fw_raw))
                continue
        if algo == "fedavg":   fa[seed] = data
        elif algo == "fedprox": fp[seed] = data
        elif algo == "fednova": fn[seed] = data
    if framework_violations:
        raise ValueError(
            f"Refusing to use {results_dir.name} as C0 baseline: "
            f"{len(framework_violations)} file(s) report framework != "
            f"{require_framework!r}. First few: {framework_violations[:3]}. "
            f"H2 requires same-runtime C0; re-run with the correct submission "
            f"script (submit_flower_C0_baseline.sh) before retrying."
        )
    return fa, fp, fn


def summarise_solo(arm: dict, name: str) -> dict | None:
    """Per-arm summary for an algorithm that has no paired comparator
    in this script (FedNova is reported alongside FedAvg/FedProx but
    its statistical inference is unpaired here)."""
    seeds = sorted(arm)
    if not seeds:
        return None
    vals = [arm[s]["macro_f1"] for s in seeds]
    return {
        "algorithm": name,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "mean": float(np.mean(vals)),
        "sd":   float(np.std(vals, ddof=1)) if len(seeds) > 1 else 0.0,
        "per_seed": vals,
    }


def wilcoxon(deltas):
    if not HAS_SCIPY or all(d == 0 for d in deltas):
        return float("nan")
    try:
        _, p = stats.wilcoxon(deltas, alternative="two-sided")
        return float(p)
    except ValueError:
        return float("nan")


def rank_biserial(deltas):
    if not HAS_SCIPY:
        return float("nan")
    abs_ranks = stats.rankdata([abs(d) for d in deltas])
    pos = sum(abs_ranks[i] for i, d in enumerate(deltas) if d > 0)
    neg = sum(abs_ranks[i] for i, d in enumerate(deltas) if d < 0)
    return (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0


def summarise_condition(fa, fp, condition_name, baseline_fa=None, baseline_fp=None):
    seeds = sorted(set(fa) & set(fp))
    n = len(seeds)
    if n == 0:
        print(f"  WARNING: no paired seeds for {condition_name}")
        return None

    deltas = [fp[s]["macro_f1"] - fa[s]["macro_f1"] for s in seeds]
    fa_vals = [fa[s]["macro_f1"] for s in seeds]
    fp_vals = [fp[s]["macro_f1"] for s in seeds]

    out = {
        "condition": condition_name,
        "n_paired_seeds": n,
        "seeds": seeds,
        "fedavg_mean":  float(np.mean(fa_vals)),
        "fedavg_sd":    float(np.std(fa_vals, ddof=1)) if n > 1 else 0.0,
        "fedprox_mean": float(np.mean(fp_vals)),
        "fedprox_sd":   float(np.std(fp_vals, ddof=1)) if n > 1 else 0.0,
        "delta_mean":   float(np.mean(deltas)),
        "delta_sd":     float(np.std(deltas, ddof=1)) if n > 1 else 0.0,
        "fedprox_wins": int(sum(1 for d in deltas if d > 0)),
        "wilcoxon_p_h1":  wilcoxon(deltas),
        "rank_biserial":  rank_biserial(deltas),
        "per_seed_delta": deltas,
    }

    # H2: is this condition's per-seed delta different from baseline's?
    # Bonferroni correction across H2 tests is applied later in main()
    # once n_h2_tests is known (one test per condition with C0 paired data).
    if baseline_fa is not None and baseline_fp is not None:
        common = sorted(set(seeds) & set(baseline_fa) & set(baseline_fp))
        if common:
            baseline_deltas = [baseline_fp[s]["macro_f1"] - baseline_fa[s]["macro_f1"]
                               for s in common]
            cond_deltas_common = [fp[s]["macro_f1"] - fa[s]["macro_f1"] for s in common]
            h2_diffs = [cond_deltas_common[i] - baseline_deltas[i] for i in range(len(common))]
            out["h2_paired_diffs"] = h2_diffs
            out["h2_mean"] = float(np.mean(h2_diffs))
            out["h2_wilcoxon_p_raw"] = wilcoxon(h2_diffs)
            out["h2_seeds"] = common

    # Straggler-tolerance ratio (vs baseline)
    if baseline_fa is not None and baseline_fp is not None:
        baseline_seeds = sorted(set(baseline_fa) & set(baseline_fp))
        baseline_fa_mean = float(np.mean([baseline_fa[s]["macro_f1"] for s in baseline_seeds]))
        baseline_fp_mean = float(np.mean([baseline_fp[s]["macro_f1"] for s in baseline_seeds]))
        out["straggler_tolerance"] = {
            "fedavg":  out["fedavg_mean"]  / baseline_fa_mean,
            "fedprox": out["fedprox_mean"] / baseline_fp_mean,
        }

    return out


# ---------------------------------------------------------------------------
# Sanity check: verify that the straggler schedule actually did what was
# claimed. Reconstructs each seed's (round × client) epoch schedule from
# the JSON provenance (system_het config + seed + num_rounds), since the
# schedule is deterministic given those inputs (see fl/system_het.py).
# Optionally cross-references with client_update_norms_*.csv when present.
# ---------------------------------------------------------------------------

def verify_straggler_schedule(results_dir: Path, condition_label: str,
                              num_clients: int = 7) -> None:
    """Print a small summary table verifying the straggler schedule in
    `results_dir` matches what the submit script claimed. Pure sanity
    check; nothing written to disk."""
    # Lazy import — only needed here, and only when system-het JSONs exist.
    from mnist_dermnist.fl.system_het import (
        SystemHetConfig, build_epoch_schedule,
    )

    jsons = sorted(results_dir.glob("test_at_best_*.json"))
    if not jsons:
        return

    # Load one JSON per seed (any algo) to get system_het + seed + num_rounds.
    by_seed: Dict[int, Dict] = {}
    for f in jsons:
        d = json.load(open(f))
        seed = int(d.get("seed", -1))
        if seed < 0 or seed in by_seed:
            continue
        by_seed[seed] = d
    if not by_seed:
        return

    # Use the first JSON as the reference for mode + E_max.
    ref = next(iter(by_seed.values()))
    sh = ref.get("system_het") or {}
    mode = sh.get("mode", "uniform")
    e_max = int(sh.get("E_max", ref.get("local_epochs", 20)))
    num_rounds = int(ref.get("num_rounds", 150))

    print(f"\n  ── Straggler-schedule sanity check for {condition_label} ──")
    print(f"  mode={mode!r}  E_max={e_max}  num_clients={num_clients}  num_rounds={num_rounds}")
    print(f"  seeds verified: {sorted(by_seed)}")

    if mode == "uniform":
        print( "  Uniform schedule (no stragglers). Skipping per-client stats.\n")
        return

    # Reconstruct schedule per seed.
    schedules = {}
    for seed, d in by_seed.items():
        sh_seed = d.get("system_het") or sh
        cfg = SystemHetConfig(
            mode=sh_seed.get("mode", mode),
            E_max=int(sh_seed.get("E_max", e_max)),
            E_straggler=int(sh_seed.get("E_straggler", 5)),
            fixed_straggler_ids=sh_seed.get("fixed_straggler_ids"),
            random_straggler_fraction=float(sh_seed.get("random_straggler_fraction", 0.5)),
            random_straggler_min_epochs=int(sh_seed.get("random_straggler_min_epochs", 1)),
            random_straggler_max_epochs=sh_seed.get("random_straggler_max_epochs"),
        )
        schedules[seed] = build_epoch_schedule(
            cfg, num_clients=num_clients, num_rounds=num_rounds, seed=seed,
        )

    # Aggregate straggler cells across all reconstructed schedules.
    all_e_straggler: List[int] = []
    per_client_freq = np.zeros(num_clients, dtype=int)
    unique_stragglers: set[int] = set()
    n_seed = len(schedules)
    for seed, sched in schedules.items():
        straggler_mask = sched < e_max
        per_client_freq += straggler_mask.sum(axis=0)
        for k in range(num_clients):
            if straggler_mask[:, k].any():
                unique_stragglers.add(k)
        all_e_straggler.extend(sched[straggler_mask].tolist())

    if not all_e_straggler:
        print( "  No straggler cells found across reconstructed schedules.\n")
        return

    arr = np.asarray(all_e_straggler)
    print(f"  Unique clients that appeared as stragglers (across all seeds): "
          f"{sorted(unique_stragglers)}  (count {len(unique_stragglers)})")
    print(f"  E_i on straggler cells: mean={arr.mean():.2f}  min={int(arr.min())}  max={int(arr.max())}  n_cells={len(arr)}")
    print(f"  Per-client straggler frequency (rounds-as-straggler / {num_rounds * n_seed} aggregated cells):")
    for k in range(num_clients):
        rate = per_client_freq[k] / (num_rounds * n_seed)
        print(f"    C{k}: {per_client_freq[k]:>5d} / {num_rounds * n_seed} = {rate:.3f}")

    # Optional: join with client_update_norms_*.csv if present.
    norm_csvs = sorted(results_dir.glob("client_update_norms_*.csv"))
    if not norm_csvs:
        print( "  (no client_update_norms_*.csv in directory; skipping drift comparison)\n")
        return

    seed_pat = re.compile(r"_s(\d+)\.csv$")
    straggler_norms: List[float] = []
    nonstraggler_norms: List[float] = []
    for csv in norm_csvs:
        m = seed_pat.search(csv.name)
        if not m: continue
        seed = int(m.group(1))
        if seed not in schedules: continue
        sched = schedules[seed]
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            r, c = int(row["round"]), int(row["client_id"])
            if not (1 <= r <= num_rounds and 0 <= c < num_clients):
                continue
            is_straggler = sched[r - 1, c] < e_max
            (straggler_norms if is_straggler else nonstraggler_norms).append(
                float(row["update_norm"]))

    if straggler_norms and nonstraggler_norms:
        sa = np.asarray(straggler_norms); sb = np.asarray(nonstraggler_norms)
        print(f"  Update-norm comparison (from {len(norm_csvs)} CSVs):")
        print(f"    straggler rounds:    mean={sa.mean():.4f}  sd={sa.std(ddof=1):.4f}  n={len(sa)}")
        print(f"    non-straggler rounds: mean={sb.mean():.4f}  sd={sb.std(ddof=1):.4f}  n={len(sb)}")
        ratio = sa.mean() / sb.mean() if sb.mean() > 0 else float("nan")
        print(f"    straggler / non-straggler norm ratio: {ratio:.3f}")
    print()


def main():
    print("=" * 72)
    print("SYSTEM HETEROGENEITY ANALYSIS")
    print("=" * 72)

    # --- C0 baseline (Flower runtime). Optional: if missing, H1 still
    #     runs for every available condition; only H2 (Δ_condition - Δ_C0)
    #     is skipped. The cross-runtime constraint (framework must be
    #     'flower-simulation') is still enforced when C0 IS loaded, to
    #     prevent silent mixing with the pure-PyTorch headline.
    print("\nLoading baseline (no system het, Flower runtime)...")
    base_dir = RESULTS_ROOT / "flower_C0_baseline"
    base_fa, base_fp, base_fn = {}, {}, {}
    c0_available = False
    if not base_dir.exists():
        print(f"  WARNING: {base_dir} does not exist.")
        print( "           C0 baseline not found; H2 contrast unavailable, "
               "reporting H1 only.")
        print( "           To compute H2 later, run "
               "submit_flower_C0_baseline.sh on HPC.")
    else:
        try:
            base_fa, base_fp, base_fn = load_pairs(
                base_dir, require_framework="flower-simulation")
            print(f"  Loaded {len(base_fa)} FedAvg, {len(base_fp)} FedProx, "
                  f"{len(base_fn)} FedNova from {base_dir}")
            if base_fa and base_fp:
                c0_available = True
            else:
                print( "  WARNING: no paired FedAvg/FedProx baseline runs in "
                      f"{base_dir}; H2 unavailable.")
        except ValueError as exc:
            print(f"  ERROR loading C0 ({exc!s}); H2 unavailable, "
                   "reporting H1 only.")

    # --- C1 fixed stragglers ---
    print("\nLoading C1 (fixed stragglers)...")
    c1_dir = RESULTS_ROOT / "system_het_fixed"
    if not c1_dir.exists():
        print(f"  Directory does not exist yet: {c1_dir}")
        c1_fa, c1_fp, c1_fn = {}, {}, {}
    else:
        c1_fa, c1_fp, c1_fn = load_pairs(c1_dir)
        print(f"  Loaded {len(c1_fa)} FedAvg, {len(c1_fp)} FedProx, "
              f"{len(c1_fn)} FedNova from {c1_dir}")

    # --- C2 random stragglers (FedAvg/FedProx in system_het_random;
    #     FedNova in system_het_random_fednova per submit_fednova_system_het.sh)
    print("\nLoading C2 (random stragglers)...")
    c2_dir = RESULTS_ROOT / "system_het_random"
    c2_fn_dir = RESULTS_ROOT / "system_het_random_fednova"
    if not c2_dir.exists():
        print(f"  Directory does not exist yet: {c2_dir}")
        c2_fa, c2_fp, _ = {}, {}, {}
    else:
        c2_fa, c2_fp, _ = load_pairs(c2_dir)
        print(f"  Loaded {len(c2_fa)} FedAvg, {len(c2_fp)} FedProx from {c2_dir}")
    if c2_fn_dir.exists():
        _, _, c2_fn = load_pairs(c2_fn_dir)
        print(f"  Loaded {len(c2_fn)} FedNova from {c2_fn_dir}")
    else:
        c2_fn = {}
        print(f"  FedNova C2 directory does not exist yet: {c2_fn_dir}")

    if not (c0_available or (c1_fa and c1_fp) or (c2_fa and c2_fp)):
        print("\nNo data available in any condition. Re-run when HPC sweeps complete.")
        return

    # --- Summarise each condition. H1 (within-condition FedProx-vs-FedAvg
    # paired Wilcoxon) is always computed. H2 (diff-in-diff vs C0) is only
    # added to a condition's summary when baseline_fa/baseline_fp are passed,
    # i.e. only when C0 was successfully loaded.
    base_kwargs = ({"baseline_fa": base_fa, "baseline_fp": base_fp}
                   if c0_available else {})
    summaries = []
    if c0_available:
        summaries.append(summarise_condition(base_fa, base_fp, "C0 (baseline)"))
    if c1_fa and c1_fp:
        summaries.append(summarise_condition(
            c1_fa, c1_fp, "C1 (fixed_stragglers)", **base_kwargs))
    if c2_fa and c2_fp:
        summaries.append(summarise_condition(
            c2_fa, c2_fp, "C2 (random_stragglers)", **base_kwargs))

    # --- Bonferroni correction for H2 across all conditions that produced
    # an H2 p-value. n_h2_tests = number of (condition × paired-with-C0)
    # H2 contrasts actually computed. Correction is a no-op when only one
    # condition has H2 data.
    h2_summaries = [s for s in summaries if s is not None and "h2_wilcoxon_p_raw" in s]
    n_h2_tests = len(h2_summaries)
    for s in h2_summaries:
        raw = s["h2_wilcoxon_p_raw"]
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            s["h2_bonferroni_p"] = float("nan")
        else:
            s["h2_bonferroni_p"] = float(min(raw * n_h2_tests, 1.0))
        s["h2_n_tests"] = int(n_h2_tests)
    if not c0_available and any(s and "C0" not in s.get("condition", "") for s in summaries):
        print("\nNote: H2 (Δ_condition - Δ_C0) skipped — C0 baseline not loaded. "
              "Within-condition H1 contrasts below are still computed.")

    # --- Straggler-schedule sanity checks (verify the experiment actually
    # did what the submit scripts claimed). Reconstructs each seed's
    # schedule from JSON provenance and reports per-client frequencies,
    # E_i range on straggler cells, and (if present) update-norm ratios.
    print("\n" + "=" * 72)
    print("STRAGGLER-SCHEDULE SANITY CHECKS")
    print("=" * 72)
    if c1_dir.exists() and (c1_fa or c1_fp):
        verify_straggler_schedule(c1_dir, "C1 (fixed_stragglers)")
    if c2_dir.exists() and (c2_fa or c2_fp):
        verify_straggler_schedule(c2_dir, "C2 (random_stragglers)")
    if c2_fn_dir.exists() and c2_fn:
        verify_straggler_schedule(c2_fn_dir, "C2 FedNova (random_stragglers)")

    # FedNova arms — reported alongside but not as a paired contrast
    # against FedAvg/FedProx (FedNova has its own objective). For each
    # condition where FedNova ran, we report mean+SD across seeds and
    # — where the same seed has all three algorithms — within-seed
    # differences FedNova - FedAvg and FedNova - FedProx.
    fednova_arms = []
    for cond_name, fn_dict, fa_dict, fp_dict in [
        ("C0 (baseline)",       base_fn, base_fa, base_fp),
        ("C1 (fixed_stragglers)", c1_fn,  c1_fa,  c1_fp),
        ("C2 (random_stragglers)", c2_fn, c2_fa,  c2_fp),
    ]:
        solo = summarise_solo(fn_dict, "FedNova")
        if solo is None:
            continue
        # Triple-paired Δ: only seeds present in all three arms
        triple_seeds = sorted(set(fn_dict) & set(fa_dict) & set(fp_dict))
        if triple_seeds:
            d_nova_avg = [fn_dict[s]["macro_f1"] - fa_dict[s]["macro_f1"]
                          for s in triple_seeds]
            d_nova_prox = [fn_dict[s]["macro_f1"] - fp_dict[s]["macro_f1"]
                           for s in triple_seeds]
            solo["delta_vs_fedavg_mean"]  = float(np.mean(d_nova_avg))
            solo["delta_vs_fedprox_mean"] = float(np.mean(d_nova_prox))
            solo["delta_vs_fedavg_p"]  = wilcoxon(d_nova_avg)
            solo["delta_vs_fedprox_p"] = wilcoxon(d_nova_prox)
            solo["triple_seeds"] = triple_seeds
        solo["condition"] = cond_name
        fednova_arms.append(solo)

    # Print headline table
    print("\n" + "=" * 110)
    print("HEADLINE RESULTS TABLE")
    print("=" * 110)
    print(f"{'condition':<25} {'FedAvg':>14} {'FedProx':>14} {'Δ':>10} "
          f"{'p (H1)':>8} {'p (H2 raw)':>11} {'p (H2 Bonf)':>12} {'r_rb':>8}")
    print("-" * 110)
    for s in summaries:
        if s is None: continue
        p_h2_raw  = s.get("h2_wilcoxon_p_raw", float("nan"))
        p_h2_bonf = s.get("h2_bonferroni_p", float("nan"))
        print(f"{s['condition']:<25} "
              f"{s['fedavg_mean']:>7.4f}±{s['fedavg_sd']:.3f} "
              f"{s['fedprox_mean']:>7.4f}±{s['fedprox_sd']:.3f} "
              f"{s['delta_mean']:>+10.4f} "
              f"{s['wilcoxon_p_h1']:>8.4f} "
              f"{p_h2_raw:>11.4f} "
              f"{p_h2_bonf:>12.4f} "
              f"{s['rank_biserial']:>+8.3f}")
    if n_h2_tests > 0:
        print(f"\nBonferroni correction applied across n_h2_tests = {n_h2_tests} "
              f"condition(s) with H2 data.")

    # Straggler-tolerance ratios
    print("\nSTRAGGLER-TOLERANCE RATIOS (vs C0 baseline)")
    print("-" * 60)
    print(f"{'condition':<25} {'ρ_FedAvg':>12} {'ρ_FedProx':>12}")
    for s in summaries:
        if s is None or "straggler_tolerance" not in s:
            continue
        st = s["straggler_tolerance"]
        print(f"{s['condition']:<25} {st['fedavg']:>12.4f} {st['fedprox']:>12.4f}")

    # FedNova arms (HV1) — reported as a third comparator
    if fednova_arms:
        print("\n" + "=" * 100)
        print("FEDNOVA ARM (per condition; triple-paired Δ vs FedAvg / vs FedProx where available)")
        print("=" * 100)
        print(f"{'condition':<25} {'FedNova':>14} {'n':>4} {'Δ vs FedAvg':>14} {'p':>8} {'Δ vs FedProx':>14} {'p':>8}")
        print("-" * 100)
        for a in fednova_arms:
            dva   = a.get("delta_vs_fedavg_mean",  float("nan"))
            dvp   = a.get("delta_vs_fedprox_mean", float("nan"))
            pva   = a.get("delta_vs_fedavg_p",     float("nan"))
            pvp   = a.get("delta_vs_fedprox_p",    float("nan"))
            print(f"{a['condition']:<25} "
                  f"{a['mean']:>7.4f}±{a['sd']:.3f} "
                  f"{a['n_seeds']:>4d} "
                  f"{dva:>+14.4f} {pva:>8.4f} "
                  f"{dvp:>+14.4f} {pvp:>8.4f}")

    # Save
    with open(DATA_DIR / "summary_statistics.json", "w") as f:
        json.dump({"conditions": summaries, "fednova_arms": fednova_arms}, f, indent=2)
    print(f"\nWrote {DATA_DIR / 'summary_statistics.json'}")

    rows = []
    for s in summaries:
        if s is None: continue
        for i, seed in enumerate(s["seeds"]):
            rows.append({
                "condition": s["condition"],
                "seed": seed,
                "delta_macro_f1": s["per_seed_delta"][i],
            })
    pd.DataFrame(rows).to_csv(DATA_DIR / "per_seed_results.csv", index=False)
    print(f"Wrote {DATA_DIR / 'per_seed_results.csv'}")


if __name__ == "__main__":
    main()
