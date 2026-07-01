"""Analyse the Li 2020 §5.2 asymmetric straggler protocol - 3-arm decomposition.

The Li et al. 2020 §5.2 comparison is widely cited as the canonical
FedProx-wins-clearly result, but it conflates two effects that this
script disentangles:

  Arm 1 - FedAvg --drop-stragglers
           reads from results/system_het_random_asymmetric/
           filenames end with '_drop_s<seed>.json'
           provenance: drop_stragglers=True

  Arm 2 - FedAvg without drop (= mu=0 FedProx, includes partial work)
           reads from results/system_het_random/
           filenames end with '_s<seed>.json' (no _drop tag)
           provenance: drop_stragglers=False

  Arm 3 - FedProx mu=0.01 (includes partial work, with proximal anchor)
           reads from results/system_het_random/
           provenance: drop_stragglers=False (algorithm=fedprox)

Three contrasts (paired by seed):
  Δ_total   = Arm 3 − Arm 1   = Li 2020 §5.2 headline ("FedProx wins")
  Δ_include = Arm 2 − Arm 1   = effect of using partial-work updates
                                 (FedAvg+include vs FedAvg+drop)
  Δ_prox    = Arm 3 − Arm 2   = pure proximal-term effect at parity
                                 (== our existing symmetric C2 delta)

If Δ_total > Δ_prox, the literature's headline "FedProx wins" is
largely attributable to Arm 1's smaller per-round sample size, not to
the proximal anchor mechanism per se.

Strictness
----------
* Filenames in ``system_het_random_asymmetric/`` MUST end with
  ``_drop_s<seed>.json`` for FedAvg files. Files lacking the ``_drop``
  tag are rejected (cannot be the asymmetric-protocol baseline).
* Each loaded JSON MUST carry a ``drop_stragglers`` provenance field
  matching the directory's protocol. Mismatch raises an error rather
  than silently mixing baselines.

Methodological caveat (for the thesis writeup)
----------------------------------------------
Flower waits for all clients to train before invoking ``aggregate_fit``
and only then discards straggler updates. The comparison measures
the *algorithmic-policy* effect (which updates are aggregated), not
the *wall-clock-deployment* effect (how long the system waits).
"""
from __future__ import annotations

import json
import os
import re
from statistics import mean, median, stdev


ASYM_DIR = "fl_dermamnist/results/system_het_random_asymmetric"
SYM_DIR  = "fl_dermamnist/results/system_het_random"
PT_DIR   = "fl_dermamnist/results/headline"

# STRICT: FedAvg files in the asymmetric directory MUST carry the _drop tag.
_ASYM_FEDAVG_PAT = re.compile(
    r"test_at_best_fedavg_mu0\.0_E20_sh-random_stragglers_drop_s(?P<seed>\d+)\.json"
)
# SYMMETRIC directory - no _drop tag permitted.
_SYM_PAT = re.compile(
    r"test_at_best_(?P<algo>fedavg|fedprox)_mu[0-9.]+_E20"
    r"_sh-random_stragglers_s(?P<seed>\d+)\.json"
)
# Pure-PyTorch headline pattern (no system-het tag).
_PT_PAT = re.compile(
    r"test_at_best_(?P<algo>fedavg|fedprox)_mu[0-9.]+_E20_s(?P<seed>\d+)\.json"
)


def _assert_provenance(doc: dict, file_path: str, *, expected_drop: bool,
                        expected_algo: str | None = None) -> None:
    """Refuse to mix baselines: each JSON must carry matching provenance."""
    actual_drop = doc.get("drop_stragglers")
    # If the provenance field is missing entirely, the file predates the
    # protocol implementation and cannot be trusted as an asymmetric-arm
    # baseline.
    if actual_drop is None and expected_drop:
        raise AssertionError(
            f"{file_path}: missing 'drop_stragglers' provenance field. "
            f"Asymmetric-protocol files require explicit provenance."
        )
    # If provided, must match the expected protocol for this directory.
    if actual_drop is not None and bool(actual_drop) is not bool(expected_drop):
        raise AssertionError(
            f"{file_path}: drop_stragglers={actual_drop} but directory "
            f"expects {expected_drop}. Aborting to prevent mixed baselines."
        )
    if expected_algo is not None and doc.get("algorithm") != expected_algo:
        raise AssertionError(
            f"{file_path}: algorithm={doc.get('algorithm')!r} but expected "
            f"{expected_algo!r}."
        )


def load_asym_fedavg_drop() -> dict:
    """Arm 1: FedAvg with --drop-stragglers (the new run)."""
    out = {}
    if not os.path.isdir(ASYM_DIR):
        return out
    for f in sorted(os.listdir(ASYM_DIR)):
        m = _ASYM_FEDAVG_PAT.match(f)
        if not m:
            # Reject anything that doesn't carry the _drop tag
            if f.startswith("test_at_best_fedavg"):
                raise AssertionError(
                    f"{ASYM_DIR}/{f}: FedAvg file in asymmetric directory "
                    f"is missing the '_drop' tag. This is suspicious — "
                    f"the asymmetric protocol requires --drop-stragglers."
                )
            continue
        path = os.path.join(ASYM_DIR, f)
        doc = json.load(open(path))
        _assert_provenance(doc, path, expected_drop=True, expected_algo="fedavg")
        out[int(m.group("seed"))] = float(doc["macro_f1"])
    return out


def load_sym(algo: str) -> dict:
    """Arm 2 (algo=fedavg) or Arm 3 (algo=fedprox): symmetric runs."""
    out = {}
    if not os.path.isdir(SYM_DIR):
        return out
    for f in sorted(os.listdir(SYM_DIR)):
        m = _SYM_PAT.match(f)
        if not m or m.group("algo") != algo:
            continue
        path = os.path.join(SYM_DIR, f)
        doc = json.load(open(path))
        _assert_provenance(doc, path, expected_drop=False, expected_algo=algo)
        out[int(m.group("seed"))] = float(doc["macro_f1"])
    return out


def load_pt(algo: str) -> dict:
    """Pure-PyTorch headline (no system-het) - for cross-runtime reference."""
    out = {}
    if not os.path.isdir(PT_DIR):
        return out
    for f in sorted(os.listdir(PT_DIR)):
        m = _PT_PAT.match(f)
        if not m or m.group("algo") != algo:
            continue
        path = os.path.join(PT_DIR, f)
        out[int(m.group("seed"))] = float(json.load(open(path))["macro_f1"])
    return out


def paired_delta(a: dict, b: dict, *, label: str = "") -> tuple[list[int], list[float]]:
    seeds = sorted(set(a) & set(b))
    deltas = [b[s] - a[s] for s in seeds]
    return seeds, deltas


def summarise(name: str, deltas: list[float]) -> float | None:
    if not deltas:
        print(f"\n{name}: (no paired data)")
        return None
    n = len(deltas)
    wins = sum(1 for d in deltas if d > 0)
    mu = mean(deltas)
    med = median(deltas)
    sd = stdev(deltas) if n > 1 else 0.0
    line = f"\n{name}: n={n}  mean={mu:+.4f}  median={med:+.4f}  SD={sd:.4f}  wins={wins}/{n}"
    print(line)
    try:
        from scipy.stats import wilcoxon
        if any(d != 0 for d in deltas):
            p = wilcoxon(deltas, alternative="two-sided").pvalue
            print(f"  Paired Wilcoxon two-sided p = {p:.4f}")
    except ImportError:
        pass
    return mu


def main() -> int:
    print("=" * 80)
    print(" Li 2020 §5.2 ASYMMETRIC STRAGGLER PROTOCOL — 3-ARM DECOMPOSITION")
    print("=" * 80)

    try:
        arm1 = load_asym_fedavg_drop()                  # FedAvg --drop
        arm2 = load_sym("fedavg")                         # FedAvg no-drop
        arm3 = load_sym("fedprox")                        # FedProx no-drop
    except AssertionError as e:
        print(f"\nAUDIT FAILED: {e}")
        return 2

    print(f"\nArm 1 (FedAvg --drop):     {len(arm1)} seeds in {ASYM_DIR}")
    print(f"Arm 2 (FedAvg include):    {len(arm2)} seeds in {SYM_DIR}")
    print(f"Arm 3 (FedProx include):   {len(arm3)} seeds in {SYM_DIR}")

    # Three primary contrasts
    print()
    print("=" * 80)
    print(" THE THREE CONTRASTS")
    print("=" * 80)

    seeds, total = paired_delta(arm1, arm3)
    mu_total = summarise("Δ_total   = FedProx-include  −  FedAvg-drop   (Li 2020 §5.2 headline)", total)

    _, include = paired_delta(arm1, arm2)
    mu_include = summarise("Δ_include = FedAvg-include  −  FedAvg-drop   (partial-work effect)", include)

    _, prox = paired_delta(arm2, arm3)
    mu_prox = summarise("Δ_prox    = FedProx-include  −  FedAvg-include (pure proximal effect; == symmetric C2)", prox)

    # Decomposition (algebraic identity, useful for sanity)
    print()
    print("=" * 80)
    print(" DECOMPOSITION (algebraic identity: Δ_total == Δ_include + Δ_prox)")
    print("=" * 80)
    if mu_total is not None and mu_include is not None and mu_prox is not None:
        print(f"  Δ_total   = {mu_total:+.4f}")
        print(f"  Δ_include = {mu_include:+.4f}    ← effect of including partial work")
        print(f"  Δ_prox    = {mu_prox:+.4f}    ← effect of the proximal anchor at parity")
        print(f"  sum       = {mu_include + mu_prox:+.4f}  (should equal Δ_total within float noise)")
        if abs(mu_total - (mu_include + mu_prox)) > 1e-3:
            print(f"  ⚠  decomposition mismatch — paired-seed sets may differ across arms")

    # Cross-reference to pure-PyTorch headline (no stragglers, no Flower)
    pt_fa = load_pt("fedavg")
    pt_fp = load_pt("fedprox")
    _, pt_d = paired_delta(pt_fa, pt_fp)
    if pt_d:
        print()
        print(f"For reference: pure-PyTorch headline (no stragglers, no Flower)")
        print(f"  Δ_PT = {mean(pt_d):+.4f}  n={len(pt_d)}")

    # Interpretation guide
    print()
    print("=" * 80)
    print(" INTERPRETATION GUIDE")
    print("=" * 80)
    print("  If Δ_total >> Δ_prox:")
    print("    Most of the literature's headline FedProx advantage comes from")
    print("    Arm 3 seeing more clients per round than Arm 1 — not from the")
    print("    proximal anchor itself. The proximal mechanism contributes only")
    print("    Δ_prox at parity-of-aggregation.")
    print()
    print("  If Δ_total ≈ Δ_prox:")
    print("    Straggler dropping doesn't matter much on this dataset; the")
    print("    proximal term alone accounts for the FedProx advantage. The")
    print("    Li 2020 §5.2 protocol asymmetry is methodologically incidental.")
    print()
    print("  Wall-clock caveat: this comparison measures aggregation-policy")
    print("  effect on accuracy-per-round, NOT real-deployment wall-clock.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
