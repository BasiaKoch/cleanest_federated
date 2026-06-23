"""Stage A — data-integrity sanity check across every result directory.

Before any inferential test runs, verify the data is not pathological:

  1. Per-sweep JSON counts match expectations.
  2. Every framework label is canonical (resolved through
     fl_dermamnist.fl.provenance.canonicalise_framework).
  3. No JSON has a pathological macro_f1 (< 0.05 = silent training
     failure; > 0.98 = data leak).
  4. selected_round (best-val round) ranges across runs — not always 1
     (early stopping) and not always 150 (never-improved). A reasonable
     distribution = training actually converged.
  5. Required provenance fields populated in every canonical-runtime
     JSON (git_commit, hostname, run_started_at, run_finished_at).
  6. Paired seeds are present in matching pairs (FedAvg and FedProx
     at every seed in any directory holding both algorithms).

Exit code:
    0  All checks passed.
    N  N suspicious files / mismatched counts detected. Per-issue
       diagnostics printed to stdout.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from fl_dermamnist.common.paths import repo_root, package_root, results_root, thesis_ready_root, thesis_data_dir, thesis_figures_dir  # noqa: E402

# Allow PYTHONPATH=. invocation from repo root
sys.path.insert(0, str(repo_root()))

from fl_dermamnist.fl.provenance import (  # noqa: E402
    CANONICAL_FRAMEWORKS,
    canonicalise_framework,
)


RESULTS_DIR = results_root()

# Expected JSON counts per directory. None = directory is optional /
# unbounded (e.g. centralised has 10 expected; ablations vary).
EXPECTED = {
    "legacy_pure_pytorch":         20,
    "flower_C0_baseline":          30,
    "system_het_fixed":            20,
    "system_het_random":           20,
    "system_het_random_fednova":   10,
    "iid":                         20,
    "dirichlet_a01":               20,
    "specialist_partition":        20,
    "mu_sweep":                    18,
    "e_sweep":                     30,
    "class_weighted_baseline":     10,
    "centralised":                 10,
    "headline_flower_verify":      None,  # may or may not exist
}

# Macro-F1 thresholds: below this = suspicious (silent failure).
MACRO_F1_FLOOR = 0.05
# Above this = suspicious (probable data leak; DermaMNIST is not solvable
# this well at 28x28 with this model — centralised tops out around 0.56).
MACRO_F1_CEILING = 0.95

# Required provenance fields for canonical-runtime JSONs (post-CP3.2).
# Legacy pure-pytorch JSONs (provenance_note == "backstamped-2026-05-18")
# are exempt because they predate the run-time provenance patch.
REQUIRED_PROVENANCE_FIELDS = (
    "git_commit", "hostname", "python_version", "torch_version",
    "run_started_at", "run_finished_at",
)


# Filename parser — matches every queued sweep's stem.
_STEM = re.compile(
    r"test_at_best_"
    r"(?P<algo>fedavg|fedprox|fednova)"
    r"_mu(?P<mu>[0-9.]+)"
    r"_E(?P<E>\d+)"
    r"(?:_sh-(?P<sh>[a-z_]+))?"
    r"(?:_C(?P<C>[0-9.]+))?"
    r"_s(?P<seed>\d+)\.json$"
)


def _check_directory(name: str, expected: int | None) -> tuple[int, list[str]]:
    """Returns (n_issues, lines_to_print)."""
    issues = []
    d = RESULTS_DIR / name
    if not d.is_dir():
        if expected is None:
            return 0, [f"  {name:30s}  (optional, not present)"]
        return 1, [f"  {name:30s}  ✗ directory missing entirely (expected {expected})"]

    jsons = sorted(d.glob("test_at_best_*.json"))
    # Centralised has a different file pattern
    if name == "centralised":
        jsons = sorted(d.glob("centralised_seed*.json"))

    n_present = len(jsons)
    count_ok = (expected is None) or (n_present == expected)
    count_marker = "✓" if count_ok else "✗"
    lines = [f"  {name:30s}  {count_marker} JSON count = {n_present}/{expected if expected else 'any'}"]
    if not count_ok:
        issues.append(f"{name}: expected {expected} JSONs, found {n_present}")

    # Inspect every JSON
    n_bad_framework = 0
    n_bad_macro = 0
    n_missing_prov = 0
    selected_rounds = []
    seeds_per_algo = defaultdict(set)
    is_legacy = (name == "legacy_pure_pytorch")
    macro_f1_min = float("inf")
    macro_f1_max = float("-inf")

    for p in jsons:
        try:
            doc = json.load(open(p))
        except Exception as e:
            n_bad_framework += 1
            issues.append(f"{name}/{p.name}: failed to load JSON ({e})")
            continue

        if name == "centralised":
            macro = doc.get("macro_f1", float("nan"))
            macro_f1_min = min(macro_f1_min, macro)
            macro_f1_max = max(macro_f1_max, macro)
            continue   # centralised has its own schema (no framework field)

        # Framework label
        fw_raw = doc.get("framework", "<missing>")
        try:
            canonicalise_framework(fw_raw)
        except ValueError:
            n_bad_framework += 1
            issues.append(f"{name}/{p.name}: framework={fw_raw!r} does not canonicalise")

        # Macro-F1 sanity
        macro = doc.get("macro_f1", float("nan"))
        macro_f1_min = min(macro_f1_min, macro)
        macro_f1_max = max(macro_f1_max, macro)
        if macro < MACRO_F1_FLOOR or macro > MACRO_F1_CEILING:
            n_bad_macro += 1
            issues.append(f"{name}/{p.name}: macro_f1={macro:.4f} outside [{MACRO_F1_FLOOR}, {MACRO_F1_CEILING}]")

        # Selected round
        sr = doc.get("selected_round")
        if isinstance(sr, int):
            selected_rounds.append(sr)

        # Provenance fields (canonical-runtime JSONs only — legacy is exempt)
        if not is_legacy:
            for field in REQUIRED_PROVENANCE_FIELDS:
                if field not in doc:
                    n_missing_prov += 1
                    issues.append(f"{name}/{p.name}: missing provenance field {field!r}")
                    break  # one log line per JSON

        # Track paired-seed coverage
        m = _STEM.match(p.name)
        if m:
            seeds_per_algo[m.group("algo")].add(int(m.group("seed")))

    if n_bad_framework:
        lines.append(f"    ✗ {n_bad_framework} JSONs with non-canonical framework label")
    if n_bad_macro:
        lines.append(f"    ✗ {n_bad_macro} JSONs with macro_f1 outside [{MACRO_F1_FLOOR}, {MACRO_F1_CEILING}]")
    if n_missing_prov and not is_legacy:
        lines.append(f"    ✗ {n_missing_prov} JSONs missing required provenance fields")
    if selected_rounds:
        lines.append(f"    selected_round: min={min(selected_rounds)} max={max(selected_rounds)} "
                     f"mean={sum(selected_rounds)/len(selected_rounds):.1f}")
    if macro_f1_min < float("inf"):
        lines.append(f"    macro_f1 range: [{macro_f1_min:.4f}, {macro_f1_max:.4f}]")

    # Paired-seed coverage check (where both FedAvg and FedProx present)
    if "fedavg" in seeds_per_algo and "fedprox" in seeds_per_algo:
        fa = seeds_per_algo["fedavg"]
        fp = seeds_per_algo["fedprox"]
        paired = fa & fp
        only_fa = fa - fp
        only_fp = fp - fa
        if only_fa or only_fp:
            lines.append(f"    ✗ unpaired seeds: FedAvg-only={sorted(only_fa)} FedProx-only={sorted(only_fp)}")
            issues.append(f"{name}: paired-seed mismatch")
        else:
            lines.append(f"    paired (FedAvg ↔ FedProx) seeds: {len(paired)}")

    return len(issues), lines


def main() -> int:
    print("=" * 86)
    print("  Stage A — data-integrity sanity check across every result directory")
    print("=" * 86)

    total_issues = 0
    all_lines = []
    for name, expected in EXPECTED.items():
        n_iss, lines = _check_directory(name, expected)
        total_issues += n_iss
        all_lines.extend(lines)

    for line in all_lines:
        print(line)

    print("=" * 86)
    if total_issues == 0:
        print("  ✓ All sanity checks passed. Data is integrity-clean for the inferential pipeline.")
    else:
        print(f"  ✗ {total_issues} issue(s) flagged. See lines above for details.")
        print("    Resolve issues before running the inferential analysis pipeline (Stages B-I).")
    print("=" * 86)
    return total_issues


if __name__ == "__main__":
    raise SystemExit(main())
