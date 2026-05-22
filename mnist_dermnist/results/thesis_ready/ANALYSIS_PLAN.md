# Analysis plan — revision 2 (2026-05-22)

This document is the canonical analysis playbook for the DermaMNIST FL
thesis. It supersedes the informal plan in the chat log; revision 2
incorporates an expert critique flagging eleven issues with the prior
plan, each addressed below.

The reference revision is committed at SHA `619bf09` (the `results/`
rename-revert + `README_PROVENANCE.md` addition).

---

## 0. Architectural changes vs the prior plan

| Critique | Resolution |
|---|---|
| Rename of `headline/` → `legacy_pure_pytorch/` was **partial + dangerous** (only 2 of 11+ scripts patched) | **Reverted.** `results/headline/README_PROVENANCE.md` (commit `619bf09`) now disambiguates legacy-vs-canonical without renaming. 11+ scripts that hard-code or default to `headline/` keep working. |
| `tables.py` is not a universal analysis engine (mispairs on `mu_sweep`; can't handle one-arm `class_weighted_baseline`) | Dedicated scripts per sweep (Stage D). `tables.py` is restricted to canonical FedAvg/FedProx two-arm directories. |
| Specialist partition omitted from "what each sh job calculates" | Promoted to first-class status (Stage C.3). |
| Equivalence script reads wrong directories; tolerance loose ("≥ 8/10") | Re-pointed at `flower_C0_baseline/`; pre-defined tolerance `|Δ| ≤ 0.02` on **every** paired (algo, seed) cell. |
| Stage A integrity check too shallow (counts + macro-F1 bounds only) | Extended to file-pair completeness, history integrity, schema invariants, directory-level invariants, SLURM log grep + sacct audit. |
| IID falsification phrased as `p > 0.05 ⇒ Δ ≈ 0` (statistically weak) | Re-framed as **equivalence test** with pre-declared smallest effect of interest (`|Δ| = 0.01`) + CI bounds. Language: "no evidence of a FedProx advantage under IID." |
| Dose-response uses partition NAMES not measured heterogeneity | New script `compute_heterogeneity_score.py`: JS divergence, label support, quantity-skew CV. Plot Δ vs measured-heterogeneity-score, not categorical labels. |
| Mechanism evidence (raw update norms) is tautological — FedProx's objective directly penalises distance | New script `analyse_update_norms.py` stratifies by (client, round, τ, condition); correlates norm reduction with per-class F1 gain. |
| Federation-tax step would crash on `results/centralised/` (different filename schema) | New script `analyse_federation_tax.py` handles `centralised_seed*.json` directly. Explicitly handles n=3 vs n=10 mismatch. |
| rsync plan contradicts itself ("don't sync raw JSONs" + `--include='*.json'`) | Corrected to include raw JSONs, NPZ predictions, SLURM logs, and a sacct audit export. |
| No confirmatory / exploratory boundary — garden-of-forking-paths risk | All tests tagged below (PRIMARY / SECONDARY / EXPLORATORY). Significance claims restricted to Tier 1. |

---

## 1. Confirmatory hierarchy

The thesis defends two **primary** hypotheses. Everything else is
reported as effect size + CI without claiming significance.

| Tier | Hypothesis | Pre-registered? | α control |
|---|---|---|---|
| **PRIMARY P1** | $\Delta_{\mathrm{Flower\ C0}} > 0$ on `balanced_paired_7_clients` (H1) | Yes (§Hypotheses pre-2026-05-18) | Single test, α = 0.05 two-sided |
| **PRIMARY P2** | $\Delta_{C_2} - \Delta_{C_0} > 0$ (H2 system-het amplification) | Yes (§Hypotheses) | Bonferroni α/2 = 0.025 over {C1, C2}; C1 descriptive-only |
| SECONDARY S1 | IID null-mechanism equivalence | Yes (§Hypotheses falsification block) | Equivalence-style with CI vs SES |
| SECONDARY S2 | Dirichlet external validity ($\Delta > 0$, smaller magnitude) | Yes (§Hypotheses external-validity block) | Two-sided Wilcoxon, magnitude descriptive |
| SECONDARY S3 | Specialist defence (4 pre-written outcomes) | Yes (SHA `9f2bb94`, 2026-05-21) | One outcome selected from data |
| SECONDARY S4 | FedNova vs FedAvg/FedProx at C2 (three-way) | No formal pre-reg | Descriptive |
| SECONDARY S5 | Cross-runtime equivalence (Flower vs pure-PyTorch) | No formal pre-reg | Max \|Δ\|, no p-value |
| EXPLORATORY E1 | Per-class breakdown + Holm | No | Holm-corrected α = 0.05 within family; no over-claiming |
| EXPLORATORY E2 | μ sensitivity curve | No | Descriptive — best operating point |
| EXPLORATORY E3 | E dose-response | No | Descriptive — predicted monotone |
| EXPLORATORY E4 | Class-weighted CE ablation | No | Three pairwise comparisons, descriptive |
| EXPLORATORY E5 | Federation-tax magnitude | No | Descriptive |
| EXPLORATORY E6 | Update-norm mechanism evidence | No | Stratified comparison, descriptive |

**Claim of significance applies ONLY to P1 and P2.** Everything else is
reported as effect sizes, CIs, and descriptive comparisons — no
"p < 0.05 therefore..." language outside Tier 1.

---

## 2. Stage A — deep integrity check (replaces v1's shallow checker)

The existing `thesis_ready/scripts/sanity_check_results.py` checks file
counts + macro-F1 bounds. Revision 2 extends it to also verify:

**File-pair completeness**
- Every `test_at_best_*.json` has a matching `history_*.csv` (same stem)
- Every `--log-update-norms` run has a matching `client_update_norms_*.csv`
- Every Flower / FedNova run has a matching `test_predictions_*.npz`

**History integrity**
- Every `history_*.csv` has exactly 150 rows (R=150) except in `e_sweep/`
  where row count varies with $E$
- `selected_round` from the JSON exists in `history.round`
- `best_val_macro_f1` from the JSON equals `max(history.val_macro_f1)`
  within `1e-6` tolerance

**Schema invariants**
- `per_class_f1` is a list of length 7 in every JSON
- No metric is NaN or inf
- No duplicate `(algo, mu, E, sh_mode, C, seed)` stems within a directory

**Directory-level invariants**
- `loss_type == "ce"` in every JSON in `flower_C0_baseline/`
- `loss_type == "class_weighted_ce"` in every JSON in `class_weighted_baseline/`
- `system_het.mode != "uniform"` in every JSON in `system_het_*/`
- `system_het.mode == "fixed_stragglers"` in `system_het_fixed/`
- `system_het.mode == "random_stragglers"` in `system_het_random*/`

**SLURM audit trail**
- Grep `mnist_dermnist/logs/*.{out,err}` for: `Traceback`, `ERROR`,
  `CANCELLED`, `TIMEOUT`
- `sacct -X` for non-zero exit codes within the last 7 days
- Cross-reference exit-code-0 jobs against JSONs on disk: every `COMPLETED`
  job should have a corresponding JSON

**Provenance** (already in v1, retained)
- `git_commit, hostname, run_started_at, run_finished_at, framework,
  runner_script, python_version, torch_version` populated on every
  canonical-runtime JSON

Output: per-violation log line + summary table + overall exit code =
number of violations.

---

## 3. Stage B — confirmatory tests (PRIMARY)

### B.1 — P1 headline (Flower C0 baseline)

```bash
# Wilcoxon + rank-biserial
PYTHONPATH=. python -m mnist_dermnist.analysis.tables \
    --results-dir mnist_dermnist/results/flower_C0_baseline

# Sign test + Hodges-Lehmann + LOSO + per-class Holm
PYTHONPATH=. python mnist_dermnist/results/thesis_ready/scripts/analyse_extra_statistics.py \
    --results-dir mnist_dermnist/results/flower_C0_baseline
```

Both required. `tables.py` alone is insufficient (gives only Wilcoxon
+ rank-biserial); `analyse_extra_statistics.py` adds the rest.

### B.2 — P2 system-het amplification

```bash
PYTHONPATH=. python mnist_dermnist/results/thesis_ready_system_het/scripts/analyse_system_het.py
```

Already does the Bonferroni correction on the {C1, C2} family.

---

## 4. Stage C — secondary tests

### C.1 — IID equivalence (re-framed)

**NEW SCRIPT** `analyse_iid_equivalence.py` (to write).

Pre-declared smallest effect size of interest:
$|\Delta_{\mathrm{SES}}| = 0.01$ macro-F1 (smaller than the headline's
0.027 by a factor of ≥ 2.5).

Outputs:
- Compute 95 % CI for $\Delta$ via Walsh-average inversion (same exact-CI
  technique as `analyse_specialist_partition.py`)
- If CI ⊂ $[-0.01, +0.01]$ → **equivalent within SES** (mechanism story
  intact)
- If CI overlaps 0 but extends beyond $\pm 0.01$ → **inconclusive**,
  reported as such (not "Δ ≈ 0")
- If CI excludes 0 → **unexpected FedProx advantage under IID**,
  falsifies mechanism

Language for the thesis: *"On IID, we found no evidence of a FedProx
advantage (95 % CI for Δ: [a, b], smallest effect of interest:
±0.01)."* Never *"$p > 0.05$ therefore Δ = 0."*

### C.2 — Dirichlet external validity

```bash
PYTHONPATH=. python -m mnist_dermnist.analysis.tables \
    --results-dir mnist_dermnist/results/dirichlet_a01
PYTHONPATH=. python mnist_dermnist/results/thesis_ready/scripts/analyse_extra_statistics.py \
    --results-dir mnist_dermnist/results/dirichlet_a01
```

Magnitude comparison against the Flower C0 headline is descriptive
(no formal test): expected $|\Delta_{\mathrm{Dir}}| < |\Delta_{\mathrm{paired}}|$.

### C.3 — Specialist defence

```bash
PYTHONPATH=. python mnist_dermnist/results/thesis_ready/scripts/analyse_specialist_partition.py
```

Picks one of 4 pre-written outcomes from
`writing/specialist_partition_scenarios.tex` based on observed Δ.

### C.4 — Cross-runtime equivalence (NOT load-bearing)

**EXISTING SCRIPT TO PATCH**
`mnist_dermnist/experiments/compare_equivalence_full_scale.py`.

Patch: re-point Flower side from `headline_flower_verify/` (the
deprecated 2-seed sweep, which was cancelled) to `flower_C0_baseline/`
(10-seed Flower headline). Tolerance: $|\Delta| \leq 0.02$ on **every**
paired (algo, seed) cell. Not "≥ 8/10" — every cell.

If any cell fails: report it plainly. The thesis cites Flower as
canonical regardless; the equivalence claim is historical reassurance,
not load-bearing.

---

## 5. Stage D — dedicated analyses (replace `tables.py` misuse)

### D.1 — `analyse_mu_sweep.py` (NEW)

`mu_sweep/` has multiple FedProx settings per seed; `tables.py` would
mispair. The new script:

- Indexes by `(seed, mu)`
- For each $\mu \in \{0.001, 0.01, 0.1, 0.5, 1.0\}$: paired Wilcoxon
  vs FedAvg(μ=0) sanity baselines at matching seeds; HL + exact CI
- Sanity check: FedProx(μ=0.0) baselines must match `flower_C0_baseline/`
  FedAvg JSONs at matching seeds within the runtime-equivalence noise
  floor

Output: CSV with one row per (seed, μ) + per-μ summary JSON.

### D.2 — `analyse_class_weighted_baseline.py` (NEW)

`class_weighted_baseline/` contains only FedAvg + CW-CE (no matched
FedProx arm). Three head-to-heads via reading two directories:

1. FedAvg+CE vs FedAvg+CW-CE: load `headline/` (or `flower_C0_baseline/`)
   FedAvg + `class_weighted_baseline/` FedAvg; paired Wilcoxon
2. FedAvg+CW-CE vs FedProx+CE: cross comparison
3. Differential: $\Delta_{\mathrm{FedProx-FedAvg(CE)}}$
   vs $\Delta_{\mathrm{FedAvg(CW-CE)-FedAvg(CE)}}$

All with paired Wilcoxon + HL + exact CI. Descriptive, no
significance claim.

### D.3 — `analyse_federation_tax.py` (NEW)

`centralised_seed*.json` has a different schema (no `framework`, no
`algorithm`). The script:

- Loads centralised JSONs + matched-seed FedAvg + FedProx from
  `flower_C0_baseline/`
- Per seed: federation tax = `centralised - FedAvg`
- Per seed: FedProx gap closure = `(FedProx - FedAvg) / (centralised - FedAvg)`
- Reports mean ± CI for both
- **Explicitly handles n=3 vs n=10 mismatch**: only seeds present in
  both centralised and Flower arms enter the matched analysis. If
  n_matched < 10, report it.

### D.4 — `analyse_e_sweep.py` (NEW)

`e_sweep/` indexes by `(seed, E)` for $E \in \{1, 5, 10, 20, 40\}$.

- $\Delta = $ FedProx($E$) − FedAvg($E$) at each $(seed, E)$ pair
- Plot Δ vs E (3 points per E, 5 E values)
- Fit a monotone-increasing regression (linear in $\log E$); report
  slope + CI

This is the canonical empirical test of the drift-control mechanism:
the proximal anchor controls drift accumulated over $E$ local steps,
so $\Delta$ should grow with $E$.

### D.5 — `analyse_update_norms.py` (NEW, stratified)

**Critique was right that raw norms are tautological.** The new
analysis reads `client_update_norms_*.csv` from `system_het_random/`
and `system_het_random_fednova/` and stratifies properly:

- Pair FedAvg vs FedProx at matching `(seed, round, client_id, τ)`
- Compute ratio `update_norm_FedProx / update_norm_FedAvg` per cell
- Aggregate by τ-bucket (τ near $E_\max$ vs τ near 1)
- **Correlate norm reduction with per-class F1 gain** on the
  classes each client holds

If FedProx's mechanism is real, the ratio should be < 1 AND the
norm-reduction should correlate positively with per-class F1 gain.
Just showing ratio < 1 alone proves only that the proximal objective
penalises distance from the anchor (tautological).

### D.6 — `compute_heterogeneity_score.py` (NEW, replaces categorical dose-response)

For each (partition, seed) currently in results:

- **Jensen-Shannon divergence** of each client's empirical class
  distribution from the global class prior; report mean and max
  across 7 clients
- **Label support** (count of classes with non-zero samples) per
  client; report mean and min
- **Quantity-skew CV**: $\mathrm{CV}(n_i) = \mathrm{std}(n_i) / \mathrm{mean}(n_i)$
- **Earth-mover-style** mean pairwise distance between client class
  priors

Then the new dose-response figure plots $\Delta$ vs measured-
heterogeneity-score, **one point per (partition, seed)**. The hand-
named "IID → specialist → Dirichlet → paired" ordering becomes
empirically derived, not a categorical claim.

---

## 6. Stage E — figures

Existing figures retained. **Three new figures** from the revised analyses:

- $\Delta$ vs measured-heterogeneity-score (replaces categorical
  dose-response; one point per (partition, seed) with the partition
  type as marker shape)
- $E$ sweep monotonicity plot (Δ vs $E$ with fitted slope + CI)
- $\mu$ sweep curve (Δ vs μ on log scale with error bars)
- Stratified update-norm comparison (FedProx/FedAvg ratio by τ-bucket
  + correlation with per-class gain)

---

## 7. Stage F — sync back to laptop (corrected)

```bash
# CORRECTED: include test_predictions_*.npz + SLURM logs + sacct
rsync -avz --include='*/' \
    --include='*.json' --include='*.csv' --include='*.png' --include='*.pdf' \
    --include='*.npz' --include='*.out' --include='*.err' \
    --exclude='*' \
    bk489@login.hpc.cam.ac.uk:/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/results/ \
    mnist_dermnist/results/

rsync -avz bk489@login.hpc.cam.ac.uk:/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/ \
    mnist_dermnist/logs/

# Export the sacct audit trail
ssh bk489@login.hpc.cam.ac.uk \
    "sacct -X -u bk489 --starttime=$(date -d '-7 days' +%Y-%m-%d) \
       --format=JobID,JobName,State,ExitCode,Elapsed,Start,Reason --parsable2" \
    > mnist_dermnist/logs/sacct_audit.csv
```

---

## 8. Implementation order

When ready to execute, the proposed sequence (each as a separate atomic
commit):

1. **Revised Stage A integrity checker** — extend existing
   `sanity_check_results.py` with the additional checks in §2.
2. **Patch `compare_equivalence_full_scale.py`** — repoint at
   `flower_C0_baseline/`; tighten tolerance to all-cell pass.
3. **`compute_heterogeneity_score.py`** (NEW, foundational — feeds the
   dose-response figure).
4. **Five new dedicated analysers** (any order; independent):
   - `analyse_iid_equivalence.py`
   - `analyse_mu_sweep.py`
   - `analyse_class_weighted_baseline.py`
   - `analyse_federation_tax.py`
   - `analyse_e_sweep.py`
5. **`analyse_update_norms.py`** (depends on `system_het_random*/`
   data being present + complete).
6. **Figure-generation updates** — integrate the three new figures
   into the existing pipeline.

Approximately 8–10 commits total. None of them touches the existing
analysis scripts (every new analyser is additive). The existing
`tables.py`, `plots.py`, `analyse_extra_statistics.py`,
`analyse_per_client.py`, `analyse_best_vs_last.py`,
`analyse_worst_case_per_class.py`, `analyse_communication_metrics.py`,
`plot_per_class_delta.py`, `generate_thesis_figures_10_12.py`,
`analyse_confusion_matrices.py`, and `analyse_system_het.py` continue
to work; the new analysers complement them rather than replacing
them.

---

## 9. References

- Pre-registered hypotheses: `results/thesis_ready/writing/09_overleaf_ready.tex` §Hypotheses + §Specialist-partition pre-registration (SHA `9f2bb94`, 2026-05-21).
- Specialist-partition outcome scenarios: `results/thesis_ready/writing/specialist_partition_scenarios.tex`.
- Provenance layout: `results/headline/README_PROVENANCE.md` (commit `619bf09`).
- Framework normalisation: `mnist_dermnist/fl/provenance.py` + `tests/test_framework_provenance.py`.
- Prior plan revision (deprecated): chat transcript, marked here as superseded.
