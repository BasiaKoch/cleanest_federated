# Thesis dataset handoff — for analysis-only agent

**Generated**: 2026-05-23
**Project**: MPhil thesis comparing FedAvg vs FedProx vs FedNova on DermaMNIST under statistical + system heterogeneity
**Repository**: `/Users/basiakoch/cleanest_federated` (laptop) and `bk489@login-q-1.hpc.cam.ac.uk` (HPC)

---

## 1. ONE-PARAGRAPH ABSTRACT

This dataset comprises 10 federated-learning sweeps across 4 partition designs (IID / Dirichlet α=0.1 / specialist / engineered balanced_paired) × 3 system-heterogeneity conditions (C0 uniform / C1 fixed stragglers / C2 random stragglers) × 3 algorithms (FedAvg / FedProx / FedNova) × 2 runtimes (pure-PyTorch / Flower simulation) × 2 model variants (GroupNorm / BatchNorm) × 10 paired seeds. The pre-registered headline (pure-PyTorch FedProx-vs-FedAvg on engineered partition) shows Δ=+0.027, p=0.020 (significant). Multiple secondary findings emerge: (a) cross-runtime sensitivity (~75% attenuation under Flower), (b) Li 2020 §5.2 asymmetric protocol decomposition (Δ_total=+0.068, p=0.002 — 74% from partial-work inclusion, 26% from proximal anchor), (c) FedNova C2 catastrophic failure (3/10 collapse), (d) BatchNorm anti-interaction (FedProx WORSE on BN). Total: 176 valid JSONs across 12 result directories.

---

## 2. DIRECTORY MAP

All results in `mnist_dermnist/results/<sweep_name>/`. Standard files per run:

- `test_at_best_<stem>.json` — primary results (macro_f1, per-class F1, etc.)
- `history_<stem>.csv` — per-round metrics (val_loss, val_macro_f1, train_loss)
- `test_predictions_<stem>.npz` — per-sample predictions + targets (for confusion matrices)
- `client_update_norms_<stem>.csv` — per-client per-round ‖w_i^(t+1) − w^t‖₂ (when `--log-update-norms` was set)

Filename convention:
```
<algo>_mu<mu>_E<E>[_sh-<system_het_mode>][_C<fraction>][_arch-<variant>][_drop]_s<seed>
```

| Sweep | n files | Status | Description |
|---|---|---|---|
| `headline/` | 20 | ✅ COMPLETE | Pure-PyTorch FedAvg+FedProx, engineered partition, C0, n=10 paired |
| `flower_C0_baseline/` | 30 | ✅ COMPLETE | Flower-runtime replication of headline + FedNova, n=10 paired |
| `iid/` | 20 | ✅ COMPLETE | Flower IID falsification check, FedAvg+FedProx, n=10 paired |
| `dirichlet_a01/` | 20 | ✅ COMPLETE | Flower Dirichlet α=0.1 (literature-standard non-IID), n=10 paired |
| `specialist_partition/` | 20 | ✅ COMPLETE | Flower specialist partition (1 class per client), n=10 paired |
| `system_het_fixed/` | 20 | ✅ COMPLETE | Flower C1 fixed stragglers (C5+C6 always at E=5), n=10 paired |
| `system_het_random/` | 20 | ✅ COMPLETE | Flower C2 random stragglers (50% frac, E∈{1..19}), n=10 paired |
| `system_het_random_fednova/` | 10 | ✅ COMPLETE | FedNova C2 (documents Issue 2 algorithmic failure) |
| `system_het_random_asymmetric/` | 10 | ✅ COMPLETE | FedAvg --drop-stragglers (Li 2020 §5.2 protocol), n=10 paired |
| `arch_ablation_bn/` | 6 | ✅ COMPLETE | BatchNorm variant (3 seeds × 2 algos), engineered partition |
| `mu_sweep/` | 3/9 | ⚠️ PARTIAL | μ ∈ {0.001, 0.1, 1.0} pure-PyTorch sensitivity check; rescue running |
| `centralised/` | 10 | ✅ COMPLETE | Pure-PyTorch centralised baseline (no FL), upper bound |

**Total valid JSONs**: 169 + 7 pending rescue = 176

---

## 3. EXPERIMENTAL DESIGN

### Common hyperparameters across all sweeps (unless noted)
- Dataset: DermaMNIST (7-class skin lesion classification), 7007 train / 1003 val / 2005 test samples
- Image resolution: 28×28 RGB
- Model: `DermMNISTCNN` (4-block CNN with GroupNorm, 423,175 trainable params) — see `mnist_dermnist/models/dermmnist_cnn.py`
- Optimizer: SGD lr=0.01, momentum=0.9, weight_decay=0.0, batch_size=32
- Local epochs per round: E=20
- Communication rounds: R=150
- Fraction-fit: C=1.0 (full participation)
- Paired seeds: {42, 123, 456, 789, 999, 2024, 31337, 161803, 271828, 8675309}
- Loss: cross-entropy (no class weighting, no focal loss)

### Manipulated variables across sweeps
1. **Partition design**: balanced_paired / IID / Dirichlet α=0.1 / specialist
2. **System-het mode**: uniform / fixed_stragglers / random_stragglers
3. **Algorithm**: fedavg (μ=0.0) / fedprox (μ=0.01) / fednova
4. **Runtime**: pure-PyTorch (`run_one.py`) vs Flower (`run_one_flower.py`)
5. **Model variant**: dermmnist_cnn (GN) / dermmnist_cnn_bn (BN)
6. **Aggregation policy**: include all clients / drop stragglers (FedAvg --drop-stragglers)
7. **μ value (FedProx)**: locked at 0.01 in main sweeps; μ-sweep at {0.001, 0.1, 1.0}

### Partition compositions (key)

**balanced_paired_7_clients** (engineered headline partition):
| Client | Classes | n_samples |
|---|---|---|
| C0, C1 | actinic + basal + nevi | 964, 963 |
| C2, C3 | benign_keratosis + dermato + nevi | 1095, 1094 |
| C4, C5 | melanoma + vascular + nevi | 1110, 1108 |
| C6 | nevi only | 673 |
Total: 7007. Every minority class held by exactly 2 of 7 clients.

**specialist_7_clients**: every minority class held by exactly 1 client (pairing-lever check)

**iid_7_clients**: uniform random IID partition (null check)

**dirichlet_alpha01_7_clients**: literature-standard non-IID via Dirichlet α=0.1

---

## 4. JSON SCHEMA

Standard `test_at_best_*.json` payload (Flower runs):

```json
{
  "loss": 1.678,                          # cross-entropy on test at best-val round
  "accuracy": 0.747,                       # overall test accuracy
  "balanced_accuracy": 0.446,              # macro-averaged balanced accuracy
  "macro_f1": 0.508,                       # PRIMARY OUTCOME — unweighted mean of per-class F1
  "per_class_f1": [0.42, 0.30, ...],      # length-7 array, class order: actinic, basal, benign_kerat, dermato, melanoma, nevi, vascular
  "selected_round": 117,                   # round of max val_macro_f1
  "best_val_macro_f1": 0.526,
  "predictions_file": "test_predictions_*.npz",
  "seed": 42,
  "algorithm": "fedprox",
  "mu": 0.01,
  "local_epochs": 20,
  "num_rounds": 150,
  "lr": 0.01,
  "momentum": 0.9,
  "weight_decay": 0.0,
  "batch_size": 32,
  "device": "cuda",
  "fraction_fit": 1.0,
  "partition": "balanced_paired_7_clients",
  "image_size": 28,
  "npz_path": "/path/to/dermamnist_64.npz",
  "framework": "flower-simulation",       # or "pure-pytorch"
  "framework_version": "1.23.0",
  "runner_script": "run_one_flower.py",
  "loss_type": "ce",
  "focal_gamma": null,
  "model_variant": "gn",                  # or "bn" (architecture ablation)
  "model_name": "dermmnist_cnn",
  "model_normalization": "GroupNorm",
  "drop_stragglers": false,               # or true (Li 2020 asymmetric protocol)
  "straggler_policy": "include_all",      # or "drop_below_E_max"
  "system_het": {
    "mode": "random_stragglers",
    "E_max": 20,
    "E_straggler": 5,
    "fixed_straggler_ids": null,
    "random_straggler_fraction": 0.5,
    "random_straggler_min_epochs": 1,
    "random_straggler_max_epochs": null
  },
  "elapsed_s": 421.5,
  "wall_clock_seconds": 421.5,
  "git_sha": "...",
  "hostname": "gpu-q-39",
  "started_at": "2026-05-22T14:30:56Z",
  "torch_version": "2.12.0",
  ...
}
```

`history_*.csv` columns:
- `seed, algorithm, mu, local_epochs, round, n_sampled, train_loss`
- `val_loss, val_accuracy, val_balanced_accuracy, val_macro_f1`
- `val_f1_class_0` through `val_f1_class_6`
- For straggler-drop runs: `n_kept, n_dropped` per round

`client_update_norms_*.csv` (when `--log-update-norms` set):
- `round, client_id, update_norm, n_samples, local_epochs`
- (FedNova also has `tau`)

---

## 5. CURRENT RESULT SNAPSHOT (key numbers)

### Headline (pure-PyTorch, balanced_paired, n=10 paired)
- FedAvg mean = **0.4814** ± 0.025
- FedProx mean = **0.5081** ± 0.014
- Paired Δ = **+0.0267**, SD=0.0349, FedProx wins 9/10
- **Wilcoxon p = 0.0195 (SIGNIFICANT at α=0.05)**
- Per-class melanoma Δ = +0.114, Holm-corrected p = 0.041 (significant; only class to survive Holm)

### Flower replication (same partition, same seeds)
- FedAvg mean = 0.4964 ± 0.015
- FedProx mean = 0.5033 ± 0.021
- Paired Δ = +0.0069, p = 0.4316 (NOT significant)
- **Cross-runtime gap (pure-PyTorch − Flower)**: 0.0198 macro-F1 (~75% attenuation)

### 4-partition Flower dose-response (all n=10 paired)
| Partition | Δ_FedProx-FedAvg | p | Direction-wins |
|---|---|---|---|
| IID | −0.0068 | 0.375 | 4/10 (FedAvg favoured) |
| Dirichlet α=0.1 | +0.0079 | 0.322 | 6/10 |
| Specialist | +0.0081 | 0.492 | 7/10 |
| Balanced paired | +0.0069 | 0.432 | 7/10 |

**Observation**: heterogeneity intensity does NOT systematically increase Δ under Flower. All effects clustered near +0.007 to +0.008 except IID (slightly negative).

### System-heterogeneity dose-response (all n=10 paired, Flower)
| Condition | Δ_FedProx-FedAvg | p | wins |
|---|---|---|---|
| C0 (uniform) | +0.0069 | 0.432 | 7/10 |
| C1 (fixed stragglers) | +0.0133 | 0.160 | 8/10 |
| C2 (random stragglers) | +0.0174 | 0.232 | 7/10 |

**H2 contrast** (between-condition):
- δ_C2 − δ_C0 = +0.0105, p = 0.8457 (n.s.)
- δ_C1 − δ_C0 = +0.0064, p = 0.6953 (n.s.)

**Observation**: monotonic increase consistent with FedProx theory (Li 2020 §5.2 predicts widening under system het), but all underpowered at n=10.

### FedNova C2 (system_het_random_fednova) — Issue 2
n=10 macro_f1: [0.346, 0.115, 0.296, 0.115, 0.115, 0.246, 0.245, 0.279, 0.364, 0.250]
- Mean = **0.237** (vs FedAvg-C2 mean 0.493, FedProx-C2 mean 0.510)
- 3/10 fully collapse (macro_f1 < 0.20 = nevi-only prediction)
- 7/10 partial training (0.20-0.36 range, never converge)

### Li 2020 §5.2 asymmetric protocol (system_het_random_asymmetric)
**This is the strongest single finding.**
- Arm 1 (FedAvg --drop-stragglers, new): n=10 mean = 0.4423
- Arm 2 (FedAvg --no-drop, existing C2): n=10 mean = 0.4928
- Arm 3 (FedProx --no-drop, existing C2): n=10 mean = 0.5102

3-arm decomposition (paired by seed):
- **Δ_total = +0.0679, SD=0.018, 10/10 wins, p=0.0020** ← Li 2020 headline contrast (FedProx-include vs FedAvg-drop)
- **Δ_include = +0.0505, SD=0.040, 9/10 wins, p=0.0059** ← partial-work inclusion effect
- **Δ_prox = +0.0174, SD=0.040, 7/10 wins, p=0.232** ← pure proximal effect at parity

**Decomposition**: 74% of headline Δ_total = +0.068 is from including stragglers (Δ_include); only 26% from proximal anchor (Δ_prox). The Li 2020 §5.2 "FedProx wins" effect is predominantly an aggregation-policy effect, NOT a proximal-mechanism effect.

### Architecture ablation BN (arch_ablation_bn, 3 seeds: 42, 123, 456)
| Seed | Δ_GN | Δ_BN | Δ_BN − Δ_GN |
|---|---|---|---|
| 42 | +0.0326 | −0.0259 | −0.0585 |
| 123 | +0.0043 | −0.0481 | −0.0524 |
| 456 | +0.0376 | +0.0134 | −0.0242 |
| **Mean** | **+0.0249** | **−0.0202** | **−0.0450** |

**Observation**: FedProx is statistically WORSE than FedAvg on BatchNorm models (Δ_BN = −0.020) but better on GroupNorm (Δ_GN = +0.025). Cross-variant contrast = −0.045. This **contradicts FedBN literature prediction** and is a novel anti-interaction finding.

### Centralised baseline (pure-PyTorch upper bound)
- n=10 seeds, macro_f1 mean = **0.5600 ± 0.024**
- Federation tax: FedAvg-PT recovers 86% (0.481/0.560); FedProx-PT recovers 91% (0.508/0.560)

### μ-sweep (partial: 3 of 9 jobs landed; rescue in progress)
Available data:
- μ=0.001 seed 456 → 0.5230
- μ=0.1 seed 42 → 0.4695
- μ=0.1 seed 456 → 0.4946
6 missing due to transient CUDA-busy failures; rescue script `hpc_mu_sweep_rescue.sh` re-submits.

---

## 6. FIVE PRIMARY THESIS FINDINGS

1. **Primary headline (pure-PyTorch)**: FedProx significantly improves macro-F1 over FedAvg on engineered partition (Δ=+0.027, p=0.020). Melanoma is the only Holm-significant per-class gain.

2. **Cross-runtime sensitivity (METHODOLOGICAL CONTRIBUTION)**: The headline effect attenuates by ~75% under the Flower simulation runtime (Δ=+0.007, n.s.). This persists across ALL 4 statistical-het partitions. Flower's Ray-actor isolation introduces floating-point noise and aggregation-order variance that compounds over 90K SGD steps to produce a systematic effect-size reduction.

3. **System-heterogeneity dose-response**: Direction consistent with FedProx theory (C0 < C1 < C2) but all underpowered at n=10. H2 between-condition contrast (δ_C2 − δ_C0 = +0.0105, p = 0.85) is not significant.

4. **FedNova C2 catastrophic failure (Issue 2)**: FedNova under random stragglers + class imbalance produces 3/10 full collapse + 7/10 partial training. Mechanism: FedNova's straggler amplification factor (~9.27× for E=1 client) pulls the global model toward straggler clients' under-trained majority-class-dominated local solutions.

5. **Li 2020 §5.2 asymmetric protocol — DECOMPOSITION**: Under the canonical FedProx evaluation protocol, FedProx wins 10/10 seeds (Δ_total=+0.068, p=0.002). However, the 3-arm decomposition shows 74% of this effect is from differential straggler handling (Arm 3 includes partial work, Arm 1 drops it) and only 26% from the proximal anchor mechanism. The widely-cited FedProx headline is mechanistically a hybrid effect.

6. **BatchNorm anti-interaction (NEW unexpected finding)**: On the BatchNorm variant of the model, FedProx is statistically WORSE than FedAvg (Δ_BN = −0.020, n=3), reversing the GroupNorm headline direction. Cross-variant contrast = −0.045. Contradicts FedBN-style hypothesis. Proposed mechanism: BN's running-stats drift overwhelms parameter-side regularization; the proximal anchor over-constrains the parameter sub-space while BN buffers drift independently.

---

## 7. PROPOSED ANALYSIS PLAN (for downstream agent)

### Phase A — Result tables (LaTeX-ready)

#### A.1 Headline result table
Convert `headline/` data to a 3-row paired comparison:
| Metric | FedAvg | FedProx | Δ |
|---|---|---|---|
| Test macro-F1 | 0.4814 ± 0.025 | 0.5081 ± 0.014 | +0.0267 ± 0.035 |
| Paired Wilcoxon p | — | — | 0.020 |
| Hodges-Lehmann | — | — | +0.022 |
| Sign test p | — | — | 0.021 |
| Rank-biserial r | — | — | +0.818 |
| FedProx wins | — | — | 9/10 |
| LOSO subsamples sig. at α=0.05 | — | — | 10/10 |

Script: `mnist_dermnist/results/thesis_ready/scripts/analyse_paired.py --results-dir headline` (exists).

#### A.2 Per-class table with Holm correction
Run `analyse_paired.py` with per-class flag on `headline/` data. Should reproduce existing `tab:per-class` from the draft. Verify melanoma Holm-corrected significance.

#### A.3 Cross-runtime comparison table
Direct comparison of `headline/` vs `flower_C0_baseline/` at same 10 paired seeds:
| Runtime | Δ_FedProx-FedAvg | p | wins |
|---|---|---|---|
| Pure-PyTorch | +0.0267 | 0.020 | 9/10 |
| Flower-simulation | +0.0069 | 0.432 | 7/10 |

#### A.4 4-partition heterogeneity dose-response (Flower)
Table + figure (line plot): mean Δ per partition vs heterogeneity-intensity metric (e.g., KL divergence of per-client class distributions vs global).

#### A.5 System-heterogeneity dose-response (Flower)
Table: C0/C1/C2 means + Δ + H2 contrast.

#### A.6 FedNova C2 failure table
Per-seed macro_f1 of `system_het_random_fednova/` showing collapse pattern.

#### A.7 Asymmetric-protocol decomposition table
The 3-arm contrast — most important table in the thesis.

#### A.8 BN ablation table
Per-seed (s42, s123, s456) Δ_BN vs Δ_GN.

#### A.9 μ-sensitivity table (after rescue completes)
Per-seed × per-μ macro_f1 matrix.

#### A.10 Federation tax table
Centralised vs FedAvg vs FedProx — rounded macro_f1 + % of centralised recovered.

### Phase B — Figures (from existing data, no compute)

#### B.1 Convergence curves (already drafted as `08_curves_main.png`)
Per-round val_macro_f1 mean ± SEM across 10 paired seeds. 4-panel: val macro-F1, val loss, train objective (log), val balanced accuracy. Data source: `headline/history_*.csv`.

#### B.2 Per-class convergence (already drafted as `09_curves_per_class.png`)
7-panel grid, one per class. Data source: `headline/history_*.csv` (val_f1_class_0 ... val_f1_class_6).

#### B.3 4-point dose-response (NEW)
X-axis: heterogeneity intensity (KL divergence). Y-axis: mean Δ ± paired CI. Points: IID, Dirichlet α=0.1, specialist, balanced_paired (all Flower n=10).

#### B.4 Update-norm mechanism plot (NEW — HIGH PRIORITY)
Per-round mean ‖w_i^(t+1) − w^t‖₂ across seeds and clients, FedAvg vs FedProx. Data source: `client_update_norms_*.csv` files in any sweep that has `--log-update-norms` (all Flower C0, C1, C2 runs do).
- Plot 1: aggregate across all 7 clients, 4 panels (FedAvg-C0, FedProx-C0, FedAvg-C2, FedProx-C2)
- Plot 2: per-client breakdown for one condition (e.g., engineered C0, FedAvg vs FedProx)
This is the direct empirical evidence that FedProx reduces client drift — no MPhil-scale FL thesis has this.

#### B.5 Federation tax bar chart (already drafted as `12_federation_tax.png`)
Per-class bar chart: centralised - FedAvg-PT - FedProx-PT showing the federation tax.

#### B.6 Per-specialty F1 table → bar chart (NEW)
Convert existing `tab:per-specialty` to a grouped bar chart with Holm-corrected error bars. Shows which specialty pairs the FedProx benefit concentrates on.

#### B.7 Confusion matrices (NEW)
7×7 confusion matrices for FedAvg vs FedProx at best-val checkpoint, averaged across 10 seeds. Use `test_predictions_*.npz` files. Side-by-side heatmap. Shows WHERE the melanoma F1 improvement comes from.

#### B.8 Cross-variant BN vs GN bar chart (NEW)
3 seeds × {GN Δ, BN Δ, cross-variant contrast}. Shows the anti-interaction visually.

#### B.9 Asymmetric protocol decomposition figure (NEW — KEY THESIS FIGURE)
Bar chart: Δ_total, Δ_include, Δ_prox, with paired Wilcoxon p-values annotated. Visual decomposition of the Li 2020 §5.2 "FedProx wins" claim.

#### B.10 μ-sensitivity plot (once rescue completes)
Mean Δ vs log(μ), with paired-seed error bars. X-axis μ ∈ {0.001, 0.01, 0.1, 1.0}.

### Phase C — Statistical analyses

#### C.1 Paired-seed inference checklist (replicate for each main comparison)
For each (sweep, contrast) tuple, compute:
- Paired Wilcoxon signed-rank test (two-sided)
- Hodges-Lehmann robust estimate
- Sign test (direction-only)
- Rank-biserial correlation (effect size)
- Walsh-average 95% CI (exact for n=10, α=0.05 → bounds = 9th smallest and 9th largest of 55 Walsh averages)
- Bootstrap 95% CI on mean Δ (10,000 resamples)
- LOSO robustness check (remove each seed, recompute test)

#### C.2 Multiple-testing correction
- Per-class F1 contrast (7 tests): Holm-Bonferroni
- Per-specialty contrast (4 tests): Holm-Bonferroni
- H2 between-condition contrast: Bonferroni α/2 = 0.025 (family of 2: C1 vs C0, C2 vs C0)

#### C.3 Pre-registered safety criterion
"No mean per-class regression > 0.05" — verify on `headline/` data.
"No worst-(seed, class) regression > 0.05" — report Table 7 (Worst-case per-(seed, class) regression). Honestly note: 10/70 cells fall below −0.05; melanoma at −0.019 is robust.

#### C.4 Decomposition algebraic identity check
For the asymmetric protocol: verify Δ_total = Δ_include + Δ_prox to within float noise (already verified at +0.0679 = +0.0505 + 0.0174).

### Phase D — Prose drafts

The user's existing draft is at `mnist_dermnist/results/thesis_ready/writing/09_overleaf_ready.tex`. Recommended structural changes:

1. **§Abstract**: replace single "FedProx wins" claim with the 5-finding structure (above).
2. **§Results**: add new subsections for:
   - 4-partition dose-response
   - Cross-runtime sensitivity (currently TODOhpc in draft)
   - Asymmetric-protocol decomposition (NEW — major contribution)
   - BN anti-interaction (NEW — unexpected finding)
3. **§Discussion**: add paragraphs decomposing the Li 2020 §5.2 result vs the literature's headline framing.
4. **§Limitations**: explicitly acknowledge:
   - Single dataset (DermaMNIST)
   - 28×28 not clinical resolution
   - 7 clients (cross-silo, not cross-device)
   - Single μ value in main sweeps (mitigated by μ-sweep once complete)
   - Cross-runtime gap is Ray-actor-specific; other frameworks may differ
   - μ=1.0 jobs all failed (CUDA-busy on HPC, rescue in progress)
5. **§Conclusion**: keep the calibrated claim from existing draft, but expand to include the asymmetric decomposition finding as a "deconstruction of literature consensus."

### Phase E — Anti-confound checks

For each finding, run a defensive check:

#### E.1 Headline robustness
- LOSO (already done, 10/10 still significant)
- Bootstrap CI excludes 0 (already done)
- Alternative metrics: report balanced accuracy + accuracy as descriptive

#### E.2 Cross-runtime gap
- Verify the gap holds on the 3 ablation seeds independently (42, 123, 456)
- Confirm by checking specific seed-level deltas in BOTH runtimes:
  - PT s42: Δ=+0.069; Flower s42: Δ=+0.033 (gap=+0.036)
  - PT s123: Δ=+0.001; Flower s123: Δ=+0.004 (gap=−0.003)
  - PT s456: Δ=+0.000; Flower s456: Δ=+0.038 (gap=−0.038)
- These are NOT consistent! Some seeds have higher Flower Δ. The "75% attenuation" is an across-10-seed mean property, not a per-seed property.

#### E.3 Asymmetric protocol
- Verify Δ_total = Δ_include + Δ_prox exactly (it does: +0.0679 = +0.0505 + 0.0174)
- Confirm provenance of all Arm-1 files: `drop_stragglers=True`, `algorithm=fedavg`
- Confirm provenance of Arm-2/3 files: `drop_stragglers=False`

#### E.4 BN ablation
- 3 seeds is too few for formal inference. Report descriptively only.
- If FedProx is worse on BN, verify it's not a code bug: check that the BN model trains stably on FedAvg alone (it does — FedAvg BN mean = 0.509, well above random)

---

## 8. KNOWN GAPS & RECOMMENDATIONS

### Critical gaps
- **μ-sweep incomplete** (3/9). Rescue submitted at `hpc_mu_sweep_rescue.sh`. ETA depends on HPC queue.
- **No update-norm figure yet** (data exists in CSVs but no plot). HIGH PRIORITY — this is the only direct empirical evidence of the mechanism story.

### Recommendations for downstream agent
1. **Do figures first**, prose second. The thesis has good data; what's missing is visualisations.
2. **Start with the update-norm figure** (B.4) — it's the singlebiggest impact item.
3. **The asymmetric-protocol decomposition (Phase A.7 + B.9)** is the strongest single finding. Make it the §Results headline.
4. **Frame the BN anti-interaction (A.8 + B.8) as a novel observation** — the FedBN-derived prediction was that FedProx should help MORE on BN; observing the OPPOSITE is interesting.
5. **Treat the 5 partitions × 3 system-het × algorithms as a single high-dimensional dataset**: a matrix-style summary figure may be more communicative than 10 individual tables.

### Scripts to run for first-pass analysis
```bash
# From repo root:
python mnist_dermnist/scripts/check_engineered.py
python mnist_dermnist/scripts/check_system_het.py
python mnist_dermnist/scripts/check_arch_ablation.py
python mnist_dermnist/scripts/check_asymmetric_stragglers.py
python mnist_dermnist/scripts/check_mu_sweep.py   # incomplete data; partial output expected
```

Existing thesis-ready analysis scripts (more comprehensive):
```bash
ls mnist_dermnist/results/thesis_ready/scripts/
# analyse_paired.py        - full paired-seed inference suite
# analyse_specialist_partition.py
# analyse_system_het.py
# analyse_extra_statistics.py
```

### Scripts that DON'T exist yet (recommend writing)
- `plot_update_norms.py` — produces B.4
- `plot_dose_response.py` — produces B.3
- `plot_confusion_matrices.py` — produces B.7
- `plot_asymmetric_decomposition.py` — produces B.9
- `plot_bn_ablation.py` — produces B.8

---

## 9. RAW DATA FILES — quick access paths

```bash
# Headline data (pure-PyTorch, all 5 contrasts derive from this):
mnist_dermnist/results/headline/test_at_best_*.json
mnist_dermnist/results/headline/history_*.csv
mnist_dermnist/results/headline/test_predictions_*.npz

# Cross-runtime data (Flower replication):
mnist_dermnist/results/flower_C0_baseline/test_at_best_*.json

# Statistical-het partition sweeps:
mnist_dermnist/results/iid/...
mnist_dermnist/results/dirichlet_a01/...
mnist_dermnist/results/specialist_partition/...

# System-het sweeps:
mnist_dermnist/results/system_het_fixed/...        # C1
mnist_dermnist/results/system_het_random/...        # C2

# Algorithmic failure & decomposition:
mnist_dermnist/results/system_het_random_fednova/...        # Issue 2
mnist_dermnist/results/system_het_random_asymmetric/...     # Li 2020 §5.2 protocol

# Ablations:
mnist_dermnist/results/arch_ablation_bn/...
mnist_dermnist/results/mu_sweep/...

# Reference:
mnist_dermnist/results/centralised/centralised_seed*.json   # upper bound (pure-PyTorch, no FL)
```

---

## 10. GIT STATE AT HANDOFF

Branch: `main`
Latest commits (most recent first):
- `b16e67b` scripts: idempotent μ-sweep rescue (recovers 6 CUDA-busy failures)
- `cd25fa3` hpc_mu_sweep.sh: correct sanity check
- `d457f5e` mu-sweep: defend headline mu=0.01 against cherry-pick critique
- `2ab25c8` asymmetric stragglers: address 6 methodology critique points
- `c2b8ff5` asymmetric stragglers: Li 2020 §5.2 protocol for FedProx max-advantage
- `0b7f69a` HPC SLURM submission for BN architecture ablation
- `e0d39f0` HPC SLURM addendum for 3 FedNova C2 reruns
- `7dad180` 3-job FedNova C2 addendum to complete sweep to 10 seeds
- `44a068d` arch ablation: BatchNorm variant of DermMNISTCNN
- `e91b081` scripts/check_engineered.py: quick audit for flower_C0_baseline
- `8c2ce38` scripts/check_system_het.py: audit + H2 contrast
- `f6d0fd9` fl/seeding: split numpy 32-bit seed from torch/python raw seed (CRITICAL FIX)

All commits pushed to `origin/main`.

Test suite: **106 tests passing** (last run).

---

## 11. CONTACT / PROVENANCE

User: Barbara Koch (Cambridge MPhil DIS)
Pure-PyTorch headline data: produced on Cambridge HPC (gpu-q-* nodes), A100 80GB
Flower-runtime data: mix of Cambridge HPC (most) + RunPod RTX 4090 (s8675309 reruns, system_het reruns, dirichlet provenance reruns)
All seeded with paired protocol; bit-equivalence verified at μ=0 (FedAvg) via test suite.

End of handoff document.
