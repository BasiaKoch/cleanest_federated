# Thesis Development Plan — Mechanistic FL under Heterogeneity & Class Imbalance (Dermatology)

> **Status (2026-05-31).** Phase 1 (Stage-A pilots and Stage-B promotions of all 7 core experiments) is **complete with 75+ result files**. This document is now reorganised as a **Phase 2 plan** focused on (a) re-analyses of existing data that fill genuine literature gaps, and (b) one cheap novel extension (asymmetric LR). The goal is no longer "more data" — it is "deeper analysis of what we have" plus a single targeted new experiment.

> **Audience.** Future-you, and any agent extending the thesis. §0 (caveats) and §1 (claim ceiling) still bind. §2-§3 are updated to reflect completed work and Phase 2 priorities. §4-§11 are unchanged historical context.

---

## §0. Pre-execution caveats (read first)

### §0.1 The `n = 2` design caveat (overrides everything else)

Your simulated federation has **two clients**. Several diagnostics from the FL literature **collapse at `n = 2`** and cannot legitimately carry a fairness or variance narrative:

| Diagnostic | What happens at `n = 2` |
|---|---|
| Cosine similarity between clients | Exactly one pair → one scalar per round, not a distribution |
| Variance of client updates | Variance over two numbers — not meaningful spread |
| Worst-client / per-client fairness (q-FFL / AFL style) | "Worst" = min of two; degenerate |

**Decision — Path 1 (default):** Lean fully into mechanism. Keep `n = 2`. Drop all *fairness/variance/worst-client* language. The justification for keeping `n = 2` is exactly that **two clients make the confounds (size, local-work, skew type) analytically clean** — the very property that breaks fairness diagnostics enables the mechanism story.

**Decision — Path 2 (optional, only if compute allows):** Spend one targeted expansion on a single client-count bump to ~8 clients on Level 4 only. This is the *only* legitimate way to unlock variance/worst-client metrics. Deferred per Phase 2 priorities below.

### §0.2 Methodological framings established in Phase 1

After Phase 1 the following framings are LOCKED IN and should appear verbatim in the thesis methods section:

1. **Symmetric vs asymmetric protocol distinction.** *"Under symmetric protocol (all algorithms see all clients), FedAvg and FedProx are statistically indistinguishable. Under the asymmetric Li 2020 §5.2 protocol (FedAvg drops stragglers, FedProx keeps γ-inexact updates), FedProx wins by Δ ≈ 0.115 macro-F1, with γ-inexact handling contributing the entire advantage."*

2. **Cross-implementation reproducibility.** *"PyTorch and Flower implementations of the same algorithm agree within 0.027 macro-F1 — within the cross-node CUDA noise floor (~0.04 macro-F1) we independently measured."*

3. **Two-implementation justification.** *"We use both run_one.py (pure-PyTorch reference) and run_one_flower.py (Flower-framework) interchangeably; numerical equivalence is verified by submit_equivalence_check.sh. The reference implementation does not depend on Ray."*

---

## §1. The thesis claim (the ceiling) — UNCHANGED

This is the strongest claim your data can support. **Do not strengthen beyond this.**

> *In a controlled, class-imbalanced dermatology FL setting (2 simulated clients, one CNN architecture, one dataset), FedProx's benefit over FedAvg is not a uniform accuracy gain. It is (i) **conditional** — appearing under the asymmetric γ-inexact protocol but absent under symmetric protocol; (ii) **mechanistically attributable** — the γ-inexact update-handling mechanism explains the entire advantage; the proximal-term-alone effect is within seed noise; (iii) **clinically concentrated** — in rare-class recall for the small specialist site under straggler conditions. The proximal strength μ that minimises update norm is not the μ that maximises macro-F1 (plasticity–stability trade-off). Under unequal local work, FedNova destabilises in the exact regime it was designed to fix — replicating Wang et al. 2024's instability finding on medical data. Results are mechanistic case-study evidence on one dataset / architecture / 2-client cohort — not a benchmark.*

---

## §2. Experiments at a glance — Phase 1 complete, Phase 2 priorities

### Phase 1 — COMPLETE (all results available, all analyses run)

| # | Experiment | Status | Headline finding |
|---|---|---|---|
| 0 | Re-analyse existing runs | ✅ done | Mechanism baseline established |
| 1 | Heterogeneity ladder (5 levels, JS-indexed) | ✅ done | Single-seed shape across 5 partitions |
| 2 | μ × heterogeneity sweep (L0/L2/L4 × 4 μ) | ✅ done | μ=0.001-0.01 best at all levels; μ=1.0 catastrophic at L4 |
| 2b | Node-pinned 3-seed L4 (variance isolation) | ✅ done | FedAvg ≈ FedProx within noise under symmetric protocol |
| 2c | Extended-rounds L3 (250r, 3 seeds) | ✅ done | Neither plateaus at 250r; FedProx gains +0.040 vs FedAvg +0.010 |
| 3 | FedNova × equal/unequal-E mechanism | ✅ done | FedNova UNSTABLE under unequal-E (drops to 0.31 at L3) |
| 4 | Federation-value matrix | ✅ partial | Existing headline + small-hospital data sufficient |
| 5 | Small-hospital rare-class case study | ✅ done | Single-seed pilot + finetune (Yu 2022 protocol) |
| 6 | Update-direction diagnostics | ⏸ deferred | Norm-only logged; full delta logging not done |
| 7 | 8-client variance/worst-client | ⏸ deferred | Out of Phase 2 scope |
| 9 | Asymmetric per-client μ (Yao 2024 ablation) | ✅ done | Direction effect within noise (negative result) |
| 10 | Li 2020 §5.2 asymmetric protocol | ✅ done | **+0.115 macro-F1; γ-inexact contributes 100%** |
| 11 | FedProx perfect-storm L4 | ✅ done | **+0.278 macro-F1 vs FedAvg+drop** (μ=0.01 ≫ μ=1.0) |

### Phase 2 — NEW EXPERIMENTS (Phase 2.1 = free re-analyses, Phase 2.2 = one new run)

Phase 2 is driven by a literature gap analysis (see §3-Phase-2). Every new experiment is justified by a 2024-2026 paper that explicitly identifies the gap.

| # | Experiment | Cost | Tier | Source gap |
|---|---|---|---|---|
| **E1** | **Best-val vs final-round vs last-K-mean protocol-flip audit** | re-analysis only | **0 (free)** | "Not All FL Algorithms Are Created Equal" (arXiv:2403.17287) explicitly notes reporting-protocol effects under-studied |
| **C1** | **Training instability replication on medical data** | re-analysis only | **0 (free)** | Same paper — their instability metric never reported on medical-FL with class imbalance |
| **A3** | **Plasticity-stability Pareto frontier for μ** | re-analysis only | **0 (free)** | DOLFIN (arXiv:2510.13567) framed this for FedCL; nobody for single-task FedProx |
| **B1** | **Early-warning rare-class collapse detector** | re-analysis + 50 LoC | **0 (free)** | FedIIC (MICCAI 2023) reacts to collapse; nobody predicts it |
| **A1** | **B-local dissimilarity empirical correlation (Li 2020 Theorem 4)** | ~5 GPU-h + 30 LoC logging | **1 (cheap)** | Yuan & Li (arXiv:2206.05187) removed B-assumption theoretically; nobody measured B(t) on medical FL |
| **B2** | **FedProx × weighted-CE 2×2 ablation** | ~10 GPU-h, 20 LoC | **1 (cheap)** | Class-imbalance-FL survey (arXiv:2303.11673) — substitutability untested |
| **D1** | **Asymmetric learning-rate protocol (extension of Li 2020 §5.2)** | ~20 GPU-h, 10 LoC | **1 (novel)** | FedLALR (arXiv:2309.09719) and Wang 2020 — LR-asymmetry FedProx-vs-FedNova mechanism untested |
| — | C2: FedExProx comparison | (deferred) | 2 | Future work — newer algorithm, out of Phase 2 |
| — | D2: Mixed optimisers per client | (deferred) | 2 | D1 is the cleaner version of the same idea |
| — | E2: Cross-node CUDA non-determinism | (deferred) | 2 | Already informally measured; its own paper |

### Phase 2 total cost: **~35 GPU-h + ~80 hours analysis/writing**

---

## §3. Phase 2 experiments in detail

For each Phase 2 experiment: **what** to run · **gap** (with cited paper) · **design** · **predicted outcome** · **contribution** · **cost** · **outputs**.

---

### Experiment E1 — Protocol-flip audit (HIGHEST ROI)

**What.** Re-analyse every (partition × algorithm × seed) cell under three reporting protocols and report how often the FedProx-vs-FedAvg ranking flips:
1. Final-round macro-F1
2. Best-val checkpoint macro-F1 (current default)
3. Mean over last 10 rounds

**Gap.** "Not All FL Algorithms Are Created Equal" (Charles et al., arXiv:2403.17287, 2024) explicitly identifies reporting-protocol effects as under-studied: *"performance stability across clients and training instability are under-reported in the FL benchmarking literature."* No paper has quantified protocol-flip rate.

**Design.** Pure re-analysis of existing `history_*.csv` and `test_at_best_*.json` files across all Phase 1 results (75+ runs).

**Predicted outcome.** ≥10% of cells flip sign between best-val and final-round. **If true** — directly motivates a "methodology reform" paragraph in your discussion (a kind of finding journals love).

**Contribution.** One table (protocol × experiment × flip count) + one paragraph in methods + one paragraph in discussion.

**Cost.** ~3 hours of analysis code, zero GPU-hours.

**Output.** `protocol_flip_audit.csv` + `F_protocol_flip_rates.{pdf,png}` + thesis discussion paragraph.

---

### Experiment C1 — Training instability metric replication

**What.** Compute Wang 2024's training-instability metric — std of test macro-F1 over the final 10 rounds — for FedAvg/FedProx/FedNova × all 5 partitions, with 3 seeds.

**Gap.** Wang et al. 2024 (arXiv:2403.17287) report this on CIFAR/FEMNIST with ≥10 clients on standard architectures. **The 2-client medical-FL regime is absent from their evaluation.** Adding our row extends their cross-method instability comparison to medical data.

**Design.** Re-analyse `history_*.csv` files. Compute `std(val_macro_f1[-10:])` per run. Group by (algorithm, partition).

**Predicted outcome.**
- FedProx more stable than FedAvg under symmetric protocol → replicates Wang 2024
- FedNova LESS stable than FedAvg under unequal-E → confirms Wang 2024's "FedNova unstable" warning
- Both predictions consistent with our existing macro-F1 findings

**Contribution.** Direct quantitative replication of Wang 2024 on a previously-untested data class. One paragraph + one table.

**Cost.** ~2 hours analysis, zero GPU-hours.

**Output.** `training_instability_summary.csv` + thesis discussion paragraph cross-citing Wang 2024.

---

### Experiment A3 — Plasticity-stability Pareto frontier for μ

**What.** For each (μ ∈ {0, 0.001, 0.01, 0.1, 1.0}, partition) decompose per-class F1 evolution into:
- **Plasticity** = Σ_c max(0, F1_c[t+1] − F1_c[t]) summed over rounds
- **Forgetting** = Σ_c max(0, F1_c[t] − F1_c[t+1]) summed over rounds

Plot the Pareto frontier (plasticity vs forgetting) as μ varies.

**Gap.** DOLFIN (arXiv:2510.13567, 2025) and Pareto Continual Learning (arXiv:2503.23390, 2025) frame the stability/plasticity tradeoff for federated *continual* learning. Nobody has applied this framing to a *single-task* FedProx run where μ is the natural plasticity knob.

**Design.** Re-analyse existing `history_*.csv` files (they already log per-round per-class val F1).

**Predicted outcome.** A clear "knee" of the Pareto curve corresponds to μ ≈ 0.01, which is the optimal μ we found via macro-F1 optimisation. This unifies two perspectives.

**Contribution.** Novel framing — first time the continual-learning plasticity-stability vocabulary is applied to FedProx μ-selection in single-task FL. One new figure.

**Cost.** ~3 hours analysis, zero GPU-hours.

**Output.** `F_plasticity_stability_pareto.{pdf,png}` + paragraph in discussion.

---

### Experiment B1 — Early-warning detector for rare-class collapse

**What.** Use round-20 features (per-class recall, KL of predicted-class distribution to uniform, per-class loss) to predict final rare-class collapse via logistic regression. Report ROC-AUC.

**Gap.** FedIIC (Wu et al., MICCAI 2023, doi.org/10.1007/978-3-031-43895-0_65) and the Confusion-Calibrated CE paper (2026) both *react* to mode collapse but neither tests whether it can be *predicted* from early-round signals. FedES (IJCAI 2024) does early stopping for label noise, not for class collapse. **The "early-warning collapse" framing is genuinely new**.

**Design.** Define "collapsed" = final per-class recall < 0.05 for ≥ 2 classes (or alternative threshold). Build features at round 20 from history CSVs. Train logistic regression. Report ROC-AUC and feature importance.

**Predicted outcome.** Round-20 per-class recall alone gives ROC-AUC > 0.85. FedProx with μ = 0.01 reduces collapse incidence by ≥ 30%.

**Contribution.** **First operational early-stopping criterion specific to FL class collapse on medical data.** Genuinely publishable as a standalone observation. One ROC curve + one feature-importance table.

**Cost.** ~4 hours of code + analysis, zero GPU-hours.

**Output.** `F_early_warning_roc.{pdf,png}` + `early_warning_features.csv` + 1-paragraph contribution to discussion.

---

### Experiment A1 — B-local dissimilarity empirical correlation (Li 2020 Theorem 4)

**What.** Add gradient-norm logging to `client.py` (~10 lines), re-run 1 seed per (level, μ) on the μ-sweep, and correlate mean B(t) with the (FedProx − FedAvg) macro-F1 gap.

**Gap.** Li et al. 2020 (FedProx, MLSys) Theorem 4 assumes bounded B-local dissimilarity. **Nobody has empirically measured B(t) across heterogeneity levels and correlated it with FedProx's win/loss direction**. Yuan & Li (arXiv:2206.05187, NeurIPS 2022) removed the bounded-B assumption theoretically but did not measure it. FedImpro (arXiv:2402.07011, 2024) and HAPI-FedProx (Springer 2025) introduced related diagnostics but on different datasets.

**Design.**
- Add `||∇F_i(w_t)||²` computation to `FlClient.fit()` (~10 LoC) — uses one extra backward pass on a mini-batch before training starts.
- Compute `B²(t) = max_i ||∇F_i(w_t)||² / ||∇F(w_t)||²` per round on the server side.
- Re-run μ-sweep ladder (3 levels × 4 μ values × 1 seed = 12 runs).
- Plot:
  - (a) B(t) trajectories per (level, μ)
  - (b) Correlation: mean B vs (FedProx − FedAvg) macro-F1 gap

**Predicted outcome.** B grows monotonically with JS divergence. The FedProx advantage emerges precisely where mean B exceeds some empirical threshold (we'd expect ~2-3 based on Li 2020's synthetic data discussion). If refuted, that's a strong empirical refutation of Li 2020's bounded-B assumption.

**Contribution.** **First empirical correlation between Li 2020's theoretical quantity B and the algorithm-choice outcome on medical FL.** Either result is novel.

**Cost.** ~5 GPU-h + ~30 LoC.

**Output.** `F_b_dissimilarity_vs_advantage.{pdf,png}` + `b_dissimilarity_summary.csv` + new methods-section paragraph linking empirical to Li 2020 Theorem 4.

---

### Experiment B2 — FedProx × class-weighted-CE: substitutes or complements?

**What.** 2×2 ablation: {standard CE, inverse-frequency weighted CE} × {FedAvg, FedProx μ=0.01} on L4, 3 seeds.

**Gap.** The class-imbalance-FL survey (arXiv:2303.11673, 2023) and FedLC (arXiv:2209.00189, 2022) compare logit calibration vs FedAvg. **Nobody has tested whether FedProx and inverse-frequency weighted CE are substitutes or complements** when the imbalance is *partition-induced* rather than dataset-level.

**Design.**
- 2 algorithms × 2 loss types × 3 seeds = 12 runs
- L4 partition (severe class skew)
- Existing `--loss-type` CLI flag already supports `class_weighted_ce`

**Predicted outcome.**
- Most likely: weighted-CE alone beats FedProx alone on macro-F1
- FedProx alone reduces variance (per-class std lower than FedAvg+CE)
- Combination roughly additive on macro-F1
- **Most interesting if non-additive** — would directly counter the "stacking" intuition in medical-FL literature

**Contribution.** First medical-FL test of FedProx × loss-reweighting compositionality. Adds one row to your federation-value matrix.

**Cost.** ~10 GPU-h, ~20 LoC if any (mostly existing flags).

**Output.** `tab:fedprox_weighted_ce.csv` + `F_fedprox_weighted_ce.{pdf,png}` + thesis paragraph.

---

### Experiment D1 — Asymmetric learning-rate protocol (THE novel extension)

**What.** Extend Li 2020 §5.2 from *asymmetric local epochs* to *asymmetric learning rates*. 2×3 grid: {LR ratio 1:1, 1:2, 1:5 between Client 0 and Client 1} × {FedAvg, FedProx μ=0.01, FedNova}, on L4, 3 seeds.

**Gap.** Wang 2020 (FedNova) explicitly proves correction only for unequal τᵢ (local epochs), not for asymmetric learning rates. FedLALR (arXiv:2309.09719, 2023) and FedEff (Nature Sci. Reports 2025, doi:10.1038/s41598-025-22672-1) propose client-specific LRs but neither isolates whether the **FedProx-vs-FedNova mechanism distinction extends to LR asymmetry**.

**Design.**
- LR pairs: (0.01, 0.01) baseline; (0.01, 0.005) ratio 1:2; (0.01, 0.002) ratio 1:5
- Algorithms: FedAvg, FedProx (μ=0.01), FedNova
- Partition: L4 (severe heterogeneity, where mechanism distinctions are clearest)
- 3 seeds → 27 jobs ≈ 20 GPU-h

**Predicted outcome.**
- **FedProx absorbs LR asymmetry** — proximal anchor bounds the bigger-step client's drift; should perform comparably to LR=1:1 baseline
- **FedNova collapses under LR asymmetry** — its derivation assumes uniform LR (only corrects for τᵢ); we predict ≥0.10 macro-F1 drop relative to LR=1:1
- **FedAvg in between** — affected but not as severely as FedNova

**Contribution.** **Cleanly separates FedNova from FedProx in a regime FedNova wasn't designed for.** Novel deployment scenario mapping to a realistic clinical setting where hospitals have different training infrastructure. One new table + one figure.

**Cost.** ~20 GPU-h, ~10 LoC for per-client LR (existing `--mu-per-client` plumbing pattern).

**Output.** `tab:asymmetric_lr.csv` + `F_asymmetric_lr.{pdf,png}` + thesis section.

---

## §4. Phase 2 execution schedule (7-day plan)

| Day | Task | Effort | Output |
|---|---|---|---|
| 1 | E1 + C1 (two protocol-related re-analyses) | 5 hours | One table + one paragraph |
| 2 | A3 + B1 (mechanism re-analyses) | 8 hours | Two figures + one paragraph |
| 3 | A1 logging change + submit re-runs | 4 hours code + 5 GPU-h | Code + queued |
| 4 | D1 design + submit (20 GPU-h) | 2 hours | 27 jobs queued |
| 5 | B2 design + submit (10 GPU-h) — while D1 runs | 2 hours | 12 jobs queued |
| 6 | Analyse D1 + B2 + A1 results | 6 hours | Three new figures |
| 7 | Update thesis with all 7 new findings | 8 hours | Complete draft |

**Total: ~35 GPU-h + 35 hours work.**

---

## §5. Decision matrix — Phase 2

| Item | Do | Defer to future work | Skip |
|---|---|---|---|
| Phase 1 experiments (all 11) | ✅ Done | | |
| E1: Protocol-flip audit | ✅ | | |
| C1: Training instability (Wang 2024 replication) | ✅ | | |
| A3: Plasticity-stability Pareto | ✅ | | |
| B1: Early-warning collapse detector | ✅ | | |
| A1: B-dissimilarity empirical | ✅ | | |
| B2: FedProx × weighted-CE | ✅ | | |
| D1: Asymmetric LR (novel extension) | ✅ | | |
| C2: FedExProx comparison | | ✅ | |
| C3: HeteRo-Select temperature ablation | | ✅ | |
| D2: Mixed optimisers per client | | ✅ | |
| E2: Cross-node CUDA non-determinism | | ✅ | |
| 8-client expansion (Exp 7) | | ✅ | |
| Cross-dataset replication | | ✅ | |
| SCAFFOLD implementation | | | ❌ |
| FedDyn / MOON / Ditto / pFedMe / FedRep | | discussion only | |
| Larger model / HAM10000 full-res | | | ❌ |
| p-values / Wilcoxon on n=3 seeds | | | ❌ (significance theatre) |

---

## §6. Top figures the thesis must keep

After Phase 2, the figure inventory is:

### Phase 1 figures (already produced)
1. **F_li2020_asymmetric_L4** — the canonical "FedProx wins" demonstration (4 conditions × 3 seeds)
2. **F_fedprox_perfect_storm_L4** — literature-canonical setup with the headline +0.278 gap
3. **F_fednova_unequal_E** — mechanism decomposition; FedNova collapse visualisation
4. **F_extended_rounds_L3** — convergence trajectories showing FedProx benefits more from training
5. **F_node_pinned_L4** — variance isolation, confirms symmetric ≈ noise
6. **F_mu_sweep_L4** — the inverted-U μ × heterogeneity story
7. **F_asymmetric_mu_L4** — Yao 2024 negative result
8. **F_val_curves_li2020_asymmetric** — validation trajectories complementing #1
9. **F_personalisation_gap** — fine-tuning Yu 2022 protocol
10. **F_pareto_small_hospital** — clinical money figure

### Phase 2 figures (new)
11. **F_b_dissimilarity_vs_advantage** — empirical Li 2020 Theorem 4 link (Exp A1)
12. **F_plasticity_stability_pareto** — μ × continual-learning vocabulary (Exp A3)
13. **F_early_warning_roc** — operational collapse detector (Exp B1)
14. **F_asymmetric_lr** — novel LR-asymmetry extension (Exp D1)
15. **F_fedprox_weighted_ce** — compositionality test (Exp B2)
16. **F_protocol_flip_rates** — methodology audit (Exp E1)

**Total: 16 figures + ~15 tables.**

---

## §7. Citation reference list

### Top-5 must-cite (UNCHANGED from earlier version)

1. **McMahan et al. (2017)** "Communication-Efficient Learning..." — AISTATS. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629). FedAvg.
2. **Li et al. (2020)** "Federated Optimization in Heterogeneous Networks" — MLSys. [arXiv:1812.06127](https://arxiv.org/abs/1812.06127). FedProx; Theorem 4; §5.2 protocol.
3. **Wang et al. (2020)** "Tackling the Objective Inconsistency Problem..." — NeurIPS. [arXiv:2007.07481](https://arxiv.org/abs/2007.07481). FedNova.
4. **Li, Diao, Chen, He (2022)** NIID-Bench — ICDE. [arXiv:2102.02079](https://arxiv.org/abs/2102.02079). Cross-method benchmark.
5. **Pati et al. (2022)** Nature Communications — [DOI:10.1038/s41467-022-33407-5](https://doi.org/10.1038/s41467-022-33407-5). Medical-FL precedent.

### NEW for Phase 2

- **Charles et al. (2024)** "Not All FL Algorithms Are Created Equal" — [arXiv:2403.17287](https://arxiv.org/abs/2403.17287). Training-instability framework (Experiments C1, E1).
- **Yuan & Li (2022)** "FedProx Convergence: Local Dissimilarity Invariant" — NeurIPS. [arXiv:2206.05187](https://arxiv.org/abs/2206.05187). Theoretical removal of bounded-B (Experiment A1).
- **Wu et al. (2023)** FedIIC — MICCAI. [DOI:10.1007/978-3-031-43895-0_65](https://doi.org/10.1007/978-3-031-43895-0_65). Class-collapse precedent (Experiment B1).
- **Yang et al. (2025)** "Stabilizing FL under Extreme Heterogeneity" — [arXiv:2508.06692](https://arxiv.org/abs/2508.06692). μ-tuning under heterogeneity.
- **Tang et al. (2024)** FedImpro — [arXiv:2402.07011](https://arxiv.org/abs/2402.07011). Update-level analysis discipline.
- **Wu et al. (2024)** FedAWARE — [arXiv:2310.02702](https://arxiv.org/abs/2310.02702). Gradient diversity diagnostic.
- **He et al. (2025)** DOLFIN — [arXiv:2510.13567](https://arxiv.org/abs/2510.13567). Plasticity-stability for FedCL (Experiment A3).
- **Class-imbalance-FL survey (2023)** — [arXiv:2303.11673](https://arxiv.org/abs/2303.11673). Survey grounding Experiment B2.
- **FedLC (Zhang et al. 2022)** — [arXiv:2209.00189](https://arxiv.org/abs/2209.00189). Logits-calibration baseline.
- **FedLALR (2023)** — [arXiv:2309.09719](https://arxiv.org/abs/2309.09719). Client-specific LR precedent (Experiment D1).
- **FedEff (Nature Sci Reports 2025)** — [DOI:10.1038/s41598-025-22672-1](https://doi.org/10.1038/s41598-025-22672-1). Per-client efficiency (D1 context).
- **Yao et al. (2024)** — [arXiv:2410.08934](https://arxiv.org/abs/2410.08934). Per-client μ theoretical framework.
- **HAPI-FedProx (Springer 2025)** — [DOI:10.1007/978-3-032-11733-5_17](https://doi.org/10.1007/978-3-032-11733-5_17). Adaptive per-client μ precedent.

---

## §8. The cited-evidence summary (defend-this-at-viva table)

UNCHANGED from earlier version. Phase 2 adds 6 new rows:

| Your design choice | Evidence | Specific support |
|---|---|---|
| Use GroupNorm, not BatchNorm | Hsieh et al. 2020 | FL feature-statistics mismatch under BN |
| Sweep μ ∈ {0.001, 0.01, 0.1, 1.0} | Li et al. 2020 FedProx | "no default μ values would work for all settings" |
| Use JS (not EMD) as heterogeneity axis | Zhao 2018 + DermaMNIST | EMD requires ground metric; categorical labels |
| Defer SCAFFOLD | NIID-Bench 2022; Charles 2024 | SCAFFOLD vulnerable to class imbalance |
| Three-seed minimum | NIID-Bench 2022 | Standard practice |
| Avoid p-values on n=3 | Methodological consensus | Underpowered |
| **Empirically measure B-local dissimilarity** | **Yuan & Li 2022; Li 2020 Theorem 4** | **Theoretical claim never empirically tested on medical FL** |
| **Report best-val + final-round + last-K-mean** | **Charles et al. 2024 (Not All Equal)** | **Reporting-protocol effects under-studied** |
| **Plasticity-stability framing for μ** | **DOLFIN 2025** | **Adapted from FedCL to single-task FedProx** |
| **Predict class collapse from round 20** | **FedIIC 2023; Confusion-Calibrated CE 2026** | **No prior operational early-stopping for class collapse** |
| **Asymmetric LR (not just asymmetric E)** | **Wang 2020; FedLALR 2023** | **FedNova proven for τᵢ only; LR-asymmetry mechanism untested** |
| **FedProx × weighted-CE compositionality** | **Class-imbalance-FL survey 2023** | **Substitutability untested on partition-induced imbalance** |

---

## §9. Strongest defensible thesis claim + future-work — UNCHANGED

(See §1 for the claim. Future-work paragraph remains the same: cross-dataset replication, SCAFFOLD comparison, multi-institutional real data.)

---

## §10b. PHASE 3 — Publishability extensions (complementary to existing findings)

After Phase 2 completed (14 findings, 75+ results, 16 figures), the four strongest findings each have a SPECIFIC publishability gap identified by 2024-2026 literature analysis. Phase 3 adds **4 complementary experiments** targeted at each gap, ranked by ROI for converting the thesis into workshop / short-paper material.

### Finding 14 (FedNova absorbs LR asymmetry) — TARGET: MICCAI DeCaF full paper

| Item | Status |
|---|---|
| Current evidence | 27 D1 runs (3 algos × 3 LR ratios × 3 seeds), L4 only |
| Publishability gap | Single perturbation magnitude (max 5:1); no breaking point; no algebraic mechanism |
| 2024-2026 landscape | No paper has empirically reported "FedNova is LR-asymmetry-invariant" |
| Source gaps | FedACS (arXiv:2505.11304, 2025): "Heterogeneity can significantly distort optimization dynamics" — does not test per-client η. FedLALR (arXiv:2309.09719, 2023): proposes per-client adaptive LR but doesn't benchmark FedAvg/FedProx/FedNova reaction. |

**Phase 3 Experiment P1 — FedNova LR-invariance dose-response (~22 GPU-h)**

Extend the LR ratio sweep with 3 extreme values to find FedNova's breaking point:
- LR pairs (C0:C1): {1:1, 2:1, 5:1, **10:1, 20:1, 50:1**}
- Already have: 1:1, 2:1, 5:1 (D1) — 27 existing runs
- NEW: 3 ratios × 3 algos × 3 seeds = **27 new runs** at 10:1, 20:1, 50:1

Output: F_fednova_lr_envelope.pdf showing macro-F1 vs log(LR ratio), 3 lines for 3 algorithms. **Headline if FedNova still absorbs at 20:1**: "FedNova's `τᵢ`-normalization produces LR-invariant per-client aggregation across a 20× range." **Headline if it breaks**: "FedNova absorbs up to N× LR asymmetry, beyond which all algorithms collapse together."

Cited in paper: Wang 2020 Eq. 7 algebraic argument that when η enters d_i and τ_i is normalized out, unit-norm direction stays invariant — combined with our empirical breaking point data, this completes the publishable claim.

**Venue rationale**: DeCaF MICCAI workshop accepts 8-page mechanism-style empirical papers; no competing 2024-2026 result on this specific question.

---

### Finding 1 (γ-inexact mechanism decomposition) — TARGET: MIDL 2026 full paper / MELBA journal

| Item | Status |
|---|---|
| Current evidence | 12 runs on L4 (4 conditions × 3 seeds); clean Condition-4 isolation |
| Publishability gap | Single partition; no cross-partition invariance test |
| 2024-2026 landscape | No paper repeats Li 2020 §5.2 decomposition with Condition-4 control; Frontiers 2025 survey explicitly notes FedProx/FedAvg results are "protocol-confounded" |

**Phase 3 Experiment P2 — Replicate 2×2 factorial on L1 quantity-only partition (~8 GPU-h)**

Replicate the Li 2020 §5.2 4-condition factorial on a SECOND partition type (L1: 86/14 quantity-only, no label skew):
- 4 conditions × 3 seeds = **12 new runs**
- Add bootstrap CIs (1000 resamples) to existing L4 data + new L1 data

**Predicted outcome**: γ-inexact attribution survives across BOTH partition types → robust mechanism claim across heterogeneity regimes. **If it doesn't survive**: the mechanism is label-skew-specific → narrower but still publishable claim.

**Venue rationale**: MIDL accepts protocol-clarification papers; MELBA has no page limit and welcomes clean ablation studies.

---

### Finding 11 (early-warning collapse detector) — TARGET: descriptive section, not standalone

| Item | Status |
|---|---|
| Current evidence | Cross-validated AUC = 0.804 on 88 mixed runs |
| Publishability gap | CV is leaky (same experiment mix used for fit + validation); single feature (macro_f1_r20) at AUC 0.825 already beats the multivariate model |
| 2024-2026 landscape | PCA + early stopping (PMC12409104, 2025) and synthetic-validation early stopping (arXiv:2511.11208, 2025) are orthogonal — neither uses trajectory features |

**Phase 3 Experiment P3 — Prospective validation on held-out experiment family (~0 GPU-h, ~50 LoC)**

Free re-analysis:
1. Hold out ALL μ-sensitivity runs from training (12 runs)
2. Refit logistic regression on remaining ~76 runs
3. Validate on the held-out μ-sweep runs
4. Add compute-saved curves: for each precision threshold {0.7, 0.8, 0.9}, plot GPU-hours saved vs collapses missed

**Honest framing**: Reframe as a *descriptive observation* ("round-20 macro-F1 < 0.30 predicts collapse with 80% precision") rather than as a "tool" — because the simple-threshold baseline matches the multivariate model.

**Venue rationale**: As a section/subsection in the P1 (DeCaF) paper, not standalone.

---

### Finding 13 (FedProx inert atop weighted-CE) — TARGET: DeCaF short paper or paper section

| Item | Status |
|---|---|
| Current evidence | 12 runs (2×2 × 3 seeds); interaction = -0.004 (additive but FedProx contributes ~0) |
| Publishability gap | One imbalance-aware loss tested (weighted-CE only); the 2026 Confusion-Calibrated CE paper VERBALLY states this finding ("FedProx remains class-agnostic") — not yet empirically isolated |
| 2024-2026 landscape | FedLC (arXiv:2209.00189): logit calibration. FedIIC (MICCAI 2023): class-imbalance in FL. Confusion-Calibrated CE (S095070512600239X, 2026) verbally states the finding |

**Phase 3 Experiment P4 — Extend 2×2 to 2×3 with focal loss column (~6 GPU-h)**

Add focal loss (γ=2) as a third loss type:
- {CE, weighted-CE, **focal-γ2**} × {FedAvg, FedProx} × 3 seeds = **18 new runs** (6 added to existing 12)
- Report interaction term per loss column with bootstrap CIs
- Add `||∇_w F||₂` averaged over rounds 50-150 per condition

**Predicted outcome**: FedProx interaction is ~0 for ALL loss-side correction methods. Mechanism: proximal-term gradient shrinks once class loss is balanced.

**Venue rationale**: DeCaF short paper (4 pages) or section in P1.

---

## §10c. Phase 3 ROI ranking and total cost

| Rank | Experiment | GPU-h | LoC | Output | Target venue |
|---|---|---|---|---|---|
| **1** | **P1 FedNova LR-envelope** | **~22** | **~80** | **F_fednova_lr_envelope.pdf** | **MICCAI DeCaF full paper** |
| 2 | P2 Cross-partition decomposition | ~8 | ~40 | F_mechanism_l1_l4.pdf | MIDL / MELBA |
| 3 | P4 FedProx × focal | ~6 | ~60 | F_fedprox_loss_compositionality.pdf | DeCaF short / section |
| 4 | P3 Prospective collapse detector | 0 | ~50 | F_early_warning_prospective.pdf | section in P1 paper |
| **Total** | | **~36** | **~230** | 4 new figures, 1 paper, ~57 jobs | |

---

## §10. Agent execution checklist — Phase 2

```text
[ ] 1. E1: implement protocol-flip audit script
[ ] 2. C1: implement training-instability metric script
[ ] 3. A3: implement plasticity-stability Pareto script
[ ] 4. B1: implement early-warning collapse-detector script
[ ] 5. A1: add gradient-norm logging to client.py + re-run μ-sweep
[ ] 6. B2: submit FedProx × weighted-CE 2×2
[ ] 7. D1: submit asymmetric-LR 2×3 (the novel one)
[ ] 8. Run all Phase 2 analyses
[ ] 9. Update thesis with Phase 2 findings
[ ] 10. Final figure/table review
[ ] 11. Stop. The thesis is complete.
```

---

*End of plan. Phase 1 (11 experiments) is complete. Phase 2 adds 7 more (4 free re-analyses + 3 new runs) for a total of ~35 GPU-h and ~80 hours of writing/analysis. The final deliverable is a thesis with 16 figures, 15 tables, 8 confirmed and 4 contradicted literature claims, and one genuinely novel extension (asymmetric LR).*
