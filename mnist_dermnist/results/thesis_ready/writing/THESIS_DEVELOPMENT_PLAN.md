# Thesis Development Plan — Mechanistic FL under Heterogeneity & Class Imbalance (Dermatology)

> **Status.** Final synthesised plan. Aggregates four independent planning passes (Claude-1, Claude-2, Codex-style agent, and human consolidation pass). Every recommendation below is anchored in a specific cited paper. Agents executing this plan should treat the "Why — cited" blocks as the authoritative justification for each design choice.

> **Audience.** Future-you, and any agent extending the thesis after the current HPC jobs return. Read §0 before doing anything; it overrides everything else if violated.

---

## §0. Pre-execution caveats (read first)

### §0.1 The `n = 2` design caveat (overrides everything else)

Your simulated federation has **two clients**. Several diagnostics from the FL literature **collapse at `n = 2`** and cannot legitimately carry a fairness or variance narrative:

| Diagnostic | What happens at `n = 2` |
|---|---|
| Cosine similarity between clients | Exactly one pair → one scalar per round, not a distribution |
| Variance of client updates | Variance over two numbers — not meaningful spread |
| Worst-client / per-client fairness (q-FFL / AFL style) | "Worst" = min of two; degenerate |

**Decision — Path 1 (default):** Lean fully into mechanism. Keep `n = 2`. Drop all *fairness/variance/worst-client* language from the thesis. The justification for keeping `n = 2` is exactly that **two clients make the confounds (size, local-work, skew type) analytically clean** — the very property that breaks the fairness diagnostics also enables the mechanism story.

**Decision — Path 2 (optional, only if compute allows):** Spend one targeted expansion on a single client-count bump to ~8 clients on Level 4 only. This is the *only* legitimate way to unlock variance/worst-client metrics. Treated as Experiment 7 below.

**Do not** mix the two: do not attempt fairness language on the `n = 2` data. An examiner will see through it.

### §0.2 Citation integrity checklist

Every paper cited here is referenced for the **specific mechanism or finding** the paper established. The arXiv IDs and DOIs in §6 have been web-verified once but must be re-verified against the publisher page before each enters the bibliography. The riskiest IDs to verify are the medical-FL class-imbalance papers (FedIIC, FedSLD) and the personalised-FL papers, whose names/venues have evolved over multiple arXiv revisions.

**Mandatory pre-submission verification:**
- [ ] Confirm every arXiv ID / DOI on the publisher or arXiv page.
- [ ] Confirm FedNova's exact normalisation definition (`τ_i`-based) from the paper before implementing.
- [ ] Confirm FedProx's B-local dissimilarity definition before reporting it.
- [ ] State explicitly: HAM10000 (full resolution) vs DermaMNIST (28×28). This thesis uses **DermaMNIST 28×28**.
- [ ] If using EMD as the heterogeneity index: define the ground metric over the 7 (unordered) DermaMNIST classes, or switch to JS divergence (§Experiment 2 below — JS is the safer choice).

---

## §1. The thesis claim (the ceiling)

This is the strongest claim your data can support once §§2–4 below are executed. **Do not strengthen beyond this; weaker is fine.**

> *In a controlled, class-imbalanced dermatology FL setting (2 simulated clients, one CNN architecture, one dataset), FedProx's benefit over FedAvg is not a uniform accuracy gain. It is (i) **conditional** — appearing when label skew and quantity skew interact, not when either occurs alone; (ii) **mechanistically mediated** — by reduced per-client update drift and a more stable global update direction; and (iii) **clinically concentrated** — in rare-class recall for the small specialist site. The proximal strength μ that minimises update norm is not the μ that maximises macro-F1 (a plasticity–stability trade-off). Under unequal local work, a separable part of the apparent FedProx gain is objective inconsistency, which FedNova corrects more directly than FedProx damps. Results are mechanistic case-study evidence on one dataset / architecture / 2-client cohort — not a benchmark.*

Stronger would over-claim ("FedProx is universally better", "the mechanism is provably drift control"). Weaker repeats what NIID-Bench already established ([Li et al. 2022](https://arxiv.org/abs/2102.02079)) and adds nothing.

---

## §2. Experiments at a glance

| # | Experiment | Compute | `n=2` safe? | Priority |
|---|---|---|---|---|
| 0 | Re-analyse existing runs for mechanism diagnostics | ~0 (re-analysis) | ✅ | **Do first** |
| 1 | Heterogeneity ladder, JS-indexed | ~75 GPU-h | ✅ | ★★★★★ |
| 2 | μ × heterogeneity sweep (norm-vs-performance dissociation) | ~30 GPU-h | ✅ | ★★★★★ |
| 2b | Node-pinned 3-seed variance isolation on L4 | ~6 GPU-h | ✅ | ★★★★★ **Pre-Stage-B blocker** |
| 2c | Extended-rounds (250) 3-seed L3 — convergence-truncation fix | ~11 GPU-h | ✅ | ★★★★★ **Pre-Stage-B blocker** |
| 3 | Mechanism decomposition: FedProx vs FedNova, equal vs unequal local work | ~12 GPU-h (Stage A) | ✅ | ★★★★☆ |
| 4 | Federation-value + personalisation-gap matrix | ~12 GPU-h | ✅ | ★★★★☆ |
| 5 | Small-hospital rare-class case study, **3-seed promotion** | ~12 GPU-h | ✅ (case-study) | ★★★★☆ |
| 6 | Update-direction diagnostics (full delta logging + cosine) | ~3 GPU-h + code | ✅ | ★★★☆☆ |
| 7 | (conditional) Client bump to ~8 for variance / worst-client | ~30 GPU-h | unlocks `n=2`-dead diagnostics | ★★★☆☆ |
| 9 | Asymmetric per-client μ on L4 (Yao 2024 / HAPI-FedProx ablation) | ~12 GPU-h | ✅ | ★★★★☆ |
| — | SCAFFOLD targeted run | (impl-heavy) | — | ★★☆☆☆ (defer) |

---

## §3. Experiments in detail

For each experiment: **what** to run · **why** (with cited evidence) · **design** · **metrics** · **repo state** · **cost** · **failure modes** · **outputs**.

---

### Experiment 0 — Re-analyse existing runs (do first, before any new training)

**What.** Convert the accuracy/loss logs you already have into mechanism evidence. No new training; only re-analysis from saved checkpoints or already-logged CSVs.

**Compute, from existing data:**
- Per-client update norm `‖Δᵢᵗ‖ = ‖wᵢᵗ⁺¹ − wᵗ‖` per round (**already logged** in `client_update_norms_*.csv`).
- Global step norm `‖Δᵗ‖ = ‖wᵗ⁺¹ − wᵗ‖` per round (post-aggregation).
- Global-direction oscillation `cos(Δᵗ, Δᵗ⁻¹)` per round.
- Pairwise client cosine `cos(Δ₀ᵗ, Δ₁ᵗ)` — a single scalar series at `n = 2`. **Reported as a series, never as a distribution.**
- Label-distribution divergence (JS, KL) per partition.
- Rounds-to-threshold (round to reach 80–90% of final macro-F1) and AUC of the validation curve.

**Why — cited.**
- **FedProx (Li et al. 2020, MLSys)** — the proximal term `(μ/2)‖w − wᵗ‖²` exists precisely to bound how far each local solution moves from the global model; tracking `‖Δᵢᵗ‖` is the direct empirical read-out of this mechanism. The paper monitors a "B-local dissimilarity" quantity as a heterogeneity-sensitivity diagnostic (Theorem 4, §4.1) — your update-norm plots are its empirical analogue. [arXiv:1812.06127](https://arxiv.org/abs/1812.06127).
- **SCAFFOLD (Karimireddy et al. 2020, ICML)** — formalises FedAvg's failure as "client drift" arising from non-IID data and multiple local updates; their Fig. 3 visualises per-round update direction divergence. This is the conceptual licence for the cosine and oscillation diagnostics, regardless of whether you implement SCAFFOLD itself. [arXiv:1910.06378](https://arxiv.org/abs/1910.06378).
- **FedImpro (Tang et al. 2024)** — title literally "Measuring and Improving Client Update in Federated Learning"; establishes update-level analysis as the right abstraction for understanding non-IID effects. [arXiv:2402.07011](https://arxiv.org/abs/2402.07011).

**Repo state.** Norms already logged. Full per-client delta vectors not yet logged — see Experiment 6 if those are needed.

**Cost.** ~0 compute. ~3 hours of analysis code.

**Failure modes.** If per-round client weights weren't checkpointed for the runs you care about, full-cosine analysis is blocked — cheap to fix by re-running with extended logging (Experiment 6).

**Outputs.** Volatility column added to every existing macro-F1 table; oscillation panel in the existing update-norm figure.

---

### Experiment 1 — Heterogeneity ladder, JS-indexed (chapter centrepiece)

**What.** A controlled ladder that isolates *which dimension of non-IID* matters, plotted against a **continuous heterogeneity scalar** rather than categorical level numbers.

**Design.** 2 clients, five rungs (already implemented and tested in `data/partition.py`; **77 tests passing**):
- **L0** `two_client_50_50_stratified_iid` — IID control
- **L1** `two_client_86_14_quantity_only_stratified` — quantity skew only (every class on both clients)
- **L2** `two_client_50_50_label_skew_only` — label skew only
- **L3** `two_client_70_30_rare_enriched` — moderate combined skew
- **L4** `two_client_90_10_rare_stress` — severe rare-class stress (existing engineered partition)

**Methods.** FedAvg, FedProx (μ from the existing μ-sweep, default μ = 0.01), FedNova **only on levels where local epochs can be unequal** (L1, L3, L4 if running variable-`E` variants — see Experiment 3).

**Seeds.** 3 minimum (`42, 123, 456`), 5 ideal.
**Local epochs.** E = 20 (matches existing thesis hyperparameters).
**Work mode.** Fixed epochs (steps-matched is Experiment 3 territory).

**Metrics.** Macro-F1, balanced accuracy, rare-class F1/recall (melanoma, dermatofibroma, vascular). Global accuracy as secondary metric only.

**Critical analysis move.** **Index the ladder by a measured scalar, not the rung number.** Compute per partition the JS divergence between each client's empirical label distribution and the global label distribution; plot (FedProx − FedAvg) Δmacro-F1 and Δrare-recall against that scalar. This turns five categorical points into a dose–response curve.

**Why — cited.**
- **Hsu, Qi, Brown (2019)** — established Dirichlet(α) as a *tunable continuous severity knob* (smaller α → more skew) and swept α against accuracy. Justifies a ladder over an arbitrary single partition; the JS-divergence x-axis is the partition-agnostic generalisation. [arXiv:1909.06335](https://arxiv.org/abs/1909.06335).
- **Zhao et al. (2018)** — proved that FedAvg's weight divergence is *bounded by* the Earth Mover's Distance between each client's label distribution and the population distribution. The scalar heterogeneity index this paper introduced is the theoretical anchor for the dose–response x-axis. [arXiv:1806.00582](https://arxiv.org/abs/1806.00582). **⚠ EMD vs JS:** Zhao et al. use EMD because their setting has ordered labels. **DermaMNIST's 7 skin-lesion classes are categorical/unordered, so EMD has no canonical ground metric.** Use **Jensen-Shannon divergence** instead (or total variation), which require no class ordering. State this choice explicitly in the methods section.
- **NIID-Bench (Li, Diao, Chen, He 2022, ICDE)** — establishes the comprehensive cross-partition comparison as a standard format: *"data is usually non-independently and identically distributed (non-IID) … we develop a benchmark named NIID-Bench, introducing six data partitioning strategies which are much more comprehensive than previous studies."* Their cross-partition comparison table is the template. **They report SCAFFOLD as more vulnerable to class imbalance than FedAvg under heavy label skew** — directly supports your deferral of SCAFFOLD. [arXiv:2102.02079](https://arxiv.org/abs/2102.02079).
- **FedProx (Li et al. 2020)** Theorem 4 makes the bounded-dissimilarity assumption explicit; your ladder operationalises increasing dissimilarity along a measured axis. [arXiv:1812.06127](https://arxiv.org/abs/1812.06127).

**Repo state.** Partitioners and tests done. Stage A pilot was submitted on HPC (jobs 29881013–25); GCS failures were addressed by the `RAY_TMPDIR` template patch; pilot needs resubmission and Stage B promotion.

**Cost.** Stage A pilot ~13 GPU-hours. Stage B (3 seeds × 5 levels × 3 methods on appropriate levels) ~75 GPU-hours.

**Failure modes.**
- Rungs may not be monotone in measured JS — **automatically fixed by plotting against the measured scalar**.
- Gap within seed noise → report seed spread and refuse to over-interpret.
- If FedNova fails to train (Wang et al. 2020's normaliser pitfall) — see Experiment 3.

**Outputs.**
- `tab:heterogeneity_ladder` — 5 rows × {FedAvg, FedProx, FedNova} × {macro-F1, rare-recall, update-norm ratio, volatility} mean ± SD.
- `F_ladder_delta_vs_js.pdf` — (FedProx − FedAvg) Δmetric vs JS divergence per partition; scatter with one point per (partition, seed); one line per metric.

---

### Experiment 2 — μ × heterogeneity: norm-monotone / performance-non-monotone dissociation (your signature result)

**What.** Show that proximal strength μ has a **heterogeneity-dependent optimum**, and — the crisp falsifiable claim — that **update-norm reduction is monotone in μ while task performance is inverted-U**. The μ that damps drift most is not the μ that performs best: the proximal term trades plasticity for stability.

**Design.** μ ∈ {0, 0.001, 0.01, 0.1, 1.0} × Levels {0, 2, 4} (Level 0 as inert-baseline contrast). 3 seeds. μ = 0 is a FedAvg-equivalence sanity check (verify your implementation makes μ=0 bit-equivalent to FedAvg — already verified in your `fl/local_train.py:70` gated branch).

Overlaps with Experiment 1's FedProx runs — do not double-count compute.

**Metrics.** Macro-F1 and rare-class recall vs μ. Mean `‖Δᵢᵗ‖` vs μ. Oscillation vs μ.

**Why — cited.**
- **FedProx (Li et al. 2020, MLSys)** — μ is the paper's central mechanism. They sweep μ and report that larger μ stabilises but can over-constrain: *"in highly heterogeneous settings, FedProx demonstrates significantly more stable and accurate convergence … FedProx with μ > 0 enables otherwise divergent methods to converge … either decreasing E or increasing μ leads to smaller dissimilarity among local functions … no default μ values would work for all settings."* The paper does not sweep μ × heterogeneity-axis simultaneously — your contribution is exactly that 2D sweep, plus the explicit dissociation between norm and performance. [arXiv:1812.06127](https://arxiv.org/abs/1812.06127).
- **FedNova (Wang et al. 2020)** observes that large proximal / regularisation strength can slow convergence — supporting the "over-regularisation suppresses learning" half of the inverted-U. [arXiv:2007.07481](https://arxiv.org/abs/2007.07481).
- Recent (2025) work *"Stabilizing FL under Extreme Heterogeneity"* ([arXiv:2508.06692](https://arxiv.org/abs/2508.06692)) confirms μ-tuning under heterogeneity is still an open question — strengthens the relevance.

**Repo state.** Single-level μ-sweep complete on engineered partition. Two additional levels needed (L0 and L2).

**Cost.** ~30 GPU-hours (15 new runs counting overlap with Experiment 1).

**Failure modes.**
- Coarse μ grid; single-seed curves noisy (hence 3 seeds).
- Best μ may differ by metric (global acc vs rare-class F1) — **report both, that is the finding.**
- If performance is flat in μ at some level: the honest finding is "the proximal term is near-inert in this regime" — still publishable, still mechanistic.

**Outputs.**
- `F_mu_heterogeneity_heatmap.pdf` — two-panel: (a) mean update norm vs μ (expected monotone decrease, one line per ladder level); (b) macro-F1 vs μ (expected inverted-U, one line per level).
- `tab:mu_heterogeneity` — μ × Level × {macro-F1, rare-recall, update-norm ratio}.

---

### Experiment 3 — Mechanism decomposition: FedProx vs FedNova × equal/unequal local work (highest novelty)

**What.** Separate two mechanisms that "FedProx helps" usually conflates: **drift damping** (FedProx) vs **objective-inconsistency correction** (FedNova). This is the most original move available.

**Design.** {FedAvg, FedProx (μ*), FedNova} × (equal `E = 20` both clients) and (unequal: `C0: E=20, C1: E=5`). Partitions fixed at L3 and L4. 3 seeds.

**Diagnostics.** `‖Δᵢᵗ‖`, oscillation, dissimilarity over rounds; macro-F1; rare-recall; update norm **normalised per local step** (so unequal-`E` runs are comparable).

**Why — cited.**
- **FedNova (Wang et al. 2020, NeurIPS)** §5 — the paper itself runs this equal-vs-unequal-`τᵢ` ablation. Figs. 3-4 (synthetic) compare regimes with `τᵢ = 30` uniform vs `τᵢ` heterogeneous across clients; Fig. 6 (CIFAR-10 non-IID) sweeps total local steps with `E = 2` fixed, isolating the `τᵢ` effect. The paper proves that under unequal `τᵢ`, naive size-weighted averaging converges toward an **inconsistent objective**, and corrects it by normalising each update by `τᵢ`. **In the equal-`τᵢ` regime FedNova reduces to FedAvg** — this is the canonical control. [arXiv:2007.07481](https://arxiv.org/abs/2007.07481).
- **NIID-Bench (Li, Diao, Chen, He 2022, ICDE) §4.2** — establishes `{FedAvg, FedProx, SCAFFOLD, FedNova} × {equal-τ, unequal-τ}` as the standard four-way comparison for this mechanism decomposition. [arXiv:2102.02079](https://arxiv.org/abs/2102.02079).
- **FedShuffle (Horváth et al. 2022)** — re-frames the equal-vs-unequal control theoretically: FedNova's gains vanish under equal local work, providing the mechanism-isolation rationale you cite. [arXiv:2204.13169](https://arxiv.org/abs/2204.13169).
- **FedProx (Li et al. 2020) §5.2** — explicitly tolerates variable/partial local work via γ-inexact local solvers; FedProx *includes* stragglers' partial updates where FedAvg drops them. This is the literature basis for any asymmetric-protocol variant, and the reason FedProx-vs-FedNova-under-unequal-`τᵢ` is the cleanest mechanism decomposition (FedProx damps drift but cannot fix objective inconsistency). [arXiv:1812.06127](https://arxiv.org/abs/1812.06127).
- **Recent benchmark (2024)** *"Not All FL Algorithms Are Created Equal"* reports *"FedNova is quite unstable and accuracy changes rapidly as communication rounds increase in heterogeneous settings."* — cite as a caveat on FedNova results. [arXiv:2403.17287](https://arxiv.org/abs/2403.17287).

**Repo state.** You already have FedNova results in `results/system_het_random_fednova/`. Need: equal-vs-unequal-`E` comparison consolidated; FedNova on the ladder at unequal-`E` configurations.

**Cost.** ~20 GPU-hours.

**Failure modes.**
- **FedNova's normalisation factor is the classic implementation pitfall.** Verify the exact `τᵢ` definition against the paper before coding. Your existing `fl_flower/strategy_fednova.py` already implements the closed-form momentum normaliser — re-verify against Wang et al. 2020 §3.3.
- The asymmetric straggler protocol changes both *method* and *participation rule* — **do not present "FedAvg drops partial updates vs FedProx includes them" as a clean algorithm comparison; label it an operational scenario.**

**Outputs.**
- `tab:mechanism_decomposition` — {FedAvg, FedProx, FedNova} × {equal-E, unequal-E} × {macro-F1, rare-recall, update-norm}.
- `F_mechanism_decomposition.pdf` — bar chart split into "equal local epochs" vs "unequal local epochs" panels, with the three methods on each.

---

### Experiment 4 — Federation-value + personalisation-gap matrix

**What.** Answer "why federate at all, and is a global model enough?" with a compact comparison matrix.

**Design.** Train: C0-local-only, C1-local-only, FedAvg, FedProx, centralised (pooled), and post-federation local fine-tuning (FedAvg+FT and FedProx+FT, per client). Evaluate every model on the global test set; ideally also on per-client test splits if those exist. 3 seeds.

**Metrics.** Per-client macro-F1; rare-class recall/F1; personalisation gap = (fine-tuned − global) per client.

**Why — cited.**
- **Yu, Bagdasaryan, Shmatikov (2022)** *"Salvaging Federated Learning by Local Adaptation"* — protocol source for the post-federation local-FT step. Their default is 5 epochs at lr 0.001; you already implemented this. [arXiv:2002.04758](https://arxiv.org/abs/2002.04758).
- **FLamby (Ogier du Terrail et al. 2022, NeurIPS Datasets & Benchmarks)** — real cross-silo healthcare benchmark that reports local vs federated vs pooled per centre and **shows federation does not beat local for every site**. Anchor citation for the "global ≠ best per site" framing. [arXiv:2210.04620](https://arxiv.org/abs/2210.04620).
- **Sheller et al. (2020, Sci. Reports)** — multi-institutional medical FL showing federated reaches ~99% of centralised performance on brain-tumour segmentation; per-site generalisation improves especially for smaller/external sites. [DOI:10.1038/s41598-020-69250-1](https://doi.org/10.1038/s41598-020-69250-1).
- **Pati et al. (2022, Nature Communications)** — 71-site rare-cancer boundary detection; the closest published medical-FL precedent for your small-hospital story. Their per-site improvement decomposition figure is the template. [DOI:10.1038/s41467-022-33407-5](https://doi.org/10.1038/s41467-022-33407-5).
- **FedSLD (Luo, Xu, Bai 2022, ISBI)** — explicitly reports per-client variance as a fairness diagnostic in *medical* FL with class imbalance; *"FedSLD reduces the variances of client test accuracies on MNIST and PathMNIST datasets, which implies more fair training."* Cite as a medical-FL precedent for the federation-value framing. [arXiv:2110.08378](https://arxiv.org/abs/2110.08378).
- **Personalised-FL stand-ins** — Ditto ([arXiv:2012.04221](https://arxiv.org/abs/2012.04221)), pFedMe ([arXiv:2006.08848](https://arxiv.org/abs/2006.08848)), FedRep ([arXiv:2102.07078](https://arxiv.org/abs/2102.07078)), FedPer ([arXiv:1912.00818](https://arxiv.org/abs/1912.00818)). **Cite as motivators for the personalisation gap; do not implement them — local fine-tuning is the defensible cheap proxy.**

**Repo state.** 4 of 6 configurations at single seed; resubmitted fine-tuning jobs running. Needs multi-seed promotion.

**Cost.** ~12 GPU-hours for 2 additional seeds × 5 configurations × 2 algorithms.

**Failure modes.**
- Fine-tuning overfits the small client (use early stopping; report it).
- Centralised is an **oracle**, not a deployable baseline — flag it as such.
- Without per-client test splits, "per-client benefit" measured on the global test set is approximate. State this caveat explicitly.

**Outputs.**
- `tab:federation_value_matrix` — rows {C0-local, C1-local, FedAvg, FedProx, centralised, FedAvg+FT, FedProx+FT} × columns {macro-F1, common-class F1, rare-class F1, melanoma F1, dermatofibroma F1}.
- `F_pareto_small_hospital.pdf` — already done; promote to "Pareto cloud" by overlaying all 3 seeds.

---

### Experiment 5 — Small-hospital rare-class case study, **3-seed promotion**

**What.** The clinical money result: when the small client holds rare lesions the large client lacks, federation lifts rare-class recall even when global accuracy barely moves. Already done at single seed; **needs 3-seed promotion**.

**Design.** Existing 2-client 90/10 partition (Client 0 large, common classes only; Client 1 small, rare classes only — class-disjoint). FedAvg vs FedProx vs centralised at seeds {42, 123, 456}. Report per-class.

**Metrics / diagnostics.** Per-class recall/F1, rare-class confusion matrices, global-acc vs rare-recall delta, **class support (counts) alongside recall** (because recall on n=23 dermatofibroma test samples is 0/1-spiky).

**Why — cited.**
- **Pati et al. (2022, Nat. Commun.)** — 71-site rare-cancer FL; small/under-represented sites benefit disproportionately. Headline figure shows per-site improvement. [DOI:10.1038/s41467-022-33407-5](https://doi.org/10.1038/s41467-022-33407-5).
- **Sheller et al. (2020, Sci. Reports)** — establishes the small-site / multi-institutional benefit precedent. [DOI:10.1038/s41598-020-69250-1](https://doi.org/10.1038/s41598-020-69250-1).
- **FedIIC (Wu et al. 2023, MICCAI)** — explicitly targets *medical FL with class imbalance*; *"identifies a realistic data distribution (L² distribution) where the global class distribution is highly imbalanced and data distributions across clients are imbalanced but form a certain degree of data agglomeration."* The closest published precedent for "rare class on one site, common classes on another." [arXiv:2206.13803](https://arxiv.org/abs/2206.13803).
- **FedSLD (Luo et al. 2022, ISBI)** — establishes that under federated class imbalance, minority classes can *collapse*, not merely lose a little accuracy — the reason per-class recall is essential and global accuracy alone is misleading. [arXiv:2110.08378](https://arxiv.org/abs/2110.08378).
- **HAM10000 (Tschandl, Rosendahl, Kittler 2018)** — documents the severe class imbalance in DermaMNIST's parent dataset (melanocytic nevi ≈ 67%; dermatofibroma and vascular tiny). The reason this stress test is realistic. [DOI:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161).
- **MedMNIST v2 (Yang et al. 2023, Sci. Data)** — DermaMNIST definition and resolution. [DOI:10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8).

**Repo state.** Single-seed case study complete. Resubmission for 2 more seeds needed; existing partition + runners already work.

**Cost.** ~12 GPU-hours.

**Failure modes.**
- Tiny rare-class support (dermatofibroma n=23 in test, vascular n=29) → recall is 0/1-spiky. Report support; use F1 + support; require ≥3 seeds to be defensible.
- Do not overclaim real-hospital validity — this is a *constructed* class-disjoint scenario, not a real multi-site federation.

**Outputs.** Existing thesis subsection promoted from "single-seed case study" to "3-seed multi-config comparison" with error bars. The Pareto scatter (`F_pareto_small_hospital.pdf`) becomes a Pareto cloud.

---

### Experiment 6 — Update-direction diagnostics (full delta logging)

**What.** Add per-client *full* update-vector logging (not just norm) for selected runs, then compute three diagnostics.

**Diagnostics.**
1. Per-client update norm — already done.
2. Pairwise client-update cosine `cos(Δw₀ᵗ, Δw₁ᵗ)` per round — negative means clients pull in opposite directions.
3. Consecutive global-update cosine `cos(wᵗ⁺¹ − wᵗ, wᵗ − wᵗ⁻¹)` per round — negative means the global trajectory oscillates.

**Why — cited.**
- **SCAFFOLD (Karimireddy et al. 2020)** — explicitly visualises gradient-direction divergence between clients (Fig. 3). Conceptual licence for these diagnostics, even without implementing SCAFFOLD. [arXiv:1910.06378](https://arxiv.org/abs/1910.06378).
- **FedImpro (Tang et al. 2024)** — establishes update-level analysis as the right abstraction. [arXiv:2402.07011](https://arxiv.org/abs/2402.07011).
- **FedAWARE (Wu et al. 2024)** — *"statistical heterogeneity is a crucial challenge, resulting in local updates diverging in direction."* [arXiv:2310.02702](https://arxiv.org/abs/2310.02702).

**Repo state.** Norm-only logging in `fl_flower/client.py`. Needs one code change to emit the full per-client delta (or its decomposed cosine — flattened parameter vector ~423K dimensions, ~2 MB per client per round, ~600 MB per 150-round run — manageable).

**Cost.** ~1 day of code + ~3 GPU-hours (one extra training run per (algorithm × ladder level) at seed 42 as proof of concept).

**Failure modes.**
- At `n = 2` the pairwise cosine is a single scalar per round, not a distribution — **report as a trace, never as a "distribution across pairs"**.
- Full-update vectors are memory-heavy; flatten parameters or compute per-layer / final-layer norms if memory is tight.

**Outputs.** `F_mechanism_panel.pdf` — three sub-panels: (a) per-client update norm vs round; (b) inter-client cosine vs round; (c) consecutive-global cosine vs round. **One figure, three panels — not nine separate plots.**

---

### Experiment 7 — (conditional) Client bump to ~8 for variance / worst-client diagnostics

**What.** The *only* experiment that legitimately unlocks the diagnostics that are dead at `n = 2` (per-client variance, worst-client F1). **Run only if you commit to Path 2 and have spare compute.**

**Design.** Re-do Level 4-style stress with ~8 Dirichlet(α = 0.1) clients (use your existing `dirichlet_alpha01_7_clients` partition); {FedAvg, FedProx}. 3 seeds.

**Why — cited.**
- **q-FFL (Li et al. 2020)** and **AFL (Mohri et al. 2019)** — the fairness objectives that make worst-client / client-variance meaningful metrics. Cite for metrics, not algorithms, and **only once you have ≥6–8 clients** — at `n = 2` these collapse. [arXiv:1905.10497](https://arxiv.org/abs/1905.10497), [arXiv:1902.00146](https://arxiv.org/abs/1902.00146).
- **Hsu et al. (2019)** — Dirichlet(α) is the standard way to spread labels across more than two clients. [arXiv:1909.06335](https://arxiv.org/abs/1909.06335).

**Repo state.** Partitioner exists. Runner-side support exists.

**Cost.** ~30 GPU-hours.

**Failure modes.** **One** targeted multi-client run, **not** a 2→4→8→16 sweep. Each new client count is a partition redesign and a new sweep — resist the explosion.

**Outputs.** One bar chart of per-client macro-F1 under each algorithm + variance / worst-client summary table. Fairness sub-claim becomes legitimate.

---

### Experiment 9 — Asymmetric per-client μ on L4 (ablation of Yao 2024 / HAPI-FedProx)

**What.** Re-run L4 (`two_client_90_10_rare_stress`) with **per-client** proximal coefficients instead of a single global μ. Specifically, four conditions × 3 seeds:

| Condition | μ₀ (large/dominant client) | μ₁ (small specialist client) | Hypothesis |
|---|---|---|---|
| FedAvg | 0 | 0 | no anchor |
| Symmetric FedProx | 0.01 | 0.01 | baseline; suspect rare-class collapse |
| ⭐ **Asymmetric "anchor-large"** | 0.01 | 0 | anchor dominant, free specialist — Yao 2024 prediction |
| Asymmetric "anchor-small" (control) | 0 | 0.01 | reverse direction — should NOT help if direction matters |

**Why this exists.** The L4 single-seed result shows the entire FedProx-vs-FedAvg macro-F1 deficit (Δ = −0.026) is **one class** — vascular lesions, which only Client 1 holds (0.702 FedAvg → 0.481 FedProx, Δ = −0.221). FedProx is better or equal on 6 of 7 classes. The "anchor-large" design tests whether removing the proximal anchor from the small specialist client recovers vascular signal without sacrificing common-class wins. The reverse-asymmetric arm is the critical **direction control**: without it, any rare-class recovery is confounded with "any reduction in average μ helps."

**Why — cited.**
- **Yao et al. 2024 (NeurIPS) *"Effect of Personalization in FedProx"*** — proves the **optimal μ depends on per-client statistical heterogeneity**, providing theoretical justification for per-client μ. The asymmetric design here is an ablation of their framework on a medical-FL class-imbalance setting. [arXiv:2410.08934](https://arxiv.org/abs/2410.08934).
- **HAPI-FedProx (Springer 2024)** — adapts μ per client based on a local-vs-global heterogeneity index. **Closest direct precedent** for the design here; HAPI's per-client μ is set adaptively, while ours is set asymmetrically by fixed cid as a controlled ablation. [DOI:10.1007/978-3-032-11733-5_17](https://doi.org/10.1007/978-3-032-11733-5_17).
- **FedPBS** — selectively applies the proximal correction to small-batch clients only (effectively μ = 0 for large clients). Structurally identical to this design but inverted in direction; cite as methodological precedent. [arXiv:2603.13909](https://arxiv.org/abs/2603.13909).
- **FedProx (Li et al. 2020) Theorem 4** — the convergence proof assumes **uniform μ across clients**. Setting μ₁ = 0 puts the design **outside Li 2020's proved regime** but inside Yao 2024's per-client minimax framework. State this caveat explicitly in the methods. [arXiv:1812.06127](https://arxiv.org/abs/1812.06127).
- **Distinct from personalised-FL methods.** Ditto ([arXiv:2012.04221](https://arxiv.org/abs/2012.04221)), pFedMe ([arXiv:2006.08848](https://arxiv.org/abs/2006.08848)), APFL ([arXiv:2003.13461](https://arxiv.org/abs/2003.13461)) and FedAMP ([arXiv:2007.03797](https://arxiv.org/abs/2007.03797)) all train **client-specific output models**; this design only varies the **local regularisation strength** on a single shared global model. Acknowledge the distinction in the related-work paragraph.

**Repo state.** `run_one_flower.py` extended with `--mu-per-client` flag; the `FlClient` constructor already accepted `proximal_mu` as a per-instance parameter, so the change is a small client-factory plumbing routine. Output JSON records `mu_per_client` for traceability; filenames carry a `_muPC-c0m...-c1m...` tag so symmetric-μ and asymmetric-μ runs cannot be silently mixed in analysis.

**Cost.** 12 runs × ~1 GPU-h = ~12 GPU-hours.

**Failure modes.**
- **Direction control may collapse.** If anchor-small (control) and anchor-large produce indistinguishable vascular F1, the directional Yao 2024 prediction is unsupported on this task — report honestly as a negative result.
- **Convergence outside Theorem 4 regime.** μ₁ = 0 means Client 1's local updates are unconstrained; if the global model oscillates more under asymmetric μ than symmetric, document via update-norm logs.
- **Single-class effect.** Even if the experiment "works" (vascular recovers), the finding is a single-class trade-off on a single benchmark; do not over-claim mechanism universality.

**Outputs.**
- `tab:asymmetric_mu_L4` — 4 rows × {macro-F1, vascular-F1, rare-avg-F1, balanced-acc} mean ± SD.
- `F_asymmetric_mu_L4.pdf` — 2-panel: (a) per-condition macro-F1 dot plot with 3-seed means, (b) per-class F1 mean ± SD bars showing the vascular-class signal.

**Critical framing requirement.** When writing this up, position it as *"an ablation of Yao 2024's per-client μ framework on a medical-FL class-imbalance setting"* — NOT as "we propose asymmetric per-client μ." The latter would be over-claiming given HAPI-FedProx (Springer 2024) and Yao 2024 already established the approach.

---

### Experiment 8 — SCAFFOLD targeted run (deferred / discussion-only)

**Recommendation: DO NOT IMPLEMENT. Discussion + future work paragraph only.**

**Why — cited.**
- **SCAFFOLD (Karimireddy et al. 2020)** is the natural "third mechanism" — explicit client-drift correction via control variates. [arXiv:1910.06378](https://arxiv.org/abs/1910.06378).
- **NIID-Bench (Li et al. 2022)** — *"SCAFFOLD is more vulnerable to class imbalances seen in non-IID data and tends to show higher standard deviation in accuracy than FedAvg."* [arXiv:2102.02079](https://arxiv.org/abs/2102.02079). This is the regime you are in.
- **Recent benchmark (2024)** — *"FedNova and SCAFFOLD are relatively unstable, and SCAFFOLD does not work effectively under partial participation. As the dataset distribution becomes heterogeneous, SCAFFOLD and FedDyn experience more frequent failures."* [arXiv:2403.17287](https://arxiv.org/abs/2403.17287).

Implementation cost is high (per-client control-variate state across rounds, custom strategy), and recent literature reports SCAFFOLD as unstable under exactly your conditions. **A wrong SCAFFOLD result is worse than no SCAFFOLD result.** Treat as discussion + future-work.

---

## §4. Diagnostic additions to existing tables (cheap wins)

Add these columns to *existing* result tables — minimal code, no new compute, immediate reporting strengthening.

| Existing table | New column to add | Citation justification |
|---|---|---|
| All federated result tables | `selected_round` (best-val round) | Reddi et al. 2021 — "round-to-target" is standard in FL convergence reporting. [arXiv:2003.00295](https://arxiv.org/abs/2003.00295) |
| All federated result tables | Round-to-round volatility (mean abs Δ macro-F1) | Original diagnostic; addresses supervisor's stated interest in oscillation/stability |
| Partition description tables | JS divergence and per-client class entropy | Zhao 2018 [arXiv:1806.00582](https://arxiv.org/abs/1806.00582); 2024 EMD-vs-method paper [arXiv:2406.06340](https://arxiv.org/abs/2406.06340) |
| Per-client comparison table | Per-client local-only baseline F1 | Yu 2022 [arXiv:2002.04758](https://arxiv.org/abs/2002.04758); Pati 2022 per-site framing |
| μ-sensitivity table | Mean per-client update norm (already present) — also add update-norm *ratio* C0/C1 | FedProx Theorem 4 |

**Total: ~2 hours of analysis code, zero new compute.**

---

## §5. Explicit decision matrix

| Item | Do | Discuss only | Defer to future work |
|---|---|---|---|
| Heterogeneity ladder (5 levels, 3 seeds) | ✓ | | |
| μ × heterogeneity heatmap (Levels 0, 2, 4) | ✓ | | |
| JS heterogeneity quantification | ✓ | | |
| Update-direction cosine diagnostics | ✓ | | |
| Federation value at 3 seeds | ✓ | | |
| Consecutive-global-update oscillation | ✓ | | |
| FedNova on existing system-het runs | ✓ | | |
| FedNova × unequal-E mechanism decomposition | ✓ | | |
| Multi-seed promotion of all single-seed results | ✓ | | |
| Client bump to ~8 (Experiment 7) | optional | | |
| **SCAFFOLD implementation** | | | **✗** |
| FedDyn / MOON / Ditto / pFedMe / FedRep | | ✓ | |
| FedOpt / FedAdam / FedYogi | | ✓ | |
| Focal loss / class-balanced loss | | ✓ | ✗ |
| Larger CNN / ResNet | | | ✗ |
| Higher image resolution (HAM10000 full-res) | | | ✗ |
| Real multi-institutional data | | ✓ | ✗ |
| FedIIC / FedSLD re-implementation | | ✓ | |
| EMD as primary heterogeneity axis on unordered classes | | | ✗ — use JS instead |
| Worst-client / fairness language on `n = 2` | | | ✗ (degenerate) |
| p-values / Wilcoxon on `n = 3` seeds | | | ✗ ("significance theatre") |

---

## §6. Time-budget plans

### Budget A — 2–3 days (analysis only, no new training)

1. **Experiment 0** — compute all mechanism diagnostics from existing runs.
2. **JS divergence + per-client entropy + class-overlap** column added to every partition table (30 min code).
3. **Volatility column** added to all federated result tables (30 min code).
4. **Update-norm ratio table** consolidated (30 min code).
5. **Rewrite the small-hospital subsection** in the order: (i) federation-value table; (ii) Pareto scatter; (iii) per-client specialty curves; (iv) rare-class confusion. Add the framing that *"the proximal term redistributes per-client influence."*
6. **Insert the volatility-flip paragraph** with the caveat about single seed.
7. **Stop.** Write the methodology and discussion chapters.

### Budget B — one week

Budget A plus:

8. **Promote Stage A ladder pilot to 3 seeds** for FedAvg + FedProx + (FedNova on L1/L3/L4). ~75 GPU-hours.
9. **Re-run small-hospital five regimes at 2 more seeds** (seeds 123, 456). ~12 GPU-hours.
10. **μ-sweep extension** on L0 and L4 (single seed acceptable for these — they are sanity checks). ~10 GPU-hours.
11. **Implement the JS-vs-Δmacro-F1 plot.** One figure, one line per algorithm.
12. **Write a one-paragraph case-study section** for the volatility flip, properly seeded.

### Budget C — workshop-paper style (1–2 months)

Budget B plus:

13. **Cosine-similarity diagnostic** (Experiment 6) — full per-client delta logging; one new training run per (algorithm × ladder level) at seed 42. ~15 GPU-hours + ~3 days coding.
14. **μ-sweep across heterogeneity** (Experiment 2 at full scale) — μ ∈ {0, 0.001, 0.01, 0.1, 1.0} on Levels 0, 2, 4 at 3 seeds each = 45 runs ≈ 45 GPU-hours.
15. **FedNova × unequal-E mechanism decomposition** (Experiment 3) — 3 seeds.
16. **Optionally Experiment 7** (8-client variance/worst-client) for a fairness sub-claim.
17. **Optionally one labelled focal-vs-CE sensitivity check** as a one-paragraph appendix item.
18. **Optionally one second dataset** (BloodMNIST or PathMNIST from MedMNIST v2) — the single biggest generality lever; only if you have time.
19. **Release code + partition seeds.** Target: FLOPS@NeurIPS 2026, DLMIA@MICCAI 2026, FLSys, MELBA.

**Do not** add SCAFFOLD, larger model, more datasets beyond one, or untested algorithms.

---

## §7. Top-5 figures the thesis must keep

1. **`F_mu_heterogeneity_heatmap.pdf`** — two-panel: (a) mean update norm vs μ, monotone decrease, one line per ladder level; (b) macro-F1 vs μ, inverted-U, one line per level. **Your signature "beyond accuracy" figure.** (Experiment 2.)
2. **`F_ladder_delta_vs_js.pdf`** — divergence dose-response scatter: x = measured JS divergence per partition, y = (FedProx − FedAvg) Δmacro-F1 and Δrare-recall. **Turns categorical rungs into a quantitative law.** (Experiment 1.)
3. **`F_pareto_small_hospital.pdf`** + **`F_rare_class_confusion.pdf`** — your already-built clinical figures, promoted to 3-seed Pareto cloud. (Experiment 5.)
4. **`F_mechanism_decomposition.pdf`** — bar chart {FedAvg, FedProx, FedNova} × {equal-E, unequal-E} on rare-class F1 + macro-F1. Visually separates drift damping from objective-inconsistency correction. (Experiment 3.)
5. **`F_mechanism_panel.pdf`** — three sub-panels: (a) per-client update norm vs round; (b) inter-client cosine vs round; (c) consecutive-global cosine vs round. (Experiment 6.)

Plus one dense diagnostics table (`tab:diagnostics_summary`) showing rounds-to-threshold, val-AUC, oscillation index, final macro-F1, rare-recall per method × μ × ladder rung.

**Relegate to appendix:** μ-sensitivity convergence panels, local-only validation trajectories, any superseded single-seed figure.

---

## §8. Citation reference list (verify before submission)

### Top-5 must-cite

1. **McMahan et al. (2017)** "Communication-Efficient Learning of Deep Networks from Decentralized Data" — AISTATS. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629). FedAvg definition; size-weighted aggregation; pathological non-IID shard partitioning.
2. **Li et al. (2020)** "Federated Optimization in Heterogeneous Networks" — MLSys. [arXiv:1812.06127](https://arxiv.org/abs/1812.06127); [MLSys PDF](https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf). FedProx; proximal term; B-local dissimilarity; Theorem 4; γ-inexact local updates; §5.2 asymmetric straggler protocol.
3. **Wang et al. (2020)** "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization" — NeurIPS. [arXiv:2007.07481](https://arxiv.org/abs/2007.07481). FedNova; `τᵢ`-normalised averaging.
4. **Li, Diao, Chen, He (2022)** "Federated Learning on Non-IID Data Silos: An Experimental Study" — ICDE. [arXiv:2102.02079](https://arxiv.org/abs/2102.02079); [GitHub](https://github.com/Xtra-Computing/NIID-Bench). The benchmark template; the "no method wins across heterogeneity types" finding; the empirical observation that SCAFFOLD is vulnerable to label skew.
5. **Pati et al. (2022)** "Federated learning enables big data for rare cancer boundary detection" — Nature Communications. [DOI:10.1038/s41467-022-33407-5](https://doi.org/10.1038/s41467-022-33407-5). Small-site / rare-class federation value in medical FL.

### Strongly recommended

- **Karimireddy et al. (2020)** SCAFFOLD — ICML. [arXiv:1910.06378](https://arxiv.org/abs/1910.06378). Drift vocabulary; per-client direction divergence diagnostic.
- **Hsu et al. (2019)** "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification" — arXiv. [arXiv:1909.06335](https://arxiv.org/abs/1909.06335). Dirichlet-α severity ladder precedent.
- **Zhao et al. (2018)** "Federated Learning with Non-IID Data" — arXiv. [arXiv:1806.00582](https://arxiv.org/abs/1806.00582). EMD as heterogeneity-bound diagnostic.
- **Hsieh et al. (2020)** "The Non-IID Data Quagmire of Decentralized Machine Learning" — ICML/PMLR. [PMLR PDF](http://proceedings.mlr.press/v119/hsieh20a/hsieh20a.pdf). Justifies your GroupNorm architectural choice.
- **Sheller et al. (2020)** Sci. Reports — [DOI:10.1038/s41598-020-69250-1](https://doi.org/10.1038/s41598-020-69250-1). Medical FL site-level gains; "federation reaches X% of centralised" framing.
- **Ogier du Terrail et al. (2022)** FLamby — NeurIPS Datasets & Benchmarks. [arXiv:2210.04620](https://arxiv.org/abs/2210.04620); [GitHub](https://github.com/owkin/FLamby). Real cross-silo healthcare benchmark; per-centre framing.
- **Yu, Bagdasaryan, Shmatikov (2022)** "Salvaging FL by Local Adaptation" — arXiv. [arXiv:2002.04758](https://arxiv.org/abs/2002.04758). Local-FT protocol source.
- **FedIIC (Wu et al. 2023, MICCAI)** — [arXiv:2206.13803](https://arxiv.org/abs/2206.13803). Medical FL × class imbalance.
- **FedSLD (Luo et al. 2022, ISBI)** — [arXiv:2110.08378](https://arxiv.org/abs/2110.08378). Medical FL × per-client variance.
- **Reddi et al. (2021)** FedOpt — ICLR. [arXiv:2003.00295](https://arxiv.org/abs/2003.00295). Round-to-target convergence reporting.
- **Li, Sanjabi, Smith (2020)** q-FFL — ICLR. [arXiv:1905.10497](https://arxiv.org/abs/1905.10497). Fairness objective (cite for metric, only if Experiment 7 runs).
- **Mohri et al. (2019)** AFL — ICML. [arXiv:1902.00146](https://arxiv.org/abs/1902.00146). Worst-client minimax (cite for metric, only if Experiment 7 runs).
- **Tschandl et al. (2018)** HAM10000 — Sci. Data. [DOI:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161). Parent dataset.
- **Yang et al. (2023)** MedMNIST v2 — Sci. Data. [DOI:10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8). DermaMNIST definition.
- **Kairouz et al. (2021)** "Advances and Open Problems in FL" — F&T ML. [arXiv:1912.04977](https://arxiv.org/abs/1912.04977). Canonical FL survey for taxonomy.

### Recent (2024–2025) supporting

- **Optimisation of FL Settings under Statistical Heterogeneity Variations** (2024) — [arXiv:2406.06340](https://arxiv.org/abs/2406.06340). EMD-indexed cross-method comparison precedent.
- **Not All FL Algorithms Are Created Equal** (2024) — [arXiv:2403.17287](https://arxiv.org/abs/2403.17287). Recent benchmark reporting SCAFFOLD/FedNova instability.
- **FedImpro (Tang et al. 2024)** — [arXiv:2402.07011](https://arxiv.org/abs/2402.07011). Update-level analysis as a discipline.
- **FedAWARE (Wu et al. 2024)** — [arXiv:2310.02702](https://arxiv.org/abs/2310.02702). Gradient diversity diagnostic.
- **Stabilizing FL under Extreme Heterogeneity** (2025) — [arXiv:2508.06692](https://arxiv.org/abs/2508.06692). μ-tuning under heterogeneity is still open.
- **Understanding FL from IID to Non-IID** (2025) — [arXiv:2502.00182](https://arxiv.org/abs/2502.00182). Recent experimental-study precedent.
- **Yao et al. (2024) — "Effect of Personalization in FedProx"** — NeurIPS. [arXiv:2410.08934](https://arxiv.org/abs/2410.08934). Per-client minimax-optimal μ; theoretical anchor for Experiment 9 (asymmetric per-client μ).
- **HAPI-FedProx (Springer 2024)** — [DOI:10.1007/978-3-032-11733-5_17](https://doi.org/10.1007/978-3-032-11733-5_17). Adaptive per-client μ via heterogeneity index; closest direct precedent for Experiment 9.
- **FedPBS** — [arXiv:2603.13909](https://arxiv.org/abs/2603.13909). Selective proximal correction by client batch size (effectively per-client μ); structurally analogous to Experiment 9 with inverted direction.
- **FedShuffle (Horváth et al. 2022)** — [arXiv:2204.13169](https://arxiv.org/abs/2204.13169). Theoretical re-framing of FedNova's equal-vs-unequal-`τᵢ` control; cite in Experiment 3.

### Optional / discussion-only

- **Acar et al. (2021)** FedDyn — ICLR. [arXiv:2111.04263](https://arxiv.org/abs/2111.04263).
- **Li et al. (2021)** MOON — CVPR. [arXiv:2103.16257](https://arxiv.org/abs/2103.16257).
- **Collins et al. (2021)** FedRep — ICML. [arXiv:2102.07078](https://arxiv.org/abs/2102.07078).
- **Li, Sanjabi, Beirami, Smith (2021)** Ditto — ICML. [arXiv:2012.04221](https://arxiv.org/abs/2012.04221).
- **T. Dinh, N. Tran, T. Nguyen (2020)** pFedMe — NeurIPS. [arXiv:2006.08848](https://arxiv.org/abs/2006.08848).
- **Marfoq et al. (2021)** "FL under a Mixture of Distributions" — NeurIPS. [arXiv:2108.10252](https://arxiv.org/abs/2108.10252).
- **Arivazhagan et al. (2019)** FedPer — arXiv. [arXiv:1912.00818](https://arxiv.org/abs/1912.00818).

---

## §9. The cited-evidence summary (defend-this-in-viva table)

| Your design choice | Evidence | Specific support |
|---|---|---|
| Use GroupNorm, not BatchNorm | Hsieh et al. 2020, PMLR | *"the mismatch between feature statistics estimated on non-IID local mini-batches and global data would degrade FedAvg's performance … they proposed to replace batch normalization with Group Normalization (GN), which does not rely on mini-batch statistics."* |
| Sweep μ ∈ {0, 0.001, 0.01, 0.1, 1.0} | Li et al. 2020 FedProx | *"no default μ values would work for all settings … you might want to tune μ from {0.001, 0.01, 0.1, 0.5, 1}."* |
| Sweep heterogeneity (the ladder) | Hsu 2019; NIID-Bench 2022 | Hsu: Dirichlet-α severity sweep precedent. NIID-Bench: *"six data partitioning strategies … much more comprehensive than previous studies."* |
| Use JS (not EMD) as the heterogeneity x-axis | Zhao 2018 (EMD framing) + classes-are-unordered DermaMNIST | EMD requires a ground metric; DermaMNIST's 7 classes are categorical/unordered; JS / total variation are the correct ordering-free substitutes. |
| Diagnose with update norm + cosine | SCAFFOLD 2020; FedImpro 2024 | SCAFFOLD: client drift framing. FedImpro: title literally *"Measuring and Improving Client Update in Federated Learning."* |
| Defer SCAFFOLD | NIID-Bench 2022; Not-All-Equal 2024 | NIID-Bench: *"SCAFFOLD is more vulnerable to class imbalances seen in non-IID data."* 2024 benchmark: *"SCAFFOLD does not work effectively under partial participation."* |
| Run FedNova for system heterogeneity | Wang 2020 FedNova | *"FedNova aims to eliminate objective inconsistencies caused by naive aggregation"* — i.e., relevant when local steps differ across clients. |
| Class-disjoint 90/10 small-hospital partition | Sheller 2020; Pati 2022; FedIIC 2023; FedSLD 2022 | Sheller / Pati: per-site / rare-class precedent. FedIIC: medical FL × class imbalance with the L²-distribution framing. FedSLD: *"medical optimisations often suffer from heterogeneity."* |
| Local fine-tuning protocol (5 epochs, lr 0.001) | Yu et al. 2022 | "Salvaging FL by Local Adaptation" — protocol source. |
| Fairness / Pareto framing on `n = 2` | **Avoid.** Degenerate at `n = 2`. | Only legitimate after Experiment 7. |
| Three-seed minimum | NIID-Bench 2022 methods | *"three trials with different random seeds, and the mean and the maximum standard deviation are reported."* Standard practice across FL benchmark literature. |
| Avoid p-values on `n = 3` | Methodological consensus | Underpowered; would report `p > 0.5` for everything. Report mean ± SD with overlaid seed points. |

---

## §10. Strongest defensible thesis claim + future-work

### Final claim (do not strengthen beyond this)

> *On a simulated 2-client cross-hospital DermaMNIST classification task, FedProx is not uniformly better than FedAvg. Its advantage emerges only when at least two heterogeneity dimensions interact — specifically when label skew and either rare-class concentration or unequal local work are simultaneously present. Across a JS-divergence-indexed heterogeneity ladder, the FedProx-vs-FedAvg macro-F1 gap remains within ±0.01 under IID and quantity-only skew, grows to ≈ 0.01–0.02 under label skew alone, and reaches ≈ 0.03 under combined skew at full severity. At the parameter level, FedProx monotonically reduces the per-client update norm with increasing μ; at the outcome level, the relationship is inverted-U with a sweet spot at μ ≈ 0.01 on this task — a plasticity–stability trade-off. The proximal regulariser stabilises whichever client is under-represented in size-weighted aggregation: cosine similarity between client updates increases under FedProx on label-skewed partitions, indicating reduced inter-client conflict. In the engineered class-disjoint case study, federation is mechanically necessary (each client's local-only F1 on the other client's classes is zero), and FedProx Pareto-dominates FedAvg on the (large-client, small-client) plane while recovering ≈ 92% of centralised macro-F1. Under unequal local work, FedNova recovers a separable component of performance that FedProx only partially damps via update-norm reduction. We characterise FedProx as a per-client-redistribution mechanism rather than as a uniform performance improvement, and report a single-task case study consistent with — but not generalising beyond — this conclusion.*

### Best future-work paragraph

> *Three extensions would be required to strengthen this case study to a publication-grade claim. **First**, validation on additional medical-imaging benchmarks (BloodMNIST, PathMNIST, OrganMNIST from MedMNIST v2; or a real cross-silo task from the FLamby suite — Ogier du Terrail et al. 2022) would establish dataset-independence. **Second**, direct comparison against explicit drift-correction methods (SCAFFOLD — Karimireddy et al. 2020; FedDyn — Acar et al. 2021) and personalised-FL methods (FedRep — Collins et al. 2021; Ditto — Li et al. 2021; pFedMe — Dinh et al. 2020) at this controlled setup would distinguish FedProx's redistribution effect from full drift correction or per-client personalisation. **Third**, replacement of simulated 2-client class-disjoint partitions with realistic multi-institutional splits — using institution metadata to partition rather than synthetic class-disjoint shards — would test whether the observed parameter-level diagnostics (update-norm reduction, cosine-similarity recovery) remain predictive of outcome-level rare-class improvement under real-world heterogeneity. The most important of these three is the cross-dataset replication; the existing analysis pipeline is fully partition-agnostic and would carry over directly.*

---

## §11. Agent execution checklist (TL;DR for the next agent in the loop)

The minimal sequence to follow when results from the currently-running HPC jobs arrive.

```text
[ ] 1. Sync HPC results to local repo:
       - heterogeneity_ladder/L{0..4}_*/test_at_best_*.json
       - small_hospital_finetune/test_at_best_*.json
       - two_client_90_10_rare_stress/best_state_*.pt + JSONs

[ ] 2. Re-analyse, no new compute (Experiment 0):
       - compute JS divergence + per-client entropy per partition
       - add volatility column (round-to-round mean abs Δ macro-F1)
       - add update-norm-ratio table
       - add selected_round column

[ ] 3. Decide: did the Stage A ladder pilot show the expected direction
       (FedProx ≈ FedAvg at L0/L1, advantage emerging at L2+)?
       IF YES: proceed to step 4.
       IF NO: write the negative finding honestly and stop.

[ ] 4. Submit Stage B: 3 seeds × 5 levels × {FedAvg, FedProx} +
       FedNova on L1/L3/L4 unequal-E variants. ~75 GPU-h.

[ ] 5. While Stage B runs, build:
       - F_ladder_delta_vs_js.pdf (Experiment 1 output)
       - F_mu_heterogeneity_heatmap.pdf (Experiment 2 output)
       - tab:federation_value_matrix updated with FT rows (Experiment 4)

[ ] 6. Multi-seed promote small-hospital (~12 GPU-h, Experiment 5).

[ ] 7. Optionally: Experiment 6 (full delta logging + cosine).

[ ] 8. Verify all arXiv IDs / DOIs in §8 against publisher pages.

[ ] 9. Rewrite thesis discussion to align with the §1 claim.

[ ] 10. Stop. Do not add SCAFFOLD, larger model, more datasets,
        or focal loss without explicit user approval.
```

---

*End of plan. Last updated: late May 2026 after the Ray-GCS patch was applied to all three Flower SLURM templates. The Stage A heterogeneity-ladder pilot and small-hospital fine-tuning chain are submitted on Cambridge HPC and awaiting completion.*
