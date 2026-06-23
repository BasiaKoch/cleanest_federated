# Federated Learning on DermaMNIST under Heterogeneity

*A Master's thesis on **when** federated optimisers preserve or destroy rare-class signal in imbalanced medical image classification — and **why**.*

## What this project is about

Federated learning (FL) lets multiple sites — think hospitals — train a shared model without pooling their data. In real medical settings that data is **non-IID** (each site sees a different mix of cases), **system-heterogeneous** (sites compute at different speeds and some drop out mid-round), and, most importantly, **severely class-imbalanced**: the clinically dangerous conditions are the rare ones.

This project studies that setting on **DermaMNIST** — a 7-class dermatology image benchmark (28×28) where melanocytic nevi make up ≈67% of the data while dermatofibroma and vascular lesions are ≈1% each and melanoma ≈11%. The central question is not "which optimiser is best" but **under which conditions** the standard aggregator **FedAvg** and the heterogeneity-oriented methods **FedProx** and **FedNova** keep the rare classes alive versus let them collapse. The contribution is a **regime map**: federated optimiser performance here is **regime-dependent**, not universal.

## Motivation

- **Class imbalance is the clinical crux.** Accuracy is misleading when one class is two-thirds of the data — a model that ignores every rare class still scores ≈67%. So the project reports **macro-F1** and **per-class** behaviour, and tracks the dominant failure mode: **rare-class collapse into the majority** (dangerous lesions silently misrouted as benign nevi).
- **Heterogeneity is unavoidable.** Sites differ in label distribution (*statistical* heterogeneity) and in compute/availability (stragglers, partial participation — *system* heterogeneity). FedProx and FedNova are proposed to cope, but the literature gives conflicting guidance on when they actually help.
- **The gap this fills.** Prior work mostly reports aggregate accuracy on balanced benchmarks; it doesn't characterise *which regime* makes each method protect (or destroy) the rare, clinically-critical classes. This thesis answers that with controlled, mechanism-isolating experiments.

## What it explores

1. **Statistical heterogeneity** — does non-IID data alone separate the optimisers? (§3.1)
2. **Proximal coefficient μ** — how sharply does FedProx depend on its tuning? (§3.2)
3. **Straggler asymmetry (centrepiece)** — when stragglers drop partial work, is FedProx's advantage from the *proximal term* or from *protocol-level handling of partial updates* (γ-inexact acceptance vs dropping)? (§3.3)
4. **Learning-rate asymmetry & FedNova** — how does FedNova's τ-normalisation help, and where does it break? (§3.4)
5. **Rare-class collapse & loss-side fixes** — is FedProx an imbalance corrector, or a drift fix, compared to class-weighted / focal loss? (§3.5)
6. **Mechanism & robustness** — what drives the effects (update norms), and do the conclusions survive analysis-choice changes? (§3.6)

## Methods

- **Task / data:** DermaMNIST (MedMNIST), 7 classes, 28×28, severe imbalance. **Primary metric: test macro-F1 at the best-validation checkpoint**; per-class F1 and rare→majority misrouting as clinical diagnostics.
- **Optimisers:** **FedAvg**, **FedProx** (μ ∈ {0, 0.001, 0.01, 0.1, 1.0}; μ=0 ≡ FedAvg), **FedNova**. Baselines: centralised (ceiling), local-only, fine-tune. Loss variants: cross-entropy, class-weighted CE, focal.
- **Heterogeneity by design:** an engineered balanced-paired 7-client partition (headline), a 2-client **L0→L4 "heterogeneity ladder"** isolating quantity- vs label-skew, Dirichlet(α=0.1), and a specialist partition; system-side: fixed/random stragglers, partial participation, unequal local epochs, and the **drop vs γ-inexact** partial-update protocols.
- **Implementation:** two equivalence-checked runtimes (a pure-PyTorch reference loop and a Flower simulation for HPC), deterministic seeding with tiered replication (n=10 paired seeds for headline comparisons, n=3 for mechanism probes), and per-run provenance.

## Key findings (the regime map)

- **Heterogeneity is necessary:** under IID, FedProx − FedAvg ≈ **−0.007** (within noise).
- **Statistical heterogeneity alone** yields only a small, runtime-sensitive FedProx edge, concentrated in the rare classes.
- **μ is sharply tuned (inverted-U):** best near **μ=0.01**, **catastrophic at μ=1.0**, with the damage landing on rare classes.
- **Centrepiece:** under straggler asymmetry, FedProx's gain is ≈**96% from γ-inexact partial-update acceptance, not the proximal term** — and a quantity-only control (L1) shows it only helps when the dropped client carries **informationally-unique rare-class signal**. Under a "perfect storm" of stressors the gap reaches **+0.404 macro-F1**.
- **FedNova is regime-dependent both ways:** robust under persistent learning-rate asymmetry (its strongest single-axis gain) but **collapses under random-τ stragglers**.
- **FedProx is not an imbalance corrector:** it acts at aggregation/drift — a loss-side fix (class-weighted CE on FedAvg) matches or beats it.
- **Central claim:** *a protocol preserves rare-class F1 iff rare-client signal reaches the global model.*

> **The report.** The submission compiles from **`report/submission_bundle/`** (`main.tex`, pdfLaTeX + BibTeX) — a self-contained four-chapter report (Introduction, Methods, Results, Discussion) plus an Appendix, with its figures and `references.bib`. `report/drafts/FULL_THESIS.tex` is the earlier full-source draft. The table below maps each report section/figure to the code, experiments, and results behind it.

---

## Repository navigation: thesis section → code → results → figures

Orientation for a reader coming from the report:

- **`fl_dermamnist/`** is the importable source package (`python -m fl_dermamnist.…`). The name is **historical** — the thesis uses **DermaMNIST** only (there is no MNIST experiment).
- **`fl_dermamnist/results/`** holds the raw/generated experiment artifacts (one directory per experiment).
- **`fl_dermamnist/results/thesis_ready/figures/`** holds the **canonical thesis-ready figure PDFs**.
- **`report/submission_bundle/`** is the **frozen submission bundle** (the `main.tex` compile target; its figures are flat copies of the canonical ones).

Path shorthands used in the table below: launchers live in `infra/slurm/`; result directories in `fl_dermamnist/results/`; analysis scripts in `fl_dermamnist/analysis/` (**`ANALYSIS/`**); figure scripts in `fl_dermamnist/figures/` (**`FIGURES/`**); conceptual-diagram scripts in `docs/whole_paper_example_audit/figure_generation/` (**`CONCEPTUAL/`**).

| Thesis item | What it explains | Main result directory | Launcher / run script | Analysis script | Figure script | Output figure/table |
|---|---|---|---|---|---|---|
| Fig 2.1 | FL training loop (conceptual) | — | — | — | `CONCEPTUAL/make_fl_workflow.py` | `F_federated_learning_workflow.pdf` |
| Fig 2.2 | DermaMNIST class imbalance | — | — | `data/load.py` | `FIGURES/plot_dataset_classes.py` | `F_dataset_classes.pdf` |
| Fig 2.3 | engineered balanced-paired partition | `partitions/` | — | `data/partition.py` (`balanced_paired_7_clients`) | `FIGURES/plot_engineered_partition_heatmap.py` | `F_engineered_partition_heatmap.pdf` |
| Fig 2.4 | L1/L4 mechanism partitions | `partitions/` | — | `data/partition.py` | `FIGURES/plot_l1_l4_partition_schematic.py` | `F_l1_l4_partition_schematic.pdf` |
| Fig 2.5 | optimiser comparison (conceptual) | — | — | — | `CONCEPTUAL/make_optimizer_comparison.py` | `F_optimizer_comparison_triptych.pdf` |
| Fig 2.6 | rare-client signal flow (conceptual) | — | — | — | `CONCEPTUAL/make_signal_flow_gamma_inexact.py` | `F_signal_flow_gamma_inexact.pdf` |
| §3.1 / Fig 3.1 | statistical heterogeneity | `flower_C0_baseline` (primary), `headline`, `iid`, `dirichlet_a01`, `specialist_partition`, `node_pinned_L4` | `submit_flower_C0_baseline.sh` (+ `submit_headline.sh`, `submit_robustness.sh`, `submit_specialist_partition.sh`, `submit_node_pinned_L4.sh`) | `ANALYSIS/analyse_statistical_heterogeneity.py` | `FIGURES/plot_engineered_per_class.py` | `F_engineered_per_class.pdf`; Table `stat-het-headline` ¹ |
| §3.2 / Fig 3.2 | μ (proximal) sensitivity | `mu_sensitivity_flower` | `submit_mu_sensitivity_clean.sh` | `ANALYSIS/analyse_mu_sensitivity.py` | `ANALYSIS/analyse_mu_sensitivity.py` (same; analyse+plot) | `F_mu_sensitivity_outcome_and_convergence.pdf`; Table `mu-sweep` |
| §3.3 / Figs 3.3–3.4 | L4 four-condition decomposition | `li2020_asymmetric_L4`, `li2020_asymmetric_L1` (control) | `submit_li2020_asymmetric_L4.sh`, `submit_li2020_quantity_skew_L1.sh` | `ANALYSIS/analyse_li2020_asymmetric_L4.py` | Fig 3.3 `FIGURES/plot_validation_curves_extreme_gaps.py`; Fig 3.4 ² | `F_val_curves_extreme_gaps.pdf`; `F_l4_four_condition_per_class_grid.pdf`; Tables `li-decomp-l4`/`-l1` |
| §3.3 (Fig 3.3 panel B) | perfect-storm L4 | `fedprox_perfect_storm_L4` | `submit_fedprox_perfect_storm_L4.sh` | `ANALYSIS/analyse_li2020_asymmetric_L4.py` | (panel of Fig 3.3) | in `F_val_curves_extreme_gaps.pdf` |
| §3.4 / Fig 3.5 | LR-asymmetry envelope | `asymmetric_lr_L4` | `submit_asymmetric_lr_L4.sh` (+ `_fednova_only.sh`, `submit_fednova_lr_envelope_L4.sh`) | `ANALYSIS/analyse_asymmetric_lr_L4.py` | `ANALYSIS/analyse_asymmetric_lr_L4.py` (same; analyse+plot) | `F_asymmetric_lr_L4.pdf`; Table `lr-envelope` |
| §3.4 / Fig 3.6 + table | FedNova random-τ stress | `system_het_random_fednova` | `submit_fednova_system_het.sh` | `ANALYSIS/analyse_system_heterogeneity.py` | `FIGURES/plot_heterogeneity_escalation.py` ³ | `F_heterogeneity_escalation.pdf`; Table `fednova-regime` |
| §3.5 / Fig 3.7 | rare-class collapse (per-class) | pools `li2020_*`, `fedprox_*`, `asymmetric_lr_L4`, `node_pinned_L4`, `mu_sensitivity_flower` | — | (per-class analysers) | `FIGURES/plot_cross_experiment_per_class_heatmap.py` | `F_cross_experiment_per_class.pdf`; Table `collapse-cells` |
| §3.5 / table | weighted-CE / focal loss | `fedprox_weighted_ce_L4` | `submit_fedprox_weighted_ce_L4.sh`, `submit_fedprox_focal_loss_L4.sh` | `ANALYSIS/analyse_fedprox_weighted_ce_L4.py` | — (table only) | Table `loss-side` |
| §3.6 / Fig 3.8 | update-norm diagnostics | `flower_C0_baseline` (run with `--log-update-norms`) | (re-run of §3.1) | `ANALYSIS/analyse_d1_mechanism.py` | `FIGURES/plot_update_norms.py` | `F_update_norms.pdf` |
| Fig 4.1 | regime map (synthesis) | — | — | — | `CONCEPTUAL/make_regime_map_summary.py` | `F_regime_map_summary.pdf`; Table `final-claims` |
| Appendix | L4 confusion matrix | `li2020_asymmetric_L4` | — | `ANALYSIS/analyse_l4_confusion_matrices.py` | `ANALYSIS/analyse_l4_confusion_matrices.py` (same) | `F_l4_confusion_li2020.pdf` ⁴ |

¹ The §3.1 forest plot `F_statistical_heterogeneity_forest.pdf` is generated and used in the older `FULL_THESIS.tex` / `RESULTS_CHAPTER.tex` drafts, but is **not included in the final submission bundle** (`report/submission_bundle/main.tex`).
² `F_l4_four_condition_per_class_grid.pdf` exists in `…/thesis_ready/figures/` and the bundle, but **no current tracked script emits that exact filename** — verify its generator before regenerating. (`FIGURES/plot_validation_curves_l4_four_condition.py` emits the appendix single-panel `F_val_curves_l4_four_condition.pdf`, not this grid.)
³ `plot_heterogeneity_escalation.py` currently embeds its summary numbers as literals rather than reading them from a CSV.
⁴ `analyse_l4_confusion_matrices.py` writes `F_l4_confusion_li2020.pdf` (the four-condition mechanism); the submission bundle includes it **renamed to `F_l4_confusion_four_condition.pdf`**.

---

## 1. Where are the training files?  (the short answer)

| You want… | It lives in |
|---|---|
| **Training code** (the loops that train models) | `fl_dermamnist/experiments/run_*.py` (entrypoints) + `fl_dermamnist/fl/` (pure-PyTorch core) + `fl_dermamnist/fl_flower/` (Flower runtime) |
| **Model / data / partition code** | `fl_dermamnist/models/`, `fl_dermamnist/data/` |
| **Training launchers** (one job → one sweep) | `infra/slurm/submit_*.sh`, `slurm_template_*.sh` |
| **Training outputs / artefacts** (the actual results) | `fl_dermamnist/results/<experiment_name>/` |
| **Analysis → thesis tables & figures** | `fl_dermamnist/{analysis,figures}/` → `…/data/` + `…/figures/` |
| **Dataset** | `dermamnist_64.npz` (root; git-ignored, ~100 MB, re-downloadable) |
| **How to run anything locally** | `infra/local/commands.sh <step>` |

There is **one directory per experiment** under `fl_dermamnist/results/`. Each
training run writes a fixed set of artefacts into its experiment directory (see
§4). **There is no `config.yaml`** — every run's full configuration is embedded
inside its `test_at_best_*.json`, and the filename encodes method/μ/epochs/seed.

---

## 2. Quickstart

```bash
cd /path/to/cleanest_federated
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# (place dermamnist_64.npz at repo root — see data/load.py)

# sanity: FedProx(μ=0) ≡ FedAvg + proximal-term + provenance unit tests
bash infra/local/commands.sh sanity

# one training run (engineered partition, seed 42, 20 local epochs, 150 rounds)
bash infra/local/commands.sh flower-fedprox     # Flower runtime
bash infra/local/commands.sh purepy-fedavg      # pure-PyTorch runtime

# regenerate thesis tables + figures from existing results
bash infra/local/commands.sh analyse

# show per-experiment completion status
bash infra/local/commands.sh status
```

Full hyperparameter sweeps were run on HPC via the `submit_*.sh` launchers
(§5); `commands.sh` exposes single-seed local smoke versions.

---

## 3. Training code — entrypoints and what each trains

All entrypoints are run as modules from the repo root with `PYTHONPATH=.`
(`python -m fl_dermamnist.experiments.<name>`), default output dir shown.

| Entrypoint | Trains | Default `--out-dir` | Used by |
|---|---|---|---|
| `experiments/run_one.py` | **Pure-PyTorch** FL (FedAvg/FedProx) — reference loop | `results/headline` | engineered headline, iid, dirichlet |
| `experiments/run_one_flower.py` | **Flower** FL (FedAvg/FedProx) — simulation runtime; supports stragglers, partial participation, loss-type, per-client lr/μ | `results/headline_flower` | most experiments |
| `experiments/run_one_fednova_flower.py` | **FedNova** arm (Flower) — normalised averaging, server-momentum/lr, τ-clip | (per launcher) | FedNova comparators |
| `experiments/run_centralised.py` | Centralised (non-federated) reference | `results/centralised` | performance ceiling |
| `experiments/run_local_only.py` | Single-client local-only training (no federation) | `results/small_hospital_local_only` | small-hospital case study |
| `experiments/run_finetune.py` | Fine-tune a saved global checkpoint (`best_state_*.pt`) on one client | `results/small_hospital_finetune` | small-hospital case study |
| `experiments/verify_flower_equivalence.py` | μ=0 pure-PyTorch ↔ Flower smoke check | — | runtime equivalence |
| `experiments/compare_equivalence_full_scale.py` | Full cross-runtime equivalence audit | — | runtime equivalence |

**Training core (imported by the entrypoints):**
`fl/server_loop.py` (paired-fair FL run), `fl/local_train.py` (CE + gated FedProx
proximal term — μ=0 ⇒ FedAvg), `fl/aggregation.py` (size-weighted averaging),
`fl/evaluation.py` (macro-F1 / balanced-acc / per-class F1), `fl/system_het.py`
(straggler schedules), `fl/class_imbalance.py` (CE / weighted-CE / focal),
`fl/seeding.py`, `fl/provenance.py`, `fl/runtime_provenance.py`;
`fl_flower/client.py`, `client_fednova.py`, `strategy_fednova.py`,
`strategy_straggler_dropping.py`; `models/dermmnist_cnn.py` (+`_bn.py` ablation);
`data/load.py`, `data/partition.py`.

> For deeper per-module detail and the spec-compliance cross-reference, see
> **`fl_dermamnist/README.md`**.

---

## 4. Training outputs — artefact schema (per run)

Every federated training run writes these into its experiment directory, with
`<stem>` = e.g. `fedprox_mu0.01_E20_s42`:

| File | Contents |
|---|---|
| `test_at_best_<stem>.json` | **Primary**: test macro-F1 + per-class F1 at best-val checkpoint, **plus the full run config** (method, μ, lr, rounds, epochs, partition, seed, fraction-fit, dataset path, git commit) |
| `history_<stem>.csv` | Per-round train/val trajectories (centralised runs use `history_centralised_seed*.json`) |
| `test_predictions_<stem>.npz` | Saved test predictions (for confusion / rare→mel-nevi misrouting) |
| `client_update_norms_<stem>.csv` | Per-round per-client ‖Δw‖ (mechanism diagnostics; only when `--log-update-norms`) |
| `best_state_<stem>.pt` | Best global model weights (only ladder + two-client dirs; consumed by `run_finetune.py`) |
| `analysis/` | Per-experiment derived summaries (`*_summary.csv`, `paired_stats.json`, …) |

---

## 5. Experiment directories — index

Every experiment writes to one directory under `fl_dermamnist/results/`. The
navigation table above links the main ones to their launchers, scripts, and
figures; this is just the **full index** of result directories and whether each
appears in the report. Run defaults (unless a launcher overrides): 150 rounds,
20 local epochs, lr 0.01, μ 0.01, CE loss; 10 paired seeds for headline
comparisons, 3 for mechanism probes.

**In the report (Results §3)**

| Directory | § | What it is |
|---|---|---|
| `flower_C0_baseline` | 3.1 | engineered 7-client headline — the primary FedAvg-vs-FedProx comparison |
| `headline` | 3.1 | pure-PyTorch headline (source of the 10 paired-seed numbers) |
| `iid`, `flower_C0_iid_baseline` | 3.1 | IID negative control |
| `dirichlet_a01` | 3.1 | Dirichlet α=0.1 robustness |
| `specialist_partition` | 3.1 | specialist 1-of-7 robustness |
| `node_pinned_L4` | 3.1 | variance control |
| `centralised` | 3.1 | non-federated performance ceiling |
| `mu_sensitivity_flower` | 3.2 | μ inverted-U sweep |
| `li2020_asymmetric_L4` | 3.3 | four-condition straggler decomposition (centrepiece) |
| `li2020_asymmetric_L1` | 3.3 | L1 quantity-only control |
| `fedprox_perfect_storm_L4` | 3.3 | extreme-stress demonstration |
| `asymmetric_lr_L4` | 3.4 | learning-rate asymmetry, 1:1 → 50:1 |
| `system_het_random_fednova` | 3.4 | FedNova random-τ collapse |
| `system_het_{fixed,random,iid_fixed,iid_random}` | 3.4/3.6 | statistical × system-heterogeneity interaction |
| `fedprox_weighted_ce_L4` | 3.5 | CE / weighted-CE / focal-loss comparison |

**Appendix / diagnostics**

| Directory | What it is |
|---|---|
| `mu_sweep_ladder`, `heterogeneity_ladder` | μ × ladder and L0→L4 ladder pilots |
| `two_client_90_10_rare_stress` | L4 deep-dive |
| `system_het_partial_C0.5` | partial participation (C = 0.5) |
| `asymmetric_mu_L4` | per-client μ |
| `fednova_unequal_E`, `e_sweep`, `e_sweep_dirichlet_a01` | local-epoch sweeps |
| `extended_rounds_L3` | longer-run convergence check |
| `small_hospital_local_only`, `small_hospital_finetune` | clinical small-site case study |

**Not in the report**

| Directory | What it is |
|---|---|
| `system_het_random_fednova_{baseline,mom0,serverlr03,servmom,tauclip320}` | FedNova collapse-mechanism probes (work in progress) |
| `arch_ablation_bn` | BatchNorm ablation (archived) |
| `partitions` | partition definitions, generated by `data/partition.py` |

---

## 6. Data flow

```
dermamnist_64.npz
   │  data/load.py + data/partition.py
   ▼
experiments/run_*.py  ──(uses)──>  fl/ , fl_flower/ , models/
   │   (launched at scale by infra/slurm/submit_*.sh)
   ▼
fl_dermamnist/results/<experiment>/        ← raw training artefacts (§4)
   │   fl_dermamnist/analysis/*.py + fl_dermamnist/figures/*.py
   ▼
results/thesis_ready/data/*.csv|json   +   results/thesis_ready/figures/F_*.pdf
   │   \input / \includegraphics
   ▼
report/submission_bundle/main.tex  →  thesis PDF
```

---

## 7. Repository top-level map

```
cleanest_federated/
├── README.md                      ← you are here (repo map + thesis→code navigation)
├── requirements.txt
├── pyproject.toml                 ← installable package (pip install -e .)
├── dermamnist_64.npz              ← dataset (git-ignored, untracked, ~100 MB, re-downloadable)
├── fl_dermamnist/                 ← the source package (Federated Learning on DermaMNIST)
│   ├── README.md                  ← detailed package + spec-compliance doc
│   ├── configs/  data/  models/  fl/  fl_flower/   ← training code
│   ├── common/                    ← paths.py (repo/results/figure path resolver)
│   ├── experiments/               ← training entrypoints (run_*.py)
│   ├── analysis/                  ← analyse_*.py (moved out of results/thesis_ready/scripts)
│   ├── figures/                   ← plot_*.py figure generators
│   ├── tests/                     ← μ=0≡FedAvg, proximal-term, provenance guards
│   └── results/                   ← one dir per experiment (raw artefacts)
│       └── thesis_ready/          ← data/ + figures/ (canonical thesis-ready outputs)
├── infra/                         ← HPC: slurm/ (templates + submitters), runpod/, local/ (commands.sh, analyse_all.sh)
├── report/                        ← all thesis text
│   ├── submission_bundle/         ← the compiled report: main.tex + figures + references.bib
│   ├── chapters/                  ← §5 section sources (5_*.tex) + bibliographies
│   └── drafts/                    ← historical drafts (FULL_THESIS.tex, RESULTS_CHAPTER.tex, …)
├── docs/                          ← verification sheet + audit notes
└── archive/                       ← legacy code (code_legacy/) + old build guides (docs_legacy/)
```

---

## 8. Organization & conventions

- **One experiment ⇒ one directory** under `fl_dermamnist/results/`, named for
  the experiment (not the method); methods/seeds are distinguished by filename.
- **Filenames are self-describing**: `<method>_mu<μ>_E<epochs>_s<seed>` (e.g.
  `fedprox_mu0.01_E20_s42`); `mu0.0` ⇒ FedAvg.
- **Config travels with the result** (inside `test_at_best_*.json`) — there is
  no separate config file to lose.
- **`thesis_ready/` is the curated bundle**: `writing/` (tex + bib),
  `figures/` (the 15 referenced `F_*.pdf`), `data/` (aggregate tables), and
  `scripts/` (analysers that regenerate them). The thesis `\graphicspath`
  points at `thesis_ready/figures/`.
- **Paths are hardcoded** in the launchers, entrypoint defaults, and the thesis
  `\graphicspath`. Result directories were therefore **not** relocated during
  cleanup — moving them would break reproduction and figure resolution. If a
  physical reorganization is wanted, it must update those references in lockstep.

## 9. Reproducing & verifying
- Implementation guards: `bash infra/local/commands.sh sanity` (run before trusting any claim).
- Pre-submission preflight: `bash infra/local/pre_submission_check.sh`.
- Claim verification ledger: `docs/VERIFICATION_SHEET.txt`. Its two historical ⚠ flags (node-pinned-L4 sign; FedProx update-norm magnitude) were re-verified in the final audit and **both resolve in the report's favour** — the report's `+0.005` and `1.029` (−32%) match the source CSVs; the sheet entries were stale (see the appended final-audit resolution).
- Full provenance / cleanup audit: `docs/repo_audit_submission_cleanup/`.

> Note: `commands.sh` and `fl_dermamnist/README.md` reference a
> `fl_dermamnist/results/PROVENANCE_AUDIT.md` that is **not present**; its role
> is currently served by `docs/VERIFICATION_SHEET.txt` and
> `docs/repo_audit_submission_cleanup/02_RESULT_TRACEABILITY_MATRIX.csv`.

---

## 10. Reproducing the report figures & tables (assessor guide)

The figures and tables in the report reproduce **from saved results without
rerunning training** (full sweeps need HPC; see the seed/HPC note below).

**Compile the report.** Upload the contents of `report/submission_bundle/` to Overleaf
(or compile locally): main document `main.tex`, compiler **pdfLaTeX**, then BibTeX.
The bundle is flat and self-contained — `main.tex`, `references.bib`, and the 15
`F_*.pdf` figures.

**Regenerate the figures/tables from saved outputs** (CPU, minutes):

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# place dermamnist_64.npz at the repo root (see fl_dermamnist/data/load.py)
bash infra/local/commands.sh sanity     # implementation guards (μ=0≡FedAvg, etc.)
bash infra/local/commands.sh analyse    # rebuild thesis tables + figures from saved results
```

**Where things live:**

| Artefact | Location |
|---|---|
| Final report (compile target) | `report/submission_bundle/main.tex` |
| Report figures (used by the PDF) | `report/submission_bundle/F_*.pdf` — copies of `fl_dermamnist/results/thesis_ready/figures/` |
| Figure/table generators | `fl_dermamnist/{analysis,figures}/` (`plot_*.py`, `analyse_*.py`) and `docs/whole_paper_example_audit/figure_generation/` (the three conceptual diagrams) |
| Aggregated tables (CSV/JSON) | `fl_dermamnist/results/thesis_ready/data/` |
| Raw per-run experiment outputs | `fl_dermamnist/results/<experiment>/` (schema in §4) |
| Numerical claim ledger | `docs/VERIFICATION_SHEET.txt` |

Saved per-run artefacts (`test_at_best_*.json`, `history_*.csv`, …) are provided so
every reported number traces back to a source file and the figures/tables can be
rebuilt without HPC.

### Seed tiers & HPC note

Seed counts differ across experiments because **HPC access was restricted**; the
count is reported per experiment in the report (Table 2.6 and Limitations).

- **n = 10 paired seeds** — headline comparisons (engineered statistical
  heterogeneity, IID control, μ-sweep, FedNova random-τ). Reported with paired
  wins / Wilcoxon where applicable.
- **n = 3 matched seeds** — mechanism and asymmetry probes (four-condition L4,
  perfect-storm, learning-rate asymmetry, weighted-CE, asymmetric-μ, node-pinned
  L4). Interpreted **directionally; no significance testing** at this sample size.
- **single-seed pilots** — the L0–L4 heterogeneity ladder; orientation only, never
  a headline number.

Full experiment reruns were performed on HPC via the `infra/slurm/submit_*.sh`
launchers and can be costly; the saved outputs above are the intended path for
reproducing the report's figures and tables.

---

## 11. AI / code-assistance statement

**Claude Code** (Anthropic's CLI), using the **Claude Opus 4.8** model, was used during
this project for coding, debugging, experiments, and repository organization. All
experiments, numerical results, their interpretation, and the scientific claims in the
report are the author's own work and remain the author's responsibility; AI-assisted
code and outputs were reviewed by the author. This statement should be read alongside,
and adjusted to comply with, the relevant course policy on AI use.
