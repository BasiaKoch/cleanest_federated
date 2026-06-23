# Federated Learning on DermaMNIST under Heterogeneity — Thesis Repository

This repository contains the code, experiments, training artefacts, and LaTeX
source for a thesis on **federated learning for imbalanced medical image
classification** (DermaMNIST, 7 skin-lesion classes). The contribution is a
**regime map** of when **FedAvg**, **FedProx**, and **FedNova** preserve or
destroy rare-class signal under statistical and system heterogeneity. Primary
metric: **test macro-F1 at the best-validation checkpoint**.

> **Where is the final report?** The submission report compiles from
> **`report/submission_bundle/`** (`main.tex`, pdfLaTeX + BibTeX) — a self-contained
> four-chapter report (Introduction, Methods, Results, Discussion/Conclusion, plus
> an Appendix) with its 15 figures and `references.bib`. `report/drafts/FULL_THESIS.tex`
> is the earlier full-source draft kept for history; `docs/repo_audit_submission_cleanup/`
> holds the full submission audit. See §10 below for the assessor reproducibility path.

---

## Repository navigation: thesis section → code → results → figures

Orientation for a reader coming from the report:

- **`fl_dermamnist/`** is the importable source package (`python -m fl_dermamnist.…`). The name is **historical** — the thesis uses **DermaMNIST** only (there is no MNIST experiment).
- **`fl_dermamnist/results/`** holds the raw/generated experiment artifacts (one directory per experiment).
- **`fl_dermamnist/results/thesis_ready/figures/`** holds the **canonical thesis-ready figure PDFs**.
- **`report/submission_bundle/`** is the **frozen submission bundle** (the `main.tex` compile target; its figures are flat copies of the canonical ones).

Path shorthands used in the table below: launchers live in `fl_dermamnist/scripts/`; result directories in `fl_dermamnist/results/`; data-figure + analysis scripts in `fl_dermamnist/results/thesis_ready/scripts/` (**`SCRIPTS/`**); conceptual-diagram scripts in `docs/whole_paper_example_audit/figure_generation/` (**`CONCEPTUAL/`**).

| Thesis item | What it explains | Main result directory | Launcher / run script | Analysis script | Figure script | Output figure/table |
|---|---|---|---|---|---|---|
| Fig 2.1 | FL training loop (conceptual) | — | — | — | `CONCEPTUAL/make_fl_workflow.py` | `F_federated_learning_workflow.pdf` |
| Fig 2.2 | DermaMNIST class imbalance | — | — | `data/load.py` | `SCRIPTS/plot_dataset_classes.py` | `F_dataset_classes.pdf` |
| Fig 2.3 | engineered balanced-paired partition | `partitions/` | — | `data/partition.py` (`balanced_paired_7_clients`) | `SCRIPTS/plot_engineered_partition_heatmap.py` | `F_engineered_partition_heatmap.pdf` |
| Fig 2.4 | L1/L4 mechanism partitions | `partitions/` | — | `data/partition.py` | `SCRIPTS/plot_l1_l4_partition_schematic.py` | `F_l1_l4_partition_schematic.pdf` |
| Fig 2.5 | optimiser comparison (conceptual) | — | — | — | `CONCEPTUAL/make_optimizer_comparison.py` | `F_optimizer_comparison_triptych.pdf` |
| Fig 2.6 | rare-client signal flow (conceptual) | — | — | — | `CONCEPTUAL/make_signal_flow_gamma_inexact.py` | `F_signal_flow_gamma_inexact.pdf` |
| §3.1 / Fig 3.1 | statistical heterogeneity | `flower_C0_baseline` (primary), `headline`, `iid`, `dirichlet_a01`, `specialist_partition`, `node_pinned_L4` | `submit_flower_C0_baseline.sh` (+ `submit_headline.sh`, `submit_robustness.sh`, `submit_specialist_partition.sh`, `submit_node_pinned_L4.sh`) | `SCRIPTS/analyse_statistical_heterogeneity.py` | `SCRIPTS/plot_engineered_per_class.py` | `F_engineered_per_class.pdf`; Table `stat-het-headline` ¹ |
| §3.2 / Fig 3.2 | μ (proximal) sensitivity | `mu_sensitivity_flower` | `submit_mu_sensitivity_clean.sh` | `SCRIPTS/analyse_mu_sensitivity.py` | `SCRIPTS/analyse_mu_sensitivity.py` (same; analyse+plot) | `F_mu_sensitivity_outcome_and_convergence.pdf`; Table `mu-sweep` |
| §3.3 / Figs 3.3–3.4 | L4 four-condition decomposition | `li2020_asymmetric_L4`, `li2020_asymmetric_L1` (control) | `submit_li2020_asymmetric_L4.sh`, `submit_li2020_quantity_skew_L1.sh` | `SCRIPTS/analyse_li2020_asymmetric_L4.py` | Fig 3.3 `SCRIPTS/plot_validation_curves_extreme_gaps.py`; Fig 3.4 ² | `F_val_curves_extreme_gaps.pdf`; `F_l4_four_condition_per_class_grid.pdf`; Tables `li-decomp-l4`/`-l1` |
| §3.3 (Fig 3.3 panel B) | perfect-storm L4 | `fedprox_perfect_storm_L4` | `submit_fedprox_perfect_storm_L4.sh` | `SCRIPTS/analyse_li2020_asymmetric_L4.py` | (panel of Fig 3.3) | in `F_val_curves_extreme_gaps.pdf` |
| §3.4 / Fig 3.5 | LR-asymmetry envelope | `asymmetric_lr_L4` | `submit_asymmetric_lr_L4.sh` (+ `_fednova_only.sh`, `submit_fednova_lr_envelope_L4.sh`) | `SCRIPTS/analyse_asymmetric_lr_L4.py` | `SCRIPTS/analyse_asymmetric_lr_L4.py` (same; analyse+plot) | `F_asymmetric_lr_L4.pdf`; Table `lr-envelope` |
| §3.4 / Fig 3.6 + table | FedNova random-τ stress | `system_het_random_fednova` | `submit_fednova_system_het.sh` | `SCRIPTS/analyse_system_heterogeneity.py` | `SCRIPTS/plot_heterogeneity_escalation.py` ³ | `F_heterogeneity_escalation.pdf`; Table `fednova-regime` |
| §3.5 / Fig 3.7 | rare-class collapse (per-class) | pools `li2020_*`, `fedprox_*`, `asymmetric_lr_L4`, `node_pinned_L4`, `mu_sensitivity_flower` | — | (per-class analysers) | `SCRIPTS/plot_cross_experiment_per_class_heatmap.py` | `F_cross_experiment_per_class.pdf`; Table `collapse-cells` |
| §3.5 / table | weighted-CE / focal loss | `fedprox_weighted_ce_L4` | `submit_fedprox_weighted_ce_L4.sh`, `submit_fedprox_focal_loss_L4.sh` | `SCRIPTS/analyse_fedprox_weighted_ce_L4.py` | — (table only) | Table `loss-side` |
| §3.6 / Fig 3.8 | update-norm diagnostics | `flower_C0_baseline` (run with `--log-update-norms`) | (re-run of §3.1) | `SCRIPTS/analyse_d1_mechanism.py` | `SCRIPTS/plot_update_norms.py` | `F_update_norms.pdf` |
| Fig 4.1 | regime map (synthesis) | — | — | — | `CONCEPTUAL/make_regime_map_summary.py` | `F_regime_map_summary.pdf`; Table `final-claims` |
| Appendix | L4 confusion matrix | `li2020_asymmetric_L4` | — | `SCRIPTS/analyse_l4_confusion_matrices.py` | `SCRIPTS/analyse_l4_confusion_matrices.py` (same) | `F_l4_confusion_li2020.pdf` ⁴ |

¹ The §3.1 forest plot `F_statistical_heterogeneity_forest.pdf` is generated and used in the older `FULL_THESIS.tex` / `RESULTS_CHAPTER.tex` drafts, but is **not included in the final submission bundle** (`report/submission_bundle/main.tex`).
² `F_l4_four_condition_per_class_grid.pdf` exists in `…/thesis_ready/figures/` and the bundle, but **no current tracked script emits that exact filename** — verify its generator before regenerating. (`SCRIPTS/plot_validation_curves_l4_four_condition.py` emits the appendix single-panel `F_val_curves_l4_four_condition.pdf`, not this grid.)
³ `plot_heterogeneity_escalation.py` currently embeds its summary numbers as literals rather than reading them from a CSV.
⁴ `analyse_l4_confusion_matrices.py` writes `F_l4_confusion_li2020.pdf` (the four-condition mechanism); the submission bundle includes it **renamed to `F_l4_confusion_four_condition.pdf`**.

---

## 1. Where are the training files?  (the short answer)

| You want… | It lives in |
|---|---|
| **Training code** (the loops that train models) | `fl_dermamnist/experiments/run_*.py` (entrypoints) + `fl_dermamnist/fl/` (pure-PyTorch core) + `fl_dermamnist/fl_flower/` (Flower runtime) |
| **Model / data / partition code** | `fl_dermamnist/models/`, `fl_dermamnist/data/` |
| **Training launchers** (one job → one sweep) | `fl_dermamnist/scripts/submit_*.sh`, `slurm_template_*.sh` |
| **Training outputs / artefacts** (the actual results) | `fl_dermamnist/results/<experiment_name>/` |
| **Analysis → thesis tables & figures** | `fl_dermamnist/results/thesis_ready/scripts/` → `…/data/` + `…/figures/` |
| **Dataset** | `dermamnist_64.npz` (root; git-ignored, ~100 MB, re-downloadable) |
| **How to run anything locally** | `fl_dermamnist/scripts/commands.sh <step>` |

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
bash fl_dermamnist/scripts/commands.sh sanity

# one training run (engineered partition, seed 42, 20 local epochs, 150 rounds)
bash fl_dermamnist/scripts/commands.sh flower-fedprox     # Flower runtime
bash fl_dermamnist/scripts/commands.sh purepy-fedavg      # pure-PyTorch runtime

# regenerate thesis tables + figures from existing results
bash fl_dermamnist/scripts/commands.sh analyse

# show per-experiment completion status
bash fl_dermamnist/scripts/commands.sh status
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

## 5. Experiment map — directory ⇄ launcher ⇄ entrypoint ⇄ role

All result dirs are under `fl_dermamnist/results/`. Launchers are under
`fl_dermamnist/scripts/`. **Role**: M=reported in main Results (Ch.5),
A=appendix/diagnostic, X=archived/not in thesis. Default hyperparameters
(unless a launcher overrides): R=150, E=20, lr=0.01, bs=32, mom=0.9, μ=0.01,
CE loss, seeds `{42,123,456,789,999,2024,31337,8675309,161803,271828}`
(mechanism/L4 use the first 3).

### Statistical heterogeneity (§5.2)
| Directory | Launcher | Entrypoint | Role |
|---|---|---|---|
| `headline/` | `submit_headline.sh` | run_one (pure-PyTorch) | M — engineered headline |
| `flower_C0_baseline/` | `submit_flower_C0_baseline.sh` | run_one_flower | M — **primary** engineered headline |
| `iid/`, `flower_C0_iid_baseline/` | `submit_robustness.sh`, `submit_flower_C0_iid_baseline.sh` | run_one / run_one_flower | M — IID negative control |
| `dirichlet_a01/` | `submit_robustness.sh`, `runpod_addendum_provenance.sh` | run_one | M — Dirichlet α=0.1 robustness |
| `specialist_partition/` | `submit_specialist_partition.sh` | run_one_flower | M — specialist 1-of-7 robustness |
| `node_pinned_L4/` | `submit_node_pinned_L4.sh` | run_one_flower | M — variance control |
| `centralised/` | `slurm_centralised.sh` | run_centralised | M — performance ceiling |

### μ sensitivity (§5.3)
| `mu_sensitivity_flower/` | `submit_mu_sensitivity_clean.sh` | run_one_flower | M — inverted-U, μ∈{0,0.001,0.01,0.1,1.0} |
| `mu_sweep_ladder/` | `submit_mu_sweep_ladder.sh` | run_one_flower | A — μ × ladder rung |

### Li-style straggler decomposition (§5.4, centrepiece)
| `li2020_asymmetric_L4/` | `submit_li2020_asymmetric_L4.sh` | run_one_flower | M — 4-condition decomposition |
| `li2020_asymmetric_L1/` | `submit_li2020_quantity_skew_L1.sh` | run_one_flower | M — L1 negative control |
| `fedprox_perfect_storm_L4/` | `submit_fedprox_perfect_storm_L4.sh` | run_one_flower | M — stress demo (μ=1.0, bs=10, straggler 0.9) |
| `two_client_90_10_rare_stress/` | `submit_two_client_90_10_rare_stress.sh` | run_one_flower | A — L4 deep dive (+`.pt`) |

### LR asymmetry & FedNova regime-dependence (§5.5)
| `asymmetric_lr_L4/` | `submit_asymmetric_lr_L4.sh` + `…_fednova_only.sh` + `submit_fednova_lr_envelope_L4.sh` | run_one_flower / run_one_fednova_flower | M — ratios 1:1…50:1 |
| `system_het_random_fednova/` | `submit_fednova_system_het.sh`, `hpc_addendum_fednova_c2.sh` | run_one_fednova_flower | M — random-τ collapse |
| `system_het_random_fednova_{baseline,mom0,serverlr03,servmom,tauclip320}/` | `submit_fednova_mechanism_fork_pilot.sh`, `submit_fednova_stage3_expand.sh` | run_one_fednova_flower | A — collapse-mechanism probes (**WIP**) |

### Rare-class collapse & loss-side correction (§5.6)
| `fedprox_weighted_ce_L4/` | `submit_fedprox_weighted_ce_L4.sh` + `submit_fedprox_focal_loss_L4.sh` | run_one_flower | M — CE / weighted-CE / focal |

### System heterogeneity & mechanism (§5.4/5.7, App C–E)
| `system_het_{fixed,random,iid_fixed,iid_random}/` | `submit_system_het.sh`, `submit_system_het_iid.sh` | run_one_flower | M — stat × system interaction |
| `system_het_partial_C0.5/` | `submit_partial_participation.sh` | run_one_flower | A — C=0.5 partial participation |
| `asymmetric_mu_L4/` | `submit_asymmetric_mu_L4.sh` | run_one_flower | A — per-client μ (Yao test) |
| `fednova_unequal_E/` | `submit_fednova_unequal_E.sh` | run_one_fednova_flower | A — unequal-E |
| `extended_rounds_L3/` | `submit_extended_rounds_L3.sh` | run_one_flower | A — R=250 convergence check |
| `heterogeneity_ladder/` | `submit_ladder_pilot.sh` | run_one_flower | A — L0→L4 pilot (+`.pt`) |
| `e_sweep/`, `e_sweep_dirichlet_a01/` | `submit_e_sweep.sh`, `submit_e_sweep_dirichlet_a01.sh` | run_one_flower | A — local-epoch sweep |
| `small_hospital_local_only/`, `small_hospital_finetune/` | `submit_small_hospital_baselines.sh` | run_local_only, run_finetune | A — clinical case study (App D) |
| `partitions/` | (generated by `data/partition.py`) | — | partition definitions |
| `arch_ablation_bn/` | `hpc_arch_ablation_bn.sh`, `runpod_arch_ablation_bn.sh` | run_one_flower | X — **archived, not in thesis** |

> **Known doc wrinkles** (see `docs/repo_audit_submission_cleanup/09_…md`):
> the older matrix in `fl_dermamnist/README.md` lists `mu_sweep/` and
> `system_het_random_asymmetric/`, which are **not present on disk** (superseded
> / HPC-only). The launcher rescue/fix/twin variants (`resubmit_*`, `runpod_*`,
> `*_rescue`, `*_fix`) are retained as run provenance.

---

## 6. Data flow

```
dermamnist_64.npz
   │  data/load.py + data/partition.py
   ▼
experiments/run_*.py  ──(uses)──>  fl/ , fl_flower/ , models/
   │   (launched at scale by scripts/submit_*.sh)
   ▼
fl_dermamnist/results/<experiment>/        ← raw training artefacts (§4)
   │   results/thesis_ready/scripts/analyse_*.py , plot_*.py
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
│   ├── scripts/                   ← launchers (submit_*.sh) + commands.sh + analyse_all.sh
│   ├── tests/                     ← μ=0≡FedAvg, proximal-term, provenance guards
│   └── results/                   ← one dir per experiment (raw artefacts)
│       └── thesis_ready/          ← data/ + figures/ (canonical) + scripts/ (analysis/plot code*)
├── report/                        ← all thesis text
│   ├── submission_bundle/         ← the compiled report: main.tex + figures + references.bib
│   ├── chapters/                  ← §5 section sources (5_*.tex) + bibliographies
│   └── drafts/                    ← historical drafts (FULL_THESIS.tex, RESULTS_CHAPTER.tex, …)
├── docs/                          ← verification sheet + audit notes
└── archive/                       ← legacy code (code_legacy/) + old build guides (docs_legacy/)

  *thesis_ready/scripts/ still holds the analysis/plot code; relocating it into
   fl_dermamnist/analysis/ + fl_dermamnist/figures/ is the one remaining reorg step.
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
- Implementation guards: `bash fl_dermamnist/scripts/commands.sh sanity` (run before trusting any claim).
- Pre-submission preflight: `bash fl_dermamnist/scripts/pre_submission_check.sh`.
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
bash fl_dermamnist/scripts/commands.sh sanity     # implementation guards (μ=0≡FedAvg, etc.)
bash fl_dermamnist/scripts/commands.sh analyse    # rebuild thesis tables + figures from saved results
```

**Where things live:**

| Artefact | Location |
|---|---|
| Final report (compile target) | `report/submission_bundle/main.tex` |
| Report figures (used by the PDF) | `report/submission_bundle/F_*.pdf` — copies of `fl_dermamnist/results/thesis_ready/figures/` |
| Figure/table generators | `fl_dermamnist/results/thesis_ready/scripts/` (`plot_*.py`, `analyse_*.py`) and `docs/whole_paper_example_audit/figure_generation/` (the three conceptual diagrams) |
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

Full experiment reruns were performed on HPC via the `scripts/submit_*.sh`
launchers and can be costly; the saved outputs above are the intended path for
reproducing the report's figures and tables.

---

## 11. AI / code-assistance statement

AI coding assistants / large language models may have been used during development
for tasks such as debugging, refactoring, plotting and figure-generation
assistance, documentation, and code/text editing. All experiments, numerical
results, their interpretation, and the scientific claims in the report are the
author's own work and remain the author's responsibility. Any code developed with
such assistance was reviewed by the author. This statement should be read alongside,
and adjusted to comply with, the relevant course policy on AI use.
