# Results directory layout

This directory holds the raw experimental output JSONs and history CSVs the
thesis cites. Each subdirectory corresponds to one HPC sweep and contains
`test_at_best_*.json` (one per (algo, mu, E, seed) run) plus matching
`history_*.csv`. Filename schema:

    test_at_best_{algo}_mu{mu}_E{E}[_sh-{mode}][_C{frac}]_s{seed}.json

## Canonical layout (post-2026-05-22)

| Subdirectory | Runtime | Role |
|---|---|---|
| `legacy_pure_pytorch/` | `framework="pure-pytorch"` | **Historical reference.** Produced by `experiments/run_one.py` before the Flower runtime was introduced. Carries the original 10-paired-seed FedAvg + FedProx headline at $E=20$, $R=150$, $\mu=0.01$ on `balanced_paired_7_clients`. Preserved as the cross-runtime validation reference (compare_equivalence_full_scale.py). **Not** the new canonical headline. |
| `flower_C0_baseline/` | `framework="flower-simulation"` | **Canonical Flower headline + system-het C0 baseline.** 10 seeds × {FedAvg, FedProx, FedNova} = 30 jobs on `balanced_paired_7_clients`, no stragglers (uniform $E=20$). |
| `system_het_fixed/` | flower-simulation | C1 system heterogeneity: clients 5,6 always at $E_{\mathrm{straggler}}=5$. Descriptive-only due to partition confound. |
| `system_het_random/` | flower-simulation | C2 system heterogeneity (Li 2020 §5.2): random stragglers, $E_i \sim \mathrm{Uniform}\{1,...,19\}$ for 4 of 7 clients per round. **Primary inferential condition for H2.** |
| `system_het_random_fednova/` | flower-simulation | FedNova arm of C2. |
| `iid/` | flower-simulation | IID partition. **Falsification probe** for the drift-control mechanism (predicts Δ ≈ 0). |
| `dirichlet_a01/` | flower-simulation | Dirichlet $\alpha=0.1$ partition. **External-validity probe** (Hsu 2019). |
| `specialist_partition/` | flower-simulation | One-client-per-minority-class sister to balanced_paired. **Engineered-partition defence**, pre-registered at SHA `9f2bb94`. |
| `mu_sweep/` | flower-simulation | Sensitivity sweep over $\mu \in \{0.001, 0.01, 0.1, 0.5, 1.0\}$, 3 seeds each + 3 FedAvg sanity baselines. |
| `e_sweep/` | flower-simulation | Sensitivity sweep over $E \in \{1, 5, 10, 20, 40\}$, 3 seeds × 2 algos. Tests the drift-control dose-response prediction. |
| `class_weighted_baseline/` | flower-simulation | FedAvg + class-weighted CE × 10 seeds. Reviewer ablation HV2 (imbalance-correction alternative). |
| `centralised/` | n/a (non-federated) | Centralised baseline, 10 seeds. Anchors the federation-tax comparison. |
| `headline_flower_verify/` | flower-simulation | (Only if `submit_equivalence_check.sh` was run.) 2 stress-test seeds × {FedAvg, FedProx} for the original 4-job equivalence check. Superseded by `flower_C0_baseline/` (10-seed) in practice. |
| `partitions/` | n/a | Per-seed partition record CSVs. Not result files. |
| `thesis_ready/`, `thesis_ready_system_het/` | n/a | Derived analysis artefacts: summary statistics JSONs, figures, LaTeX source. |

## How to tell legacy from canonical at a glance

Every result JSON carries:

- `framework: "pure-pytorch"` (legacy) or `"flower-simulation"` (canonical)
- `runner_script: "run_one.py"` (legacy) or `"run_one_flower.py"` / `"run_one_fednova_flower.py"` (canonical)
- `provenance_note: "backstamped-2026-05-18"` (legacy only; pre-dates run-time provenance)
- `git_commit, hostname, run_started_at, run_finished_at` (canonical only; post the CP3.2 runtime-provenance patch)

The `mnist_dermnist/fl/provenance.py:canonicalise_framework()` helper maps both to the canonical label, with a unit-test guard at
`tests/test_framework_provenance.py:test_all_20_headline_jsons_canonicalise_to_pure_pytorch` that reads the actual `legacy_pure_pytorch/` JSONs every CI run.

## Sanity check

Before any inferential test, run:

    PYTHONPATH=. python mnist_dermnist/results/thesis_ready/scripts/sanity_check_results.py

This walks every subdirectory, verifies expected JSON counts, confirms framework
labels are canonical, checks for pathological macro-F1 values (≈ 0 = silent
training failure), and prints a per-sweep summary.
