# Thesis-ready folder — System Heterogeneity experiment

> **Separation notice.** This folder is **sibling to** `thesis_ready/` (which
> covers statistical heterogeneity). Nothing in here references or modifies
> the contents of `thesis_ready/`, and vice versa. Each folder is a
> self-contained thesis section.

## What this section investigates

System heterogeneity — clients have **different computational budgets** per
round, so some perform less local work than others. This is the canonical
motivation for FedProx in its original paper (Li et al., 2020, MLSys §3, §5.2):
the proximal regulariser was introduced specifically to allow ``γ-inexact''
updates from stragglers to remain useful for aggregation. Showing FedProx >
FedAvg under this regime is therefore the primary FedProx demonstration; the
statistical-heterogeneity result in `thesis_ready/` is the secondary one.

## Experimental design

Same paired-seed protocol as the headline (`balanced_paired_7_clients`
partition, 10 paired seeds, $E_\max = 20$, $R = 150$, μ = 0.01) — but with
the local-epoch count varied per client per round.

| Condition | Description | Source |
|---|---|---|
| **C0 — Headline (no system het)** | All 7 clients perform $E = 20$ every round | Already in `thesis_ready/` |
| **C1 — Fixed stragglers** | Clients 5 and 6 always perform $E = 5$; others $E = 20$ | This work |
| **C2 — Random stragglers (Li-style)** | Each round, 50% of clients randomly designated stragglers with $E_i \sim \text{Unif}[1, 19]$; others $E = 20$ | Li et al. (2020) §5.2 |

C2 follows the canonical FedProx system-heterogeneity setup exactly. C1
is closer to Marija (2025) §3.8.4 but adapted to our 7-client partition.
Together they probe the deterministic and stochastic extremes of the
straggler regime.

## Files in this folder

```
thesis_ready_system_het/
├── README.md                          ← this file
├── data/                              ← machine-readable outputs (HPC pending)
│   ├── per_seed_results.csv           (pending)
│   ├── per_class_results.csv          (pending)
│   ├── summary_statistics.json        (pending)
│   ├── system_het_vs_baseline.json    (pending — comparison to C0)
│   └── curves_aggregated.csv          (pending)
├── figures/                           ← publication-quality plots (HPC pending)
├── writing/
│   ├── 01_methodology.md              ← drop-in methodology section
│   ├── 02_results_placeholder.md      ← results scaffolding with HPC TODOs
│   └── 03_overleaf_ready_system_het.tex ← drop-in .tex for Overleaf
├── conclusions/
│   └── hypothesis_and_expected_outcomes.md  ← what we expect to find and why
└── scripts/                           ← analysis scripts (to be run when HPC results land)
```

## Relationship to the statistical-heterogeneity section

| Aspect | `thesis_ready/` (statistical het) | `thesis_ready_system_het/` (this folder) |
|---|---|---|
| What varies | Partition (which classes go to which client) | Local computation budget (E_i per client per round) |
| Partition | `balanced_paired_7_clients` (custom non-IID) | Same partition — isolates the system-het variable |
| FedProx motivation tested | Secondary (drift mitigation under non-IID) | **Primary** (γ-inexact updates from stragglers) |
| Headline question | Does FedProx help when data is non-IID? | Does FedProx help when clients have heterogeneous compute? |
| Expected result | Δ macro-F1 > 0, p < 0.05 (confirmed: +0.027, p = 0.020) | Δ macro-F1 ≫ 0 (FedProx advantage should be larger here) |

## How to use this folder

1. **While HPC is pending:** Read `writing/01_methodology.md` and
   `conclusions/hypothesis_and_expected_outcomes.md`. Polish the drop-in
   `.tex` file with the placeholder system from `thesis_ready/`.

2. **After HPC sweeps finish (40 jobs total):**
   - Files land in `mnist_dermnist/results/system_het_fixed/` and `mnist_dermnist/results/system_het_random/`.
   - Run the analysis scripts in `scripts/` (to be written) to populate `data/`.
   - Update the `\TODOhpc{...}` placeholders in `writing/03_overleaf_ready_system_het.tex`.

## Citing this work

The two-folder structure mirrors a thesis chapter with two clearly
separated experimental sections. Reviewers should be able to read each
folder in isolation and understand the contribution. Cross-references in
the main thesis text should link the two sections explicitly so the
reader sees them as complementary tests of different FedProx claims.
