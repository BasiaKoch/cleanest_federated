# Thesis-ready FedProx-vs-FedAvg analysis

This folder is the **dissertation-writing source-of-truth** for the FedAvg vs FedProx experimental chapter. Everything you need to write the methodology, results, and discussion is here. Numbers, plots, prose templates, tables, and bibliography — ready to paste.

## TL;DR

> Across 10 paired seeds on DermaMNIST with the `balanced_paired_7_clients` partition, **FedProx (μ = 0.01) improves test macro-F1 over FedAvg by Δ = +0.027** (paired Wilcoxon **p = 0.020**, rank-biserial **r = +0.818**). Largest per-class gain on **melanoma (+0.114 F1, p = 0.006)**. No per-class regression exceeds 0.012. FedProx additionally reduces across-seed variance by 45 %.

## Folder layout

```
thesis_ready/
├── README.md                          ← you are here
├── data/                              ← machine-readable analysis output
│   ├── per_seed_results.csv           ← 10-row paired table (the raw data)
│   ├── per_class_results.csv          ← 7-row per-class table with Wilcoxon p
│   ├── summary_statistics.json        ← every aggregate stat in one JSON
│   └── curves_aggregated.csv          ← per-round mean/SD across 10 seeds (for curve regen)
├── figures/                           ← publication-quality plots (PNG @ 300 dpi + PDF vector)
│   ├── 01_headline_summary.{png,pdf}      ← bar chart with significance
│   ├── 02_paired_forest.{png,pdf}         ← per-seed forest plot
│   ├── 03_delta_strip.{png,pdf}           ← Δ distribution + bootstrap CI
│   ├── 04_per_class_bars.{png,pdf}        ← per-class grouped bars
│   ├── 05_per_class_delta.{png,pdf}       ← per-class Δ with significance
│   ├── 06_distribution.{png,pdf}          ← box+strip plot, paired connectors
│   ├── 07_summary_panel.{png,pdf}         ← three-panel summary
│   ├── 08_curves_main.{png,pdf}           ← 4-panel convergence curves (val macro-F1, val loss, train loss, val bACC)
│   ├── 09_curves_per_class.{png,pdf}      ← 7-panel per-class val F1 curves
│   └── 10_overfitting_diagnostic.{png,pdf} ← train vs val loss per algorithm
├── writing/                           ← prose ready to drop into your thesis
│   ├── 01_abstract_paragraph.md           ← three lengths
│   ├── 02_methods.md                      ← full methodology chapter
│   ├── 03_results.md                      ← full results chapter
│   ├── 04_discussion.md                   ← full discussion chapter
│   ├── 05_figure_captions.md              ← captions for each figure
│   ├── 06_tables.md                       ← LaTeX + Markdown tables
│   ├── 07_bibliography.bib                ← BibTeX entries for every citation
│   └── 08_curves_analysis.md              ← curve-by-curve analysis + drop-in prose
├── conclusions/                       ← analysis findings, claim-by-claim
│   ├── thesis_claims_summary.md           ← bullet-point list of every claim + evidence
│   ├── numerical_differences_from_curves.md  ← every numerical Δ visible in the curves
│   ├── metric_relationship_macro_vs_per_class.md  ← how Fig 8 and Fig 9 relate
│   └── partition_mechanism_mel_nevi.md    ← why mel_nevi shows no FedProx advantage
└── scripts/
    └── generate_all_figures.py            ← reproduces all figures from /data
```

## How to use this folder for thesis writing

### 1. Drop the headline sentence into your abstract
Open `writing/01_abstract_paragraph.md` — three pre-written versions (80 words, 150 words, one-sentence elevator pitch). Pick one, paste, adjust to your overall abstract flow.

### 2. Build the methods chapter
Open `writing/02_methods.md` — covers problem statement, dataset, model, FL setup, partition design, both algorithms, paired-seed protocol, metrics, statistical analysis, environment, reproducibility, threats to validity. All twelve sections drop-in ready. Citations use `[Author Year]` format mapped to entries in `07_bibliography.bib`.

### 3. Build the results chapter
Open `writing/03_results.md` — covers headline result, per-seed comparison, per-class breakdown, single-seed deep-dive, variance reduction, computational cost, summary. Figure references are placeholders (`Fig. 1`, `Tab. 2`); replace with your final figure numbers at LaTeX compile.

### 4. Build the discussion chapter
Open `writing/04_discussion.md` — covers what the results add to the literature, the mechanism behind the per-class pattern, variance as a secondary contribution, clinical significance of the melanoma result, limitations (single dataset, single partition, μ selection, no IID control, resolution), future work, concluding remark.

### 5. Insert figures
The seven figures in `figures/` are designed to mirror conventions of major FL papers:
- **Fig. 1** (headline bar chart) — style of McMahan 2017, Fig. 2
- **Fig. 2** (forest plot) — style of clinical-trials meta-analyses
- **Fig. 4** (per-class grouped bars) — style of Li 2020, per-task breakdowns
- **Fig. 5** (per-class Δ with tolerance) — original to this thesis, communicates the "no regression" criterion at a glance
- **Fig. 6** (box+strip with paired connectors) — style of Bouthillier 2021 variance benchmarks

Captions in `writing/05_figure_captions.md` are written to LaTeX standards (declarative, self-contained, no "see text" forward references).

### 6. Insert tables
Open `writing/06_tables.md` — four tables in both Markdown and LaTeX:
- Tab. 1 — headline paired comparison
- Tab. 2 — per-class F1 with significance
- Tab. 3 — per-seed paired results (full)
- Tab. 4 — partition composition

### 7. Add citations
Append `writing/07_bibliography.bib` to your overall `.bib` file. All references in the prose chapters cite entries from this file. 16 entries covering FL, FedProx, non-IID partitioning, MedMNIST, statistical methods, and variance benchmarks.

### 8. Check your claims
Before submitting, open `conclusions/thesis_claims_summary.md` and verify every claim you make in the thesis maps to evidence listed here.

## Reproduce the figures from scratch

If `data/` changes (e.g. you re-run analysis with corrected seeds), regenerate all figures:

```bash
cd /Users/basiakoch/cleanest_federated
python mnist_dermnist/results/thesis_ready/scripts/generate_all_figures.py
```

Takes about 5 seconds. Writes PNG + PDF for all 7 figures.

## Provenance of the data in this folder

The numbers in `data/` are derived from the HPC headline sweep:
- **Source:** 20 SLURM jobs on Cambridge CSD3 Ampere partition (A100 GPUs)
- **Sweep:** 10 paired seeds × 2 algorithms (FedAvg μ=0; FedProx μ=0.01)
- **Date completed:** mid-May 2026
- **Raw outputs:** `mnist_dermnist/results/headline/` (per-run JSON + CSV files)
- **Aggregation:** `mnist_dermnist/results/headline/analysis/` (via `tables.py`)
- **This folder:** summary, plots, prose templates derived from the aggregated analysis

If a reviewer or examiner challenges a number in the thesis, the audit trail is:
`thesis_ready/data/*.csv` ← `headline/analysis/*.csv` ← `headline/test_at_best_*.json` ← SLURM job

## What is NOT in this folder

- Per-round convergence CSVs for all 20 runs (those live in `mnist_dermnist/results/headline/`, ~40 files). The single-seed convergence figure (Fig. 8) draws on the Colab single-seed FedProx CSV recovered separately.
- Per-seed per-class F1 data — we only have aggregated means and Wilcoxon p-values. This is sufficient for the table and figure, but a future re-run could save per-seed per-class F1 to support richer plots (e.g. box plots per class).
- IID-control results — not run in this thesis (flagged as future work in discussion §5.4).
- Dirichlet-α robustness check — not run in this thesis (flagged as future work in discussion §5.2).
- μ-sweep at n ≥ 5 seeds — pilot was 3 seeds; flagged as future work in discussion §5.3.

## Quick stats lookup (no need to open files)

| Question | Answer |
|---|---|
| Mean FedAvg test macro-F1 | 0.481 ± 0.025 |
| Mean FedProx test macro-F1 | 0.508 ± 0.014 |
| Mean Δ macro-F1 | +0.027 ± 0.035 |
| Wilcoxon p (two-sided) | 0.020 |
| Wilcoxon p (one-sided FedProx > FedAvg) | 0.010 |
| Rank-biserial r | +0.818 |
| Win rate | 9/10 |
| Best per-class Δ | melanoma +0.114 (p = 0.006) |
| Second-best per-class Δ | actinic +0.067 (p = 0.020) |
| Largest per-class regression | vascular −0.012 (p = 0.70) |
| Variance ratio (FedProx SD / FedAvg SD) | 0.55 (45 % lower) |
| Compute cost (full sweep) | ~25 A100-hours |
| Seeds | 42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828 |

## Maintenance

If you re-run the headline sweep (e.g. after a bug fix), redo this folder:

1. Edit `data/per_seed_results.csv`, `data/per_class_results.csv`, and `data/summary_statistics.json` with the new numbers (or write a script that imports from the new `headline/analysis/`).
2. Run `python scripts/generate_all_figures.py` to regenerate figures.
3. Update any verbatim numbers in `writing/*.md` (search for the old mean Δ value to find them).
4. Re-check `conclusions/thesis_claims_summary.md`.

The prose templates and figure scripts are robust to minor number changes; only the verbatim values need updating.
