# thesis-ready outputs (data + figures)

This folder holds the **canonical thesis-ready outputs** the report draws from:
the aggregate tables and the final figure PDFs. It contains derived artefacts
only — nothing here is trained or hand-edited.

## Layout

```
thesis_ready/
├── data/        # aggregate tables / summaries (CSV + JSON) behind the report's tables
└── figures/     # the canonical F_*.pdf figures referenced by the report
```

## How these are produced (reproducibility)

The generators live in `fl_dermamnist/analysis/` (`analyse_*.py` — tables and
numeric summaries) and `fl_dermamnist/figures/` (`plot_*.py` — the `F_*.pdf`
figures). They rebuild everything here from the saved per-run results under
`fl_dermamnist/results/<experiment>/` **without retraining**:

```bash
bash infra/local/commands.sh analyse     # run from the repo root
```

The report bundle (`report/supporting/`) carries flat copies of the `F_*.pdf`
figures next to `main.tex`.

## Where to look

- **Section → script → figure/table map:** the navigation table in the
  top-level [`README.md`](../../../README.md).
- **Per-number provenance:** `docs/provenance/result_traceability_matrix.csv`
  (claim → result → script → figure) and
  `docs/provenance/numerical_verification_sheet.txt` (every headline number
  re-derived from the raw artefacts).
