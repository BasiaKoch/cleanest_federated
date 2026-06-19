# Final submission summary (one/two-page)

Standalone summary of the thesis for a second / time-limited reviewer.

- **`summary.tex`** — self-contained `article` document (compiles on its own).
- **`F_regime_map_summary.pdf`**, **`F_l4_four_condition_per_class_grid.pdf`** —
  the two figures it uses, copied here so the folder compiles in isolation.

## Compile

```bash
pdflatex summary.tex      # or upload this folder to Overleaf, main doc = summary.tex
```

Target length: **one page where possible, two pages maximum**. All numbers match
the main report (`OVERLEAF_FLAT_BUNDLE/main.tex`); the figures are identical copies
of the report's `F_regime_map_summary.pdf` and `F_l4_four_condition_per_class_grid.pdf`.
No LaTeX engine was available locally, so a final compile/visual check is still
required (Overleaf or local `pdflatex`).
