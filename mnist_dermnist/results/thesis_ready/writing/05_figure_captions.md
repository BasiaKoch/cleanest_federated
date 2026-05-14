# Figure captions (ready to drop into LaTeX)

## Figure 1 — Headline summary
**File:** `figures/01_headline_summary.png` / `.pdf`

> **Figure 1. FedAvg vs FedProx on DermaMNIST.** Mean test macro-F1 (± standard deviation across 10 paired seeds). FedProx ($\mu = 0.01$) achieves 0.508 ± 0.014 against FedAvg's 0.481 ± 0.025, a within-pair improvement of $\Delta = +0.027 \pm 0.035$. The paired Wilcoxon signed-rank test (two-sided) yields $p = 0.020$; significance is indicated above the bracket.

## Figure 2 — Paired forest plot
**File:** `figures/02_paired_forest.png` / `.pdf`

> **Figure 2. Per-seed paired comparison.** Each row shows the test macro-F1 for FedAvg (blue) and FedProx (orange) at one seed; the connecting line is coloured green if FedProx wins and red if FedAvg wins. The annotation at the right of each row is the paired difference $\Delta = M^{\text{FedProx}} - M^{\text{FedAvg}}$. Vertical dashed lines mark the across-seed means. FedProx outperforms FedAvg on 9 of 10 paired seeds; the single reversal (seed 789) has small magnitude ($\Delta = -0.020$).

## Figure 3 — Delta strip plot
**File:** `figures/03_delta_strip.png` / `.pdf`

> **Figure 3. Distribution of paired differences with bootstrap confidence interval.** Each circle is one seed's paired $\Delta$ (FedProx − FedAvg); green = positive (FedProx wins), red = negative. The black diamond is the mean $\Delta = +0.027$, with the bootstrap 95% confidence interval shown as horizontal error bars. The interval excludes zero, supporting the headline significance test.

## Figure 4 — Per-class comparison
**File:** `figures/04_per_class_bars.png` / `.pdf`

> **Figure 4. Per-class test F1 by algorithm.** Mean per-class F1 across 10 paired seeds, FedAvg (blue) vs FedProx (orange). X-axis labels include each class's prevalence in the training set, in parentheses. Asterisks above bar pairs indicate per-class paired Wilcoxon significance at $p < 0.05$. Significant improvements are observed on melanoma ($\Delta = +0.114$, $p = 0.006$) and actinic keratoses ($\Delta = +0.067$, $p = 0.020$); all other classes are statistically indistinguishable.

## Figure 5 — Per-class delta with significance and tolerance
**File:** `figures/05_per_class_delta.png` / `.pdf`

> **Figure 5. Per-class FedProx advantage with significance markers and design tolerance.** Bars show mean $\Delta$ F1 (FedProx − FedAvg) per class across 10 paired seeds. Green bars indicate statistically significant positive $\Delta$ (paired Wilcoxon $p < 0.05$); grey bars are not significant. Red dotted lines mark the pre-registered ±0.05 per-class regression tolerance (Methods §9d). The largest mean regression is on vascular lesions ($\Delta = -0.012$, $p = 0.70$), comfortably within tolerance.

## Figure 6 — Distribution comparison
**File:** `figures/06_distribution.png` / `.pdf`

> **Figure 6. Distribution of test macro-F1 across 10 paired seeds.** Box plots show the per-algorithm distribution (median, IQR, whiskers to 1.5 × IQR). Individual seed values are overlaid as filled circles, and grey lines connect each seed's paired FedAvg and FedProx values. The FedProx distribution has higher median, smaller inter-quartile range, and 45 % lower across-seed standard deviation (0.014 vs 0.025).

## Figure 7 — Summary panel
**File:** `figures/07_summary_panel.png` / `.pdf`

> **Figure 7. Three-panel summary of the FedProx vs FedAvg comparison.** **(A)** Mean test macro-F1 (± SD); $\Delta = +0.027$. **(B)** Rank-biserial effect size with conventional magnitude thresholds; $r_{\text{rb}} = +0.818$ falls in the "very large" range. **(C)** Across-seed standard deviation of test macro-F1; FedProx exhibits 45 % less variance than FedAvg.

## (Optional) Figure 8 — Single-seed deep-dive
**File:** `figures/08_single_seed_report.png` (copy from `colab_recovered/single_seed_report_s42.png` if including)

> **Figure 8. Detailed analysis of seed = 42 (single-seed deep-dive).** Six-panel breakdown of one representative paired run. **(A)** Validation macro-F1 vs round; horizontal lines show final test macro-F1 of each algorithm. **(B–C)** Validation and training loss vs round; log scale on (C) shows FedProx training loss is held above FedAvg's by the proximal term, evidence the regularisation is active. **(D)** Final test metrics: macro-F1, balanced accuracy, accuracy. **(E)** Per-class test F1. **(F)** Per-class $\Delta$: FedProx wins on dermato (+0.239), melanoma (+0.059), vascular (+0.051); regresses on actinic (−0.056). This single-seed actinic regression is not consistent with the 10-seed aggregate (mean +0.067, $p = 0.020$), illustrating why per-class claims must be made on the aggregate rather than individual seeds.
