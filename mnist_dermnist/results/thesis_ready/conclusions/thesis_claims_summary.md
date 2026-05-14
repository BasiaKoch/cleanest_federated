# Thesis claims — bullet summary

This document distils the FedProx-vs-FedAvg comparison into the exact claims your thesis defends. Each claim is paired with the evidence supporting it.

## Primary claim

**Claim 1.** On a clinically realistic non-IID partition of DermaMNIST (`balanced_paired_7_clients`), FedProx (μ = 0.01) significantly improves test macro-F1 over FedAvg.

**Evidence:**
- Mean Δ test macro-F1 = **+0.027** ± 0.035 across n = 10 paired seeds
- Paired Wilcoxon signed-rank: **p = 0.020** (two-sided)
- Rank-biserial effect size: **r = +0.818** ("very large")
- Bootstrap 95% CI on mean Δ excludes 0
- FedProx wins **9 of 10** paired seeds

## Secondary claims

**Claim 2.** The improvement is concentrated on the clinically critical melanoma class.
- Mean melanoma Δ F1 = **+0.114**
- Per-class paired Wilcoxon p = **0.006**
- Relative improvement: 61 % (F1: 0.187 → 0.301)

**Claim 3.** FedProx satisfies the pre-registered safety criterion.
- Largest mean per-class regression: vascular −0.012 F1
- Design tolerance: ±0.05
- Criterion **satisfied with margin**

**Claim 4.** FedProx reduces across-seed variance.
- FedAvg test macro-F1 SD across 10 seeds: 0.025
- FedProx test macro-F1 SD across 10 seeds: 0.014
- **45 % variance reduction**
- Implication: more reproducible global models under non-IID

**Claim 5.** A second statistically significant class-level improvement exists.
- Mean actinic keratoses Δ F1 = **+0.067**
- Per-class paired Wilcoxon p = **0.020**
- Relative improvement: 17 % (F1: 0.404 → 0.471)

## Negative findings (honest reporting)

**One paired seed regresses:** seed 789, Δ = −0.020. Small in magnitude, not statistically significant in either direction, consistent with random initialisation variance.

**No significant per-class advantage on:** majority class (mel_nevi), basal cell carcinoma, benign keratosis, dermatofibroma, vascular lesions. FedProx is statistically indistinguishable from FedAvg on these classes — neither better nor worse.

## What this thesis does **not** claim

- That FedProx is the best non-IID FL algorithm (we did not compare to SCAFFOLD [Karimireddy 2020], FedNova, etc.)
- That these results generalise to other medical-imaging datasets or higher resolutions
- That FedProx with μ = 0.01 is optimal — the pilot μ-sweep was small (n = 3)
- That this partition is the only valid non-IID configuration — we did not run Dirichlet-α controls
- That F1 = 0.301 on melanoma is clinically deployable (it is not; this is a benchmark study)

## Practical recommendation arising from this work

> For practitioners building federated medical-imaging classifiers under non-IID conditions, FedProx with small μ (~0.01) should be the **default baseline** rather than FedAvg. The performance gain is statistically robust, the per-class behaviour is favourable on the most clinically important class, and the variance reduction makes single-run deployment more reliable. The computational overhead (~50 % training-time premium) is modest given the gains.

## Single-sentence headline for the abstract

> FedProx improves test macro-F1 over FedAvg by +0.027 (paired Wilcoxon p = 0.020, rank-biserial r = +0.818, n = 10 seeds) on a non-IID DermaMNIST partition, with the largest per-class gain on melanoma (+0.114 F1, p = 0.006).
