# Per-client specialty analysis and best-vs-last round comparison

Two analyses inspired by Marija (2025) §3.8.2, computed from the existing
20 HPC test_at_best JSONs and 20 history CSVs — no new compute required.

## 1 Per-specialty test F1

The `balanced_paired_7_clients` partition forms 3 specialty pairs and one
generalist:

| Specialty | Clients | Classes | FedAvg | FedProx | Δ | Wins | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| Pair 0 | C0+C1 | actinic, basal | 0.439 | 0.472 | **+0.033** | 7/10 | 0.131 |
| Pair 1 | C2+C3 | benign_kerat, dermato | 0.362 | 0.370 | +0.008 | 4/10 | 0.922 |
| **Pair 2** | **C4+C5** | **melanoma, vascular** | **0.442** | **0.493** | **+0.051** | **9/10** | **0.010** |
| Generalist | C6 | mel_nevi only | 0.884 | 0.886 | +0.002 | 7/10 | 0.375 |
| All minorities | (combined) | 6 non-nevi classes | 0.414 | 0.445 | **+0.031** | 8/10 | **0.027** |

**Key observations:**

1. **Pair 2 (melanoma + vascular) drives the headline result.** Δ = +0.051, p = 0.010, FedProx wins 9 of 10 seeds. This pair includes the clinically critical melanoma class.

2. **The mel_nevi generalist sees no FedProx advantage** (Δ = +0.002, p = 0.375). Mel_nevi is in every client's training data, so no client drift exists for the proximal anchor to correct.

3. **Pair 1 (benign_kerat + dermato) is the weakest specialty.** Only 4 of 10 seeds favour FedProx. These two classes are the second-most prevalent and least prevalent respectively; the very rare dermato class (1.15% prevalence) introduces large per-seed variance.

4. **Pair 0 (actinic + basal) shows a positive but marginally non-significant effect.** Δ = +0.033, 7/10 wins, p = 0.131. The magnitude is meaningful but n = 10 is not powered to detect this within a single specialty.

5. **The "all minorities" aggregate is significant** (Δ = +0.031, p = 0.027), confirming that across the full set of non-majority classes, FedProx provides a measurable improvement.

**Interpretation:** FedProx's drift-mitigation mechanism engages most strongly where there is most drift to mitigate. Melanoma is a clinically critical class held by only two clients; FedAvg has only those two clients' gradient signal to learn it, and the other five clients actively pull the model away (their nevi-only training pushes the melanoma logit toward zero). FedProx caps how far the five non-specialist clients can pull, preserving the C4/C5 melanoma signal during aggregation.

The Pair 1 weakness (benign_kerat + dermato) is consistent with the observation that the C2+C3 pair's specialty includes dermato, the rarest class (n ≈ 81 train samples), where per-seed variance in F1 is largest.

## 2 Best vs final round comparison

Validation macro-F1 at peak round vs at round 150, mean across 10 seeds:

| Algorithm | Peak val_F1 (mean ± SD) | Final val_F1 (mean ± SD) | Drop (peak → final) |
|---|---|---|---|
| FedAvg | 0.519 ± 0.017 | 0.473 ± 0.034 | 0.046 ± 0.036 |
| FedProx | 0.554 ± 0.027 | 0.511 ± 0.042 | 0.043 ± 0.034 |

**Mean peak round:**
- FedAvg: round 127 (median 134; range [70, 150])
- FedProx: round 118 (median 122; range [90, 137])

**Paired Wilcoxon on the post-peak drop:** p = 0.92 — drops are statistically indistinguishable between algorithms.

**Key observations:**

1. **FedProx maintains a higher plateau throughout, not just at the peak.** Mean peak Δ = +0.035 (0.554 vs 0.519); mean final-round Δ = +0.038 (0.511 vs 0.473). The advantage is stable across the late-training regime.

2. **Both algorithms overfit by similar absolute amounts** (Δ drop ≈ -0.003, p = 0.92). This is a different finding from Marija's two-client setup (where FedAvg dropped substantially more): in the high-E, 10-seed regime, both algorithms exhibit comparable overfitting magnitudes.

3. **FedProx peaks ~9 rounds earlier on average** (118 vs 127). Combined with the slightly tighter range ([90, 137] vs [70, 150]), this indicates more predictable training dynamics.

4. **The test-at-best-val protocol is essential.** Both algorithms drop ~0.04 in val_F1 between their peak and round 150, which would translate to a corresponding test drop. Always reporting the final-round test number (which neither study does) would understate the achievable performance of both methods by approximately equal amounts.

## 3 What these analyses add to the thesis

| Claim | Strength of evidence |
|---|---|
| "FedProx specifically helps the melanoma/vascular pair" | Direct: Pair 2 Δ = +0.051, p = 0.010, 9/10 wins |
| "FedProx leaves majority-class performance untouched" | Direct: Generalist Δ = +0.002, p = 0.375 |
| "Aggregate minority-class improvement is significant" | Direct: All-minorities Δ = +0.031, p = 0.027 |
| "FedProx maintains a higher convergence plateau" | Direct: peak F1 +0.035, final F1 +0.038 |
| "Both algorithms overfit similarly in absolute magnitude" | Direct: drop Δ = -0.003, p = 0.92 |
| "FedProx converges to a predictable peak round" | Direct: range [90, 137] vs FedAvg's [70, 150] |

## 4 What these analyses do NOT show

- They do not constitute Marija-style "per-client local models" — those would require training each client on its own data and comparing to the FL global model, which is a different experiment (~5 GPU-hours of additional work).
- The per-specialty significance tests are computed per pair (n = 10 each); they have not been Bonferroni-corrected for the 4 specialties tested. The headline "all minorities" aggregate (p = 0.027) is the more conservative figure to cite.
- The post-peak drop equivalence is a property of the high-E regime (E = 20); at lower E, the relative overfitting balance may differ (consistent with Marija's lower-E observations).
