# Relationship between Figure 8 (macro-F1) and Figure 9 (per-class F1)

> Both figures measure the **same model predictions on the same validation
> set**, but at different levels of aggregation. The macro-F1 curve is
> mathematically the unweighted mean of the seven per-class F1 curves.

## 1 The mathematical relationship

For every round $r$, every seed $s$, every algorithm:

$$\text{val\_macro\_f1}(r) = \frac{1}{7} \sum_{c=0}^{6} \text{val\_f1\_class\_}c(r)$$

This is **not** a coincidence; it is the definition of macro-F1.

## 2 Verified with one concrete data point

At round 50, FedAvg's mean across the 10 seeds:

| Class | val_f1_class_c |
|---|---|
| 0 actinic | 0.278 |
| 1 basal | 0.439 |
| 2 benign_kerat | 0.336 |
| 3 dermato | 0.269 |
| 4 melanoma | 0.062 |
| 5 mel_nevi | 0.871 |
| 6 vascular | 0.574 |

$(0.278+0.439+0.336+0.269+0.062+0.871+0.574)/7 = 2.829/7 = \mathbf{0.404}$

Cross-check with the macro-F1 curve at round 50: **0.404** ✓. Identical.

## 3 What each panel computes

### Per-class F1 (Figure 9)

For each class $c$:
1. Filter validation predictions to those where true label = $c$ **or** predicted = $c$.
2. Compute precision and recall for class $c$.
3. $F1_c = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$.

### Macro-F1 (Figure 8 Panel A)

Computes per-class F1 for all 7 classes, then takes the **unweighted mean**:

$$\text{macro\_F1} = \frac{F1_0 + F1_1 + F1_2 + F1_3 + F1_4 + F1_5 + F1_6}{7}$$

## 4 Why "unweighted by class size" matters

The 7 classes have wildly different sizes in the validation set (~1,003 samples total):

| Class | Approx. val samples | Prevalence |
|---|---|---|
| mel_nevi | 673 | 67.05 % |
| melanoma | 111 | 11.11 % |
| benign_kerat | 110 | 10.97 % |
| basal | 51 | 5.13 % |
| actinic | 33 | 3.27 % |
| vascular | 14 | 1.41 % |
| dermato | 12 | 1.15 % |

A size-weighted metric (like accuracy) would let mel_nevi dominate at 67%. A model that gets only mel_nevi right would score very high on accuracy (~0.67), masking complete failure on every other class.

**Macro-F1 deliberately ignores size.** A model that gets only mel_nevi right scores 0.88/7 ≈ 0.13 — appropriately low. This is why macro-F1 is the standard metric for imbalanced classification and why your headline test uses it.

## 5 What macro-F1 hides that per-class reveals

**Example: FedAvg at round 50.** Macro-F1 = 0.404 — looks mediocre.

Per-class breakdown:
- mel_nevi F1 = 0.87 (excellent)
- melanoma F1 = **0.06** (almost complete failure)
- actinic F1 = 0.28 (poor)

The macro-F1 of 0.404 is an average of one excellent class and six mediocre-to-failing ones. **The melanoma collapse is invisible at the macro level.** This is why both figures must appear in the thesis.

## 6 What each figure lets you say

### Things provable from Figure 8 alone

- "FedProx achieves higher overall validation macro-F1 than FedAvg from round 5 onwards"
- "The largest gap is at round 50 (Δ = +0.094)"
- "Both algorithms begin overfitting around round 30"

### Things requiring Figure 9 (not provable from macro-F1 alone)

- "FedProx's mid-training advantage is concentrated on melanoma" — Fig 9 Panel E shows FedAvg at 0.06 and FedProx at 0.30 at round 50. Macro-F1 alone gives Δ = +0.094 without saying *where* it comes from.
- "The majority class is learned equally well by both algorithms" — Fig 9 Panel F shows mel_nevi F1 ≈ 0.87 by round 10 for both. The macro averaging hides this.
- "FedAvg never recovers full melanoma performance" — Fig 9 Panel E shows FedAvg's melanoma F1 plateauing at ~0.26 across the full 150 rounds.
- "Per-class trajectories differ in shape" — e.g., actinic peaks around round 50 then declines slightly; basal grows monotonically.
- "Minority classes have wider uncertainty bands than the majority class" — F1 on 12-sample classes (dermato) is noisier than on 673-sample classes (mel_nevi).

## 7 Summary table

|  | Figure 8 Panel A (macro-F1) | Figure 9 (per-class F1) |
|---|---|---|
| What's plotted | 1 curve per algorithm | 7 curves per algorithm |
| What's measured | Quality "on average across classes" | Quality "on each class individually" |
| Sensitive to class size? | No (equal weighting) | No (each class isolated) |
| Number reported | 1 number per round per seed | 7 numbers per round per seed |
| Where it goes in the thesis | Headline figure + statistical test | Per-class analysis + safety check |
| What it can hide | Per-class collapse | Aggregate quality |
| What it reveals | Stable headline trajectory | Which classes drive the gap |

## 8 Why both belong in the thesis

If you only showed Figure 8 (macro-F1), a reviewer would ask: *"Is the gain on a clinically important class, or just an average over noise on rare classes?"*

If you only showed Figure 9 (per-class), a reviewer would ask: *"What's the overall result?"*

**Together they tell the complete story:**
- Macro-F1 is the headline (FedProx wins by Δ = +0.027, p = 0.020)
- Per-class explains the mechanism (gain concentrated on melanoma +0.114, actinic +0.067)
- Per-class also provides the safety check (no class regresses by more than 0.012)

The two figures are **complementary, not redundant**: same data, different aggregation levels, both informative.
