# Discussion — full chapter draft

> Use this as the scaffold of the discussion chapter; expand or trim sections
> to fit the chapter target length.

## 1 What our results add to the FedProx literature

FedProx was introduced by Li et al. (2020) primarily on benchmark datasets (Synthetic, FEMNIST, Shakespeare, Sentiment140) chosen for their controllable degree of statistical heterogeneity. Their experiments used predominantly low E (≤ 1) configurations. Our results extend this evidence base in three concrete ways:

1. **A non-toy medical-imaging benchmark.** DermaMNIST has clinically realistic class imbalance (58:1 prevalence ratio between the most and least frequent class) that the original FedProx benchmarks do not capture. Our finding that FedProx significantly improves macro-F1 under such imbalance is novel.

2. **The high-E regime.** With E = 20, our experiments are in the regime where client drift is most pronounced — exactly where FedProx's proximal term is theoretically motivated. The 9/10 win rate at this E confirms the mechanism scales beyond the low-E settings reported by Li et al.

3. **A clinically informed per-class analysis.** Macro-F1 averages 7 classes equally; reporting per-class F1 with significance tests reveals **where** the improvement materialises. The +0.114 F1 improvement on melanoma (p = 0.006) is the strongest single per-class result we are aware of in FedProx-vs-FedAvg comparisons on medical imaging.

## 2 Why FedProx wins specifically on rare classes

The mechanism we hypothesise — and that our per-class results are consistent with — is as follows.

Under our `balanced_paired_7_clients` partition, the majority class (mel_nevi, 67 %) is present in every client. The minority classes (especially melanoma and dermato) are present in only two clients each. During FedAvg aggregation, the size-weighted mean pulls the global model toward the **majority-class-friendly** parameter region, because every client has gradient signal pushing toward correct mel_nevi prediction. Minority-class specialist clients' updates are diluted.

FedProx's proximal term explicitly penalises deviation from the *current* global model. This has two effects:

1. It dampens any single client's update magnitude, including the majority-class generalist (C6, nevi-only), which is otherwise unconstrained.
2. It allows minority-class signal — which is consistent across the two specialist clients for each minority class — to survive aggregation without being washed out by drifting majority gradients.

The result is that minority-class predictive accuracy improves without compromising majority-class performance — exactly the per-class pattern we observe: melanoma +0.114, actinic +0.067, with mel_nevi virtually unchanged (+0.002).

## 3 Variance reduction as a secondary contribution

FedProx's 45 % reduction in across-seed standard deviation is a finding that, to our knowledge, has not been quantified in prior FL papers comparing FedAvg and FedProx. Most papers report central tendency only. Bouthillier et al. (2021) argue that variance is itself a benchmark-worthy property: a method with lower variance can be deployed with higher confidence in a single run.

For federated medical-imaging deployment, where re-running the entire training pipeline is operationally expensive (each silo must re-train, re-aggregate, re-validate), an algorithm that produces more reproducible global models has practical value beyond mean accuracy.

## 4 Clinical significance of the melanoma result

Melanoma is the most clinically consequential class in DermaMNIST — missed melanoma diagnoses lead to delayed treatment and substantial mortality risk. A baseline classifier (FedAvg) achieves only F1 = 0.187 on melanoma in our setup, reflecting both the small training fraction (11.1 %) and the visual similarity of early melanomas to benign nevi. FedProx improves this to F1 = 0.301 — a **61 % relative improvement** on the most important class, statistically significant at p = 0.006.

This is not, by itself, a clinically deployable system: F1 = 0.301 is far below screening-grade. But the magnitude of improvement from a single algorithmic change suggests that FedProx-style proximal regularisation should be considered a baseline default for non-IID medical-imaging FL, alongside other interventions (model architecture, class re-weighting, ensembling) that any production system would stack.

## 5 Limitations

### 5.1 Single dataset

Our results are on DermaMNIST. Generalisation to other medical-imaging datasets (chest X-rays, histopathology, retinal images) is plausible but not demonstrated. We treat the consistency of the macro-F1 win and the per-class mechanism as evidence that the effect is unlikely to be DermaMNIST-specific, but a multi-dataset extension is future work.

### 5.2 Single partition design

We evaluate on one partition (`balanced_paired_7_clients`), which we designed with three pre-registered desiderata. The most common alternative in FL literature — Dirichlet-α partitioning [Hsu et al. 2019] — is not run here. The Dirichlet-α family allows tuning the non-IID severity along a single axis, and is the de facto standard for FL benchmarks. Including a Dirichlet-α robustness check at α ∈ {0.1, 0.5} would strengthen external validity. This is the most important single piece of future work flagged by this thesis.

### 5.3 μ chosen on small sweep

We selected μ = 0.01 on a 3-seed pilot sweep over {0.001, 0.01, 0.1}. With n = 3 the selected μ may not be the population-best; a larger sweep at n = 5+ seeds would refine this estimate. We chose to commit GPU budget to the headline 10-seed comparison rather than a broader μ sweep, on the rationale that even a sub-optimal μ that yields significant Δ is sufficient for the headline claim. A "μ-optimal" version of the comparison may yield a larger Δ.

### 5.4 No IID falsification check

A control experiment under an IID partition (uniform random sharding) should yield Δ ≈ 0 by theory: under IID, client gradients are unbiased estimators of the global gradient and FedAvg suffices. We do not run this control in this thesis. Adding 3 paired-seed IID runs (~6 A100-hours) would close this falsification gap.

### 5.5 28×28 resolution

We use 28×28 images to match the MedMNIST v1 baseline. Modern dermoscopy classifiers operate at higher resolution (224×224, 384×384). Our absolute macro-F1 numbers therefore reflect a heavily downsampled benchmark, not state-of-the-art dermatology AI. The *relative* claim (FedProx > FedAvg) is invariant to resolution under fixed model capacity.

## 6 Future work

Beyond the limitations above, three concrete extensions are within scope of follow-up work:

1. **Dirichlet-α robustness study.** Run paired-seed FedAvg vs FedProx at α ∈ {0.1, 0.3, 0.5, 1.0} to map FedProx's advantage as a function of heterogeneity severity.
2. **Cross-dataset extension.** Repeat the experiment on PathMNIST and BloodMNIST. These two MedMNIST datasets have similar class-imbalance structure but different visual modalities, providing tests of generalisation.
3. **Mechanism ablation.** Replace the proximal term with explicit norm-clipping of local updates. If norm-clipping reproduces FedProx's gains, the mechanism is "update magnitude restriction" rather than "anchoring to global model" — a methodologically interesting distinction.

## 7 Concluding remark

FedProx is not new (2020), and proximal regularisation is not novel as an algorithmic idea. Our contribution is **clear empirical evidence that, on a non-IID medical-imaging benchmark under high local-epoch counts, FedProx produces a statistically significant, clinically interpretable improvement over FedAvg, with the gain concentrated on the most clinically critical class.** For practitioners building federated medical-imaging systems, FedProx with small μ (~0.01) should be the default baseline rather than FedAvg.
