# Methodology — full chapter draft

> Drop-in section for the experimental methodology. Cite-as-you-go in the
> bracketed [Author Year] form; corresponding BibTeX entries in
> `07_bibliography.bib`.

## 1 Problem statement

Real-world medical imaging is partitioned across institutions that cannot share patient data due to GDPR and HIPAA constraints. Each institution observes a non-representative slice of the global distribution — a specialist dermatology clinic sees mostly melanoma; a general hospital, mostly benign nevi. Federated Learning (FL) [McMahan 2017] trains a single global model across these silos via parameter aggregation, without raw data leaving any client.

The canonical FL algorithm, FedAvg, assumes that local stochastic gradients average to an unbiased estimate of the global gradient. This assumption breaks under heterogeneous (non-IID) client distributions: each client's loss landscape has a different minimum, and after many local update steps, the client models drift apart — a phenomenon known as **client drift**. FedProx [Li 2020] addresses client drift by augmenting each client's local objective with a proximal term that anchors local updates to the round-start global model.

This dissertation quantifies whether FedProx improves over FedAvg on a clinically realistic non-IID partition of DermaMNIST [Yang 2023; Tschandl 2018], and whether any improvement is statistically distinguishable from random seed variance.

## 2 Dataset

DermaMNIST is the dermatology subset of MedMNIST v2 [Yang 2023], derived from the HAM10000 collection [Tschandl 2018]. It contains 10,015 dermatoscopic images of skin lesions across seven diagnostic categories, with the canonical split:

| Split | Samples |
|---|---|
| Train | 7,007 |
| Validation | 1,003 |
| Test | 2,005 |

The classification target is one of seven lesion types. The class distribution is **severely imbalanced**, faithful to underlying medical prevalence:

| Class | Code | Train prevalence |
|---|---|---|
| Melanocytic nevi | mel_nevi | 67.05 % |
| Melanoma | melanoma | 11.11 % |
| Benign keratosis-like | benign_kerat | 10.97 % |
| Basal cell carcinoma | basal | 5.13 % |
| Actinic keratoses | actinic | 3.27 % |
| Vascular lesions | vascular | 1.41 % |
| Dermatofibroma | dermato | 1.15 % |

The imbalance ratio of 58:1 between mel_nevi and dermato is **not** corrected by resampling. This is a deliberate choice: a clinically deployable FL system must learn under such priors, and any reported gains should reflect realistic behaviour.

Images are loaded from the official `dermamnist_64.npz` archive at 64×64 resolution and resized to 28×28 to match the MedMNIST v1 baseline.

## 3 Model architecture

The classifier is a 4-block convolutional network (`DermMNISTCNN`, ≈ 423K
parameters). Each block contains a single 3×3 convolution (padding=1)
followed by GroupNorm, ReLU, and a 2×2 max-pool (or adaptive global
average pool for the final block). The exact computational graph
implemented in `mnist_dermnist/models/dermmnist_cnn.py` is:

```
Conv(3→32,   k=3, p=1) → GN(groups=4,  channels=32)  → ReLU → MaxPool(2)
Conv(32→64,  k=3, p=1) → GN(groups=8,  channels=64)  → ReLU → MaxPool(2)
Conv(64→128, k=3, p=1) → GN(groups=16, channels=128) → ReLU → MaxPool(2)
Conv(128→256,k=3, p=1) → GN(groups=16, channels=256) → ReLU → AdaptiveAvgPool(1×1)
Flatten → Linear(256→128) → ReLU → Dropout(0.2) → Linear(128→7)
```

Parameter count: 423,175 (verified at runtime). The use of
`AdaptiveAvgPool2d((1, 1))` in the final block allows the same
architecture to accept any spatial input size from 28×28 upwards without
modification.

### Why GroupNorm rather than BatchNorm

BatchNorm computes running mean/variance statistics over each mini-batch. Under federated training, each client's mini-batch distribution differs from the global distribution — particularly under non-IID partitions where a client may see only a subset of classes. Per-client BatchNorm statistics therefore encode local-distribution information that cannot be safely averaged across clients [Hsieh 2020]. GroupNorm normalises across feature-group dimensions within a single sample, independent of batch composition; its parameters average correctly. All convolutional blocks use GroupNorm.

## 4 Federated learning setup

**Framework.** Flower 1.x [Beutel 2020] in simulation mode. The server orchestrates a fixed pool of seven clients; each runs inside a Ray actor with one A100 GPU allocated (`num_gpus = 1.0`).

**Communication round structure.** At each of R = 150 rounds:

1. Server broadcasts global parameters $w^t$ to all clients.
2. Each client $i$ performs E = 20 local epochs of SGD on its private partition.
3. Server collects updated parameters $w_i^{t+1}$ from all clients (full participation, `fraction-fit = 1.0`).
4. Server aggregates by the weighted-mean rule:
   $$w^{t+1} = \sum_i \tfrac{n_i}{\sum_j n_j} w_i^{t+1}$$
   where $n_i$ is client $i$'s training-set size.
5. Server evaluates the aggregated model on a held-out validation set.

**Hyperparameters (fixed across all runs):**

| Parameter | Value | Justification |
|---|---|---|
| Clients (K) | 7 | Matches partition design |
| Fraction fit (C) | 1.0 | Full participation isolates the drift effect |
| Rounds (R) | 150 | Sufficient for FedAvg convergence |
| Local epochs (E) | 20 | High-E regime where drift dominates [Li 2020 Thm 4] |
| Optimizer | SGD + momentum | [McMahan 2017] |
| Learning rate | 0.01 | Standard for CNNs at this scale |
| Momentum | 0.9 | Standard |
| Weight decay | 0.0 | Avoid conflation with FedProx's L2-like term |
| Batch size | 32 | Capped per-client to min(32, client_size) |

The choice E = 20 is critical: with E = 1, local SGD takes a single step and the proximal anchor has no drift to constrain, so FedAvg and FedProx degenerate to the same algorithm. E = 20 corresponds to the regime [Li 2020] motivates.

## 5 Partition design: `balanced_paired_7_clients`

The single most consequential methodological decision in non-IID FL benchmarking is the partition. Three desiderata constrained our design:

1. **Non-IID enough to challenge FedAvg.** Each client must see a class distribution far from the global prior.
2. **No client dominates aggregation.** Under FedAvg's size-weighted mean, a 90%-of-data hospital silo overwhelms minority-class specialists, masking any algorithmic difference.
3. **No class held by exactly one client.** Singleton ownership penalises the only client that can learn that class, producing catastrophic per-class regression — observed in pilot experiments with a `balanced_specialist` partition.

We instantiate these criteria with the following composition:

| Client | Composition | Size |
|---|---|---|
| C0 | actinic + basal + nevi | 964 |
| C1 | actinic + basal + nevi | 963 |
| C2 | benign_kerat + dermato + nevi | 1,095 |
| C3 | benign_kerat + dermato + nevi | 1,094 |
| C4 | melanoma + vascular + nevi | 1,110 |
| C5 | melanoma + vascular + nevi | 1,108 |
| C6 | nevi only | 673 |

Total = 7,007 (entire DermaMNIST training set). Client sizes are within a 1.65× range (max/min = 1,110/673), so no client dominates aggregation. Every minority class is held by **two** clients, so the FedProx anchor can pull a drifting specialist back toward a peer specialist rather than toward a majority-only generalist. C6 is a hold-out generalist on the dominant class.

This partition is a deliberate departure from the more common Dirichlet-α partitioner [Hsu 2019]; we acknowledge this in §10 (Threats to validity).

## 6 Algorithms

### 6.1 FedAvg (baseline)

Each client minimises the local cross-entropy loss for E epochs:
$$\mathcal{L}_i^{\text{FedAvg}}(w) = \tfrac{1}{|\mathcal{D}_i|} \sum_{(x, y) \in \mathcal{D}_i} \mathrm{CE}\bigl(f_w(x), y\bigr)$$
The server aggregates by the weighted-mean rule above.

### 6.2 FedProx

FedProx augments the local objective with a proximal term:
$$\mathcal{L}_i^{\text{FedProx}}(w) = \mathcal{L}_i^{\text{FedAvg}}(w) + \tfrac{\mu}{2}\, \lVert w - w^t \rVert_2^2$$

When μ = 0, FedProx is bit-identical to FedAvg; for μ > 0, clients are anchored to the round-start global model $w^t$. We fix μ = 0.01 for the headline sweep, selected on the same partition via a pilot 3-seed sweep over $\{0.001, 0.01, 0.1\}$.

**Implementation note.** $w^t$ is snapshotted at the start of each round and detached from the autograd graph, so no gradient flows through the anchor. The proximal term is summed over all trainable parameters. FedAvg and FedProx share the same implementation; FedAvg is the special case μ = 0, ensuring no algorithm-divergent code paths.

## 7 Paired-seed experimental protocol

The single most important methodological control is **paired seeding**: for each seed $s \in S$, we run FedAvg and FedProx with **identical** random state — identical partition, identical model initialisation, identical dataloader shuffling, identical SGD batch ordering. The seed governs:

- `numpy`, `torch`, and `random` global RNG state
- The partition function (which classes land on which client)
- The model's weight initialisation
- The dataloader's batch generator, seeded per-(seed, round, cid) as $seed \times 10^4 + round \times 10^2 + cid$ to break aliasing across clients within a round.

Thus the within-pair Δ at a given seed reflects only the algorithmic difference (presence vs absence of the proximal term), not random-state variation. Each pair is treated as a within-subject observation and analysed with paired statistics [Demšar 2006].

**Sweep:** $S = \{42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828\}$ (10 paired seeds × 2 algorithms = 20 SLURM jobs on the CSD3 Ampere partition; ~1 GPU-hour per job).

Per-round validation metrics are written to `history_*.csv`; final-test metrics at the round of best validation macro-F1 are written to `test_at_best_*.json`. The test set is evaluated **only** at this best-val round, never at every round, to avoid test-set selection bias.

## 8 Metrics

**Primary:**
- **Test macro-F1** — unweighted mean of per-class F1 scores. Required under severe class imbalance; accuracy is misleadingly high (≈ 0.67) for any model that predicts the majority class for every input.

**Secondary:**
- Test balanced accuracy (mean per-class recall).
- Test accuracy (reported for completeness, never used as headline).
- Per-class F1 (7 classes) — catches per-class regression that the macro mean would hide.
- Validation macro-F1 trajectory — convergence diagnostic.

The headline Δ is:
$$\Delta_s = M_s^{\text{FedProx}} - M_s^{\text{FedAvg}}$$
where $M$ is test macro-F1 and $s$ indexes the seed.

## 9 Statistical analysis

With *n* = 10 paired seeds we report:

1. **Paired Wilcoxon signed-rank test** on the per-seed Δ values. We use Wilcoxon over a paired *t*-test because *n* = 10 is too small to reliably check normality, and Δ may be skewed by the long tail of minority-class F1 [Demšar 2006]. We report the two-sided p-value as the headline statistic and the one-sided (FedProx > FedAvg) p-value as a secondary, given the pre-registered direction.

2. **Rank-biserial effect size** $r_{\text{rb}} \in [-1, 1]$ — the signed fraction of pairs in which FedProx wins [Kerby 2014]. $|r_{\text{rb}}| < 0.1$ = negligible; $0.1$ – $0.3$ = small; $0.3$ – $0.5$ = medium; $> 0.5$ = large.

3. **Bootstrap 95% CI on mean Δ** — 10,000 resamples of the 10 pairs with replacement [Efron & Tibshirani 1993].

4. **Per-class paired Wilcoxon** for each of 7 classes — reported with raw p-values; Bonferroni-corrected threshold $\alpha / 7 \approx 0.007$ flagged where relevant.

A finding will be reported as "FedProx improves over FedAvg" only if (a) the median Δ is positive, (b) the two-sided Wilcoxon p < 0.05, (c) the rank-biserial $|r_{\text{rb}}| \geq 0.3$, and (d) no per-class F1 regresses by more than 0.05 on average. Criterion (d) prevents a positive macro-F1 from masking a per-class collapse.

## 10 Computational environment

- **HPC:** Cambridge CSD3, Ampere partition, NVIDIA A100-SXM4-80GB, CUDA 12.x.
- **Submission:** SLURM via `mnist_dermnist/scripts/submit_headline.sh`. Each job: 1 A100, 4 CPUs, 8-hour wall-clock cap.
- **Per-job runtime:** ≈ 1 hour for FedAvg, ≈ 1.5 hours for FedProx (proximal term adds ~30 % per-batch overhead).
- **Total compute:** ≈ 25 GPU-hours for the 20-job headline sweep.
- **Software:** Python 3.9, PyTorch 2.x, Flower 1.x. RNG seeded deterministically at NumPy, PyTorch global, and dataloader levels; CUDA-kernel non-determinism is treated as part of the algorithmic noise floor (estimated < 0.005 macro-F1 from repeated identical-seed runs).

## 11 Reproducibility

- Repository: [redacted-during-review] (`mnist_dermnist/` subtree).
- Dataset: DermaMNIST 64×64, official MedMNIST v2 archive [Yang 2023].
- Seeds: 10 listed above.
- Run all 20 jobs: `bash mnist_dermnist/scripts/submit_headline.sh`.
- Regenerate analysis: `python -m mnist_dermnist.analysis.tables --results-dir mnist_dermnist/results/headline`.
- Regenerate figures: `python mnist_dermnist/results/thesis_ready/scripts/generate_all_figures.py`.

## 12 Threats to validity

**Internal.** Paired-seed design controls for initialisation variance; CUDA non-determinism within a seed is estimated < 0.005 macro-F1 and is treated as noise.

**Construct.** Macro-F1 weights all 7 classes equally, including the 1.15%-prevalence dermato class. We complement with per-class F1 to detect any class collapse.

**External.** Results are specific to (i) DermaMNIST at 28×28 resolution, (ii) `balanced_paired_7_clients` partition, (iii) μ = 0.01, and (iv) R = 150, E = 20. Generalisation to other medical-imaging datasets, partition schemes (e.g. Dirichlet-α [Hsu 2019]), or hyperparameter regimes is not claimed.

**Partition design.** Our partition is custom-designed for this experiment and not directly comparable to the Dirichlet-α partitions standard in the broader FL literature. We justify this in §5 by the three pre-registered desiderata. We do not run a Dirichlet-α control in this work; this is identified as future work.

**μ selection.** μ = 0.01 was chosen on a 3-seed validation sweep; with *n* = 3 the chosen μ may not be the true population-best. A larger μ sweep is left to future work.
