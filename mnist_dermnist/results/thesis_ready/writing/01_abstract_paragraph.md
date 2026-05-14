# Abstract paragraph (drop into thesis abstract)

## Short form (~80 words)

> We compare FedAvg and FedProx on DermaMNIST under a custom non-IID partition (`balanced_paired_7_clients`) designed to eliminate singleton-class ownership while preserving clinically realistic class heterogeneity. Across 10 paired seeds, FedProx (μ = 0.01) achieves a mean test macro-F1 of 0.508 ± 0.014 versus FedAvg's 0.481 ± 0.025, an improvement of Δ = +0.027 ± 0.035 (paired Wilcoxon p = 0.020, rank-biserial r = +0.818). The largest per-class gain is on melanoma (Δ = +0.114, p = 0.006), the most clinically critical lesion.

## Long form (~150 words) — for thesis abstract or executive summary

> Federated Learning (FL) trains models across data silos without centralising patient data, but the canonical FedAvg algorithm (McMahan et al., 2017) suffers under client heterogeneity. We evaluate FedProx (Li et al., 2020), which adds a proximal regulariser to local objectives, on DermaMNIST — a 7-class skin-lesion classification dataset with severe class imbalance (58:1 between majority and minority classes). We design a custom non-IID partition (`balanced_paired_7_clients`) in which every minority class is held by exactly two clients, avoiding the catastrophic per-class regression observed under singleton-class partitions. Across 10 paired-seed runs (R = 150 rounds, E = 20 local epochs, μ = 0.01), FedProx achieves a mean test macro-F1 of 0.508 ± 0.014 against FedAvg's 0.481 ± 0.025 — a within-pair improvement of Δ = +0.027 ± 0.035 (paired Wilcoxon p = 0.020, rank-biserial r = +0.818). FedProx wins on 9 of 10 paired seeds. The largest per-class gain is on melanoma (Δ = +0.114, p = 0.006), the most clinically critical lesion. No class regresses by more than 0.012 F1, satisfying the pre-registered safety criterion. FedProx additionally reduces across-seed variance by 45%, suggesting more stable convergence under non-IID conditions.

## One-sentence elevator pitch

> FedProx improves test macro-F1 over FedAvg by +0.027 (Wilcoxon p = 0.020, r = +0.818, n = 10 paired seeds) on a non-IID DermaMNIST partition, with the largest gain on the clinically critical melanoma class (+0.114 F1).
