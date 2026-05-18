# System heterogeneity — hypotheses and expected outcomes

Read this **before** seeing results, so you can recognise whether they
match theory or fail it (a falsified prediction is also a publishable
finding).

## 1 Background

The proximal regulariser in FedProx,

$$\mathcal{L}_i^{\text{FedProx}}(w) = \mathcal{L}_i^{\text{FedAvg}}(w) + \tfrac{\mu}{2}\,\lVert w - w^t \rVert_2^2,$$

was introduced by Li et al. (2020) primarily to handle **inexact local
updates** — partial work submitted by stragglers — in a principled manner.
Their Theorem~4 proves a convergence guarantee for FedProx under γ-inexact
updates: each client may submit a parameter vector $w_i^{t+1}$ that does not
fully minimise its local objective, provided that the resulting "inexactness"
is bounded. FedAvg has no such guarantee under heterogeneous local work and
its size-weighted aggregation can be biased by whichever clients drift
furthest.

## 2 Primary hypothesis (H1)

Under system heterogeneity at the same statistical-heterogeneity partition
(`balanced_paired_7_clients`),

$$\Delta_\text{system-het} \;>\; \Delta_\text{baseline} \;=\; +0.027,$$

where $\Delta = $ test macro-F1$(\text{FedProx}) - $ test macro-F1$(\text{FedAvg})$.

In other words: **FedProx's advantage over FedAvg should be LARGER when
clients have heterogeneous compute budgets than when they don't**, because
the regime is closer to the one FedProx was theoretically designed for.

**How tested:** paired Wilcoxon signed-rank on the per-seed differences
$\Delta^{\text{sh}}_s - \Delta^{\text{baseline}}_s$, $n = 10$ seeds.
$p < 0.05$ supports H1.

## 3 Secondary hypothesis (H2): FedProx is more straggler-tolerant

Define the **straggler-tolerance ratio** for each algorithm as

$$\rho_\text{algo} \;=\; \frac{M^\text{system-het}_\text{algo}}{M^\text{baseline}_\text{algo}},$$

where $M$ is test macro-F1. Then H2 predicts

$$\rho_\text{FedProx} \;>\; \rho_\text{FedAvg}.$$

That is, the relative performance drop introduced by stragglers is smaller
for FedProx. If FedAvg drops from macro-F1 = 0.481 to 0.42 (a 13% drop)
while FedProx drops from 0.508 to 0.495 (a 3% drop), then $\rho_{\text{FA}} =
0.87$ and $\rho_{\text{FP}} = 0.97$ — FedProx is more straggler-tolerant.

## 4 What we expect to see — best-case scenario

If Li et al.'s theory transfers cleanly:

| Quantity | Headline (no system het) | C1 (fixed stragglers) | C2 (random stragglers) |
|---|---|---|---|
| FedAvg test macro-F1 | $0.481 \pm 0.025$ | $\sim 0.42$–$0.46$ | $\sim 0.40$–$0.45$ |
| FedProx test macro-F1 | $0.508 \pm 0.014$ | $\sim 0.49$–$0.51$ | $\sim 0.47$–$0.50$ |
| $\Delta$ | $+0.027$ ($p = 0.020$) | $\sim +0.04$–$+0.06$ | $\sim +0.05$–$+0.08$ |
| FedAvg straggler-tolerance $\rho$ | 1.00 (reference) | $\sim 0.90$ | $\sim 0.85$ |
| FedProx straggler-tolerance $\rho$ | 1.00 (reference) | $\sim 0.97$ | $\sim 0.95$ |

These ranges are conservative estimates — actual numbers depend on the
strength of statistical heterogeneity already present in the partition.

## 5 What would falsify the hypotheses

Equally important to know in advance:

| Observation | Interpretation |
|---|---|
| $\Delta_\text{system-het} \approx \Delta_\text{baseline}$ | System heterogeneity adds no extra advantage to FedProx. The statistical-heterogeneity story stands, but the FedProx-as-system-het-solution claim fails. Honest discussion required. |
| $\Delta_\text{system-het} < \Delta_\text{baseline}$ | System heterogeneity actually *hurts* FedProx relative to FedAvg. Would be surprising; would suggest the proximal term is over-regularising under high-variance updates. |
| Both algorithms tank equally | $\rho_\text{FedAvg} \approx \rho_\text{FedProx} \approx 0.7$. Suggests neither algorithm handles severe system het well; FedProx's theoretical advantage doesn't transfer. |
| FedProx loses on individual seeds | If LOSO Wilcoxon fails on system-het sweeps, the result is weaker than the headline. |

## 6 What this section adds to the thesis story

The statistical-heterogeneity section (`thesis_ready/`) demonstrates that
FedProx helps on a designed non-IID partition. By itself this is one
finding. **Adding system heterogeneity tests the more fundamental FedProx
claim**: that the proximal term enables principled aggregation when local
work is inexact. If the system-het result also favours FedProx, the thesis
defends a stronger overall claim: FedProx improves over FedAvg on two
distinct dimensions of FL heterogeneity, in a manner consistent with the
original theoretical motivation in Li et al. (2020). If the system-het
result is null, the statistical-heterogeneity claim is unaffected — but a
narrower claim is defended ("FedProx helps under non-IID data but not
under straggler regimes in our setup").

Either outcome is publishable. The pre-registered hypotheses above ensure
the analysis is honest in either direction.

## 7 Connections to related work

| Paper | What they tested | What we test additionally |
|---|---|---|
| **Li et al. (2020) FedProx** | Random stragglers on Synthetic / FEMNIST / Shakespeare with $E \in \{1, 5, 20\}$. Reported up to 90% straggler fractions. | Same setup on medical imaging (DermaMNIST) at $E = 20$, with paired-seed protocol and Wilcoxon test. |
| **Wang et al. (2020) FedNova** | Heterogeneous $E_i$ with FedNova normalisation. Compared FedNova vs FedAvg vs FedProx. | We do not implement FedNova; future work could add it as a competitor. |
| **Karimireddy et al. (2020) SCAFFOLD** | Control variates for client drift; tested heterogeneous local work. | We do not implement SCAFFOLD; future work. |
| **Marija (2025) thesis §3.8.4** | 2-client fixed-straggler setup with U-Net segmentation; single seed. | Our setup has 7 clients, 10 paired seeds, classification, and full statistical inference. |

## 8 Pre-registration statement

The hypotheses above are pre-registered: they are written before the system-
heterogeneity SLURM jobs have completed, and the test procedures (paired
Wilcoxon on $\Delta^{\text{sh}}_s - \Delta^{\text{baseline}}_s$, two-sided
$\alpha = 0.05$, $n = 10$) are committed to in advance. This protects
against post-hoc rationalisation of whatever the data shows.
