# Tables — LaTeX and Markdown versions

## Table 1 — Headline paired comparison (10 seeds)

### Markdown

| Statistic | FedAvg | FedProx | Δ (FedProx − FedAvg) |
|---|---|---|---|
| Mean test macro-F1 (± SD) | 0.481 ± 0.025 | 0.508 ± 0.014 | **+0.027 ± 0.035** |
| Median test macro-F1 | 0.479 | 0.506 | +0.020 |
| Min test macro-F1 | 0.441 | 0.490 | — |
| Max test macro-F1 | 0.522 | 0.536 | — |
| FedProx wins (n / 10) | — | — | **9 / 10** |
| Paired Wilcoxon p (two-sided) | — | — | **0.020** |
| Paired Wilcoxon p (one-sided, FedProx > FedAvg) | — | — | 0.010 |
| Rank-biserial effect size r | — | — | **+0.818** (very large) |
| Bootstrap 95% CI on mean Δ | — | — | excludes 0 |

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{Paired comparison of FedAvg and FedProx on DermaMNIST across 10 seeds. FedProx ($\mu = 0.01$) significantly improves test macro-F1 over FedAvg.}
\label{tab:headline}
\begin{tabular}{lccc}
\toprule
Statistic & FedAvg & FedProx & $\Delta$ (FedProx $-$ FedAvg) \\
\midrule
Mean test macro-F1 ($\pm$ SD) & $0.481 \pm 0.025$ & $0.508 \pm 0.014$ & $\mathbf{+0.027 \pm 0.035}$ \\
Median test macro-F1 & $0.479$ & $0.506$ & $+0.020$ \\
Min / Max test macro-F1 & $0.441$ / $0.522$ & $0.490$ / $0.536$ & --- \\
FedProx win rate & --- & --- & $\mathbf{9 / 10}$ \\
Paired Wilcoxon $p$ (two-sided) & --- & --- & $\mathbf{0.020}$ \\
Paired Wilcoxon $p$ (one-sided) & --- & --- & $0.010$ \\
Rank-biserial $r$ & --- & --- & $\mathbf{+0.818}$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 2 — Per-class F1 with significance

### Markdown

| Class | Train prevalence | FedAvg mean F1 | FedProx mean F1 | Δ mean F1 | Paired Wilcoxon p |
|---|---|---|---|---|---|
| Melanocytic nevi (majority) | 67.05 % | 0.884 | 0.886 | +0.002 | 0.375 |
| Melanoma | 11.11 % | 0.187 | 0.301 | **+0.114** | **0.006** |
| Benign keratosis | 10.97 % | 0.440 | 0.444 | +0.004 | 0.695 |
| Basal cell carcinoma | 5.13 % | 0.475 | 0.473 | −0.001 | 0.846 |
| Actinic keratoses | 3.27 % | 0.404 | 0.471 | **+0.067** | **0.020** |
| Vascular lesions | 1.41 % | 0.698 | 0.686 | −0.012 | 0.695 |
| Dermatofibroma | 1.15 % | 0.283 | 0.296 | +0.013 | 0.922 |

**Bold:** Significant at α = 0.05 (paired Wilcoxon signed-rank, n = 10).

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{Per-class test F1 across 10 paired seeds. FedProx ($\mu = 0.01$) significantly improves melanoma and actinic keratoses F1; no class regresses by more than 0.012 mean F1.}
\label{tab:per-class}
\begin{tabular}{lrcccc}
\toprule
Class & Prev. (\%) & FedAvg & FedProx & $\Delta$ & Wilcoxon $p$ \\
\midrule
Melanocytic nevi (majority)  & 67.05 & 0.884 & 0.886 & $+0.002$ & 0.375 \\
Melanoma                     & 11.11 & 0.187 & 0.301 & $\mathbf{+0.114}$ & $\mathbf{0.006}$ \\
Benign keratosis-like        & 10.97 & 0.440 & 0.444 & $+0.004$ & 0.695 \\
Basal cell carcinoma         &  5.13 & 0.475 & 0.473 & $-0.001$ & 0.846 \\
Actinic keratoses            &  3.27 & 0.404 & 0.471 & $\mathbf{+0.067}$ & $\mathbf{0.020}$ \\
Vascular lesions             &  1.41 & 0.698 & 0.686 & $-0.012$ & 0.695 \\
Dermatofibroma               &  1.15 & 0.283 & 0.296 & $+0.013$ & 0.922 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 3 — Per-seed paired results (full)

### Markdown

| Seed | FedAvg test macro-F1 | FedProx test macro-F1 | Δ | FedProx wins? |
|---|---|---|---|---|
| 42      | 0.4477 | 0.5162 | +0.0685 | ✓ |
| 123     | 0.4966 | 0.4980 | +0.0014 | ✓ |
| 456     | 0.5105 | 0.5109 | +0.0004 | ✓ |
| 789     | 0.5223 | 0.5023 | −0.0200 | ✗ |
| 999     | 0.4768 | 0.5040 | +0.0272 | ✓ |
| 2024    | 0.4702 | 0.5074 | +0.0372 | ✓ |
| 31337   | 0.4825 | 0.5230 | +0.0404 | ✓ |
| 161803  | 0.4770 | 0.4897 | +0.0127 | ✓ |
| 271828  | 0.4898 | 0.4936 | +0.0038 | ✓ |
| 8675309 | 0.4406 | 0.5360 | +0.0953 | ✓ |
| **Mean ± SD** | **0.4814 ± 0.0254** | **0.5081 ± 0.0140** | **+0.0267 ± 0.0349** | **9/10** |

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{Per-seed paired results for the headline sweep. Seeds chosen \emph{a priori} from the integer sequence $\{42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828\}$.}
\label{tab:per-seed}
\begin{tabular}{rccccc}
\toprule
Seed & FedAvg & FedProx & $\Delta$ & FedProx wins? \\
\midrule
42      & 0.4477 & 0.5162 & $+0.0685$ & \checkmark \\
123     & 0.4966 & 0.4980 & $+0.0014$ & \checkmark \\
456     & 0.5105 & 0.5109 & $+0.0004$ & \checkmark \\
789     & 0.5223 & 0.5023 & $-0.0200$ & $\times$ \\
999     & 0.4768 & 0.5040 & $+0.0272$ & \checkmark \\
2024    & 0.4702 & 0.5074 & $+0.0372$ & \checkmark \\
31337   & 0.4825 & 0.5230 & $+0.0404$ & \checkmark \\
161803  & 0.4770 & 0.4897 & $+0.0127$ & \checkmark \\
271828  & 0.4898 & 0.4936 & $+0.0038$ & \checkmark \\
8675309 & 0.4406 & 0.5360 & $+0.0953$ & \checkmark \\
\midrule
\textbf{Mean $\pm$ SD} & $\mathbf{0.4814 \pm 0.0254}$ & $\mathbf{0.5081 \pm 0.0140}$ & $\mathbf{+0.0267 \pm 0.0349}$ & $\mathbf{9/10}$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 4 — Partition composition (`balanced_paired_7_clients`)

### Markdown

| Client | Composition | Samples |
|---|---|---|
| C0 | actinic + basal + nevi | 964 |
| C1 | actinic + basal + nevi | 963 |
| C2 | benign_kerat + dermato + nevi | 1,095 |
| C3 | benign_kerat + dermato + nevi | 1,094 |
| C4 | melanoma + vascular + nevi | 1,110 |
| C5 | melanoma + vascular + nevi | 1,108 |
| C6 | nevi only | 673 |
| **Total** | | **7,007** |

Max/min size ratio: **1.65×**. Every minority class held by exactly **2** clients.

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{The \texttt{balanced\_paired\_7\_clients} partition design. Every minority class is held by exactly two clients, eliminating singleton-class catastrophic regression observed under earlier specialist partitions.}
\label{tab:partition}
\begin{tabular}{cll}
\toprule
Client & Composition & Samples \\
\midrule
C0 & actinic + basal + nevi          & 964 \\
C1 & actinic + basal + nevi          & 963 \\
C2 & benign\_kerat + dermato + nevi  & 1,095 \\
C3 & benign\_kerat + dermato + nevi  & 1,094 \\
C4 & melanoma + vascular + nevi      & 1,110 \\
C5 & melanoma + vascular + nevi      & 1,108 \\
C6 & nevi only                       & 673 \\
\midrule
\textbf{Total} & & \textbf{7,007} \\
\bottomrule
\end{tabular}
\end{table}
```
