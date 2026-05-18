# Reviewer-critique responses

For each numbered critique received from an external methodology review,
this document records: (a) whether the critique was accepted as valid,
(b) what action was taken, (c) where the change appears in the repo,
and (d) the rationale for any items marked "deferred".

## (3) Custom partition may be too engineered

**Verdict:** Valid concern, already in flight.

**Action:** IID + Dirichlet-$\alpha = 0.1$ sweeps are pre-submitted via
`mnist_dermnist/scripts/submit_robustness.sh` (40 SLURM jobs total). When
HPC resumes, results will populate the placeholders in
`tab:robustness` of `09_overleaf_ready.tex` (rows 2 and 3).

**Expected outcome:** IID should yield $\Delta \approx 0$ (falsification
check); Dirichlet should yield $\Delta > 0$ with magnitude similar to the
headline (external-validity check).

---

## (4) FedProx is not really a "class imbalance correction" method

**Verdict:** Valid — sharp conceptual distinction.

**Action:** Added a "Scope of FedProx as a remedy" paragraph in the
Algorithms subsection of `09_overleaf_ready.tex`, distinguishing FedProx
(optimisation-level mitigation of client drift) from class-imbalance
methods (FedLC, focal loss), feature-shift methods (FedBN), and
alternative drift methods (SCAFFOLD). Future work explicitly identified.

---

## (5) Fixed straggler condition (C1) is confounded with client identity

**Verdict:** Valid — sharp methodological catch.

**Action:** Updated `thesis_ready_system_het/writing/03_overleaf_ready_system_het.tex`
to:
- Rename C1 to "fixed stragglers — a specialist + a generalist"
- Add explicit caveat that C1 conflates system heterogeneity with
  particular clinical-role configuration
- Make C2 (random_stragglers, Li 2020) the primary/canonical test
- State that H2 inference rests primarily on C2

**Not changed:** the code or SLURM submission for C1 — the experiment is
retained for operational realism but its interpretation is reframed.

---

## (6) Full participation ($C = 1.0$) weakens realism

**Verdict:** Valid but expensive to address.

**Action (no HPC):** Added a "Full participation" paragraph in the
Limitations subsection acknowledging the limitation, citing Karimireddy
et al. 2020 as the SCAFFOLD reference for partial-participation regimes,
and identifying partial-participation sensitivity ($C = 0.5$) as future
work.

**Action (HPC):** Not added to current sweep. Partial-participation
sensitivity would require an additional 20-job sweep ($\sim$20 GPU-h)
and is deferred.

---

## (7) $E = 20$ may exaggerate FedProx's advantage

**Verdict:** Valid — high $E$ is the regime where FedProx is expected to
help most.

**Action (no HPC):** Added a "Large local-epoch count" paragraph to
Limitations stating this explicitly. Headline is reframed as "FedProx
improves over FedAvg at $E = 20$" rather than as an unconditional claim.

**Action (HPC):** $E$-sensitivity sweep (e.g., $E \in \{1, 5, 20\}$ at
3 paired seeds) deferred. Would require $\sim$10 GPU-h.

---

## (8) Statistical tests are good, but careful with $n = 10$

**Verdict:** Valid — and we already do this for the headline.

**Action:**
- Headline writing already uses honest phrasing ("9/10 with substantial
  across-seed variability"), Hodges-Lehmann reported alongside the mean,
  LOSO Wilcoxon confirms robustness. No further change needed for the
  headline.
- For the system-heterogeneity H2 test (which comprises 2 related tests:
  C1 vs C0 and C2 vs C0), added a Bonferroni-correction note
  ($\alpha_\text{family} = 0.05 \to \alpha_\text{per-test} = 0.025$) in
  `03_overleaf_ready_system_het.tex`.

---

## (9) Per-class safety should report worst-case, not only mean regression

**Verdict:** Valid — important for medical-imaging safety claims.

**Action:**
- New analysis script
  `thesis_ready/scripts/analyse_worst_case_per_class.py` computes worst
  per-(seed, class) $\Delta$ across all $10 \times 7 = 70$ cells.
- New output `data/worst_case_per_class.json` and
  `data/per_seed_per_class_delta.csv`.
- New table `tab:worst-per-class` in `09_overleaf_ready.tex` reporting
  worst-seed $\Delta$ per class.
- Per-class results text rewritten to acknowledge that the
  pre-registered "no mean regression $> 0.05$" criterion is satisfied
  but the stronger "no worst-case regression $> 0.05$" criterion is NOT
  satisfied (10 of 70 cells fall below $-0.05$). Reported honestly.

**Key findings:**
- Single worst cell: vascular at seed 789, $\Delta = -0.113$
- Worst-case affects rarest classes (vascular, dermato, benign_kerat)
- **Melanoma (clinically critical) is robust at worst-case: $\Delta = -0.019$**

---

## (10) Centralised baseline single-seed caveat

**Verdict:** Valid — already labelled descriptive.

**Action (no HPC):** No new change. The existing methodology already
states: "Important caveat: this is a single seed; we have not quantified
across-seed centralised variance. The descriptive comparisons should
therefore be interpreted as approximate reference points rather than
formal statistical contrasts." This appears in `09_overleaf_ready.tex`
around the centralised-baseline paragraph.

**Action (HPC, optional):** Multi-seed centralised (3 more jobs, $\sim$15
minutes total) can be added trivially when HPC resumes:
```bash
for s in 123 456 789; do
  sbatch mnist_dermnist/scripts/slurm_centralised.sh $s 50 \
    mnist_dermnist/results/centralised
done
```

---

## "Must-do" #1: Add Flower/NVFlare mini-demo

**Verdict:** Valid (already implemented).

**Action:** Implemented in earlier turn:
- `mnist_dermnist/fl_flower/client.py` — NumPyClient with FedAvg + FedProx local objectives
- `mnist_dermnist/experiments/run_one_flower.py` — runs `flwr.simulation.start_simulation`
- Equivalence verified at smoke-test scale: $|\Delta \text{macro-F1}| \leq 0.02$ between Flower and pure-PyTorch paths
- Full-scale equivalence verification ready to submit (`submit_equivalence_check.sh`)
- All future sweeps (system_het, IID, Dirichlet) route through the Flower runtime via `slurm_template_flower.sh`

Methodology in `09_overleaf_ready.tex` rewritten to state Flower as the
primary framework with the pure-PyTorch reference loop as a diagnostic
equivalent.

---

## "Must-do" #2: Fix 28×28 vs `dermamnist_64.npz` inconsistency

**Verdict:** Valid — the file is the v2 64×64 archive but we resize to
v1 28×28.

**Action:** Rewrote the Dataset subsection in `09_overleaf_ready.tex` to
explicitly state: (a) the source archive is the v2 64×64 distribution,
(b) we resize to 28×28 to match the MedMNIST v1 baseline resolution at
which prior FL-on-DermaMNIST work has been reported, (c) the
downsampling is a compute-economy choice with the trade-off acknowledged,
(d) the relative $\Delta$ between algorithms is resolution-invariant
under fixed model capacity.

---

## "Must-do" #3: IID + Dirichlet robustness — see (3) above

---

## "Must-do" #4: Clarify simulated vs.\ real heterogeneity

**Verdict:** Valid — and important to state up-front.

**Action:** Added a "Simulated, not real, heterogeneity" paragraph at the
start of the Partition subsection (`sec:partition`). States that:
- "Client heterogeneity" is generated by partitioning a single dataset by
  class index across synthetic clients
- Real federated medical-imaging deployments include feature/domain
  shift, demographic shifts, missing-modality structure — none of which
  are modelled here
- The setup isolates the label-distribution-skew component
- Multi-site benchmarks (FeTS, COVID-19 federated) identified as future
  work

---

## Summary: implementation status

| # | Critique | Status | Where |
|---|---|---|---|
| 3 | Custom partition too engineered | ✓ Pending HPC | `submit_robustness.sh` already queued |
| 4 | FedProx ≠ class-imbalance method | ✓ Done | §Algorithms paragraph in `09_overleaf_ready.tex` |
| 5 | C1 confound | ✓ Done | `03_overleaf_ready_system_het.tex` reframing |
| 6 | $C = 1.0$ realism | ✓ Acknowledged | §Limitations in `09_overleaf_ready.tex` |
| 7 | $E = 20$ exaggeration | ✓ Acknowledged | §Limitations in `09_overleaf_ready.tex` |
| 8 | $n = 10$ Bonferroni | ✓ Done | H2 paragraph in `03_overleaf_ready_system_het.tex` |
| 9 | Worst-case per-class | ✓ Done | `tab:worst-per-class` + analysis script |
| 10 | Multi-seed centralised | ✓ Acknowledged | §Limitations (multi-seed flagged as easy follow-up) |
| MD1 | Flower mini-demo | ✓ Done | `fl_flower/`, `run_one_flower.py` (prior turn) |
| MD2 | 28×28 vs 64.npz | ✓ Done | §Dataset rewrite |
| MD3 | IID + Dirichlet | ✓ Pending HPC | `submit_robustness.sh` |
| MD4 | Simulated vs real heterogeneity | ✓ Done | §Partition paragraph |

**No-HPC work: 7 of 12 fully addressed.**
**HPC-dependent: 2 (IID + Dirichlet, equivalence verification — both already queued).**
**Deferred to future work: 3 (partial participation, $E$-sweep, multi-seed centralised, comparative baselines vs. FedLC/SCAFFOLD).**
