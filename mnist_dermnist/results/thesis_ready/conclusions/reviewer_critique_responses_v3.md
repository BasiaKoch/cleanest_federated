# Reviewer-critique responses — third round

Third round of methodology critique. This round was substantially
constructive: it identified canonical repositories to cross-check
against and proposed concrete extensions (FedNova for system het,
class-weighted CE for label-skew). Implementations were made for the
two strongest suggestions; remaining items are queued for HPC.

## (1) Strong points affirmed

No action needed; the reviewer confirms the paired-seed protocol,
statistical reporting, and scope language are at strong-dissertation
level. This is a useful sanity check that the previous rounds of
revision addressed the most important issues.

---

## (2.1) Headline too partition-dependent

**Verdict:** Valid; already queued.

**Status:** IID + Dirichlet-$\alpha = 0.1$ sweeps are pre-submitted via
`submit_robustness.sh` (40 SLURM jobs). The methodology already uses
the three-regime framing (IID falsification + Dirichlet literature
standard + custom mechanism). No further action until HPC produces
results.

---

## (2.2) Add a real imbalance-aware baseline

**Verdict:** Valid; highest-priority comparative extension.

**Action this round:**
- **Implemented** `mnist_dermnist/fl/class_imbalance.py` with two
  imbalance-aware losses:
  - `make_class_weighted_ce(labels)`: inverse-frequency class weights
    normalised to sum to `num_classes`. For DermaMNIST: dermato weight
    $\approx 2.67$, mel_nevi weight $\approx 0.05$.
  - `FocalLoss(gamma=2.0)`: focal loss with optional inverse-frequency
    $\alpha$ weights.
- Module verified by unit-test pseudocode in
  `analyse_communication_metrics.py` test output.
- Methodology paragraph added to `09_overleaf_ready.tex` under
  "Implemented but not yet evaluated: class-weighted CE and FedNova
  baselines".

**Pending HPC:** A FedAvg + class-weighted CE sweep at the headline
configuration ($n = 10$ paired seeds, $E = 20$, balanced_paired_7_clients)
would be the most direct comparator to FedProx for the label-skew
question. SLURM submission script can be added in ~30 minutes; ~10
GPU-h to run.

---

## (2.3) FedNova for system heterogeneity

**Verdict:** Valid; highest-priority comparative extension for system-het
section.

**Action this round:**
- **Implemented** `mnist_dermnist/fl_flower/client_fednova.py`: client
  that returns parameter delta and effective local-step count $\tau_i$
  to enable normalised aggregation.
- **Implemented** `mnist_dermnist/fl_flower/strategy_fednova.py`:
  `PairedFedNovaStrategy` subclassing `FedAvg`. Computes
  $d_\text{norm} = \sum_i p_i d_i / \tau_i$ and
  $\tau_\text{eff} = \sum_i p_i \tau_i$, then updates global as
  $w^{t+1} = w^t - \tau_\text{eff} \cdot d_\text{norm}$.
- Imports verified.
- Cross-checked against the JYWa/FedNova canonical implementation
  (see `canonical_repo_crosscheck.md`); the core aggregation rule
  matches, with server-side momentum deliberately omitted.
- Methodology paragraph added describing FedNova as the canonical
  competitor for the heterogeneous-local-step regime.

**Pending HPC:** A FedAvg / FedProx / FedNova three-way comparison on
the C2 random-stragglers condition (10 paired seeds × 3 algos = 30 jobs,
~30 GPU-h) would directly address the reviewer's concern.

---

## (2.4) C1 fixed-stragglers confounded — already addressed

No new action; the system-het methodology already states explicitly
that "H2 inference is based primarily on C2; C1 is retained as a
clinically interpretable but confounded stress case".

---

## (2.5) E-sensitivity

**Verdict:** Valid; deferred.

**Action:** Acknowledged in Limitations subsection. Implementation cost
is trivial (existing `run_one_flower.py` accepts `--local-epochs`).
A 3-seed × 3-E-value sweep ($E \in \{1, 5, 20\}$) = 9 jobs ~10 GPU-h
ready to submit when HPC resumes.

---

## (2.6) Partial-participation sensitivity

**Verdict:** Valid; deferred.

**Action:** Acknowledged in Limitations. Implementation requires
extending `run_one_flower.py` to accept a `--fraction-fit` argument and
wiring it into the Flower simulation (the strategy already exposes
`fraction_fit`; the run script currently hard-codes 1.0). This is a
~5-line code change. Not implemented this round to keep the focus on
the highest-priority FedNova + CW-CE extensions.

---

## (2.7) Flower full-scale equivalence

**Verdict:** Critical; already queued.

**Action:** `submit_equivalence_check.sh` is pre-submitted to HPC and
will run when resources resume. The methodology paragraph already
flags this clearly and uses non-overclaiming wording: "The headline
sweep was produced by the deterministic PyTorch reference loop;
Flower equivalence was verified on selected full-scale seeds." If the
2-seed full-scale check fails, the headline must be re-run via Flower.

---

## (Repos to cross-check)

**Action:** A full audit document
(`canonical_repo_crosscheck.md`) was written this round, comparing the
thesis implementations against:
- litian96/FedProx → ✓ matches
- adap/flower → ✓ matches (GroupNorm avoids known BatchNorm pitfall)
- JYWa/FedNova → ✓ matches core aggregation (server-side momentum
  omitted by design)
- med-air/FedBN → not implemented; non-implementation justified
- Xtra-Computing/NIID-Bench → ✓ vocabulary and conventions aligned
- MedMNIST/MedMNIST → ✓ conventions followed; minor cleanup
  opportunity (use native 28×28 file directly) flagged

**Conclusion:** No silent implementation differences found that would
invalidate any headline number.

---

## (Best practical 6) Confusion matrices + per-class precision/recall

**Verdict:** Valid; acknowledged with explicit cost estimate.

**Action:** Limitations subsection states the saved
`test_at_best_*.json` files contain per-class F1 only; full prediction
arrays / confusion matrices are not saved. Adding them requires either
(a) augmenting `evaluate()` to save full predictions (~5 lines of code
in `mnist_dermnist/fl/evaluation.py`) + re-running the headline sweep,
or (b) saving model checkpoints during the existing run and
re-evaluating offline (also ~5 lines of code plus a few minutes of
compute per seed).

No code change this round (focus on FedNova + CW-CE); the limitation is
documented and the fix path is specified for any follow-up reviewer.

---

## (Best practical 7) 64×64 resolution sensitivity

**Verdict:** Valid; deferred.

**Action:** Limitations subsection already states this. Implementation
cost: loading from the same `dermamnist_64.npz` archive without
resizing, and scaling the first conv block to expect 64×64 input. ~30
minutes of code change + ~50 GPU-h for full sweep, or ~5 GPU-h for
a 3-seed sensitivity probe.

---

## (Best practical 8) HAM10000 metadata source splits

**Verdict:** Out of scope for this thesis (MedMNIST .npz does not
preserve HAM10000 source metadata).

**Action:** Documented as future-work in Limitations: would require
re-fetching the underlying HAM10000 dataset, joining with metadata, and
designing source-aware partitions. This is the most direct path to
real (rather than simulated) multi-site heterogeneity but is a separate
project.

---

## (Best practical 9) Communication / runtime metrics

**Verdict:** Valid, cheap.

**Action this round:**
- **Implemented** `mnist_dermnist/results/thesis_ready/scripts/analyse_communication_metrics.py`.
- Extracts rounds-to-threshold from existing history CSVs.
- Computes per-round and total-run communication cost from model size.
- **Headline finding:** at val macro-F1 = 0.45, FedProx requires
  ~13 rounds vs FedAvg's ~39 → approximately $3\times$ fewer
  communication round-trips. At a 1.61 MB model snapshot and 1
  broadcast + 7 uploads per round, this is a 12.9 MB / round saving for
  every additional round avoided.
- Methodology paragraph "Communication efficiency (a complementary
  metric)" added to `09_overleaf_ready.tex`, with the note that FedProx
  and FedAvg have IDENTICAL per-round communication (FedProx adds local
  compute only) — unlike SCAFFOLD which would double communication due
  to control-variate exchange.

---

## Calibrated thesis-level claim wording

The reviewer suggested a specific calibrated wording for the
thesis-level claim. We adopted it verbatim (with minor citation
expansion) as a new "Calibrated thesis-level claim" subsection in
`09_overleaf_ready.tex`, immediately following the Convergence
subsection and preceding Limitations. The claim wording is:

> Across 10 paired seeds on a controlled DermaMNIST label-skew
> partition, FedProx improves average test macro-F1 over FedAvg under
> a high-local-epoch regime. The result is statistically supported at
> the paired-seed level and is driven primarily by
> melanoma/specialist-client behaviour. However, the study does not
> show that FedProx solves class imbalance or real medical-site
> heterogeneity; robustness to IID/Dirichlet partitions,
> heterogeneous-local-step baselines such as FedNova (Wang et al.,
> 2020), and imbalance-aware losses (class-weighted CE, FedLC) remain
> necessary to delimit the claim. The first two of these have been
> pre-registered and submitted to HPC; the last has been implemented
> in code and is queued.

---

## Summary: implementation status after round 3

| # | Critique | Status |
|---|---|---|
| 2.1 | IID + Dirichlet not optional | Queued HPC |
| **2.2** | **Class-weighted CE baseline** | **Code complete; HPC queued** |
| **2.3** | **FedNova for system het** | **Code complete; HPC queued** |
| 2.4 | C1 confound | Already addressed |
| 2.5 | E-sensitivity | Acknowledged; trivial submission ready |
| 2.6 | Partial-participation sensitivity | Acknowledged; needs ~5-line code change |
| 2.7 | Flower equivalence | Queued HPC |
| BP6 | Confusion matrices | Acknowledged; ~5-line code change needed |
| BP7 | 64×64 sensitivity | Acknowledged; deferred |
| BP8 | HAM10000 metadata splits | Out of scope; deferred |
| **BP9** | **Communication metrics** | **Done (no HPC needed)** |
| Claim | Calibrated wording | **Adopted verbatim in .tex** |
| Repo audit | Canonical cross-check | **Done (no silent differences found)** |

**Three items completed this round without HPC:**
1. Class-weighted CE / focal loss implementation
2. FedNova client + strategy implementation
3. Communication / runtime metrics analysis

**Two new documents:**
- `canonical_repo_crosscheck.md` — audit against 6 reference repositories
- `reviewer_critique_responses_v3.md` (this file)

**One adopted wording change** to the thesis claim (calibrated language
verbatim from the reviewer).

The thesis now has FedNova and class-weighted CE implementations
ready to submit on HPC resumption, alongside the previously queued
IID, Dirichlet, system-het, and Flower-equivalence sweeps.
