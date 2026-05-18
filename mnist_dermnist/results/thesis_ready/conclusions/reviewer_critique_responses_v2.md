# Reviewer-critique responses — second round

Second round of methodology critiques received after the first revision.
This document captures the response to each.

## (1) Tone down "real hospital heterogeneity" claims

**Verdict:** Valid. The disclaimer already existed but could be sharper.

**Action:**
- Reinforced the "Simulated, not real, heterogeneity" paragraph at the
  start of §sec:partition with stronger explicit language including
  the safe-claim sentence verbatim ("controlled synthetic label-skew
  heterogeneity in DermaMNIST, not real multi-site clinical heterogeneity").
- Added an explicit "We do not claim that FedProx solves medical-site
  heterogeneity" disclaimer.
- Added bullet-list of axes NOT modelled: feature/domain shift,
  demographic shift, missing-modality structure.
- Cited FedBN \citep{Li et al., 2021} as the canonical method for feature
  shift, stating it has been reported to outperform FedAvg and FedProx
  under that regime.

---

## (2) Custom partition may look FedProx-friendly --- frame as 3 partitions together

**Verdict:** Valid structural improvement.

**Action:**
- Rewrote the lead-in to §Centralised Baseline and Robustness as a
  bullet-listed three-regime structure:
  - IID (falsification check, expects $\Delta \approx 0$)
  - Dirichlet-$\alpha = 0.1$ (literature-standard non-IID)
  - `balanced_paired_7_clients` (mechanism experiment, this work)
- Explicitly states the custom partition is the *mechanism* experiment,
  with the two literature-standard partitions as the *external validity*
  pair, rather than the custom partition being the whole story with
  others as appendix.

**Pending:** the IID and Dirichlet sweep results (HPC queued in
`submit_robustness.sh`) will populate the table rows when HPC resumes.

---

## (3) FedNova caveat for system heterogeneity

**Verdict:** Valid — FedNova is the direct competitor for the
heterogeneous-local-steps regime.

**Action:** Added a "Scope: FedProx vs. FedNova for heterogeneous local
steps" subsection to `03_overleaf_ready_system_het.tex`. States:
- This section restricts itself to FedAvg vs. FedProx per the project brief
- FedNova \citep{Wang et al., 2020} is the more specialised method for
  this regime, addressing objective inconsistency under heterogeneous
  local steps via normalised averaging
- A direct FedNova comparison is the most natural follow-up and is
  recommended

---

## (4) E = 20 may exaggerate FedProx's advantage --- soften headline

**Verdict:** Valid. Already in Limitations; needed tightening in headline.

**Action:**
- The per-class results paragraph is rewritten to use precise phrasing:
  "FedProx improves macro-F1 on average, with the clearest
  multiple-comparison-corrected per-class benefit on melanoma."
- The Limitations subsection already states the claim should be read
  as "FedProx improves over FedAvg at $E = 20$" rather than as an
  unconditional claim.

**Deferred (HPC):** $E$-sensitivity sweep at $E \in \{1, 5, 20\}$ with
3 paired seeds (~10 GPU-h). Flagged as the highest-priority
follow-up sensitivity check.

---

## (5) Multi-seed centralised baseline

**Verdict:** Valid, cheap.

**Action (no HPC):** Running 2 additional centralised seeds (123, 456)
on CPU laptop, ~5 min each. Combined with the existing seed=42 result,
the centralised baseline is now a 3-seed mean ± SD.

**Action (.tex):** Methodology rewritten as "Centralised upper bound
(3-seed sweep)" with `\TODOhpc{...}` placeholders for the mean ± SD
and recovery percentages. Will be filled once the 2 additional CPU runs
complete (~10 min).

**Action (HPC, optional):** Extending to all 10 seeds would be
straightforward (~5 min × 10 seeds on A100); easy follow-up after HPC
resumes.

---

## (6) "Resolution-invariant" wording is unsafe

**Verdict:** Valid — sharp methodological catch.

**Action:** Rewrote the Dataset paragraph to remove the
"resolution-invariant under fixed model capacity" claim. The new
phrasing:
- Acknowledges that "many dermatology-relevant features are heavily
  compressed at 28$\times$28" (subtle pigment networks, asymmetry
  boundaries)
- States explicitly: "28$\times$28 is not intended to estimate
  clinical-grade dermatology performance"
- States: "we do not claim that the magnitude of the FedProx-vs-FedAvg
  gap observed here would necessarily transfer to higher resolutions"
- Acknowledges that at higher resolution, learnable features change,
  minority-class collapse severity may differ, and FedProx's advantage
  may be larger or smaller
- Flags resolution-generalisation as future work

---

## (7) Test-set usage --- already correct

**Verdict:** Valid, already done correctly. No change needed.

**Action:** None. Test-at-best-val protocol stays as documented; we do
not imitate any best-test-round framing.

---

## (8) Don't overclaim per-class improvement

**Verdict:** Valid. Already largely addressed; sharpened.

**Action:** The per-class results paragraph now explicitly says:
"FedProx improves macro-F1 on average, with the clearest
multiple-comparison-corrected per-class benefit on melanoma. Other
minority-class improvements are suggestive but exploratory, and the
worst-case per-(seed, class) regressions show that FedProx is not
uniformly safer across all classes." The headline claim is macro-F1;
per-class is exploratory + safety check.

---

## Most-important-fix list (from the critique)

| Fix | Status | Notes |
|---|---|---|
| Finish IID and Dirichlet sweeps | ✓ Pending HPC | Queued in `submit_robustness.sh` |
| Complete Flower full-scale equivalence check | ✓ Pending HPC | Queued in `submit_equivalence_check.sh` |
| Run 3-seed centralised baseline | **✓ Done** | Now running on CPU laptop |
| Add E sensitivity | Deferred | Flagged as highest-priority follow-up |
| Add one imbalance baseline (CE-weighted or focal) | Deferred | ~10 GPU-h additional |
| Don't hide worst-case per-class regressions | ✓ Done | `tab:worst-per-class` reports them; headline phrased honestly |

## Summary

All 8 critiques addressed in the .tex source. Three depend on HPC and
are already queued. One (3-seed centralised) is running on laptop CPU.
Two ($E$-sensitivity, FedLC/focal baseline) are deferred as
acknowledged future work with HPC cost estimates.
