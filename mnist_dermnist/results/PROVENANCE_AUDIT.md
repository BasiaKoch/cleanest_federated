# Provenance audit — result JSONs across all experiment directories

Snapshot date: 2026-05-27. Audited fields: `git_commit`, `framework`,
`framework_version`, `runner_script`, `npz_path`, `partition`,
`provenance_note`.

The headline pure-PyTorch JSONs were written by an older
`run_one.py` build that did not yet stamp a git commit hash into the
output JSON. All Flower-runtime JSONs and all later pure-PyTorch
re-runs carry `git_commit`. The centralised JSONs are the weakest:
they carry only experimental fields (no framework, no runner script,
no git commit).

This file is the authoritative provenance ledger; treat it as the
appendix item the chapter should cite if asked "what code produced
which numbers".

## Per-directory provenance ledger

| Directory | Partition | Runtime | Runner script | `git_commit` in JSON | `framework_version` | Notes |
|---|---|---|---|---|---|---|
| `headline/` | `balanced_paired_7_clients` | pure-PyTorch | `run_one.py` | **missing** | `n/a` | Primary engineered headline. `provenance_note: "backstamped-2026-05-18"`. |
| `iid/` | `iid_7_clients` | Flower 1.23.0 | `run_one_flower.py` | `2ab422b4aef82d0a78ddab129a291da5bf2e3d05` | `1.23.0` | IID mechanism-null control. |
| `dirichlet_a01/` | `dirichlet_7_clients` (α = 0.1) | Flower 1.23.0 | `run_one_flower.py` | `8c2ce383f2795183dcc6ef5c05fdb72c7af2497e` | `1.23.0` | Dirichlet non-IID alternative. |
| `flower_C0_baseline/` | `balanced_paired_7_clients` | Flower 1.23.0 | `run_one_flower.py` | `2ab422b4aef82d0a78ddab129a291da5bf2e3d05` | `1.23.0` | Runtime replication of headline. |
| `flower_C0_iid_baseline/` | `iid_7_clients` | Flower 1.23.0 | `run_one_flower.py` | `2ab422b4aef82d0a78ddab129a291da5bf2e3d05` | `1.23.0` | IID under Flower. |
| `specialist_partition/` | `specialist_7_clients` | Flower 1.23.0 | `run_one_flower.py` | `2ab422b4aef82d0a78ddab129a291da5bf2e3d05` | `1.23.0` | Falsification probe. |
| `system_het_fixed/` | `balanced_paired_7_clients` + fixed stragglers | Flower 1.23.0 | `run_one_flower.py` | `2ab422b4aef82d0a78ddab129a291da5bf2e3d05` | `1.23.0` | S1 condition. |
| `system_het_random/` | `balanced_paired_7_clients` + random stragglers | Flower 1.23.0 | `run_one_flower.py` | `8c2ce383f2795183dcc6ef5c05fdb72c7af2497e` | `1.23.0` | S2 condition. |
| `system_het_iid_fixed/` | `iid_7_clients` + fixed stragglers | Flower 1.23.0 | `run_one_flower.py` | (Flower runner — see filenames) | `1.23.0` | IID system-het control. |
| `system_het_iid_random/` | `iid_7_clients` + random stragglers | Flower 1.23.0 | `run_one_flower.py` | (Flower runner — see filenames) | `1.23.0` | IID system-het control. |
| `system_het_random_asymmetric/` | `balanced_paired_7_clients` + asymmetric dropout | Flower 1.23.0 | `run_one_flower.py` | (Flower runner — see filenames) | `1.23.0` | Asymmetric-straggler protocol. |
| `system_het_random_fednova/` | `balanced_paired_7_clients` + random stragglers | Flower 1.23.0 | `run_one_fednova_flower.py` | `0b7f69ae3742dd90ef60e57fa7c2656a7ac94853` | `1.23.0` | FedNova arm. |
| `system_het_partial_C0.5/` | `balanced_paired_7_clients` + partial participation (C=0.5) | Flower 1.23.0 | `run_one_flower.py` | `5c5d1d3e2ca41cd476c435dd5b26cf7a48ddd4c5` | `1.23.0` | Partial-participation arm. |
| `mu_sensitivity_flower/` | `balanced_paired_7_clients` | Flower 1.23.0 | `run_one_flower.py` | `6e30985ee3205b43f8d9128856e1b6fc7071041d` | `1.23.0` | μ ∈ {0.001, 0.01, 0.1, 1.0}, n = 10 seeds. |
| `mu_sweep/` | `balanced_paired_7_clients` | pure-PyTorch | `run_one.py` | (varies — see JSONs) | `n/a` | Older pure-PyTorch μ sweep, n = 3 seeds. Superseded by `mu_sensitivity_flower/`. |
| `arch_ablation_bn/` | `balanced_paired_7_clients` | Flower 1.23.0 | `run_one_flower.py` | `0b7f69ae3742dd90ef60e57fa7c2656a7ac94853` | `1.23.0` | BatchNorm ablation, n = 3 seeds. **Not used in main thesis** — see `README_PROVENANCE.md` in that directory. |
| `centralised/` | pooled (no FL partition) | centralised PyTorch | `run_centralised.py` (inferred) | **missing** | (missing) | Reference upper bound. Only experimental fields stored; no framework / git / runner provenance in JSON. Traceable via filename pattern `centralised_seed*.json`. |

## Hyperparameters (shared across federated experiments unless noted)

| Hyperparameter | Value |
|---|---|
| Communication rounds | 150 |
| Local epochs $E$ | 20 |
| Optimizer | SGD, lr = 0.01, momentum = 0.9, weight decay = 0 |
| Batch size | 32 |
| FedProx $\mu$ | 0.01 (sweep arms vary; see `mu_sensitivity_flower/` and `mu_sweep/`) |
| FedAvg equivalence | $\mu = 0$ (gated branch in `fl/local_train.py`) |
| Loss | cross-entropy |
| Image resolution | 28×28 (resized from 64×64 npz) |
| Seeds (n = 10) | 42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828 |

## Recoverable provenance for the headline pure-PyTorch JSONs

The 20 `test_at_best_*.json` files in `headline/` carry
`provenance_note: "backstamped-2026-05-18"` but **no `git_commit`
field**. They were committed to git on 2026-05-22 in commit
`f9024126fa5554313fbd7041ca134a7ef40a16af` ("Revert results/ rename;
document separation via README_PROVENANCE.md instead").

The runner code (`run_one.py`, `fl/server_loop.py`, `fl/local_train.py`,
`fl/aggregation.py`) was last modified on or before 2026-05-18 in
one of three candidate commits:

| SHA | Timestamp | Subject |
|---|---|---|
| `134c201b61d23b1a6ca00fb4e72de59ebcddcae6` | 2026-05-17 17:09 +0100 | new inspiered changes for hpc |
| `e45f2978fb7a281863120c09b196a83663e33a7e` | 2026-05-18 16:21 +0100 | system hetero |
| `baabbaff9d45cfc9a2ec931da0932c910b7667d0` | 2026-05-18 17:29 +0100 | further improvements while hpc is down |

The runs themselves were executed on a separate HPC machine (see
`npz_path: /Users/basiakoch/cleanest_federated/dermamnist_64.npz` —
this is the laptop-side npz path, indicating a local re-run path, or
that the JSONs were transferred and re-stamped on the laptop). The
exact HPC-side runner SHA is not recoverable from the JSONs alone.

**Honest statement for the thesis appendix:**
> The 20 pure-PyTorch headline JSONs in `mnist_dermnist/results/headline/`
> carry a textual `provenance_note: "backstamped-2026-05-18"` but were
> produced by a runner build that pre-dated the introduction of
> `git_commit` stamping in the output JSON. The applicable runner code
> is the state of `mnist_dermnist/experiments/run_one.py`,
> `mnist_dermnist/fl/server_loop.py`, `mnist_dermnist/fl/local_train.py`,
> and `mnist_dermnist/fl/aggregation.py` on or immediately before
> 2026-05-18 (candidate commits `134c201`, `e45f297`, `baabbaf` — the
> precise SHA cannot be uniquely recovered from the JSON). Independent
> traceability is provided by: (a) the directory's
> `README_PROVENANCE.md`; (b) the per-JSON fields `framework`,
> `runner_script`, `partition`, `npz_path`, `num_rounds`, `local_epochs`,
> `mu`, `seed`, `fraction_fit`, `lr`, `momentum`, `batch_size`,
> `num_classes`, `image_size`, `device`, `loss_type`,
> `best_val_macro_f1`, `selected_round`; (c) the matching
> `history_*.csv` files in the same directory; and (d) the unit test
> `tests/test_framework_provenance.py` that asserts every JSON in this
> directory canonicalises to `framework = "pure-pytorch"`. The
> numerical claims drawn from these JSONs reproduce on the current
> head of `main` if the runner is re-invoked at the documented seeds
> and hyperparameters.

## Remaining provenance gaps

1. **`headline/*.json`** — no `git_commit`, three candidate commits documented above; not uniquely recoverable.
2. **`centralised/centralised_seed*.json`** — no `git_commit`, no `framework`, no `runner_script`, no `npz_path`. The fields are limited to experimental settings (`regime`, `seed`, `num_epochs`, `lr`, `momentum`, `batch_size`, `device`, metrics, `per_class_f1`). Runner script is inferred as `experiments/run_centralised.py` from the regime field; consider regenerating with full provenance fields before submission if time allows.
3. **`mu_sweep/*.json`** — older pure-PyTorch sweep at n = 3 seeds. Provenance varies by file; not used in the main chapter (`mu_sensitivity_flower/` at n = 10 seeds supersedes it).

## How to add `git_commit` to future runs

`run_one_flower.py` already writes the field. The matching block in
`run_one.py` (and any future centralised re-run script) should add at
write time:

```python
import subprocess
git_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
).strip()
result_json["git_commit"] = git_commit
```

Adding this single line at the point of JSON write would close the gap
for any future pure-PyTorch or centralised re-runs.
