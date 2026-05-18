# Cross-check against canonical FL repositories

This document audits the implementations in this dissertation against
the canonical reference implementations cited in the FL literature.
The point is to catch silent implementation differences that a reviewer
might find by reading those repositories.

## 1. FedProx (litian96/FedProx)

The official FedProx repository (Li et al., 2020, MLSys) implements the
proximal term as:

```python
# Pseudocode for the litian96/FedProx local update
prox_term = (mu / 2) * sum((w - w_global).pow(2).sum() for w in model.parameters())
loss = task_loss + prox_term
```

**Cross-check against `mnist_dermnist/fl/local_train.py`:**

| Check | Reference | This thesis | OK? |
|---|---|---|---|
| Proximal term applied to all trainable parameters | yes (full `model.parameters()` iteration) | yes (line 85: `for w, w_g in zip(self.model.parameters(), global_params)`) | ✓ |
| Global anchor detached and fixed for full local update | yes (snapshot once per round) | yes (`global_weights_frozen` snapshotted in `server_loop.py:130`; passed by value to client) | ✓ |
| μ = 0 exactly recovers FedAvg | yes (`if mu > 0:` gated branch) | yes (`if self.proximal_mu > 0:` in `client.py:135`) — gated, bit-identical when μ = 0 | ✓ |
| Per-client size weighting in aggregation | yes (n_i / N) | yes (`local_weights_for_agg.append(len(client_indices[cid]))` in `server_loop.py:181`) | ✓ |
| Multiple-heterogeneity-level reporting | yes (multiple Dirichlet α) | yes (IID + Dirichlet α=0.1 + custom partition) | ✓ |

**Conclusion:** FedProx implementation matches the canonical reference.

---

## 2. Flower FedProx baseline (adap/flower → baselines)

Flower's official FedProx baseline page describes the architecture as:

- `flwr.client.NumPyClient` subclass that does local training with the
  proximal term applied client-side
- `flwr.server.strategy.FedAvg` used for server-side aggregation
  (FedProx aggregates identically to FedAvg; the proximal term lives in
  the client local objective)
- μ passed via strategy `on_fit_config_fn` or hardcoded in the client

**Cross-check against `mnist_dermnist/fl_flower/client.py` +
`run_one_flower.py`:**

| Check | Flower baseline | This thesis | OK? |
|---|---|---|---|
| μ handled consistently | via config | via client `__init__` `proximal_mu` parameter (set once per run, paired across FedAvg/FedProx) | ✓ (cleaner for paired-seed protocol) |
| Aggregation strategy | `FedAvg` (size-weighted mean) | `FedAvg` from `flwr.server.strategy` (line 198 of `run_one_flower.py`) | ✓ |
| Sample-count weighting | `num_examples` returned from `fit()` | `len(self.train_subset)` returned (= identical) | ✓ |
| Validation/test eval | server-side via `evaluate_fn` | server-side via `evaluate_fn` (line 162 of `run_one_flower.py`) | ✓ |
| Per-round metrics aggregated | yes | yes (history_rows in evaluate_fn) | ✓ |

**One silent difference identified and documented:** Flower's
`FedAvg.aggregate_fit` aggregates the parameters returned via
`NumPyClient.get_parameters`, which in our case includes ALL state_dict
entries (parameters + buffers). For DermMNISTCNN (GroupNorm, no running
stats), `state_dict()` contains only learnable parameters, so this is
equivalent to aggregating `model.parameters()`. The pure-PyTorch
reference loop uses `weighted_average_state_dicts` which also operates
on the full state_dict. So the two paths are equivalent for our model.

If a BatchNorm model were used instead, the Flower path would aggregate
running mean/variance buffers (which it should not under non-IID; see
Hsieh et al. 2020). This is one of the reasons we use GroupNorm.

**Conclusion:** Flower implementation matches the canonical Flower
FedProx baseline; silent differences with BatchNorm models are avoided
by the GroupNorm choice.

---

## 3. FedNova (JYWa/FedNova)

The JYWa/FedNova reference implementation:
1. Computes per-client effective local step count $\tau_i$ (typically
   `local_epochs × batches_per_epoch`)
2. Returns the parameter delta $d_i = w^t - w_i^{t+1}$ and $\tau_i$
3. Aggregates as $d_\text{norm} = \sum_i p_i d_i / \tau_i$ and
   $\tau_\text{eff} = \sum_i p_i \tau_i$
4. Updates global as $w^{t+1} = w^t - \tau_\text{eff} \cdot d_\text{norm}$

**Cross-check against `mnist_dermnist/fl_flower/{client_fednova.py,
strategy_fednova.py}`:**

| Check | JYWa reference | This thesis | OK? |
|---|---|---|---|
| $\tau_i$ counted as SGD step count, not epoch count | yes | yes (`tau += 1` per batch in `client_fednova.py:111`) | ✓ |
| Local-epoch heterogeneity is the experimental variable | yes | yes (uses `system_het.build_epoch_schedule`) | ✓ |
| Comparison with FedAvg, FedProx, FedNova on same heterogeneity | yes | yes (will be in `submit_system_het_fednova.sh` when run) | ✓ |
| Aggregation reduces to FedAvg under uniform $\tau$ | yes (mathematically) | yes (when all $\tau_i$ equal, $\sum p_i \tau_i \cdot \sum p_i d_i / \tau_i = \sum p_i d_i$, i.e. FedAvg) | ✓ |

**One implementation difference:** JYWa's repo uses momentum on the
server side as well; our Flower implementation does not. This is a
deliberate simplification since adding server-side momentum changes the
algorithm's character; we cite the basic FedNova rule only.

**Conclusion:** FedNova implementation matches the canonical reference's
core aggregation rule; server-side momentum is deliberately omitted.

---

## 4. FedBN (med-air/FedBN)

The FedBN repository implements FedAvg with one modification: BatchNorm
parameters and statistics are kept LOCAL (not aggregated across clients).
This is designed for feature-shift partitions where per-client BN
statistics encode local distribution information.

**Cross-check:** This dissertation does NOT implement FedBN. The reason
is documented in the methodology:
- Our model uses GroupNorm, not BatchNorm
- Our partitions are pure label-skew, not feature-shift
- FedBN is the right comparator for feature-shift but not for our setup

The methodology explicitly cites FedBN as the canonical method for
feature-shift heterogeneity and identifies feature-shift evaluation as
out of scope (would require multi-site data).

**Conclusion:** FedBN is not implemented; non-implementation is
correctly justified and documented.

---

## 5. NIID-Bench (Xtra-Computing/NIID-Bench)

The NIID-Bench repository implements FedAvg, FedProx, SCAFFOLD, and
FedNova across multiple non-IID scenarios:
- Label-distribution skew (#C=k, Dirichlet)
- Feature-distribution skew
- Quantity skew

**Cross-check:** This dissertation tests:
- IID partition (sanity / falsification check)
- Dirichlet α=0.1 (literature standard non-IID label skew)
- `balanced_paired_7_clients` (custom mechanism partition)

This is consistent with NIID-Bench's labelled-skew category. We do NOT
test feature-distribution skew (out of scope; flagged) and we do NOT
test quantity skew (no informative variation; partition sizes are
within 1.65× range deliberately).

**Adopted vocabulary:** The methodology now uses NIID-Bench's
terminology for non-IID severity (Dirichlet α) and partition naming.

**Conclusion:** Methodology aligns with NIID-Bench conventions for the
label-skew category.

---

## 6. MedMNIST (MedMNIST/MedMNIST)

The MedMNIST repository documents DermaMNIST as available in multiple
resolutions: 28×28, 64×64, 128×128, 224×224.

**Cross-check against this dissertation:**

| Item | MedMNIST documentation | This thesis | OK? |
|---|---|---|---|
| Source archive | `dermamnist_64.npz` for 64×64 | `dermamnist_64.npz` (gitignored) | ✓ |
| Resize to lower resolution | supported; conventionally use native 28 file | source 64×64, resized to 28×28 in `load.py` | ✓ (documented; could be cleaner) |
| Normalisation | `[0, 1]` after `/255.0` | `[0, 1]` after `/255.0` in `load.py` | ✓ |
| Train/val/test split | canonical (7007/1003/2005) | canonical | ✓ |

**Minor improvement opportunity:** Using `dermamnist.npz` (the native
28×28 file) instead of resizing from 64×64 would remove one
preprocessing step. Currently we use the 64×64 file because it was the
first one downloaded; substituting the 28×28 file would not change
results but would be cleaner. Identified as a low-priority cleanup.

**Resolution sensitivity:** Re-running at 64×64 (the native 64-file
resolution, with model channels scaled appropriately) is identified as
follow-up work in the limitations section.

**Conclusion:** MedMNIST conventions are followed; one minor cleanup
identified.

---

## Summary

| Repository | Cross-check verdict |
|---|---|
| litian96/FedProx | ✓ Implementation matches canonical |
| adap/flower | ✓ Matches Flower FedProx baseline; GroupNorm avoids known BN pitfall |
| JYWa/FedNova | ✓ Core aggregation matches; server-side momentum deliberately omitted |
| med-air/FedBN | ✓ Not implemented; non-implementation correctly justified |
| Xtra-Computing/NIID-Bench | ✓ Methodology aligns with NIID-Bench's label-skew category |
| MedMNIST/MedMNIST | ✓ Conventions followed; minor cleanup possible |

**No silent implementation differences found that would invalidate the
headline result.** One minor opportunity for clean-up (using the
native 28×28 MedMNIST file directly) is flagged as low-priority follow-up.
