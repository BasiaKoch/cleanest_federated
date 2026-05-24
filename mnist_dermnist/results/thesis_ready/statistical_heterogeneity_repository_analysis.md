# Statistical Heterogeneity Experiments: Repository Documentation and Thesis Analysis

## 1. Executive Summary

This document audits the repository evidence for statistical heterogeneity experiments only: IID versus non-IID client data distributions, their FedAvg/FedProx comparisons, and how they should be reported in the dissertation. It deliberately excludes system heterogeneity except where filenames or output fields expose it.

The current canonical dataset is DermaMNIST, a 7-class medical image classification benchmark loaded from `dermamnist_64.npz` and evaluated at 28x28 RGB resolution in the current `mnist_dermnist` pipeline. The split present on disk is 7,007 train / 1,003 validation / 2,005 test samples, with severe global class imbalance: 4,693 / 7,007 training samples are melanocytic nevi. The loader and deterministic preprocessing are in `mnist_dermnist/data/load.py:1-67`; class names are defined in `mnist_dermnist/data/partition.py:45-55`.

The implemented statistical partition families are richer than the completed thesis evidence. The current usable Flower statistical-heterogeneity results cover:

- `balanced_paired_7_clients`: engineered non-IID label-skew partition.
- `iid_7_clients`: IID falsification/control partition.
- `dirichlet_alpha01_7_clients`: severe Dirichlet label skew with alpha = 0.1.
- `specialist_7_clients`: non-IID singleton-specialist counterfactual with the same per-client sample counts as the engineered partition.

The pure-PyTorch `headline/` result supports a positive FedProx-FedAvg effect on the engineered partition: mean paired delta = +0.0267 macro-F1, Wilcoxon two-sided p = 0.0195, FedProx wins 9/10 seeds. However, the Flower replication on the same partition is much weaker: mean paired delta = +0.0069, p = 0.4316, wins 7/10. The safest thesis claim is therefore not “FedProx generally improves statistical non-IID medical FL”. The defensible claim is:

> In an engineered DermaMNIST label-skew partition designed to create client drift, FedProx showed a positive macro-F1 effect in the pure-PyTorch reference implementation, but the effect attenuated under the Flower simulation runtime and did not reach significance in the Flower replication. Across IID, Dirichlet alpha = 0.1, and specialist non-IID controls, FedProx effects were small and non-significant. The evidence supports a context-dependent benefit, not a universal advantage.

The largest methodological weakness is that the strongest positive result comes from a partition whose source comments explicitly say it was designed to maximise FedProx's expected advantage (`mnist_dermnist/data/partition.py:66-73`, `398-408`). This is not fatal if it is reported honestly as an engineered stress test, backed by controls. It is fatal if it is presented as representative hospital heterogeneity.

## 2. Repository Map

### Current Canonical Pipeline

| Path | Role for statistical heterogeneity |
|---|---|
| `mnist_dermnist/data/load.py:1-67` | Loads DermaMNIST NPZ, converts NHWC uint8 images to CHW float tensors, optionally resizes to 28x28, applies ImageNet normalization. |
| `mnist_dermnist/data/partition.py:45-656` | Defines all current partition functions: engineered paired, specialist, IID, Dirichlet alpha, quantity skew, medical skew, and pathological split. |
| `mnist_dermnist/experiments/run_one_flower.py:55-73` | Flower runner partition registry, including alpha = 0.1 and alpha = 0.5 Dirichlet wrappers. |
| `mnist_dermnist/experiments/run_one_flower.py:76-140` | Main Flower CLI: algorithm, mu, seed, rounds, local epochs, optimizer, batch size, partition, model variant. |
| `mnist_dermnist/experiments/run_one_flower.py:177-184` | Loads train/val/test and creates client indices from the selected partition using the run seed. |
| `mnist_dermnist/experiments/run_one_flower.py:227-267` | Central validation evaluation every round; validation macro-F1 is used for best-checkpoint selection. |
| `mnist_dermnist/experiments/run_one_flower.py:304-337` | Server strategy. FedAvg aggregation is used; FedProx differs only through client-side local objective. |
| `mnist_dermnist/experiments/run_one_flower.py:432-513` | Test once at best validation checkpoint and write JSON, history CSV, optional predictions, provenance. |
| `mnist_dermnist/fl_flower/client.py:112-197` | Flower client local training, including the FedProx proximal term and update-norm metric. |
| `mnist_dermnist/fl/local_train.py:24-112` | Pure-PyTorch local training; proximal term is gated on `proximal_mu > 0`. |
| `mnist_dermnist/fl/server_loop.py:1-18` | Pure-PyTorch design invariants: paired init, partition, client sampling, batch order, best-validation test. |
| `mnist_dermnist/fl/server_loop.py:274-323` | Pure-PyTorch weighted aggregation, validation, best-val checkpoint, final test. |
| `mnist_dermnist/fl/aggregation.py:13-46` | Size-weighted state-dict averaging, used by FedAvg and FedProx in the pure-PyTorch loop. |
| `mnist_dermnist/fl/evaluation.py:17-76` | Metrics: cross-entropy loss, accuracy, balanced accuracy, macro-F1, per-class F1, optional predictions. |
| `mnist_dermnist/models/dermmnist_cnn.py:1-67` | Headline GroupNorm CNN architecture. |
| `mnist_dermnist/models/dermmnist_cnn_bn.py:1-86` | BatchNorm variant used only for exploratory architecture ablation. |
| `mnist_dermnist/analysis/tables.py:1-260` | Generic paired FedAvg/FedProx result table generator; writes per-seed table, paired stats, per-class differences. |
| `mnist_dermnist/results/thesis_ready/scripts/analyse_statistical_heterogeneity.py:1-54` | Existing thesis-grade statistical heterogeneity analysis script, but its framing should be updated to avoid overclaiming pure-PyTorch as sole truth. |

### Submission and Run Scripts

| Path | Implemented/submitted intent |
|---|---|
| `mnist_dermnist/scripts/submit_flower_C0_baseline.sh:1-25` | Flower engineered-partition baseline: FedAvg, FedProx, FedNova, 10 seeds, E=20, R=150, C=1.0. |
| `mnist_dermnist/scripts/submit_robustness.sh:1-50` | Flower robustness sweep: Dirichlet alpha = 0.1 and IID, 10 seeds x 2 algorithms. |
| `mnist_dermnist/scripts/submit_specialist_partition.sh:1-61` | Specialist partition sweep, same 10 seeds and hyperparameters, with only partition changed. |
| `mnist_dermnist/scripts/submit_mu_sweep.sh:1-38` | Intended mu sweep: mu in {0.001, 0.01, 0.1, 0.5, 1.0} over 3 seeds plus FedAvg baselines. Actual result directory is incomplete. |
| `mnist_dermnist/scripts/runpod_arch_ablation_bn.sh:1-86` | BatchNorm architecture ablation: 3 paired seeds x FedAvg/FedProx on engineered partition. Exploratory only. |
| `mnist_dermnist/scripts/slurm_template_flower.sh` | HPC execution template for Flower FedAvg/FedProx runs. |
| `mnist_dermnist/scripts/slurm_centralised.sh:1-36` | Centralised baseline execution template; not a federated partition experiment but useful as an upper-bound reference. |

### Current Result Directories

| Result family | Expected | Found | Paired? | Complete artifact set? | Safe to use? | Caveats |
|---|---:|---:|---|---|---|---|
| `mnist_dermnist/results/headline/` | 20 JSONs | 20 | 10 FedAvg/FedProx pairs | Histories yes; predictions no; update norms no | Use as reference/legacy primary only if labelled pure-PyTorch | No runtime provenance fields; no predictions; effect does not replicate strongly in Flower. |
| `mnist_dermnist/results/flower_C0_baseline/` | 30 JSONs | 30 | 10 FedAvg/FedProx pairs | Histories/predictions yes; update norms 24/30 | Use as canonical Flower engineered-partition result | Mixed git commits, hostnames, Python/Torch versions; FedProx effect non-significant. |
| `mnist_dermnist/results/flower_C0_iid_baseline/` | 30 JSONs | 30 | 10 FedAvg/FedProx pairs | Histories/predictions/update norms yes | Use as richer IID baseline | Duplicates `iid/` conceptually but includes FedNova and update norms; do not double-count as independent evidence. |
| `mnist_dermnist/results/iid/` | 20 JSONs | 20 | 10 FedAvg/FedProx pairs | Histories/predictions yes; no update norms | Use as IID falsification/control | Duplicate IID evidence relative to `flower_C0_iid_baseline`; no full per-job logs in directory. |
| `mnist_dermnist/results/dirichlet_a01/` | 20 JSONs | 20 | 10 FedAvg/FedProx pairs | Histories/predictions yes; no update norms | Use as secondary robustness result | Only alpha = 0.1 completed in current pipeline; no alpha dose-response. |
| `mnist_dermnist/results/specialist_partition/` | 20 JSONs | 20 | 10 FedAvg/FedProx pairs | Histories/predictions yes; no update norms | Use as secondary counterfactual | Still engineered, not natural hospital heterogeneity. |
| `mnist_dermnist/results/arch_ablation_bn/` | 6 JSONs | 6 | 3 FedAvg/FedProx pairs | Histories/predictions/update norms yes | Descriptive/exploratory only | n=3; different model normalization; not valid for statistical claims. |
| `mnist_dermnist/results/mu_sweep/` | 18 JSONs intended | 9 | No FedAvg rows in this directory | Histories/predictions yes | Do not use as a complete sweep | Only FedProx mu = 0.001, 0.1, 1.0 over 3 seeds. Missing mu = 0.01, 0.5, and FedAvg baselines in this directory. |
| `mnist_dermnist/results/centralised/` | 10 JSONs | 10 | Not paired FL | Histories yes | Use as non-federated reference | Not a statistical heterogeneity experiment; no clients or partitions. |
| `fl-dermamnist-starter/results/` | Legacy outputs | Multiple | Some 3-seed pairs | Has per-client metrics/plots | Use only as historical/appendix evidence | Different codebase, 64x64, often 10 clients, 100 rounds, E=5 or C=0.5. Do not mix with current Flower results. |

No canonical statistical result directory contains NaN metrics or suspicious selected-round collapse. All current `test_at_best_*.json` files inspected had 150-row history files where applicable. However, complete per-job SLURM/RunPod logs are not present beside the current `mnist_dermnist/results/*` directories, so "completed" is inferred from valid JSON/history/prediction artifacts, not from a full log archive.

## 3. Datasets

### DermaMNIST

DermaMNIST is the only dataset used in the current statistical heterogeneity experiments. The loader reads `dermamnist_64.npz`, returns train/validation/test datasets, and applies deterministic preprocessing (`mnist_dermnist/data/load.py:56-67`). Images are RGB arrays converted to float tensors and optionally resized to 28x28 (`mnist_dermnist/data/load.py:45-52`). ImageNet mean/std normalization is hardcoded (`mnist_dermnist/data/load.py:17-18`). I found no augmentation in the current canonical runner; local training uses deterministic `DataLoader(..., shuffle=True)` with seeded generators (`mnist_dermnist/fl_flower/client.py:128-143`).

Observed split and class counts from the NPZ:

| Split | Samples |
|---|---:|
| Train | 7,007 |
| Validation | 1,003 |
| Test | 2,005 |

Training class counts:

| Class id | Class name | Train samples |
|---:|---|---:|
| 0 | actinic_keratoses | 228 |
| 1 | basal_cell_carcinoma | 359 |
| 2 | benign_keratosis_like_lesions | 769 |
| 3 | dermatofibroma | 80 |
| 4 | melanoma | 779 |
| 5 | melanocytic_nevi | 4,693 |
| 6 | vascular_lesions | 99 |

This dataset is appropriate for a medical FL non-IID study because it combines image classification, clinically meaningful minority classes, and strong class imbalance. The limitation is equally important: this repository simulates clients from a single centralized benchmark split. It does not contain true hospital/site labels, scanner metadata, demographic metadata, or natural institution-specific distributions. Claims about "hospital-like" heterogeneity should therefore be phrased as simulated label-skew heterogeneity, not real multi-institution heterogeneity.

## 4. Statistical Heterogeneity Design

### 4.1 IID Baseline

The current IID partition is `iid_7_clients`, implemented in `mnist_dermnist/data/partition.py:592-611`. It shuffles all training indices with `np.random.default_rng(seed)` and splits them into 7 equal shards with `np.array_split`. Each client has 1,001 samples. Because labels are not stratified explicitly, each client matches the global prior only in expectation; in practice all 10 audited seeds gave all 7 classes to every client and very small Jensen-Shannon divergence from the global class prior.

The IID baseline exists in two result directories:

- `mnist_dermnist/results/iid/`: FedAvg/FedProx only, 10 paired seeds.
- `mnist_dermnist/results/flower_C0_iid_baseline/`: FedAvg/FedProx/FedNova, 10 paired seeds, with update norms.

These should not be treated as independent experiments for a statistical claim. Use one as the main IID result and the other as a richer diagnostic replication.

### 4.2 Non-IID Partitioning

#### Engineered Paired Partition: `balanced_paired_7_clients`

The main engineered non-IID partition is specified in `mnist_dermnist/data/partition.py:66-85` and implemented in `mnist_dermnist/data/partition.py:398-445`. Its source comments explicitly state that it is "designed to maximise FedProx's expected advantage" (`mnist_dermnist/data/partition.py:70-73`). Every minority class is held by exactly two clients, while the majority class, melanocytic nevi, is present in all clients. This is an engineered label-skew and class-overlap design, not a random or naturally observed hospital split.

Per-client composition:

| Client | Samples | Classes | Composition |
|---:|---:|---:|---|
| C0 | 964 | 3 | actinic 114, basal 180, nevi 670 |
| C1 | 963 | 3 | actinic 114, basal 179, nevi 670 |
| C2 | 1,095 | 3 | benign keratosis 385, dermatofibroma 40, nevi 670 |
| C3 | 1,094 | 3 | benign keratosis 384, dermatofibroma 40, nevi 670 |
| C4 | 1,110 | 3 | melanoma 390, vascular 50, nevi 670 |
| C5 | 1,108 | 3 | melanoma 389, vascular 49, nevi 670 |
| C6 | 673 | 1 | nevi 673 |

Research role: stress-test whether FedProx helps when multiple clients share minority classes but have highly non-IID local objectives. This partition can support a mechanistic hypothesis, but only if the engineered nature is foregrounded.

#### Specialist Counterfactual: `specialist_7_clients`

The specialist partition is specified in `mnist_dermnist/data/partition.py:88-112` and implemented in `mnist_dermnist/data/partition.py:448-518`. It exactly matches the engineered paired partition's per-client sample counts while assigning each minority class to only one client. This isolates the structural lever "paired minority class ownership" from quantity skew.

Per-client composition:

| Client | Samples | Classes | Composition |
|---:|---:|---:|---|
| C0 | 964 | 2 | actinic 228, nevi 736 |
| C1 | 963 | 2 | basal 359, nevi 604 |
| C2 | 1,095 | 2 | benign keratosis 769, nevi 326 |
| C3 | 1,094 | 2 | dermatofibroma 80, nevi 1,014 |
| C4 | 1,110 | 2 | melanoma 779, nevi 331 |
| C5 | 1,108 | 2 | vascular 99, nevi 1,009 |
| C6 | 673 | 1 | nevi 673 |

Research role: a counterfactual to ask whether the engineered paired result depends on its FedProx-favourable pairing structure. It is useful, but it is still synthetic and should not be called natural hospital heterogeneity.

#### Dirichlet Label Skew: `dirichlet_7_clients`

The Dirichlet partition is implemented in `mnist_dermnist/data/partition.py:614-656`. For each class, it samples a 7-client probability vector from a symmetric Dirichlet distribution and distributes class-specific examples by those proportions. Lower alpha produces more severe label skew; alpha -> infinity approaches IID. The implementation retries up to 50 times if any client is empty (`mnist_dermnist/data/partition.py:633-652`).

Current runner wrappers expose:

- alpha = 0.1: `_dir_a01` in `mnist_dermnist/experiments/run_one_flower.py:55-56`.
- alpha = 0.5: `_dir_a05` in `mnist_dermnist/experiments/run_one_flower.py:59-60`.

Completed current results exist only for alpha = 0.1 (`mnist_dermnist/results/dirichlet_a01/`). I found no completed current canonical alpha = 0.5 result directory. Relative to common FL visual-classification practice, alpha = 0.1 is a severe non-IID setting; alpha = 0.5 is moderate-to-severe and closer to what many papers use as a less extreme robustness condition. Missing alpha = 0.3, 0.5, and 1.0 means the current dissertation cannot show a true heterogeneity dose-response over alpha.

#### Other Implemented but Not Canonically Completed Partitions

The repository also implements:

- `simple_pathological_3_clients`: fixed class sets for 3 clients (`mnist_dermnist/data/partition.py:58-63`, `225+`).
- `medical_skew_7_clients`: dominant/secondary class hospital-style synthetic skew (`mnist_dermnist/data/partition.py:162-170`, `272+`).
- `balanced_specialist_7_clients`: older FedProx-favourable 7-client specialist split (`mnist_dermnist/data/partition.py:143-159`, `328+`).
- `quantity_skew_improved`: explicit quantity-skewed referral-network style split (`mnist_dermnist/data/partition.py:115-140`, `521-589`).

Partition CSVs exist in `mnist_dermnist/results/partitions/` for several of these. They should be described as implemented or visualised partition designs unless matching FedAvg/FedProx result directories are present.

### 4.3 Dirichlet Alpha Values

| Alpha | Implemented in runner? | Completed current result? | Interpretation |
|---:|---|---|---|
| IID | Yes, via `iid_7_clients` | Yes | Null/control condition; clients approximately share the global prior. |
| 0.1 | Yes | Yes, `dirichlet_a01/` | Severe label skew; standard stress-test value in FL literature. |
| 0.5 | Yes | No current canonical result found | Moderate-to-severe label skew; useful missing robustness point. |
| 0.3 | Not in current `mnist_dermnist` runner wrappers | Only in legacy `fl-dermamnist-starter` outputs | Legacy evidence uses different code/hyperparameters; do not mix with current claims. |
| 1.0 | Not in current wrappers | No | Useful missing mild non-IID point. |

### 4.4 Empirical Partition Statistics

These statistics were computed from the current partition functions and `dermamnist_64.npz`. They are not persisted in the current result JSON files. The thesis should either include a generated CSV/figure of these statistics or state that they were computed during analysis.

| Partition | K | n per client mean ± SD | n min-max | Classes/client mean ± SD | Classes min-max | Mean normalized entropy | Mean JS divergence to global prior |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID seed 42 | 7 | 1001.0 ± 0.0 | 1001-1001 | 7.00 ± 0.00 | 7-7 | 0.579 | 0.001 |
| Balanced paired seed 42 | 7 | 1001.0 ± 158.8 | 673-1110 | 2.71 ± 0.76 | 1-3 | 0.355 | 0.166 |
| Specialist seed 42 | 7 | 1001.0 ± 158.8 | 673-1110 | 1.86 ± 0.38 | 1-2 | 0.219 | 0.243 |
| Dirichlet alpha = 0.1 seed 42 | 7 | 1001.0 ± 1283.4 | 91-3721 | 4.71 ± 1.25 | 3-7 | 0.414 | 0.382 |
| Dirichlet alpha = 0.5 seed 42 | 7 | 1001.0 ± 822.2 | 244-2318 | 6.29 ± 0.95 | 5-7 | 0.530 | 0.138 |
| Quantity skew improved seed 42 | 7 | 1001.0 ± 936.2 | 150-2420 | 2.86 ± 0.69 | 2-4 | 0.388 | 0.287 |
| Medical skew seed 42 | 7 | 1001.0 ± 1164.4 | 60-2942 | 3.57 ± 3.21 | 1-7 | 0.200 | 0.536 |
| Simple pathological seed 42 | 3 | 2335.7 ± 2779.2 | 179-5472 | 2.33 ± 0.58 | 2-3 | 0.355 | 0.550 |

Across the 10 paired seeds, the IID partition was stable by construction: all clients had all 7 classes and near-zero JS divergence. Dirichlet alpha = 0.1 was highly variable: average per-seed client-size SD was 1350.6 samples, and the mean minimum number of classes per client was 2.1. Dirichlet alpha = 0.5 would be milder: average client-size SD 805.6 and mean minimum classes/client 4.7. This supports the interpretation that alpha = 0.1 is a severe stress test, but it also exposes a limitation: alpha = 0.1 conflates label skew with substantial quantity skew because client sizes vary dramatically.

## 5. Federated Learning Configuration

Current statistical-heterogeneity Flower runs use a common cross-silo federation:

| Setting | Value | Evidence |
|---|---|---|
| Number of clients | 7 | Partition functions produce 7 clients except legacy/simple pathological; runner uses `len(client_indices)` (`run_one_flower.py:182-184`). |
| Client participation | C = 1.0 | CLI default `--fraction-fit 1.0` (`run_one_flower.py:95-98`); scripts do not override it for statistical sweeps. |
| Communication rounds | 150 | CLI default (`run_one_flower.py:82-83`); scripts state R = 150 (`submit_flower_C0_baseline.sh:19`, `submit_robustness.sh:6`). |
| Local epochs | E = 20 | CLI default (`run_one_flower.py:82`); scripts set `LOCAL_EPOCHS=20` (`submit_flower_C0_baseline.sh:32`, `submit_robustness.sh:14`). |
| Batch size | 32 | CLI default (`run_one_flower.py:87`). |
| Optimizer | SGD | Client creates `torch.optim.SGD` (`fl_flower/client.py:148-153`). |
| Learning rate | 0.01 | CLI default (`run_one_flower.py:84`). |
| Momentum | 0.9 | CLI default (`run_one_flower.py:85`). |
| Weight decay | 0.0 | CLI default (`run_one_flower.py:86`). |
| Evaluation | Central validation every round; test once at best validation macro-F1 | `run_one_flower.py:227-267`, `432-443`. |
| Main metric | Test macro-F1 at best validation checkpoint | Metrics computed in `fl/evaluation.py:65-71`; best-val test in `run_one_flower.py:432-443`. |
| Secondary metrics | Accuracy, balanced accuracy, loss, per-class F1 | `fl/evaluation.py:65-71`. |
| Random seeds | 42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828 | Declared in `submit_flower_C0_baseline.sh:33`, `submit_robustness.sh:15`, `submit_specialist_partition.sh:29`. |

Variables controlled across current FedAvg/FedProx statistical comparisons:

- Dataset and split.
- Model variant, except `arch_ablation_bn/`.
- Number of clients.
- Participation fraction.
- Rounds, local epochs, optimizer, learning rate, momentum, batch size.
- Seed list.
- Partition function within each paired comparison.
- Central validation and test sets.

Variables intentionally varied:

- Algorithm: FedAvg versus FedProx.
- FedProx mu: fixed at 0.01 in the main runs.
- Partition: engineered paired, IID, Dirichlet alpha = 0.1, specialist.
- Model normalization only in the exploratory BN ablation.

Important missing control: a complete current mu sweep is not available, so the choice mu = 0.01 is inherited from literature and prior assumptions rather than validated comprehensively on this exact Flower/DermaMNIST setting.

## 6. Methods Compared

### 6.1 FedAvg

FedAvg uses size-weighted aggregation of client model states. In the pure-PyTorch path this is explicit in `weighted_average_state_dicts` (`mnist_dermnist/fl/aggregation.py:13-46`) and called after local training (`mnist_dermnist/fl/server_loop.py:274-276`). In Flower, the standard `fl.server.strategy.FedAvg` strategy is used unless the system-heterogeneity-only drop-straggler flag is enabled (`mnist_dermnist/experiments/run_one_flower.py:304-337`). For statistical heterogeneity runs, `system_het.mode` is uniform and `drop_stragglers` is absent or false.

### 6.2 FedProx

FedProx uses the same server aggregation as FedAvg; the only intended change is the local objective. In the Flower client, the proximal anchor is cloned once after receiving the round-start parameters (`mnist_dermnist/fl_flower/client.py:145-146`), and the loss adds `(mu / 2) * sum(||w - w_g||^2)` when `proximal_mu > 0` (`mnist_dermnist/fl_flower/client.py:163-168`). The pure-PyTorch implementation follows the same gating and uses detached cloned global weights (`mnist_dermnist/fl/local_train.py:70-92`, `105-112`).

The `mu = 0` path is explicitly designed to be bit-identical to FedAvg in the local-training branch (`mnist_dermnist/fl/local_train.py:3-13`, `87-92`). That is correct and important for sanity checks.

### 6.3 Other Methods

FedNova appears in `flower_C0_baseline/` and `flower_C0_iid_baseline/`, but it is not central to the statistical heterogeneity question in this document. It can be reported as an additional baseline only when the implementation and hyperparameters are discussed separately. It should not be allowed to distract from the FedAvg/FedProx statistical-heterogeneity analysis.

The BatchNorm architecture ablation uses the same FedAvg/FedProx methods but changes model state dynamics by adding BatchNorm buffers. With only 3 seeds, it is exploratory.

## 7. Models

The headline model is `DermMNISTCNN`, a 423,175-parameter GroupNorm CNN. Its architecture is documented in `mnist_dermnist/models/dermmnist_cnn.py:1-20` and implemented in `mnist_dermnist/models/dermmnist_cnn.py:28-50`. It has four convolutional blocks, GroupNorm layers, adaptive average pooling, a 128-unit hidden layer, dropout, and a 7-class output. It has no BatchNorm buffers.

The BatchNorm variant `DermMNISTCNN_BN` is architecturally identical except that the four GroupNorm layers are replaced with BatchNorm2d (`mnist_dermnist/models/dermmnist_cnn_bn.py:1-24`, `40-69`). It has the same 423,175 trainable parameters but 964 buffers. Because BatchNorm buffers alter the state that is exchanged/aggregated, results from `arch_ablation_bn/` should not be mixed with the headline GroupNorm results.

The model capacity is reasonable for 28x28 DermaMNIST, but it is not a state-of-the-art dermoscopy model. The thesis should frame absolute macro-F1 values as benchmark results, not clinical performance.

## 8. Experimental Matrix

| Dataset | Method | Partition type | Alpha / level | Clients | Clients/round | E | Batch | LR | Mu | Seeds | Result files |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| DermaMNIST 28x28 | FedAvg/FedProx | Engineered paired label skew | N/A | 7 | 7 | 20 | 32 | 0.01 | 0 / 0.01 | 10 | `results/headline/` pure-PyTorch |
| DermaMNIST 28x28 | FedAvg/FedProx/FedNova | Engineered paired label skew | N/A | 7 | 7 | 20 | 32 | 0.01 | 0 / 0.01 | 10 | `results/flower_C0_baseline/` |
| DermaMNIST 28x28 | FedAvg/FedProx | IID random sharding | IID | 7 | 7 | 20 | 32 | 0.01 | 0 / 0.01 | 10 | `results/iid/` |
| DermaMNIST 28x28 | FedAvg/FedProx/FedNova | IID random sharding | IID | 7 | 7 | 20 | 32 | 0.01 | 0 / 0.01 | 10 | `results/flower_C0_iid_baseline/` |
| DermaMNIST 28x28 | FedAvg/FedProx | Dirichlet label/quantity skew | alpha = 0.1 | 7 | 7 | 20 | 32 | 0.01 | 0 / 0.01 | 10 | `results/dirichlet_a01/` |
| DermaMNIST 28x28 | FedAvg/FedProx | Specialist singleton skew | N/A | 7 | 7 | 20 | 32 | 0.01 | 0 / 0.01 | 10 | `results/specialist_partition/` |
| DermaMNIST 28x28 | FedAvg/FedProx | Engineered paired label skew | N/A | 7 | 7 | 20 | 32 | 0.01 | 0 / 0.01 | 3 | `results/arch_ablation_bn/`, BN model |
| DermaMNIST 28x28 | FedProx only | Engineered paired label skew | N/A | 7 | 7 | 20 | 32 | 0.01 | 0.001 / 0.1 / 1.0 | 3 | `results/mu_sweep/`, incomplete |
| DermaMNIST 28x28 | Centralised SGD | No federation | N/A | 0 | N/A | 50 epochs | 32 | 0.01 | N/A | 10 | `results/centralised/` |

Legacy `fl-dermamnist-starter` results include Dirichlet alpha = 0.3, 10 clients, 64x64, 100 rounds, E = 5, batch size 64, and 3 paired seeds. These are useful for historical context and per-client metric examples, but invalid as direct comparators to current 7-client, 28x28, E=20 results.

## 9. Results

### 9.1 Final Performance

All current FL results are test performance at the best validation macro-F1 checkpoint, not necessarily final-round test performance.

| Result family | Runtime | Partition | n pairs | FedAvg macro-F1 mean ± SD | FedProx macro-F1 mean ± SD | Mean delta | Wilcoxon p | FedProx wins |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `headline/` | pure-PyTorch | balanced paired | 10 | 0.4814 ± 0.0254 | 0.5081 ± 0.0140 | +0.0267 ± 0.0349 | 0.0195 | 9/10 |
| `flower_C0_baseline/` | Flower | balanced paired | 10 | 0.4964 ± 0.0150 | 0.5033 ± 0.0206 | +0.0069 ± 0.0243 | 0.4316 | 7/10 |
| `flower_C0_iid_baseline/` | Flower | IID | 10 | 0.5846 ± 0.0197 | 0.5786 ± 0.0190 | -0.0061 ± 0.0187 | 0.5566 | 5/10 |
| `iid/` | Flower | IID | 10 | 0.5853 ± 0.0205 | 0.5785 ± 0.0204 | -0.0068 ± 0.0216 | 0.3750 | 4/10 |
| `dirichlet_a01/` | Flower | Dirichlet alpha = 0.1 | 10 | 0.4796 ± 0.0355 | 0.4875 ± 0.0341 | +0.0079 ± 0.0237 | 0.3223 | 6/10 |
| `specialist_partition/` | Flower | specialist | 10 | 0.4365 ± 0.0236 | 0.4446 ± 0.0172 | +0.0081 ± 0.0346 | 0.4922 | 7/10 |
| `arch_ablation_bn/` | Flower | balanced paired, BN model | 3 | 0.5089 ± 0.0052 | 0.4887 ± 0.0289 | -0.0202 ± 0.0311 | 0.5000 | 1/3 |

Interpretation:

- The pure-PyTorch engineered result is the only statistically positive FedProx-FedAvg result in the current statistical heterogeneity corpus.
- The Flower engineered result is directionally positive but non-significant and much smaller.
- IID controls behave as expected: FedProx does not outperform FedAvg, and the mean effect is slightly negative.
- Dirichlet alpha = 0.1 and specialist controls are directionally positive but small and non-significant.
- The BN ablation must not be called a statistical result; with n=3 it is a pilot observation only.

### 9.2 Best Performance

The logged test JSON files are already "test at best validation checkpoint". The selected validation round is stored as `selected_round`, and `best_val_macro_f1` is stored in each JSON (`run_one_flower.py:442-443`, `481-513`). Selected rounds were reasonable in the current result directories:

| Family | Selected round min | Median | Max |
|---|---:|---:|---:|
| `headline/` | 70 | 126.5 | 150 |
| `flower_C0_baseline/` | 93 | 123.0 | 150 |
| `iid/` | 18 | 35.0 | 83 |
| `dirichlet_a01/` | 60 | 124.5 | 148 |
| `specialist_partition/` | 84 | 134.5 | 150 |

The earlier failure fingerprint of constant validation macro-F1 and selected round 1 is not present in the current statistical result directories.

### 9.3 Convergence Speed

Per-round validation history CSVs exist for all current statistical FL results. The repository already has scripts for convergence curves and communication metrics, for example `mnist_dermnist/results/thesis_ready/scripts/analyse_communication_metrics.py` and `generate_curves.py`. However, the final thesis should not claim communication efficiency until a threshold rule is pre-declared and computed consistently.

Recommended computation:

- Use validation macro-F1 curves from `history_*.csv`.
- Define target thresholds before looking at method wins: e.g. 90% of the centralised mean macro-F1, 95% of each method's own best validation macro-F1, or fixed clinically interpretable thresholds such as 0.45 / 0.50 macro-F1.
- Report median rounds-to-threshold across seeds, plus number of seeds that never reach the threshold.
- Do not use test performance curves; test is only evaluated once at best-val checkpoint.

### 9.4 Stability

Stability is not fully analysed in the current result JSONs. It can be derived from history CSVs using final-K-round validation macro-F1 variance, but that value is not currently included in the result summaries. A thesis-ready stability table should report:

- Final 20-round validation macro-F1 SD per seed.
- Across-seed SD of test-at-best macro-F1.
- Collapse/divergence counts, if any.

At present, "FedProx improves stability" is not established across Flower statistical sweeps. The pure-PyTorch headline has lower across-seed SD for FedProx than FedAvg, but the Flower result does not justify a broad stability claim without more formal curve-based analysis.

### 9.5 Per-Client Performance

The current `mnist_dermnist` Flower/pure-PyTorch pipeline evaluates centrally on global validation and test sets. It does not log per-client validation/test performance for the canonical statistical runs. The old `fl-dermamnist-starter` results do contain `per_client_metrics.csv`, `client_summary.json`, and local class distributions, but they belong to a different codebase/configuration.

Do not include per-client fairness claims for the current thesis unless new analysis code evaluates the global model on each client's local held-out subset or client-specific slice. The current global test split is not client-specific.

### 9.6 Failure Cases

No current canonical statistical result family contains NaN metrics, macro-F1 below 0.05, macro-F1 above 0.95, missing histories, or non-150-row histories. The incomplete item is not a failed run but an incomplete design: `mu_sweep/` contains only FedProx rows for mu = 0.001, 0.1, and 1.0 over 3 seeds, with no in-directory FedAvg baselines and no mu = 0.01 or 0.5 rows.

## 10. Literature-Aligned Interpretation

The dissertation should follow reporting conventions from standard FL papers rather than overfitting the write-up to whichever result looks strongest.

Relevant literature anchors:

- McMahan et al. introduced FedAvg and reported communication-round behaviour under unbalanced and non-IID settings: https://arxiv.org/abs/1602.05629
- Li et al. introduced FedProx as a modification of the local objective for statistical and systems heterogeneity, with FedAvg recovered when the proximal coefficient is zero: https://proceedings.mlsys.org/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf
- Hsu, Qi, and Brown popularised Dirichlet label-skew analysis for federated visual classification: https://arxiv.org/abs/1909.06335
- LEAF emphasises realistic federated benchmarks and heterogeneity-aware evaluation: https://arxiv.org/abs/1812.01097
- FedNova and SCAFFOLD are useful comparators in the wider FL literature, but they address different mechanisms and should not be folded into the statistical FedAvg/FedProx claim without care: https://arxiv.org/abs/2007.07481 and https://arxiv.org/abs/1910.06378
- MedMNIST v2 documents the benchmark family and lightweight biomedical classification framing: https://arxiv.org/abs/2110.14795

Reporting style to emulate:

- Report the exact partition-generation rule, not merely "non-IID".
- Report number of clients, participation fraction, local epochs, rounds, optimizer, batch size, learning rate, model, and seed count.
- Report mean ± SD over seeds.
- Use paired tests only where the same seeds and partitions are aligned.
- Separate final-test-at-best-validation from final-round metrics.
- Report negative or null results with the same prominence as positive results.
- Show class distribution heatmaps for synthetic non-IID settings.
- Avoid mechanism language unless there is a direct measurement.

## 11. Recommended Thesis Tables

### Table 1: Experimental Matrix

Use the table in Section 8. Add a column marking status: completed, exploratory, incomplete, or legacy.

### Table 2: Final Performance Across Heterogeneity Levels

Do not use the requested alpha columns if no data exist. A truthful version is:

| Method | IID | Dirichlet alpha = 0.1 | Specialist | Engineered paired, Flower | Engineered paired, pure-PyTorch |
|---|---:|---:|---:|---:|---:|
| FedAvg | 0.5853 ± 0.0205 | 0.4796 ± 0.0355 | 0.4365 ± 0.0236 | 0.4964 ± 0.0150 | 0.4814 ± 0.0254 |
| FedProx mu = 0.01 | 0.5785 ± 0.0204 | 0.4875 ± 0.0341 | 0.4446 ± 0.0172 | 0.5033 ± 0.0206 | 0.5081 ± 0.0140 |
| Delta | -0.0068 | +0.0079 | +0.0081 | +0.0069 | +0.0267 |

Mark alpha = 0.5, 0.3, 1.0 as "not run in current canonical pipeline" rather than leaving blanks that imply missing analysis.

### Table 3: Communication Efficiency

Recommended but not yet thesis-ready. Compute from `history_*.csv` only after defining thresholds. Suggested columns:

- Partition.
- Method.
- Threshold.
- Median rounds to threshold.
- Seeds reached / total.
- Median delta vs FedAvg.

### Table 4: Stability

Recommended but not yet thesis-ready. Suggested rows:

- Across-seed SD of test-at-best macro-F1.
- Mean final-20-round validation macro-F1 SD.
- Number of seeds with selected_round < 10.
- Number of histories with NaNs or flat curves.

### Table 5: Per-Client Performance / Fairness

Not available for current canonical runs. If added, report:

- Mean client test macro-F1.
- SD across clients.
- Worst-client macro-F1.
- Best-client macro-F1.
- Best-worst gap.

Until this exists, do not make fairness claims.

## 12. Recommended Thesis Figures

### Essential

1. **Partition heatmap for engineered paired, specialist, IID, and Dirichlet alpha = 0.1.**
   - Source: partition functions in `mnist_dermnist/data/partition.py`; optionally regenerate CSVs under `mnist_dermnist/results/partitions/`.
   - X-axis: clients; y-axis: classes; color: sample count or client class proportion.
   - Message: the experiments simulate distinct forms of label skew, and the engineered partition is visibly non-IID.
   - Criticism to pre-empt: engineered paired is not natural hospital data.

2. **Paired seed delta plot for engineered paired partition, pure-PyTorch and Flower side by side.**
   - Source: `results/headline/` and `results/flower_C0_baseline/`.
   - X-axis: seed; y-axis: FedProx - FedAvg test macro-F1.
   - Message: pure-PyTorch positive effect attenuates under Flower.
   - Criticism to pre-empt: do not claim Flower "causes" attenuation; call it runtime/implementation sensitivity.

3. **Robustness table/forest plot across partitions.**
   - Source: `flower_C0_baseline/`, `iid/` or `flower_C0_iid_baseline/`, `dirichlet_a01/`, `specialist_partition/`.
   - X-axis: paired delta with uncertainty; y-axis: partition.
   - Message: Flower effects are small across statistical partitions.
   - Criticism to pre-empt: alpha dose-response is incomplete.

4. **Convergence curves for Flower engineered paired and IID.**
   - Source: `history_fedavg_*.csv`, `history_fedprox_*.csv`.
   - X-axis: communication round; y-axis: validation macro-F1; show mean ± SEM or SD.
   - Message: show training dynamics instead of only final/best test numbers.
   - Criticism to pre-empt: curves are validation, not test.

### Useful

5. **Per-class test F1 comparison for the engineered partition.**
   - Source: `per_class_f1` fields in `headline/` and/or `flower_C0_baseline/`.
   - Message: identify which classes drive the aggregate delta.
   - Caveat: per-class tests are exploratory and require Holm correction.

6. **Federation tax table or bar chart.**
   - Source: `centralised/` versus engineered FL results.
   - Message: how far FL models sit below pooled centralised training.
   - Caveat: centralised is not a partition experiment and should not be over-interpreted.

### Optional / Exploratory

7. **BN versus GN ablation plot.**
   - Source: `arch_ablation_bn/` and matching seeds from `flower_C0_baseline/`.
   - Status: n=3 pilot only.
   - Safe message: normalization may interact with FL dynamics; no conclusion.

8. **Mu sensitivity plot.**
   - Source: `mu_sweep/` plus matching baseline/headline rows if carefully merged.
   - Status: incomplete; do not present as full sweep.
   - Safe message: high mu values appear worse in the partial pilot; optimal mu not fully tuned.

### Remove / Avoid

- Per-client performance box plots for current canonical runs: data not logged.
- Per-client specialty curves pretending validation classes correspond to clients: this is not true per-client evaluation.
- Confusion matrices for a single seed as main evidence: useful appendix, weak main figure.
- Decorative multi-panel figures that repeat table values without clarifying a claim.

## 13. Critical Gaps and Risks

### Too few alpha levels

Only alpha = 0.1 is completed in the current canonical pipeline. The runner can implement alpha = 0.5 (`run_one_flower.py:59-60`), but no current result directory was found. Without alpha = 0.5/1.0, the thesis cannot claim a smooth relationship between heterogeneity severity and FedProx benefit.

Minimal fix: no mandatory rerun if time is tight; reframe as "severe Dirichlet robustness check" instead of "dose-response". If compute is available, alpha = 0.5 at 10 paired seeds is the cleanest single addition.

### Engineered main partition

The main positive result is from a partition explicitly designed to give FedProx a mechanism to act on (`partition.py:66-73`, `398-408`). This is acceptable only if called an engineered stress test.

Minimal fix: lead with the design rationale and then immediately show IID, Dirichlet, and specialist controls.

### Runtime inconsistency

The pure-PyTorch and Flower engineered results disagree in effect size. Current result directories also contain mixed commits/hosts/Torch versions in some Flower sweeps. This does not invalidate the artifacts, but it weakens any claim that the effect is implementation-invariant.

Minimal fix: make Flower the canonical implementation for current claims, and present pure-PyTorch as historical/reference evidence. Or, if the thesis keeps pure-PyTorch as primary, explicitly call Flower a non-confirmatory replication.

### Incomplete mu tuning

The intended mu sweep is not complete. `submit_mu_sweep.sh:1-38` planned 18 jobs, but `results/mu_sweep/` contains only 9 FedProx JSONs and no FedAvg rows in that directory. It covers mu = 0.001, 0.1, 1.0 only.

Minimal fix: do not claim mu = 0.01 is empirically optimal. Say it is a literature-standard value selected a priori.

### No per-client evaluation

The current canonical pipeline uses central validation/test evaluation. This is standard for benchmark comparison but insufficient for fairness claims.

Minimal fix: remove per-client fairness claims from the main thesis. Add as future work or implement a small post-hoc evaluator using prediction on client-indexed train/validation subsets if strictly needed.

### No true natural site heterogeneity

The dataset is federated synthetically. Even the "medical skew" partition is hand-coded and not backed by hospital metadata.

Minimal fix: use "simulated hospital-like label skew" rather than "hospital heterogeneity".

## 14. Minimal Action Plan Before Thesis Submission

1. **Stop adding new statistical experiments unless alpha = 0.5 can be completed cleanly.**
   - Must-have reruns: none identified for current statistical artifacts.
   - Strongly recommended if compute remains: complete Dirichlet alpha = 0.5 with 10 paired Flower seeds.
   - Nice to have: complete mu sweep, but this is less important than alpha = 0.5.

2. **Choose the thesis hierarchy now.**
   - Primary confirmatory result: FedAvg vs FedProx on engineered paired partition, but explicitly discuss runtime sensitivity.
   - Secondary robustness: Flower IID, Dirichlet alpha = 0.1, specialist.
   - Exploratory: BN ablation, incomplete mu sweep, legacy starter results.

3. **Generate partition heatmaps and a result forest plot.**
   - These are more defensible than many single-seed curves.

4. **Compute communication/stability metrics only if they will be reported consistently.**
   - Do not add ad hoc "rounds-to-target" claims after seeing favourable curves.

5. **Rewrite claims conservatively.**
   - Use "consistent with", "suggests", and "in this engineered stress test" where appropriate.
   - Avoid "proves client drift reduction" and "FedProx generally outperforms FedAvg".

## Appendix A: File-to-Experiment Mapping

| Experiment | Implementation files | Submission/run files | Results |
|---|---|---|---|
| Pure-PyTorch engineered headline | `run_one.py`, `server_loop.py`, `local_train.py`, `partition.py` | Historical local/reference path; not current HPC default | `mnist_dermnist/results/headline/` |
| Flower engineered paired | `run_one_flower.py`, `fl_flower/client.py`, `partition.py` | `submit_flower_C0_baseline.sh` | `mnist_dermnist/results/flower_C0_baseline/` |
| IID control | `iid_7_clients` in `partition.py` | `submit_robustness.sh`; possibly separate IID C0 script | `mnist_dermnist/results/iid/`, `flower_C0_iid_baseline/` |
| Dirichlet alpha = 0.1 | `dirichlet_7_clients`, `_dir_a01` wrapper | `submit_robustness.sh` | `mnist_dermnist/results/dirichlet_a01/` |
| Specialist counterfactual | `specialist_7_clients` | `submit_specialist_partition.sh` | `mnist_dermnist/results/specialist_partition/` |
| Mu sweep | FedProx mu CLI in `run_one_flower.py` / pure PyTorch `run_one.py` | `submit_mu_sweep.sh` | `mnist_dermnist/results/mu_sweep/`, incomplete |
| BN architecture ablation | `DermMNISTCNN_BN`, `--model-variant bn` | `runpod_arch_ablation_bn.sh` | `mnist_dermnist/results/arch_ablation_bn/` |
| Centralised reference | `run_centralised.py` | `slurm_centralised.sh` | `mnist_dermnist/results/centralised/` |
| Legacy alpha = 0.3 starter | Legacy configs/results only | Logs under `fl-dermamnist-starter/logs/` | `fl-dermamnist-starter/results/fedavg_dir03_E5_s*`, `fedprox_dir03_E5_s*` |

## Appendix B: Full Hyperparameter Listing

Current main Flower statistical runs:

- Dataset: DermaMNIST from `dermamnist_64.npz`.
- Image size: 28x28.
- Normalization: ImageNet mean/std.
- Model: `DermMNISTCNN`, GroupNorm, 423,175 trainable parameters.
- Algorithms: FedAvg, FedProx with mu = 0.01.
- Clients: 7.
- Participation: 1.0.
- Rounds: 150.
- Local epochs: 20.
- Optimizer: SGD.
- Learning rate: 0.01.
- Momentum: 0.9.
- Weight decay: 0.0.
- Batch size: 32.
- Loss: cross-entropy.
- Evaluation: central validation each round; central test once at best validation macro-F1.
- Seeds: 42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828.

Legacy starter alpha = 0.3 runs:

- Dataset: DermaMNIST, 64x64.
- Clients: 10.
- Rounds: 100.
- Local epochs: 5 for `dir03_E5`; 20 for showcase.
- Batch size: 64.
- Partition: Dirichlet alpha = 0.3.
- Seeds: 42, 123, 456 for paired FedAvg/FedProx `dir03_E5`.
- Do not mix directly with current 28x28, 7-client, E=20 results.

## Appendix C: Missing Metrics and Suggested Code Changes

### Missing: Persisted partition statistics

Add a small utility that writes, for every run:

- Client sample count.
- Per-client class count vector.
- Number of classes per client.
- Normalized entropy.
- Jensen-Shannon or KL divergence to global class prior.

This could be written once per run beside the result JSON or once per partition/seed under `results/partitions/`.

### Missing: Per-client evaluation

Add a post-hoc evaluator that loads the best global checkpoint or predictions and evaluates on client-specific subsets. Because the current pipeline does not save checkpoints, this would require either saving best model state or evaluating per-client metrics before the model is discarded.

Minimal design:

- During final test stage, also evaluate the best model on each client's local training subset or a client-specific validation subset if available.
- Write `per_client_metrics_<stem>.csv`.
- Include client id, n, local class distribution, accuracy, balanced accuracy, macro-F1, per-class F1.

### Missing: Complete mu sweep

If the thesis wants to defend mu = 0.01 empirically, complete:

- FedAvg baseline rows for seeds 42, 123, 456.
- FedProx mu = 0.01 and mu = 0.5 rows for seeds 42, 123, 456.
- Ideally use Flower rather than pure-PyTorch for consistency with current canonical runs.

### Missing: Alpha dose-response

If one more statistical experiment is possible, run Dirichlet alpha = 0.5 at 10 paired Flower seeds. This gives a moderate non-IID bridge between IID and alpha = 0.1 and is more thesis-critical than additional architecture pilots.

### Suggested safe thesis wording

Do not write:

> FedProx significantly improves DermaMNIST federated learning under non-IID data.

Write:

> On an engineered label-skew partition designed to stress client drift, the pure-PyTorch reference implementation showed a positive FedProx-FedAvg macro-F1 difference. In the canonical Flower simulation and in IID, Dirichlet alpha = 0.1, and specialist controls, the FedProx effect was smaller and not statistically significant. These results suggest that FedProx's benefit on DermaMNIST is sensitive to the exact partition and runtime implementation, and should be interpreted as context-dependent rather than universal.

