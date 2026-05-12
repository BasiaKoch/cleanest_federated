# Experiment Matrix

## Core Experiments

| ID | Config | Aggregation | Client objective | Partition | Alpha | Clients | Rounds | Local epochs | Seeds |
|---|---|---|---|---|---:|---:|---:|---:|---|
| C01 | fedavg_iid | FedAvg | CE | IID | - | 10 | 100 | 5 | 42,123,456 |
| C02 | fedavg_dir10 | FedAvg | CE | Dirichlet | 1.0 | 10 | 100 | 5 | 42,123,456 |
| C03 | fedavg_dir05 | FedAvg | CE | Dirichlet | 0.5 | 10 | 100 | 5 | 42,123,456 |
| C04 | fedavg_dir03 | FedAvg | CE | Dirichlet | 0.3 | 10 | 100 | 5 | 42,123,456 |
| C05 | fedavg_dir01 | FedAvg | CE | Dirichlet | 0.1 | 10 | 100 | 5 | 42,123,456 |
| C06 | fedavg_pathological | FedAvg | CE | Pathological(2) | - | 10 | 100 | 5 | 42,123,456 |
| C07 | fedprox_dir10 | FedAvg | CE + FedProx | Dirichlet | 1.0 | 10 | 100 | 5 | 42,123,456 |
| C08 | fedprox_dir05 | FedAvg | CE + FedProx | Dirichlet | 0.5 | 10 | 100 | 5 | 42,123,456 |
| C09 | fedprox_dir03 | FedAvg | CE + FedProx | Dirichlet | 0.3 | 10 | 100 | 5 | 42,123,456 |
| C10 | fedprox_dir01 | FedAvg | CE + FedProx | Dirichlet | 0.1 | 10 | 100 | 5 | 42,123,456 |
| C11 | fedprox_pathological | FedAvg | CE + FedProx | Pathological(2) | - | 10 | 100 | 5 | 42,123,456 |
| C12 | fedavg_dirq05 | FedAvg | CE | Dirichlet + quantity skew | 0.5 | 10 | 100 | 5 | 42,123,456 |
| C13 | fedprox_dirq05 | FedAvg | CE + FedProx | Dirichlet + quantity skew | 0.5 | 10 | 100 | 5 | 42,123,456 |

## Mitigation Experiments

| ID | Base | Partition | Alpha | Loss Variant | Seeds |
|---|---|---|---:|---|---|
| M01 | FedAvg | Dirichlet | 0.5 | Weighted CE | 42,123,456 |
| M02 | FedProx | Dirichlet | 0.5 | Weighted CE | 42,123,456 |
| M03 | FedAvg | Dirichlet | 0.5 | Focal gamma=2 | 42,123,456 |
| M04 | FedProx | Dirichlet | 0.5 | Focal gamma=2 | 42,123,456 |
| M05 | FedAvg | Dirichlet | 0.5 | Weighted Sampler | 42,123,456 |
| M06 | FedProx | Dirichlet | 0.5 | Weighted Sampler | 42,123,456 |
| M07 | FedProx | Dirichlet | 0.5 | Focal + Sampler | 42,123,456 |

## Ablation Experiments

| ID | Variable | Values | Fixed |
|---|---|---|---|
| A01 | Local epochs | 1, 5, 10, 20 | FedProx, Dir(0.5), 100 rounds |
| A02 | FedProx mu | 0, 0.001, 0.01, 0.1, 1 | Dir(0.5), E=5, 100 rounds |
| A03 | Dirichlet alpha | 0.05, 0.1, 0.3, 0.5, 1, 5 | FedAvg, E=5, 100 rounds |
| A04 | Num clients | 3, 5, 10, 20 | FedAvg, Dir(0.5), 100 rounds |
| A05 | Participation | 0.3, 0.5, 0.7, 1.0 | FedAvg, Dir(0.5), 10 clients |

## Optional Extensions

| ID | Extension | Description |
|---|---|---|
| E01 | Pretrained ResNet-18 64x64 | Compare cold-start vs warm-start FL |
| E02 | Local fine-tuning | Fine-tune global model locally per client |
| E03 | SCAFFOLD | Variance-reduction aggregation |
| E04 | Poisoning robustness | 1 malicious client plus robust aggregation |

## Compute Stages

| Stage | Clients | Rounds | Subset | Purpose | Where |
|---|---:|---:|---:|---|---|
| debug | 2 | 2 | 500 | Code correctness | Local CPU |
| smoke | 5 | 5 | full | Pipeline check | Local GPU |
| pilot | 10 | 30 | full | Hyperparameter check | HPC |
| final | 10 | 100 | full | Dissertation results | HPC |

Important: use DermaMNIST validation data for checkpoint selection and DermaMNIST test data only for final reporting.
