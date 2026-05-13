# new_clean_fed — FedAvg vs FedProx on DermaMNIST

Self-contained FL implementation mirroring
[nedeljkovicmajaa/Federated-Learning-And-Class-Imbalances](https://github.com/nedeljkovicmajaa/Federated-Learning-And-Class-Imbalances)
but adapted for:
- **PyTorch** (their reference is TensorFlow)
- **Classification on DermaMNIST** (their reference is segmentation on breast MRI)
- **Specified CNN architecture** (4 conv blocks 32→64→128→256 + FC head)
- **Flower simulation API** (single machine; their reference uses TCP networking)

## Layout

```
new_clean_fed/
├── configs/
│   ├── fedavg_config.py        # plain SGD + CE
│   └── fedprox_config.py       # adds MU for the proximal term
├── src/
│   ├── fedavg/
│   │   ├── model.py            # CNNClassifier
│   │   ├── prepare_data.py     # DermaMNIST loader + Dirichlet partition
│   │   └── client.py           # Flower client with plain local CE
│   └── fedprox/
│       ├── model.py            # re-exports fedavg.model
│       ├── prepare_data.py     # re-exports fedavg.prepare_data
│       └── client.py           # Flower client with proximal-term loss
├── scripts/
│   ├── fedavg/server.py        # Flower simulation entry point (FedAvg)
│   └── fedprox/server.py       # Flower simulation entry point (FedProx)
├── tests/
│   ├── test_fedavg.py          # CNN architecture + state-dict round-trip
│   └── test_fedprox.py         # μ=0 ≡ FedAvg + proximal-term correctness
├── requirements.txt
└── README.md
```

## Model

The CNN backbone is identical for both algorithms (single source of truth in
`src/fedavg/model.py`):

```
Conv(3 → 32)  + BatchNorm + ReLU + MaxPool
Conv(32 → 64) + BatchNorm + ReLU + MaxPool
Conv(64 → 128) + BatchNorm + ReLU + MaxPool
Conv(128 → 256) + BatchNorm + ReLU + AdaptiveAvgPool(1)
FC(256 → 128) + ReLU + Dropout(0.2)
FC(128 → 7)
```

~470K trainable parameters. Works at any input resolution because of
`AdaptiveAvgPool2d(1)`. The npz file is 64×64 by default; the model handles it
unchanged.

## FedAvg vs FedProx — the only difference

FedAvg local loss per batch:
```
loss = CrossEntropy(model(x), y)
```

FedProx local loss per batch (Li et al. 2020):
```
loss = CrossEntropy(model(x), y)  +  (μ / 2) * sum(‖w - w_global‖²)
```

`w_global` is a detached snapshot of the model weights at the *start* of the
round; it does not receive gradient. When `μ=0`, the second branch is gated
off and FedProx reduces to FedAvg numerically (verified by
`tests/test_fedprox.py::test_mu_zero_matches_fedavg_gradient`).

## Setup

From the parent directory `/Users/basiakoch/cleanest_federated`:

```bash
# Activate your existing venv OR create a fresh one
source .venv/bin/activate
pip install -r new_clean_fed/requirements.txt

# Update the NPZ_PATH in both config files to point to your dermamnist_64.npz
# (default already points to /Users/basiakoch/cleanest_federated/dermamnist_64.npz)
```

## Running

```bash
# From the parent dir of new_clean_fed/
cd /Users/basiakoch/cleanest_federated

# Run tests first
PYTHONPATH=. python -m pytest new_clean_fed/tests/ -v

# FedAvg
PYTHONPATH=. python -m new_clean_fed.scripts.fedavg.server

# FedProx
PYTHONPATH=. python -m new_clean_fed.scripts.fedprox.server
```

Outputs land in `new_clean_fed/results/{fedavg,fedprox}/metrics_history.csv`
with one row per round containing centralized validation `loss`, `accuracy`,
`macro_f1`, `balanced_accuracy`, plus the weighted client `train_loss` from
`fit_metrics_aggregation_fn`.

## Comparison

After both runs complete:

```bash
PYTHONPATH=. python -c "
import pandas as pd
fa = pd.read_csv('new_clean_fed/results/fedavg/metrics_history.csv')
fp = pd.read_csv('new_clean_fed/results/fedprox/metrics_history.csv')
print('FedAvg final  — round', int(fa.iloc[-1]['round']), 'macro_F1', round(fa.iloc[-1]['macro_f1'], 4))
print('FedProx final — round', int(fp.iloc[-1]['round']), 'macro_F1', round(fp.iloc[-1]['macro_f1'], 4))
print('Δ =', round(fp.iloc[-1]['macro_f1'] - fa.iloc[-1]['macro_f1'], 4))
"
```

## Notes on faithfulness to the reference repo

| Reference repo | This repo | Reason |
|---|---|---|
| TensorFlow + Keras | PyTorch | Cleaner classification ergonomics; gradient tape replaced with `loss.backward()` |
| U-Net + Dice loss (segmentation) | CNNClassifier + CE loss (classification) | DermaMNIST is a classification task |
| `fl.client.start_client(server_address=...)` (TCP) | `fl.simulation.start_simulation(...)` | Single-machine reproducibility; same Flower APIs underneath |
| 2 clients, 10 rounds, 3 local epochs | 10 clients, 50 rounds, 5 local epochs | Standard FL benchmark config; configurable in `configs/*.py` |
| `PROBLEM_TYPE = "normal/" / "statistical_het/" / "system_het/"` | `PARTITION_STRATEGY = "dirichlet"` with `α=0.3` | Dirichlet is the standard non-IID benchmark |
| `MU = 0.01` default | `MU = 0.1` default | Larger μ tends to win for stronger drift conditions (paper §5.1) |

The FedProx client implementation in `src/fedprox/client.py` follows the
reference repo's structure (custom training loop with explicit proximal-term
computation) almost line-for-line — just translated from TensorFlow's
`GradientTape` to PyTorch's `loss.backward()`.
