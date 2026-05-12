# Federated Learning for Medical Imaging with Class Imbalances - DermaMNIST

Starter codebase for an MPhil dissertation project studying FedAvg and FedProx under class imbalance and client heterogeneity on DermaMNIST.

## Main focus

- Simulated federated clients using Flower + PyTorch.
- DermaMNIST medical image classification.
- IID, Dirichlet, pathological, quantity-skew, and combined quantity+label partitions.
- FedAvg vs FedProx.
- Global, per-class, and per-client evaluation.
- Class imbalance mitigation using weighted CE, focal loss, weighted sampling, and logit adjustment.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import flwr; print(flwr.__version__)"
```

After the Flower smoke test passes, freeze the exact environment:

```bash
pip freeze > requirements.lock.txt
```

## Dataset licence notice

DermaMNIST is derived from HAM10000 and is used under CC BY-NC 4.0 for academic/non-commercial research only.

## Dataset loading

The default config loads DermaMNIST through the `medmnist` package. To load a local standard MedMNIST archive instead, set:

```yaml
dataset:
  source: npz
  npz_path: datasets/medmnist/dermamnist.npz
```

## First commands

```bash
pytest tests/test_partition.py -v
python tests/test_flower_smoke.py
python experiments/run_exploration.py
python experiments/run_experiment.py --config configs/fedavg_dir05.yaml --debug_subset 500 --num_rounds_override 3 --device cpu
```

## HPC usage

Edit the virtualenv path in `scripts/slurm_template.sh` if needed, then submit:

```bash
mkdir -p /home/bk489/federated/federated-thesis/experiments/dermamnist/logs
sbatch scripts/slurm_template.sh configs/fedavg_dir05.yaml 42
```
