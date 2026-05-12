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

Edit the virtualenv path in `scripts/slurm_template.sh` if needed, then submit a single test job:

```bash
mkdir -p /home/bk489/federated/federated-thesis/experiments/dermamnist/logs
sbatch scripts/slurm_template.sh configs/fedavg_dir05.yaml 42
```

### Recommended: dissertation-focused submission

The default submission script for the dissertation is:

```bash
bash scripts/submit_dissertation_core.sh
```

It submits a curated subset (31 jobs + one μ ablation sweep) that targets the main scientific question — how class heterogeneity affects FedAvg, and whether FedProx plus selected mitigations help. Specifically:

- 7 core configs × 3 seeds = 21 jobs
  - `fedavg_iid` (control)
  - `fedavg_dir05`, `fedprox_dir05` (moderate heterogeneity)
  - `fedavg_dir01`, `fedprox_dir01` (severe heterogeneity)
  - `fedavg_pathological`, `fedprox_pathological` (specialist clients, k=2)
- 3 mitigation configs × 3 seeds = 9 jobs
  - `fedprox_dir05_weighted`, `fedprox_dir05_focal`, `fedavg_dir05_weighted`
- 1 μ ablation job (seed 42)

Use `bash scripts/check_results.sh` to see which dissertation jobs have finished and which are still missing.

### Broader matrix (optional)

The following scripts are still available but are NOT part of the default dissertation run. Only invoke them manually if compute budget allows:

```bash
bash scripts/submit_all_core.sh        # 11 core configs × 3 seeds = 33 jobs
bash scripts/submit_mitigation.sh      # 7 mitigation configs × 3 seeds = 21 jobs
bash scripts/submit_ablation.sh <name> # local_epochs | alpha | clients | participation
```
