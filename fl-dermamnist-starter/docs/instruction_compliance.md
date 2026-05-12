# Instruction Compliance Checklist

This checklist maps `instructions.md` to the current project implementation.

## Phase 0

- Prompt 0: Research questions and experiment matrix are in `docs/`.
- Prompt 1: Project structure, requirements, README, and seed utility are present.
- Prompt 2: Flower smoke test is present and validates `state_dict` serialisation.
- Prompt 3: Dataset utilities support MNIST, MedMNIST package loading, and direct standard MedMNIST `.npz` loading.
- Prompt 4: IID, Dirichlet, pathological, quantity-skew, and combined partitioning are implemented with validation.
- Prompt 5: Partition tests cover validity, reproducibility, Dirichlet behavior, min-sample enforcement, impossible configs, and rare-class warnings.
- Prompt 6: Data visualisation functions save figures, return figures, and close saved figures.
- Prompt 7: SimpleCNN, ResNet-18, model factory, parameter counting, and model smoke tests are implemented.
- Prompt 8: Evaluation metrics, plots, AUROC handling, and `MetricsLogger` are implemented.
- Prompt 9: Focal loss, weighted CE helpers, zero-count handling, and logit adjustment are implemented.
- Prompt 10: Centralised trainer tracks validation balanced accuracy and returns test metrics plus history.
- Prompt 11: Centralised MNIST and DermaMNIST baseline scripts are present.
- Prompt 12: DermaMNIST exploration script is present.

## Phase 1

- Prompt 13: Flower client uses `state_dict`, handles FedProx client objective, and converts labels with `view(-1).long()`.
- Prompt 14: Flower server uses FedAvg with checkpointing, validation-based best model selection, final test evaluation, and per-client metrics.
- Prompt 15: Core YAML configs and config loader are present; experiment names include local epochs and seed.
- Prompt 16: FL MNIST validation script is present.
- Prompt 17: Unified runner and all-core runner are present; results include config, history, metrics, plots, and checkpoints.
- Prompt 18: Per-class tracker includes F1/recall curves, convergence heatmap, and convergence-round computation.

## Phase 2

- Prompt 19: Weighted CE, focal loss, weighted sampler, and mitigation configs are present.
- Prompt 20: Training-time and post-hoc logit adjustment are implemented; calibration analysis script is present.

## Phase 3

- Prompt 21: Ablation runner is present.
- Prompt 22: Drift measurement in the strategy and drift analysis script are present.
- Prompt 23: Fairness metrics and fairness analysis script are present.
- Prompt 24: Statistical analysis script is present.

## Phase 4

- Prompt 25: Figure/table generation script is present and emits the named outputs when corresponding experiment results exist.

## Optional Extensions

Prompts E1-E4 are explicitly marked optional in `instructions.md`; the core project does not depend on them. The current ResNet implementation already supports pretrained 64x64 experiments if those optional configs are added later.

## HPC Scripts

SLURM template, core submission, mitigation submission, ablation submission, single submission, and result checking scripts are present under `scripts/`.
