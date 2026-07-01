"""Flower-based runtime: clients and strategies for FedAvg, FedProx and FedNova.

A Flower-framework wrapper that produces results equivalent to the
pure-PyTorch reference loop in `fl_dermamnist/runtimes/pytorch/`. The
underlying algorithms are shared from `fl_dermamnist/core/`; only the
orchestration framework differs (Flower's `start_simulation` vs a manual
round loop). This path carries the system-heterogeneity, partial-participation
and FedNova experiments.

Use the pure-PyTorch path in `fl_dermamnist/runtimes/pytorch/` when
fine-grained control over RNG ordering is needed (bit-exact paired-seed
reproducibility). Equivalence between the two paths is verified by
`fl_dermamnist/experiments/verify_flower_equivalence.py`.
"""
