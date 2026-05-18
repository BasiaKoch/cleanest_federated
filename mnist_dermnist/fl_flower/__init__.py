"""Flower-based runtime for FedAvg / FedProx.

This subpackage provides a Flower-framework wrapper that produces results
equivalent to the pure-PyTorch FL loop in `mnist_dermnist/fl/`. The
underlying FedAvg and FedProx algorithms are identical; only the
orchestration framework differs (Flower's `start_simulation` vs a manual
round loop).

Use this path when the experimental brief requires a federated-learning
framework (e.g., Flower or NVFlare). Use the pure-PyTorch path in `fl/`
when fine-grained control over RNG ordering is needed (e.g., for
bit-exact paired-seed reproducibility audits).

Equivalence between the two paths is verified by
`mnist_dermnist/experiments/verify_flower_equivalence.py`.
"""
