"""Shared federated-learning algorithms used by both runtimes.

Local training (cross-entropy + the gated FedProx proximal term),
size-weighted aggregation, evaluation (macro-F1 / balanced accuracy /
per-class F1), loss variants (weighted-CE, focal), straggler schedules,
deterministic seeding, and run provenance. The runtime orchestrators in
`fl_dermamnist/runtimes/` call into this package.
"""
