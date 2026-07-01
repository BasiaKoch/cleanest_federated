"""Runtime orchestrators for the shared algorithms in `fl_dermamnist/core/`.

`pytorch/` is the pure-PyTorch reference FL loop; `flower/` is the
Flower-based simulation runtime (clients and strategies). Both drive the
same FedAvg / FedProx / FedNova algorithms.
"""
