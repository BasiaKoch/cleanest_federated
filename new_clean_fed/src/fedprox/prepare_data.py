"""Re-export shared loading/partitioning helpers (symmetry with reference repo)."""
from new_clean_fed.src.fedavg.prepare_data import (   # noqa: F401
    DermaMNISTDataset,
    load_dermamnist,
    dirichlet_partition,
    print_distribution,
)
