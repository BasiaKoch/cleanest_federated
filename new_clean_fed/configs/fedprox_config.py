"""FedProx config — DermaMNIST classification.

Identical to fedavg_config plus the proximal coefficient MU (Li et al. 2020).
"""

# Paths
NPZ_PATH = "/Users/basiakoch/cleanest_federated/dermamnist_64.npz"
RESULTS_DIR = "results/fedprox"

# Federation
NUM_CLIENTS = 10
NUM_ROUNDS = 50
FRACTION_FIT = 0.5
NUM_LOCAL_EPOCHS = 5

# Partition
PARTITION_STRATEGY = "dirichlet"
DIRICHLET_ALPHA = 0.3
MIN_SAMPLES_PER_CLIENT = 5

# Training
BATCH_SIZE = 32
LR = 0.02
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
DROPOUT = 0.2

# FedProx — proximal coefficient
MU = 0.1                       # 0 ≡ FedAvg

# Misc
SEED = 42
DEVICE = "cpu"
LOG_EVERY = 1
