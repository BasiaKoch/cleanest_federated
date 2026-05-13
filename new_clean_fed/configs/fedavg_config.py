"""FedAvg config — DermaMNIST classification.

Mirrors the structure of nedeljkovicmajaa/Federated-Learning-And-Class-Imbalances
but adapted for PyTorch + classification (not TensorFlow segmentation).
"""

# Paths
NPZ_PATH = "/Users/basiakoch/cleanest_federated/dermamnist_64.npz"
RESULTS_DIR = "results/fedavg"

# Federation
NUM_CLIENTS = 10
NUM_ROUNDS = 50
FRACTION_FIT = 0.5            # 5 clients sampled per round
NUM_LOCAL_EPOCHS = 5

# Partition (Dirichlet label skew — strong non-IID)
PARTITION_STRATEGY = "dirichlet"
DIRICHLET_ALPHA = 0.3
MIN_SAMPLES_PER_CLIENT = 5

# Training
BATCH_SIZE = 32
LR = 0.02
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
DROPOUT = 0.2

# Misc
SEED = 42
DEVICE = "cpu"                # set to "cuda" on HPC
LOG_EVERY = 1                 # validate every N rounds
