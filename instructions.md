# Federated Learning DermaMNIST — Corrected Claude Code Prompts
# Focused: FedAvg + FedProx core. Extensions clearly marked as optional.


---

## PHASE 0 — Project Scaffolding & Smoke Tests

---

### Prompt 0: Research Questions & Experiment Matrix 🖥️ LOCAL

```
Create the following documentation files for a federated learning dissertation project.

File: docs/research_questions.md

Contents:

# Research Questions

RQ1: How does label distribution skew (controlled via Dirichlet α) degrade per-class
     performance in federated learning on DermaMNIST, and are clinically rare classes
     (Dermatofibroma, Vascular Lesions) disproportionately harmed?

RQ2: Does FedProx improve stability and minority-class performance over FedAvg under
     increasing label heterogeneity?

RQ3: Do client-side class-aware loss functions (weighted CE, focal loss) provide
     additional benefit on top of FedProx, or is aggregation-level correction sufficient?

RQ4: Can global performance improve while per-client or per-class fairness degrades?
     What is the trade-off between aggregate balanced accuracy and worst-client /
     worst-class metrics?

RQ5: How does the number of local training epochs interact with class imbalance severity
     — does more local training exacerbate minority-class degradation?


File: docs/experiment_matrix.md

Contents:

# Experiment Matrix

## Core Experiments (must complete)

| ID   | Aggregation | Client Objective | Partition       | α    | Clients | Rounds | Local E | Seeds       |
|------|-------------|-----------------|-----------------|------|---------|--------|---------|-------------|
| C01  | FedAvg      | CE              | IID             | —    | 10      | 100    | 5       | 42,123,456  |
| C02  | FedAvg      | CE              | Dirichlet       | 1.0  | 10      | 100    | 5       | 42,123,456  |
| C03  | FedAvg      | CE              | Dirichlet       | 0.5  | 10      | 100    | 5       | 42,123,456  |
| C04  | FedAvg      | CE              | Dirichlet       | 0.3  | 10      | 100    | 5       | 42,123,456  |
| C05  | FedAvg      | CE              | Dirichlet       | 0.1  | 10      | 100    | 5       | 42,123,456  |
| C06  | FedAvg      | CE              | Pathological(2) | —    | 10      | 100    | 5       | 42,123,456  |
| C07  | FedAvg      | CE + FedProx    | Dirichlet       | 1.0  | 10      | 100    | 5       | 42,123,456  |
| C08  | FedAvg      | CE + FedProx    | Dirichlet       | 0.5  | 10      | 100    | 5       | 42,123,456  |
| C09  | FedAvg      | CE + FedProx    | Dirichlet       | 0.3  | 10      | 100    | 5       | 42,123,456  |
| C10  | FedAvg      | CE + FedProx    | Dirichlet       | 0.1  | 10      | 100    | 5       | 42,123,456  |
| C11  | FedAvg      | CE + FedProx    | Pathological(2) | —    | 10      | 100    | 5       | 42,123,456  |

## Mitigation Experiments (complete after core)

| ID   | Base        | Partition  | α   | Loss Variant    | Seeds      |
|------|-------------|-----------|-----|-----------------|------------|
| M01  | FedAvg      | Dir       | 0.5 | Weighted CE     | 42,123,456 |
| M02  | FedProx     | Dir       | 0.5 | Weighted CE     | 42,123,456 |
| M03  | FedAvg      | Dir       | 0.5 | Focal (γ=2)     | 42,123,456 |
| M04  | FedProx     | Dir       | 0.5 | Focal (γ=2)     | 42,123,456 |
| M05  | FedProx     | Dir       | 0.5 | Weighted Sampler | 42,123,456 |
| M06  | FedProx     | Dir       | 0.5 | Logit Adjusted  | 42,123,456 |

## Ablation Experiments (complete after mitigation)

| ID   | Variable         | Values                 | Fixed                          |
|------|------------------|------------------------|--------------------------------|
| A01  | Local epochs     | 1, 5, 10, 20          | FedProx, Dir(0.5), 100 rounds |
| A02  | FedProx μ        | 0, 0.001, 0.01, 0.1, 1| Dir(0.5), E=5, 100 rounds     |
| A03  | Dirichlet α      | 0.05,0.1,0.3,0.5,1,5  | FedAvg, E=5, 100 rounds       |
| A04  | Num clients      | 3, 5, 10, 20          | FedAvg, Dir(0.5), 100 rounds  |
| A05  | Participation    | 0.3, 0.5, 0.7, 1.0    | FedAvg, Dir(0.5), 10 clients  |

## Optional Extensions (only if time permits — pick ONE)

| ID   | Extension                     | Description                               |
|------|-------------------------------|-------------------------------------------|
| E01  | Pretrained ResNet-18 (64×64)  | Compare cold-start vs warm-start FL       |
| E02  | Local fine-tuning             | Fine-tune global model locally per client |
| E03  | SCAFFOLD                      | Variance-reduction aggregation            |
| E04  | Poisoning robustness          | 1 malicious client + trimmed mean defence |

## Compute Stages

| Stage  | Clients | Rounds | Subset | Purpose          | Where     |
|--------|---------|--------|--------|------------------|-----------|
| debug  | 2       | 2      | 500    | Code correctness | Local CPU |
| smoke  | 5       | 5      | full   | Pipeline check   | Local GPU |
| pilot  | 10      | 30     | full   | Hyperparameter   | HPC       |
| final  | 10      | 100    | full   | Dissertation     | HPC       |

Do NOT create any code yet. This is documentation only.
```

---

### Prompt 1: Project Structure & Dependencies 🖥️ LOCAL

```
Create a Python project for a federated learning dissertation. Set up this directory structure:

fl-dermamnist/
├── configs/              # YAML config files for experiments
├── data/                 # Data download and partitioning
│   ├── __init__.py
│   ├── download.py
│   ├── partition.py
│   └── visualise.py
├── models/               # Model architectures
│   ├── __init__.py
│   ├── simple_cnn.py
│   └── resnet.py
├── client/               # Flower client logic
│   ├── __init__.py
│   └── flower_client.py
├── server/               # Flower server + custom strategy
│   ├── __init__.py
│   └── flower_server.py
├── trainers/             # Centralised training loop
│   ├── __init__.py
│   └── centralised.py
├── losses/               # Custom loss functions
│   ├── __init__.py
│   ├── focal_loss.py
│   └── weighted_ce.py
├── metrics/              # Evaluation utilities
│   ├── __init__.py
│   └── evaluation.py
├── utils/                # Logging, config, helpers
│   ├── __init__.py
│   ├── config_loader.py
│   ├── experiment_tracker.py
│   └── seed.py
├── tests/                # Unit tests
│   └── test_partition.py
├── experiments/          # Experiment runner scripts
│   └── run_experiment.py
├── scripts/              # HPC job submission
├── results/              # Output logs, plots, checkpoints
├── docs/                 # Research questions, experiment matrix
├── requirements.txt
└── README.md

Create the directory structure with empty __init__.py files.

Write requirements.txt:
```
torch>=2.0
torchvision>=0.15
flwr[simulation]>=1.11
medmnist>=3.0
numpy
pandas
matplotlib
seaborn
scikit-learn
pyyaml
tensorboard
tqdm
scipy
```

Write README.md with:
- Title: "Federated Learning for Medical Imaging with Class Imbalances — DermaMNIST"
- Brief description of the project
- Setup instructions: `pip install -r requirements.txt`
- Dataset licence notice:
  "DermaMNIST is derived from HAM10000 and used under CC BY-NC 4.0 for
   academic/non-commercial research only."
- How to run: pointer to experiments/run_experiment.py

Write utils/seed.py:
```python
import random, numpy as np, torch

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

After creating everything, print the Flower version:
  python -c "import flwr; print(flwr.__version__)"
```

---

### Prompt 2: Flower Smoke Test 🖥️ LOCAL

```
IMPORTANT: Run this BEFORE implementing any real FL logic. This validates that
the installed Flower version works with our simulation setup.

Create `tests/test_flower_smoke.py`:

1. Print the installed Flower version.
2. Create a trivial model: a single nn.Linear(2, 2).
3. Create a trivial dataset: 100 random (x, y) pairs with labels 0 or 1.
4. Implement a minimal Flower NumPyClient that:
   - Serialises and deserialises using model.state_dict() (NOT model.parameters())
   - Trains for 1 local epoch on the trivial data
   - Evaluates and returns loss + accuracy
5. Run fl.simulation.start_simulation (or the current equivalent) with:
   - 2 clients
   - 1 communication round
   - FedAvg strategy with a centralised evaluate function
   - client_resources={"num_cpus": 1, "num_gpus": 0.0}
6. After simulation completes, print:
   - "Flower version: X.Y.Z"
   - "Simulation API used: fl.simulation.start_simulation" (or whatever worked)
   - "Round 1 metrics: {accuracy: ..., loss: ...}"
   - "SMOKE TEST PASSED"

If the start_simulation API has changed in the installed version, adapt to
whatever the current Flower docs recommend. The goal is to confirm the exact
API calls that work, so we can use them in all subsequent prompts.

Also verify that state_dict serialisation round-trips correctly:
   - Get state dict, convert to numpy, convert back to state dict, load into model
   - Assert all parameters are identical after round-trip (torch.allclose)

Print the exact import paths and function signatures that worked, so we can
copy them into the real client/server code.
```

---

### Prompt 3: Data Download Utilities 🖥️ LOCAL

```
In `data/download.py`, implement dataset loading functions.

IMPORTANT: Do NOT hard-code class names anywhere. Load them from medmnist:

    from medmnist import INFO
    info = INFO["dermamnist"]
    class_names = [info["label"][str(i)] for i in range(len(info["label"]))]

1. `load_mnist()` -> returns (train_dataset, test_dataset) as PyTorch datasets
   Transforms: ToTensor(), Normalize((0.1307,), (0.3081,))

2. `load_dermamnist(size=28)` -> returns (train_dataset, val_dataset, test_dataset)
   - size must be in [28, 64, 128, 224]. Assert this.
   - Use medmnist.DermaMNIST(split=..., download=True, size=size, ...)
   - Apply ToTensor()
   - For size 224: normalise with ImageNet stats (mean=[0.485,0.456,0.406],
     std=[0.229,0.224,0.225])
   - For size 28/64/128: normalise with DermaMNIST computed stats or ImageNet stats
     (consistent choice; document which)
   - CRITICAL: MedMNIST labels have shape (N, 1). Squeeze to (N,) so they work
     with standard CrossEntropyLoss. Do this in a wrapper or transform.

3. `get_class_distribution(dataset)` -> dict {class_idx: count}
   Works with any dataset that has .targets or .labels attribute.
   For MedMNIST, labels are in dataset.labels (numpy array).

4. `get_dataset_info(dataset_name)` -> dict with:
   - num_classes
   - class_names (loaded from medmnist INFO, not hard-coded)
   - input_channels
   - supported_sizes

Include `if __name__ == "__main__"`:
- Download DermaMNIST (28) and MNIST
- Print class distributions with names
- Print: "DermaMNIST — Train: 7007, Val: 1003, Test: 2005" (verify these numbers)
- Print imbalance ratio: max_class_count / min_class_count
- Report melanoma count explicitly (clinically important class)
```

---

### Prompt 4: Data Partitioning 🖥️ LOCAL

```
In `data/partition.py`, implement federated data partitioning strategies.

Each function takes a dataset and number of clients K, returns List[List[int]]
— a list of K lists of sample indices.

All functions must satisfy these invariants (enforced by assertions at the end):
- Every sample index appears exactly once across all clients
- No duplicate indices within or across clients
- Every client has at least 1 sample
- len(all_indices) == len(dataset)

1. `iid_partition(dataset, num_clients, seed=42)`:
   - Shuffle all indices, split evenly. Remainder distributed round-robin.

2. `dirichlet_partition(dataset, num_clients, alpha, min_samples_per_client=10,
                        seed=42, max_retries=100)`:
   - Extract labels from dataset (handle both .targets and .labels attributes)
   - For each class c, draw proportions p_c ~ Dir(alpha * ones(num_clients))
   - Assign samples of class c to clients according to p_c
   - RETRY LOGIC: if any client has fewer than min_samples_per_client total samples,
     resample with a new random draw. Retry up to max_retries times.
   - If still failing after max_retries, raise ValueError with message:
     "Could not satisfy min_samples={min_samples_per_client} with alpha={alpha}
      and {num_clients} clients after {max_retries} retries. Try larger alpha,
      fewer clients, or smaller min_samples."
   - Run the invariant assertions at the end.

3. `pathological_partition(dataset, num_clients, classes_per_client=2, seed=42)`:
   - Assign each client exactly classes_per_client classes
   - For 10 clients and 7 classes with k=2: some classes assigned to more clients.
     Distribute class assignments as evenly as possible (each class appears in
     floor(k*K/C) or ceil(k*K/C) clients).
   - WARN if a rare class (fewer than 50 samples) is assigned to more than 2 clients.
   - Save and return the class-client assignment as a separate matrix.
   - Split each class's samples evenly among its assigned clients.

4. `quantity_skew_partition(dataset, num_clients, alpha=0.5, seed=42)`:
   - Draw client sizes from Dir(alpha * ones(num_clients)) * len(dataset)
   - Round to integers, ensure sum == len(dataset)
   - Assign samples randomly (IID labels, unequal sizes)

5. `dirichlet_quantity_partition(dataset, num_clients, label_alpha=0.5,
                                 quantity_alpha=0.5, seed=42)`:
   - Combine label skew AND quantity skew
   - First assign unequal quantities, then apply Dirichlet label skew within
     each client's allocation
   - This is the most realistic scenario

6. Helpers:
   - `get_client_class_distribution(dataset, client_indices)` -> dict {class: count}
   - `get_all_client_distributions(dataset, all_client_indices)` -> pd.DataFrame
     (rows=clients, columns=classes)
   - `compute_distribution_entropy(client_distribution)` -> float
     Shannon entropy of the client's normalised class distribution

Include type hints. Docstrings with parameter explanations. Use numpy for randomness.
```

---

### Prompt 5: Partition Unit Tests 🖥️ LOCAL

```
Create `tests/test_partition.py` with comprehensive unit tests for all partitioning
functions. Use pytest.

Tests for EVERY partition function (iid, dirichlet, pathological, quantity_skew,
dirichlet_quantity):

1. test_all_samples_assigned:
   - Total indices across all clients == len(dataset)

2. test_no_duplicates:
   - No index appears more than once across all clients

3. test_all_clients_nonempty:
   - Every client has at least 1 sample

4. test_class_totals_preserved:
   - Sum of per-client class counts == global class counts for every class

5. test_reproducibility:
   - Same seed produces identical partitions on two calls

Dirichlet-specific tests:

6. test_dirichlet_high_alpha_approx_iid:
   - With alpha=100, client distributions should be roughly uniform
   - Use chi-squared test or check max deviation from uniform < threshold

7. test_dirichlet_low_alpha_heterogeneous:
   - With alpha=0.1, at least some clients should have >60% of samples from one class
   - More heterogeneous than alpha=1.0 (compare entropy)

8. test_dirichlet_min_samples_enforced:
   - With min_samples_per_client=10, every client has >= 10 samples

9. test_dirichlet_impossible_raises:
   - With num_clients=1000, alpha=0.001, min_samples=100, should raise ValueError

Pathological-specific tests:

10. test_pathological_class_count:
    - Each client has exactly classes_per_client distinct classes

11. test_pathological_rare_class_warning:
    - With 7 classes, 10 clients, k=2: check that rare-class warnings trigger

Use a small synthetic dataset for speed:
    Create a dataset with 1000 samples, 7 classes with counts [400, 150, 120, 30, 130, 100, 70]
    (mimics DermaMNIST proportions at 1/10 scale)

Run: pytest tests/test_partition.py -v
All tests must pass.
```

---

### Prompt 6: Data Distribution Visualisation 🖥️ LOCAL

```
In `data/visualise.py`, implement visualisation functions. All functions save to a
path AND return the matplotlib figure.

1. `plot_global_class_distribution(dataset, class_names, save_path)`:
   - Horizontal bar chart, bars annotated with count and percentage
   - Colour intensity proportional to frequency

2. `plot_client_distributions(client_distributions_df, class_names, save_path, title="")`:
   - Stacked bar chart: x=clients, y=samples, segments=classes
   - Use tab10 palette. Legend with class names.

3. `plot_client_heatmap(client_distributions_df, class_names, save_path, title="")`:
   - Seaborn heatmap: rows=clients, columns=classes
   - Normalise row-wise (proportions). Annotate with raw counts.

4. `plot_partition_comparison(distributions_dict, class_names, save_path)`:
   - Takes {"IID": df, "Dir(0.1)": df, "Dir(0.5)": df, ...}
   - Grid of heatmaps, one per strategy, as subplots

5. `plot_sample_images(dataset, class_names, num_per_class=5, save_path=None)`:
   - Grid: rows=classes, columns=samples. Title each row with class name and count.

Use matplotlib + seaborn. DPI 150. tight_layout(). Optional figsize parameter.
Use a colourblind-friendly palette throughout. Close figures after saving to avoid
memory leaks (plt.close(fig)).
```

---

### Prompt 7: Model Architectures 🖥️ LOCAL

```
Implement model architectures.

File: models/simple_cnn.py

class SimpleCNN(nn.Module):
    """Small CNN for 28×28 or 64×64 images."""
    def __init__(self, in_channels=3, num_classes=7):
        Architecture:
        - Conv2d(in_channels, 32, 3, padding=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2)
        - Conv2d(32, 64, 3, padding=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2)
        - Conv2d(64, 128, 3, padding=1) -> BatchNorm2d(128) -> ReLU -> AdaptiveAvgPool2d(1)
        - Flatten -> Linear(128, 64) -> ReLU -> Dropout(0.5) -> Linear(64, num_classes)
    Forward returns logits (no softmax).

File: models/resnet.py

def get_resnet18(in_channels=3, num_classes=7, pretrained=False, image_size=28):
    """
    Returns a ResNet-18 adapted to the input resolution.

    CRITICAL for small images (28, 64):
      Replace the ImageNet stem (7×7 conv, stride 2, maxpool) with a CIFAR-style stem:
        Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        Replace maxpool with nn.Identity()
      This prevents over-aggressive downsampling on small images.

    For 224×224 (pretrained experiments only):
      Keep the standard ImageNet stem.
      If in_channels != 3, replace conv1 but copy weights for the 3 overlapping channels.

    For pretrained weights, use the modern torchvision API:
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
      Do NOT use the deprecated pretrained=True kwarg.

    Always replace the final fc:
        model.fc = nn.Linear(512, num_classes)
    """

File: models/__init__.py

def get_model(model_name, in_channels, num_classes, image_size=28, pretrained=False):
    """Factory function."""
    if model_name == "simple_cnn":
        return SimpleCNN(in_channels, num_classes)
    elif model_name == "resnet18":
        return get_resnet18(in_channels, num_classes, pretrained, image_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

Include a smoke test in `if __name__ == "__main__"`:
- Instantiate SimpleCNN(3, 7) and pass a random (1, 3, 28, 28) tensor, verify output shape (1, 7)
- Instantiate get_resnet18(3, 7, image_size=28) and pass (1, 3, 28, 28), verify output shape
- Instantiate get_resnet18(3, 7, image_size=64) and pass (1, 3, 64, 64), verify output shape
- Print parameter counts for each
```

---

### Prompt 8: Evaluation Metrics 🖥️ LOCAL

```
In `metrics/evaluation.py`, implement evaluation utilities.

1. `evaluate_model(model, dataloader, device, num_classes)`:
   Returns a dict with:
   - "loss": average CE loss
   - "accuracy": overall accuracy
   - "balanced_accuracy": sklearn.metrics.balanced_accuracy_score
   - "per_class_f1": list of length num_classes
   - "macro_f1": macro-averaged F1
   - "per_class_recall": list of length num_classes
   - "per_class_precision": list of length num_classes
   - "per_class_accuracy": list of length num_classes
   - "worst_class_recall": min of per_class_recall
   - "worst_class_f1": min of per_class_f1
   - "confusion_matrix": numpy array (num_classes × num_classes)

   For AUROC:
   - Compute softmax probabilities and attempt macro AUROC (one-vs-rest)
   - If any class is missing from the batch, set auroc to np.nan and log a warning
   - Include "macro_auroc" and "present_classes" (list of class indices that appeared)

   IMPORTANT: Do NOT compute "worst_class_accuracy" — it is misleading under
   class imbalance. Use worst_class_recall and worst_class_f1 instead.

   Use torch.no_grad(). Collect all predictions and targets first, then compute.

2. `plot_confusion_matrix(cm, class_names, save_path, title="", normalize=True)`:
   Seaborn heatmap. Option for normalised (row-wise percentages) or raw counts.

3. `plot_per_class_metrics(metrics_dict, class_names, save_path, title="")`:
   Grouped bar chart: F1, recall, precision per class.
   Highlight clinically important class (Melanoma) with a marker or annotation.

4. `plot_training_curves(history_df, save_path, title="")`:
   Takes a DataFrame with columns: round, accuracy, balanced_accuracy, loss
   Dual y-axis: accuracy (left), loss (right).

5. Class MetricsLogger:
   - __init__(self, save_dir, class_names)
   - log_round(self, round_num, metrics_dict): append to internal list
   - save(self): write to metrics_history.csv
   - get_dataframe(self): return pd.DataFrame
   - plot_all(self): generate training curves, per-class plots
```

---

### Prompt 9: Loss Functions 🖥️ LOCAL

```
Implement custom loss functions.

File: losses/focal_loss.py

class FocalLoss(nn.Module):
    """Focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
    Focuses learning on hard (minority) examples by down-weighting easy ones."""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        - alpha: optional class weights tensor of shape (num_classes,), or None for uniform
        - gamma: focusing parameter (0 = standard CE, higher = more focus on hard examples)

    def forward(self, logits, targets):
        - logits: (N, C) raw logits
        - targets: (N,) class indices
        - Compute using log_softmax for numerical stability
        - Apply alpha weighting per class if provided
        - Apply focal modulation: (1 - p_t)^gamma

File: losses/weighted_ce.py

def compute_class_weights(class_counts, strategy="inverse"):
    """
    Compute class weights from a dict or array of class counts.

    Strategies:
    - "inverse": weight_c = total / (num_classes * count_c)
    - "sqrt_inverse": weight_c = sqrt(max_count / count_c)
    - "effective": weight_c = (1 - beta) / (1 - beta^count_c), beta=0.9999

    CRITICAL: Handle zero counts safely.
      counts = np.maximum(counts, 1)  # avoid division by zero
    For absent classes (count=0), set weight to 0.0 — the client cannot learn
    that class anyway.

    Returns: torch.FloatTensor of shape (num_classes,)
    """

def compute_local_class_weights(dataset, client_indices, num_classes, strategy="inverse"):
    """Compute weights based on THIS CLIENT's local class distribution."""

def compute_global_estimated_weights(all_client_distributions_df, num_classes, strategy="inverse"):
    """
    Estimate global class distribution by summing client-reported class counts.
    NOTE: This requires clients to share aggregate class counts rather than raw
    images. This is lower-risk metadata sharing but still reveals label-distribution
    information. Document this privacy trade-off.
    """

Include unit tests in `if __name__ == "__main__"`:
- Compute weights for DermaMNIST-like distribution [327, 514, 1099, 115, 1113, 6705, 142]
- Verify: minority classes (DF, VASC) get highest weights
- Verify: class with zero count gets weight 0
- Verify focal loss with gamma=0 matches standard CE
```

---

### Prompt 10: Centralised Training Loop 🖥️+GPU

```
In `trainers/centralised.py`, implement a centralised (non-FL) training loop.

class CentralisedTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 device, num_classes, lr=1e-3, weight_decay=1e-4,
                 loss_fn="ce", class_weights=None, focal_gamma=2.0):
        - Store all parameters
        - Loss selection:
          "ce": nn.CrossEntropyLoss(weight=class_weights) if class_weights else nn.CrossEntropyLoss()
          "weighted_ce": nn.CrossEntropyLoss(weight=class_weights) — require class_weights
          "focal": FocalLoss(alpha=class_weights, gamma=focal_gamma)
        - Optimiser: Adam(lr=lr, weight_decay=weight_decay)
        - Scheduler: ReduceLROnPlateau(patience=5, factor=0.5, mode='max') monitoring val balanced_accuracy

    def train_epoch(self):
        Standard training loop for one epoch. Returns avg training loss.

    def evaluate(self, dataloader):
        Calls evaluate_model from metrics/. Returns metrics dict.

    def train(self, num_epochs, save_dir, class_names=None):
        - Train for num_epochs, evaluate val set each epoch
        - Track best model by balanced_accuracy on val
        - Save best checkpoint: {save_dir}/best_model.pt
        - At the end, load best model, evaluate on test set
        - Log all metrics using MetricsLogger
        - Print progress with tqdm (epoch, train_loss, val_bacc, val_worst_f1)
        - Return: {"test_metrics": ..., "history": ..., "best_epoch": ...}
```

---

### Prompt 11: Centralised Baselines 🖥️+GPU (short runs) / 🔥 HPC (full 50 epochs)

```
Create two scripts:

1. `experiments/run_centralised_mnist.py`:
   - Load MNIST, create DataLoaders (batch=64, shuffle train)
   - Train SimpleCNN(in_channels=1, num_classes=10) for 10 epochs, lr=1e-3
   - Save to results/centralised/mnist/
   - Print final test accuracy
   - Sanity check: should converge smoothly. Do NOT assert a specific accuracy number.

2. `experiments/run_centralised_dermamnist.py`:
   - Load DermaMNIST (size=28)
   - Load class_names from medmnist INFO (not hard-coded)
   - Run three separate training runs:
     a. Standard CE — save to results/centralised/dermamnist/ce/
     b. Weighted CE (inverse freq weights) — save to results/centralised/dermamnist/weighted_ce/
     c. Focal loss (gamma=2) — save to results/centralised/dermamnist/focal/
   - For each: 50 epochs, lr=1e-3, batch=64
   - After all three, generate comparison plots:
     - Per-class F1 grouped bar chart (3 bars per class)
     - Per-class recall grouped bar chart
     - Confusion matrices (3 panels)
     - Training curves overlay
   - Save comparison to results/centralised/dermamnist/comparison/
   - Print summary table: loss_type | accuracy | bACC | macro_f1 | worst_f1 | melanoma_recall

Both scripts accept: --device (cuda/cpu), --seed (default 42), --epochs, --batch_size, --lr
Call set_all_seeds(seed) at the start.
```

---

### Prompt 12: DermaMNIST Exploration Script 🖥️ LOCAL

```
Create `experiments/run_exploration.py` — full exploratory analysis of DermaMNIST.

1. Load DermaMNIST (size=28). Print dataset sizes (train/val/test).
   Verify: 7007 / 1003 / 2005

2. Class distribution:
   - Print exact counts and percentages per class (names from medmnist INFO)
   - Compute imbalance ratio: max_count / min_count
   - Print melanoma count and percentage explicitly

3. Sample visualisation:
   - 7 rows × 8 columns grid of random samples, one row per class
   - Title each row: "ClassName (N=count)"
   - Save as results/exploration/dermamnist_samples.png

4. Partition previews — for 10 clients, generate and save:
   - IID
   - Dirichlet(0.1), Dirichlet(0.3), Dirichlet(0.5), Dirichlet(1.0)
   - Pathological(k=2)
   For each: save stacked bar chart + heatmap
   Save a combined 2×3 grid of heatmaps (one per strategy)

5. For each partition, also print:
   - Number of clients with zero samples of Dermatofibroma
   - Number of clients with zero samples of Vascular Lesions
   - Min and max client dataset sizes
   - Mean client distribution entropy

Save all to results/exploration/
```

---

## PHASE 1 — Core Federated Learning Pipeline

---

### Prompt 13: Flower Client 🖥️ LOCAL

```
In `client/flower_client.py`, implement the Flower FL client.

IMPORTANT — Use the EXACT Flower API calls that worked in Prompt 2's smoke test.
If the smoke test used different imports or function signatures, use those here.

CRITICAL — Model serialisation MUST use model.state_dict(), NOT model.parameters().
This is essential because our SimpleCNN uses BatchNorm, which has running_mean and
running_var buffers that model.parameters() does not include. Ignoring these creates
silent inconsistency between clients.

Utility functions (put at top of file):

def get_state_dict_as_numpy(model):
    """Convert model state_dict (parameters AND buffers) to list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_state_dict_from_numpy(model, np_arrays):
    """Load numpy arrays into model state_dict. Handles all entries including
    BatchNorm running_mean, running_var, num_batches_tracked."""
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    assert len(keys) == len(np_arrays), (
        f"Parameter count mismatch: model has {len(keys)}, received {len(np_arrays)}"
    )
    new_state_dict = {}
    for key, np_arr in zip(keys, np_arrays):
        new_state_dict[key] = torch.tensor(np_arr, dtype=state_dict[key].dtype)
    model.load_state_dict(new_state_dict, strict=True)


class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device,
                 num_classes, local_epochs=1, lr=0.01, momentum=0.9,
                 loss_fn="ce", class_weights=None, focal_gamma=2.0,
                 proximal_mu=0.0):
        """
        proximal_mu: if > 0, adds FedProx proximal term to the local objective.
        FedProx is a CLIENT-SIDE modification. The server still uses standard
        FedAvg aggregation. FedProx loss:
            L_total = L_task + (mu / 2) * ||w_local - w_global||^2
        """
        - Store parameters
        - Set up loss: CE, weighted CE (with class_weights), or focal
        - Optimiser: SGD(model.parameters(), lr=lr, momentum=momentum)

    def get_parameters(self, config):
        return get_state_dict_as_numpy(self.model)

    def set_parameters(self, parameters):
        set_state_dict_from_numpy(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # If proximal_mu > 0, store a COPY of global parameters for the proximal term
        if self.proximal_mu > 0:
            global_params = [p.clone().detach() for p in self.model.parameters()]

        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for epoch in range(self.local_epochs):
            for batch in self.train_loader:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                # Handle MedMNIST label shape
                if labels.dim() > 1:
                    labels = labels.squeeze()
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                # FedProx proximal term
                if self.proximal_mu > 0:
                    proximal_term = 0.0
                    for local_p, global_p in zip(self.model.parameters(), global_params):
                        proximal_term += ((local_p - global_p) ** 2).sum()
                    loss = loss + (self.proximal_mu / 2.0) * proximal_term

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"train_loss": avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = evaluate_model(self.model, self.val_loader, self.device, self.num_classes)
        return metrics["loss"], len(self.val_loader.dataset), {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "worst_class_f1": metrics["worst_class_f1"],
        }


def create_client_fn(model_fn, train_loaders, val_loaders, device, num_classes, config):
    """Factory for Flower simulation. Returns a function cid -> Client."""
    def client_fn(cid: str):
        cid_int = int(cid)
        model = model_fn().to(device)
        return FLClient(
            model=model,
            train_loader=train_loaders[cid_int],
            val_loader=val_loaders[cid_int],
            device=device,
            num_classes=num_classes,
            local_epochs=config.get("local_epochs", 5),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9),
            loss_fn=config.get("loss_fn", "ce"),
            class_weights=config.get("class_weights", None),
            proximal_mu=config.get("proximal_mu", 0.0),
        ).to_client()
    return client_fn
```

---

### Prompt 14: Flower Server & Strategy with Checkpointing 🖥️ LOCAL

```
In `server/flower_server.py`, implement the server-side logic.

IMPORTANT: FedProx is NOT a separate server-side strategy. FedProx modifies the
client's local objective (adding the proximal term). The server aggregation is
still standard FedAvg. The config distinguishes them:

    aggregation:
      strategy: fedavg       # server-side: always FedAvg for core experiments

    client_objective:
      proximal_mu: 0.0       # 0.0 = pure FedAvg, >0 = FedProx


1. Custom strategy with checkpointing:

class SaveModelFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that stores the latest and best global model parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters = None
        self.best_parameters = None
        self.best_balanced_accuracy = -1.0

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
        return aggregated_parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        # After evaluation, check if this is the best round
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        # Note: centralised evaluate metrics will be checked separately
        return loss, metrics

    def update_best(self, balanced_accuracy):
        """Call this from the centralised evaluate fn to track best model."""
        if balanced_accuracy > self.best_balanced_accuracy:
            self.best_balanced_accuracy = balanced_accuracy
            self.best_parameters = self.latest_parameters


2. Centralised evaluate function:

def create_centralised_evaluate_fn(model_fn, test_loader, device, num_classes,
                                    class_names, strategy_ref):
    """
    Returns a function for Flower's evaluate_fn that evaluates the global model
    on the full test set after each round.

    strategy_ref: reference to SaveModelFedAvg instance so we can call update_best.
    """
    def evaluate(server_round, parameters_ndarrays, config):
        model = model_fn().to(device)
        set_state_dict_from_numpy(model, parameters_ndarrays)
        metrics = evaluate_model(model, test_loader, device, num_classes)

        # Track best model
        strategy_ref.update_best(metrics["balanced_accuracy"])

        # Build per-class metric keys for Flower history
        result_metrics = {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "worst_class_f1": metrics["worst_class_f1"],
            "worst_class_recall": metrics["worst_class_recall"],
        }
        for i, name in enumerate(class_names):
            short_name = name[:20].replace(" ", "_")
            result_metrics[f"f1_{short_name}"] = metrics["per_class_f1"][i]
            result_metrics[f"recall_{short_name}"] = metrics["per_class_recall"][i]

        return metrics["loss"], result_metrics

    return evaluate


3. Simulation runner:

def run_simulation(config):
    """
    Run a complete FL experiment from a config dict.
    Returns: (History, final_global_metrics, per_client_metrics)
    """
    Steps:
    a. Load dataset, partition, create client DataLoaders (90/10 train/val per client)
    b. Create global test DataLoader
    c. Create per-client test DataLoaders (each client's val portion, for client-level eval)
    d. Instantiate SaveModelFedAvg with:
       - fraction_fit, fraction_evaluate from config
       - min_fit_clients = num_clients * fraction_fit
       - evaluate_fn = create_centralised_evaluate_fn(...)
    e. Run Flower simulation
    f. Save training history
    g. CRITICAL — Per-client evaluation of final global model:
       After simulation, load the best global model parameters.
       For each client, evaluate the global model on that client's local test data.
       Save per_client_metrics.csv with columns:
         client_id, num_train_samples, num_test_samples, local_class_distribution,
         accuracy, balanced_accuracy, macro_f1, worst_class_f1, per_class_recall_json
       Also compute and save:
         - average client balanced_accuracy
         - worst-client balanced_accuracy
         - best-client / worst-client gap
         - client balanced_accuracy standard deviation
    h. Return everything

Accept --debug_subset N (use only N training samples for fast iteration)
Accept --num_rounds_override M (override config's num_rounds)
Accept --dry_run (parse config, partition data, print summary, exit without training)
```

---

### Prompt 15: YAML Configs 🖥️ LOCAL

```
Create YAML configs for all core experiments.

IMPORTANT design: FedProx is a client_objective setting, NOT a separate strategy.

Base config `configs/base.yaml`:

dataset:
  name: dermamnist
  size: 28

model:
  name: simple_cnn
  in_channels: 3
  num_classes: 7

federation:
  num_clients: 10
  num_rounds: 100
  local_epochs: 5
  fraction_fit: 1.0

training:
  lr: 0.01
  momentum: 0.9
  batch_size: 64
  weight_decay: 0.0

loss:
  type: ce
  focal_gamma: 2.0
  weight_strategy: none

client_objective:
  proximal_mu: 0.0

partition:
  strategy: dirichlet
  alpha: 0.5
  classes_per_client: 2  # only for pathological

misc:
  seed: 42
  device: cuda
  save_dir: results/

Create `utils/config_loader.py`:
- load_config(path): load YAML, merge with base.yaml defaults
- merge_configs(base, override): deep merge, override wins
- config_to_experiment_name(config): generate descriptive name like
  "fedavg_dir05_E5_s42" or "fedprox_mu001_dir01_E5_s42"

Core experiment configs (each overrides base):

configs/fedavg_iid.yaml          — partition.strategy: iid
configs/fedavg_dir10.yaml        — partition.alpha: 1.0
configs/fedavg_dir05.yaml        — partition.alpha: 0.5
configs/fedavg_dir03.yaml        — partition.alpha: 0.3
configs/fedavg_dir01.yaml        — partition.alpha: 0.1
configs/fedavg_pathological.yaml — partition.strategy: pathological
configs/fedprox_dir10.yaml       — client_objective.proximal_mu: 0.01, partition.alpha: 1.0
configs/fedprox_dir05.yaml       — client_objective.proximal_mu: 0.01, partition.alpha: 0.5
configs/fedprox_dir03.yaml       — client_objective.proximal_mu: 0.01, partition.alpha: 0.3
configs/fedprox_dir01.yaml       — client_objective.proximal_mu: 0.01, partition.alpha: 0.1
configs/fedprox_pathological.yaml — client_objective.proximal_mu: 0.01, partition.strategy: pathological
```

---

### Prompt 16: FL MNIST Validation 🖥️+GPU

```
Create `experiments/run_fl_mnist.py` to validate the FL pipeline on MNIST
before running expensive DermaMNIST experiments.

Run 3 experiments sequentially:

Exp 1: FedAvg + IID
  - 5 clients, IID, 30 rounds, 5 local epochs, lr=0.01, proximal_mu=0

Exp 2: FedAvg + Non-IID
  - 5 clients, Dirichlet(0.5), 30 rounds, 5 local epochs, lr=0.01, proximal_mu=0

Exp 3: FedProx + Non-IID
  - 5 clients, Dirichlet(0.5), 30 rounds, 5 local epochs, lr=0.01, proximal_mu=0.01

For each:
  - Save client distribution plots
  - Save accuracy/loss curves
  - Save final metrics (global test + per-client)

After all 3:
  - Overlay accuracy curves on one plot
  - Print comparison table
  - Save to results/fl_mnist/

Expected sanity checks (NOT hard numbers — these depend on hyperparameters):
  - IID FedAvg converges smoothly and approaches centralised performance
  - Non-IID FedAvg converges but is less stable, may plateau lower
  - FedProx should be at least as stable as FedAvg on non-IID (not worse)
  - None should diverge. If any diverges, there is a bug.

Accept: --device, --seed, --debug (2 clients, 3 rounds for fast check)
```

---

### Prompt 17: Unified Experiment Runner 🖥️+GPU (debug) / 🔥 HPC (full)

```
Create `experiments/run_experiment.py` — the main entry point for all FL experiments.

Usage:
  python experiments/run_experiment.py --config configs/fedavg_dir05.yaml
  python experiments/run_experiment.py --config configs/fedavg_dir05.yaml --seed 123
  python experiments/run_experiment.py --config configs/fedavg_dir05.yaml --debug_subset 500 --num_rounds_override 3
  python experiments/run_experiment.py --config configs/fedavg_dir05.yaml --dry_run

The script:

1. Parse args: --config (required), --seed (override), --device (override),
   --debug_subset N, --num_rounds_override M, --dry_run

2. Load config with config_loader, apply overrides

3. set_all_seeds(seed)

4. Load dataset, partition data, create DataLoaders

5. Save data distribution plots to {save_dir}/data_distributions/

6. If --dry_run: print config summary, partition stats, exit

7. Create SaveModelFedAvg strategy with centralised evaluate fn

8. Run Flower simulation with client_resources:
   - If device == "cuda": {"num_cpus": 1, "num_gpus": 0.1}  (10 clients share 1 GPU)
   - If device == "cpu": {"num_cpus": 1, "num_gpus": 0.0}

9. After training, evaluate global model on:
   a. Pooled global test set — save global_test_metrics.json
   b. Each client's local test set — save per_client_metrics.csv
   Compute: avg client bACC, worst client bACC, best-worst gap, client bACC stdev

10. Save to {save_dir}/{experiment_name}/:
    - config.yaml (copy of config used)
    - metrics_history.csv (round-by-round: global accuracy, bACC, loss, per-class F1/recall)
    - global_test_metrics.json
    - per_client_metrics.csv
    - confusion_matrix.png
    - per_class_f1_bar.png
    - accuracy_curve.png, loss_curve.png
    - per_class_convergence.png (7 lines, one per class)
    - best_model.pt

11. Print formatted summary:
    Experiment: fedavg_dir05_E5_s42
    Global: bACC=0.XXX | Macro F1=0.XXX | Worst F1=0.XXX | Melanoma Recall=0.XXX
    Clients: Avg bACC=0.XXX | Worst bACC=0.XXX | Gap=0.XXX
    Completed in XX:XX minutes


Also create `experiments/run_all_core.py`:
- Loop over all configs in configs/ that start with "fedavg_" or "fedprox_"
- For each, run with seeds [42, 123, 456]
- After all complete, generate master_comparison.csv with mean ± std
- Generate master_comparison.tex (LaTeX booktabs table)
- Handle errors: if one experiment fails, log error and continue
```

---

### Prompt 18: Per-Class Convergence Tracker 🖥️ LOCAL (analysis of saved results)

```
Create `metrics/per_class_tracker.py` to track and visualise per-class performance
across communication rounds.

class PerClassTracker:
    def __init__(self, num_classes, class_names, save_dir):
        self.data = []  # list of dicts

    def log_round(self, round_num, metrics_dict):
        """Extract per-class metrics from the Flower centralised evaluate metrics.
        Keys like 'f1_Melanocytic_Nevi', 'recall_Dermatofibroma', etc."""
        record = {"round": round_num}
        for i, name in enumerate(self.class_names):
            short = name[:20].replace(" ", "_")
            record[f"f1_{short}"] = metrics_dict.get(f"f1_{short}", np.nan)
            record[f"recall_{short}"] = metrics_dict.get(f"recall_{short}", np.nan)
        record["balanced_accuracy"] = metrics_dict.get("balanced_accuracy", np.nan)
        record["worst_class_f1"] = metrics_dict.get("worst_class_f1", np.nan)
        self.data.append(record)

    def save(self, path=None):
        """Save to per_class_history.csv"""

    def plot_per_class_f1_curves(self, save_path):
        """One line per class. x=round, y=F1.
        Use THICKER dashed lines for minority classes (Dermatofibroma, Vascular Lesions).
        Use solid lines for majority classes.
        Distinct colours per class (colourblind-safe palette)."""

    def plot_per_class_recall_curves(self, save_path):
        """Same as above but for recall.
        Annotate Melanoma recall with a special marker (clinically important)."""

    def plot_convergence_heatmap(self, metric="f1", save_path=None):
        """Heatmap: x=rounds (sampled every 5 or 10 rounds), y=classes.
        Shows which classes converge early vs late."""

    def compute_convergence_round(self, threshold=0.3, metric="f1"):
        """For each class, find first round where metric >= threshold.
        Returns dict {class_name: round_number_or_None}.
        Useful for quantifying minority-class convergence delay."""

Integrate into run_experiment.py: create a PerClassTracker, populate it from
the Flower History object's centralised metrics, save outputs alongside other results.
```

---

## PHASE 2 — Class Imbalance Mitigation

---

### Prompt 19: Client-Side Mitigation Strategies 🖥️ LOCAL (code) / 🔥 HPC (experiments)

```
Add client-side class imbalance mitigation, toggled via config.

1. Update config schema:

loss:
  type: ce                # ce | weighted_ce | focal
  weight_strategy: none   # none | local_inverse | local_sqrt | global_estimated
  focal_gamma: 2.0

augmentation:
  use_weighted_sampler: false   # Use WeightedRandomSampler instead of random shuffle
  # NOTE: do NOT implement heavy minority oversampling yet. Start with weighted sampler.


2. In `data/augmentation.py`:

def create_weighted_sampler(dataset, client_indices, num_classes):
    """
    Create a WeightedRandomSampler that oversamples minority classes.
    Weight per sample = 1 / count_of_its_class.
    Returns a sampler to pass to DataLoader(sampler=...).

    This is MUCH safer than duplicating minority samples because:
    - No risk of extreme overfitting on duplicated samples
    - Naturally integrates with DataLoader
    - Each epoch sees a different random draw
    """


3. Update flower_client.py to apply loss variants:
   - If config loss.weight_strategy != "none":
     Compute class weights using the chosen strategy from losses/weighted_ce.py
     Pass them to the loss function
   - If config loss.type == "focal":
     Use FocalLoss with the computed weights as alpha and configured gamma
   - If config augmentation.use_weighted_sampler:
     Replace the train DataLoader's default sampler with the weighted sampler


4. Create experiment configs:

configs/mitigation/fedavg_dir05_weighted.yaml   — FedAvg + Dir(0.5) + local_inverse weights
configs/mitigation/fedavg_dir05_focal.yaml      — FedAvg + Dir(0.5) + focal(γ=2)
configs/mitigation/fedavg_dir05_sampler.yaml     — FedAvg + Dir(0.5) + weighted sampler
configs/mitigation/fedprox_dir05_weighted.yaml  — FedProx + Dir(0.5) + local_inverse weights
configs/mitigation/fedprox_dir05_focal.yaml     — FedProx + Dir(0.5) + focal(γ=2)
configs/mitigation/fedprox_dir05_sampler.yaml    — FedProx + Dir(0.5) + weighted sampler
configs/mitigation/fedprox_dir05_combined.yaml  — FedProx + Dir(0.5) + focal + sampler
```

---

### Prompt 20: Logit Adjustment 🖥️ LOCAL (code) / 🔥 HPC (experiments)

```
Implement logit adjustment for training-time and post-hoc correction.

File: losses/logit_adjustment.py

IMPORTANT: Be explicit about the formula. Do NOT let the sign be ambiguous.

class LogitAdjustedCrossEntropy(nn.Module):
    """
    Training-time logit adjustment (Menon et al., 2020).

    The standard CE loss is: L = -log(softmax(z)_y)
    Logit-adjusted loss modifies logits before softmax:
        z_adjusted_c = z_c - tau * log(pi_c)
    where pi_c is the prior probability of class c and tau is a temperature.

    Effect: subtracting log(pi_c) penalises the model less for predicting
    frequent classes, effectively raising the bar for majority classes and
    lowering it for minority classes.

    Formula:
        L_LA = CrossEntropy(z - tau * log(pi), y)
    """
    def __init__(self, class_priors, tau=1.0):
        """
        class_priors: tensor of shape (num_classes,) — estimated P(Y=c).
                      Must sum to ~1.0 and all entries > 0.
        tau: temperature. tau=0 recovers standard CE. tau=1 is standard adjustment.
        """
        super().__init__()
        assert (class_priors > 0).all(), "All priors must be positive"
        # The adjustment is SUBTRACTED from logits
        self.register_buffer("adjustment", tau * torch.log(class_priors + 1e-12))

    def forward(self, logits, targets):
        adjusted_logits = logits - self.adjustment  # SUBTRACT log-priors
        return F.cross_entropy(adjusted_logits, targets)


def post_hoc_logit_adjustment(logits, class_priors, tau=1.0):
    """
    Apply logit adjustment at inference time without retraining.
    adjusted_logits = logits - tau * log(class_priors)
    Then take argmax of adjusted_logits for predictions.

    This changes predicted labels (unlike temperature scaling).
    """
    adjustment = tau * torch.log(class_priors + 1e-12)
    return logits - adjustment


UNIT TESTS (in if __name__ == "__main__"):

1. Create toy logits: [[5.0, 1.0, 1.0]] (model confidently predicts class 0)
   Priors: [0.8, 0.1, 0.1] (class 0 is majority)
   After adjustment: logits[0] decreases most (subtract log(0.8) ≈ 0.22),
   logits[1] and [2] decrease less (subtract log(0.1) ≈ 2.30)
   Net effect: class 0's logit goes from 5.0 to ~4.78, class 1 goes from 1.0 to ~3.30
   Verify that adjustment helps minority classes.

2. With tau=0: adjusted logits == original logits (no adjustment)

3. With uniform priors [1/3, 1/3, 1/3]: adjustment is the same for all classes
   (no net effect on predictions)


Also create `experiments/run_calibration_analysis.py`:
- Load a saved global model checkpoint
- Apply post-hoc logit adjustment with tau ∈ {0.5, 1.0, 1.5, 2.0}
- For each tau: compute balanced_accuracy, macro_f1, worst_class_f1, per_class_recall
- Generate: plot of bACC vs tau, per-class recall vs tau
- NOTE: do NOT include temperature scaling as an imbalance mitigation method.
  Temperature scaling calibrates confidence (improves ECE) but does not change
  predicted labels or balanced accuracy with a single scalar T.
  If you want to report calibration, compute ECE and Brier score separately.
- Save to results/analysis/logit_adjustment/
```

---

## PHASE 3 — Ablations & Analysis

---

### Prompt 21: Ablation Studies 🔥 HPC

```
Create `experiments/run_ablations.py` — systematic ablation studies.

Accept: --ablation {local_epochs, mu, alpha, clients, participation} --device --seed

Each ablation fixes all variables except one and sweeps it.

1. --ablation local_epochs
   Fix: FedProx(μ=0.01), Dir(0.5), 10 clients, 100 rounds
   Vary: local_epochs ∈ {1, 5, 10, 20}
   Outputs:
   - bACC vs rounds (4 lines, one per E)
   - Per-class F1 at final round (4 groups)
   - worst_class_f1 vs E (bar chart)
   Save to results/ablations/local_epochs/

2. --ablation mu
   Fix: Dir(0.5), 10 clients, 100 rounds, E=5
   Vary: proximal_mu ∈ {0.0, 0.001, 0.01, 0.1, 1.0}
   (mu=0.0 is FedAvg, the rest are FedProx variants)
   Outputs:
   - bACC vs rounds (5 lines)
   - Final bACC and worst_class_f1 vs mu (line plot with markers)
   Save to results/ablations/mu/

3. --ablation alpha
   Fix: FedAvg, 10 clients, 100 rounds, E=5
   Vary: alpha ∈ {0.05, 0.1, 0.3, 0.5, 1.0, 5.0}
   Outputs:
   - Final bACC vs alpha
   - worst_class_f1 vs alpha
   - Show that as alpha increases, performance approaches IID
   Save to results/ablations/alpha/

4. --ablation clients
   Fix: FedAvg, Dir(0.5), 100 rounds, E=5
   Vary: num_clients ∈ {3, 5, 10, 20}
   Outputs:
   - bACC vs num_clients
   - worst_client_bACC vs num_clients
   Save to results/ablations/clients/

5. --ablation participation
   Fix: FedAvg, Dir(0.5), 10 clients, 100 rounds, E=5
   Vary: fraction_fit ∈ {0.3, 0.5, 0.7, 1.0}
   Outputs:
   - Accuracy stability (std of bACC over last 10 rounds) vs participation
   Save to results/ablations/participation/

COMPUTE NOTE: Each ablation runs 4–6 experiments × 100 rounds × 10 clients.
Use --num_rounds_override 30 for pilot runs, then 100 for final.
Run each ablation as a separate HPC job.
```

---

### Prompt 22: Client Drift & Update Analysis 🖥️ LOCAL (analysis code) / 🔥 HPC (data collection)

```
Add client drift measurement INSIDE the custom strategy (not via a fragile global dict).

Modify `server/flower_server.py`:

class SaveModelFedAvg(fl.server.strategy.FedAvg):
    # ... existing code ...

    def __init__(self, *args, track_drift=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_drift = track_drift
        self.drift_history = []  # list of {round, client_id, cosine_sim, l2_dist}
        self._previous_global_params = None

    def aggregate_fit(self, server_round, results, failures):
        """
        In aggregate_fit, we have access to client results BEFORE aggregation.
        This is the natural place to measure drift:
        drift = distance(client_update, previous_global_model)
        """
        if self.track_drift and self._previous_global_params is not None:
            for client_proxy, fit_res in results:
                client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                cosine_sim = compute_cosine_similarity(
                    self._previous_global_params, client_params
                )
                l2_dist = compute_l2_distance(
                    self._previous_global_params, client_params
                )
                self.drift_history.append({
                    "round": server_round,
                    "client_id": client_proxy.cid if hasattr(client_proxy, 'cid') else str(client_proxy),
                    "cosine_similarity": cosine_sim,
                    "l2_distance": l2_dist,
                    "num_samples": fit_res.num_examples,
                })

        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        # Store current global params for next round's drift computation
        if aggregated_parameters is not None:
            self._previous_global_params = fl.common.parameters_to_ndarrays(aggregated_parameters)

        return aggregated_parameters, metrics


Helper functions (in metrics/drift_analysis.py):

def compute_cosine_similarity(params1, params2):
    """Flatten both parameter lists, compute cosine similarity."""
    flat1 = np.concatenate([p.flatten() for p in params1])
    flat2 = np.concatenate([p.flatten() for p in params2])
    return float(np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2) + 1e-12))

def compute_l2_distance(params1, params2):
    flat1 = np.concatenate([p.flatten() for p in params1])
    flat2 = np.concatenate([p.flatten() for p in params2])
    return float(np.linalg.norm(flat1 - flat2))


Create `experiments/run_drift_analysis.py`:
- Run FedAvg + Dir(0.1), 10 clients, 50 rounds, track_drift=True
- Load client partition info to get per-client distribution entropy
- Generate:
  a. Drift (L2) over rounds: one line per client
  b. Scatter: x = client entropy, y = mean drift, colour = client worst_class_f1
  c. Correlation: drift vs client bACC (do high-drift clients perform worse?)
- Save to results/analysis/drift/
```

---

### Prompt 23: Fairness Analysis 🖥️ LOCAL (runs on saved results)

```
Create `metrics/fairness.py`:

Class-level fairness:
- equity_gap(per_class_metrics): max - min
- coefficient_of_variation(per_class_metrics): std / mean (0 = perfect equity)
- worst_to_best_ratio(per_class_metrics): min / max (1.0 = perfectly fair)
- minority_macro_f1(per_class_f1, minority_indices=[3, 6]):
  Average F1 over Dermatofibroma and Vascular Lesions only

Client-level fairness:
- client_bACC_stdev(per_client_bACCs): std of client balanced accuracies
- worst_client_gap(per_client_bACCs): max - min
- client_worst_class_recall(per_client_metrics): for each client, worst recall


Create `experiments/run_fairness_analysis.py`:
- Glob all experiment result directories that have per_client_metrics.csv
- For each, compute all fairness metrics
- Generate comparison table:
  Experiment | bACC | Macro F1 | Worst F1 | Minority F1 | Equity Gap | Client Gap | Melanoma Recall
- Generate Pareto frontier plot:
  x = balanced accuracy, y = equity gap (lower = fairer)
  Each dot = one experiment, labeled
  Highlight Pareto-optimal experiments
- Generate per-class radar chart overlaying FedAvg vs FedProx vs best mitigation
- Save to results/analysis/fairness/
```

---

### Prompt 24: Statistical Testing 🖥️ LOCAL (runs on saved results)

```
Create `experiments/statistical_analysis.py`:

Assumes each core experiment has been run with 3 seeds (42, 123, 456).

1. Load results:
   For each config × seed, load global_test_metrics.json
   Group by config, compute mean ± std for each metric

2. Pairwise comparisons:
   For each pair of methods at the same heterogeneity level (e.g., FedAvg vs FedProx at Dir(0.5)):
   - Paired t-test (or Wilcoxon signed-rank if only 3 seeds → check normality with Shapiro-Wilk)
   - Report: mean_A ± std_A vs mean_B ± std_B, p-value, effect size (Cohen's d)
   - Mark significance: * p<0.05, ** p<0.01, *** p<0.001

3. Effect of alpha:
   For FedAvg across alpha levels: one-way ANOVA (or Kruskal-Wallis)
   Post-hoc pairwise with Bonferroni correction

4. Output files:
   - results/analysis/statistics/pairwise_comparisons.csv
   - results/analysis/statistics/anova_results.csv
   - results/analysis/statistics/results_summary.txt:
     Natural-language summary of key findings, e.g.:
     "FedProx (μ=0.01) significantly outperformed FedAvg at α=0.1
      (bACC: 0.612 ± 0.015 vs 0.543 ± 0.022, p=0.031, d=3.64)"

5. Master results table:
   - results/analysis/statistics/master_results.tex
   - LaTeX booktabs table with mean ± std, bold best per column, significance markers
```

---

## PHASE 4 — Dissertation Figures

---

### Prompt 25: Publication-Quality Figures 🖥️ LOCAL

```
Create `experiments/generate_figures.py` — reads all results and generates
dissertation-quality figures.

Formatting for ALL figures:
- plt.style.use('seaborn-v0_8-paper') or 'seaborn-v0_8-whitegrid'
- Font: 11pt labels, 9pt ticks
- Width: single column = 3.5 inches, double column = 7.2 inches
- Colourblind-friendly palette (seaborn "colorblind" or Wong palette)
- Save as PDF + PNG (300 DPI)
- constrained_layout=True
- Close figures after saving

Figures:

1. fig_class_distribution.pdf — DermaMNIST global class distribution bar chart
   Width: single column. Horizontal bars. Annotate with count + %.

2. fig_partition_comparison.pdf — 2×3 grid of client distribution heatmaps
   IID | Dir(1.0) | Dir(0.5) | Dir(0.3) | Dir(0.1) | Pathological
   Width: double column.

3. fig_strategy_comparison.pdf — bACC vs round, line plots
   2×2 subplots: α=0.1, 0.3, 0.5, 1.0
   Lines: FedAvg, FedProx. (Add shaded ± std if 3-seed data available)
   Width: double column.

4. fig_per_class_f1.pdf — grouped bar chart, per-class F1 at final round
   Groups: 7 classes. Bars: FedAvg / FedProx / Best mitigation.
   For Dir(0.5). Width: double column.

5. fig_per_class_convergence.pdf — 7 lines, one per class, F1 vs round
   Best method, Dir(0.5).
   Thick dashed for DF and VASC. Annotate convergence delay.
   Width: single column.

6. fig_mitigation_comparison.pdf — grouped bars
   Groups: baseline / weighted CE / focal / sampler / logit adj
   Metrics: bACC and worst_class_f1 side by side.
   Width: double column.

7. fig_ablation_panel.pdf — 2×2 subplot
   (a) bACC vs local epochs, (b) bACC vs FedProx μ,
   (c) bACC vs Dirichlet α, (d) bACC vs num clients.
   Width: double column.

8. fig_client_scatter.pdf — scatter
   x = client distribution entropy, y = client bACC
   Colour = client dataset size. Different markers for FedAvg vs FedProx.
   Width: single column.

9. fig_pareto.pdf — Pareto frontier
   x = balanced accuracy, y = equity gap
   Label each point. Highlight Pareto-optimal.
   Width: single column.

10. LaTeX tables saved to results/tables/:
    - table_centralised.tex
    - table_core_comparison.tex (mean ± std, significance markers)
    - table_mitigation.tex
    - table_ablation_summary.tex
    Each using booktabs. Numbers to 3 decimal places.
```

---

## PHASE 5 — Optional Extensions (pick ONE if time permits)

---

### Prompt E1: Pretrained ResNet on 64×64 🔥 HPC

```
OPTIONAL — only attempt after Phases 0–4 are complete.

Implement pretrained model comparison at 64×64 resolution.

1. In models/resnet.py, update get_resnet18:
   - For size=64 with pretrained=True:
     Load ResNet18_Weights.DEFAULT
     Keep ImageNet stem (7×7 conv is acceptable for 64×64)
     Replace fc with Linear(512, 7)
   - For size=64 with pretrained=False:
     Use CIFAR-style stem (3×3 conv, no maxpool)

2. Experiment configs:
   configs/ext/pretrained_fedavg_dir05.yaml  — ResNet18 pretrained, Dir(0.5), size=64
   configs/ext/pretrained_fedprox_dir05.yaml — ResNet18 pretrained, FedProx, Dir(0.5), size=64
   configs/ext/scratch_resnet_dir05.yaml     — ResNet18 from scratch, Dir(0.5), size=64
   configs/ext/scratch_cnn_dir05.yaml        — SimpleCNN, Dir(0.5), size=64

   Reduce batch_size to 32 for ResNet. Reduce num_rounds to 50 (pretrained converges faster).

3. Create experiments/run_pretrained_comparison.py:
   - Run all 4 configs
   - Compare: pretrained vs scratch convergence speed
   - Key question: does pretraining help minority classes more or less than majority?
   - Generate: convergence curves, per-class F1 comparison, rounds-to-threshold table
   - Save to results/extensions/pretrained/
```

---

### Prompt E2: Local Fine-Tuning Personalisation 🔥 HPC

```
OPTIONAL — only attempt after Phases 0–4 are complete.

Implement local fine-tuning of the global model as a simple personalisation baseline.

This is MUCH simpler than FedRep/FedBABU and gives you personalised FL analysis
without implementing a new algorithm.

Create `experiments/run_local_finetuning.py`:

1. Load a saved best global model from a completed FL experiment
   (e.g., results/fedprox_dir05_E5_s42/best_model.pt)

2. For each client:
   a. Load the global model
   b. Evaluate on client's local test set (BEFORE fine-tuning)
   c. Fine-tune on client's local train set for E_ft ∈ {1, 3, 5, 10} epochs
      - Option A: fine-tune all layers (lr=1e-4)
      - Option B: freeze feature extractor, fine-tune only classifier head (lr=1e-3)
   d. Evaluate on client's local test set AFTER each fine-tuning step
   e. Save per-client before/after metrics

3. Analysis:
   - For each client: plot bACC before vs after fine-tuning at each E_ft
   - Scatter: x = client train size, y = improvement from fine-tuning
   - Key question: does fine-tuning help clients with extreme class skew?
   - Does it hurt generalisation to unseen classes?
   - Compare Option A vs Option B

4. Summary table:
   E_ft | Avg client bACC (before) | Avg client bACC (after) | Worst client (before) | Worst client (after)

Save to results/extensions/personalisation/
```

---

### Prompt E3: SCAFFOLD 🔥 HPC

```
OPTIONAL — only attempt after FedAvg/FedProx pipeline is fully stable and analysed.

SCAFFOLD is complex. It maintains control variates on server and each client.

WARNING: In Flower simulation, clients may run in separate Ray actors.
Do NOT use a Python global dict for client state — it will not work reliably.
Instead, persist client control variates to disk (pickle files per client)
or use Flower's built-in client state mechanisms.

[Full SCAFFOLD implementation details here — same as original Prompt 14
but with the disk-persistence approach and the explicit warning about
simulation workers.]

Only create 2 configs:
  configs/ext/scaffold_dir05.yaml
  configs/ext/scaffold_dir01.yaml

Compare against FedAvg and FedProx at the same settings.
```

---

### Prompt E4: Poisoning Robustness (Appendix) 🔥 HPC

```
OPTIONAL — appendix-level analysis only. Do NOT prioritise over core results.

Keep scope minimal:
- 1 attack type: label flipping (1 out of 10 clients)
- 1 defence: coordinate-wise median aggregation
- 1 heterogeneity setting: Dir(0.5)

[Details of label-flipping client and median aggregation — simplified from
original Prompt 24, reduced to 3 configs total]

Key question: does existing class imbalance make the system MORE vulnerable to
poisoning? (Hypothesis: minority classes are easier to corrupt because the
aggregation has fewer "correct" updates to outvote the attacker.)

Save to results/extensions/poisoning/
```

---

## HPC JOB SCRIPTS

---

### Prompt HPC: Job Submission Scripts 🖥️ LOCAL

```
Create SLURM job submission scripts. Adapt to your cluster's specifics
(partition name, account, module names).

Template `scripts/slurm_template.sh`:

#!/bin/bash
#SBATCH --job-name=fl-derm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu          # CHANGE to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=YOURPROJECT    # CHANGE to your account

# Load modules — ADAPT THESE to your cluster
module purge
module load python/3.10
module load cuda/12.1

# Activate environment
source ~/fl-derm-env/bin/activate

# Create log directory
mkdir -p logs

# Run experiment
python experiments/run_experiment.py --config "$1" --seed "$2" --device cuda

echo "Job completed: config=$1 seed=$2"


Script `scripts/submit_single.sh`:
  Usage: sbatch scripts/slurm_template.sh configs/fedavg_dir05.yaml 42


Script `scripts/submit_all_core.sh`:
  #!/bin/bash
  # Submit all core experiments with 3 seeds each
  mkdir -p logs
  for config in configs/fedavg_*.yaml configs/fedprox_*.yaml; do
    for seed in 42 123 456; do
      sbatch --job-name="$(basename $config .yaml)_s${seed}" \
             scripts/slurm_template.sh "$config" "$seed"
      sleep 1  # avoid overwhelming the scheduler
    done
  done
  echo "Submitted all core experiments"


Script `scripts/submit_ablation.sh`:
  Usage: sbatch --time=08:00:00 scripts/slurm_template.sh <ablation_config>
  # Ablations need longer time (8h) because they run multiple sub-experiments


Script `scripts/submit_mitigation.sh`:
  #!/bin/bash
  for config in configs/mitigation/*.yaml; do
    for seed in 42 123 456; do
      sbatch --job-name="mit_$(basename $config .yaml)_s${seed}" \
             scripts/slurm_template.sh "$config" "$seed"
      sleep 1
    done
  done


Script `scripts/check_results.sh`:
  #!/bin/bash
  # Check which experiments completed successfully
  echo "=== Completed experiments ==="
  find results/ -name "global_test_metrics.json" | sort
  echo ""
  echo "=== Expected but missing ==="
  for config in configs/fedavg_*.yaml configs/fedprox_*.yaml; do
    name=$(basename $config .yaml)
    for seed in 42 123 456; do
      dir="results/${name}_E5_s${seed}"
      if [ ! -f "${dir}/global_test_metrics.json" ]; then
        echo "MISSING: ${dir}"
      fi
    done
  done


WHAT RUNS WHERE:

| Task                          | Where      | Time Est.  | GPU? |
|-------------------------------|------------|------------|------|
| Project setup, tests          | Local      | minutes    | No   |
| Flower smoke test             | Local      | <1 min     | No   |
| Data exploration              | Local      | <5 min     | No   |
| Centralised MNIST             | Local GPU  | ~5 min     | Yes  |
| Centralised DermaMNIST (3×50) | Local GPU  | ~30 min    | Yes  |
| FL MNIST validation           | Local GPU  | ~15 min    | Yes  |
| Single core FL experiment     | HPC        | 1–2 hours  | Yes  |
| All 11 core × 3 seeds        | HPC        | ~33 jobs   | Yes  |
| Single ablation               | HPC        | 3–6 hours  | Yes  |
| All ablations                 | HPC        | ~5 jobs    | Yes  |
| Mitigation experiments        | HPC        | ~21 jobs   | Yes  |
| Analysis / figures / stats    | Local      | minutes    | No   |
| Extension experiment          | HPC        | 2–4 hours  | Yes  |
```

---

