"""Quantitative descriptors of the engineered partition's heterogeneity.

Reports for the balanced_paired_7_clients partition (seed=42):
  - samples per client
  - per-client class distribution (counts and proportions)
  - label entropy per client (in bits)
  - Jensen-Shannon divergence from the global training distribution
  - dominant class per client (id + proportion)
  - classes absent per client (count)
  - effective number of classes per client: exp(H), where H is in nats

Outputs a single descriptive table to stdout. The thesis cites a small
selection of these numbers (samples per client, JS divergence range,
classes absent range) to make the "non-IID" qualifier concrete.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fl_dermamnist.data.load import load_dermmnist
from fl_dermamnist.data.partition import balanced_paired_7_clients


THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
NPZ_PATH = REPO_ROOT / "dermamnist_64.npz"


CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if len(p) else 0.0


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric Jensen-Shannon divergence in bits (base-2 log)."""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def main():
    print(f"Loading DermaMNIST from {NPZ_PATH} ...")
    train, _, _ = load_dermmnist(str(NPZ_PATH), image_size=28)
    labels = np.asarray(train.labels).flatten()
    client_indices, _ = balanced_paired_7_clients(labels, seed=42)
    K = len(client_indices)
    C = 7

    counts = np.zeros((K, C), dtype=int)
    for k, idx in enumerate(client_indices):
        labels_k = labels[idx]
        for c in range(C):
            counts[k, c] = int((labels_k == c).sum())

    global_dist = counts.sum(axis=0) / counts.sum()

    print()
    print(f"Engineered partition (balanced_paired_7_clients, seed=42)")
    print(f"Global training-set class distribution (proportion):")
    for c in range(C):
        print(f"  {c} {CLASS_NAMES[c]:<22} {global_dist[c]:.4f}")
    print()
    print(f"{'Client':>6} {'n':>5} {'H_bits':>7} {'JS_bits':>8} "
          f"{'dominant':<10} {'absent':>6} {'eff_K':>6}")
    print("-" * 60)

    js_vals = []
    absent_counts = []
    for k in range(K):
        n = counts[k].sum()
        p = counts[k] / n if n > 0 else np.zeros(C)
        H = entropy_bits(p)
        js = js_divergence(p, global_dist)
        dom_c = int(np.argmax(p))
        dom_prop = p[dom_c]
        absent = int((counts[k] == 0).sum())
        eff_K = float(np.exp(H * np.log(2)))  # convert bits->nats->exp
        js_vals.append(js)
        absent_counts.append(absent)
        print(f"  C{k:<4} {n:>5} {H:>7.3f} {js:>8.4f} "
              f"{dom_c}({dom_prop:.2f}){'':<2} {absent:>6} {eff_K:>6.2f}")

    print()
    print(f"Summary:")
    print(f"  samples-per-client range:      {min(counts.sum(axis=1))} - {max(counts.sum(axis=1))}")
    print(f"  JS-divergence-from-global:     {min(js_vals):.4f} - {max(js_vals):.4f} bits")
    print(f"  classes-absent-per-client:     {min(absent_counts)} - {max(absent_counts)}")
    print(f"  global entropy (max possible): {entropy_bits(global_dist):.3f} bits "
          f"(max for K=7: {np.log2(C):.3f} bits)")


if __name__ == "__main__":
    main()
