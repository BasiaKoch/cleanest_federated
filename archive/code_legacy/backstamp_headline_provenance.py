"""One-shot back-stamper for the headline JSONs (audit fix I1).

The 20 headline `test_at_best_*.json` files in
`mnist_dermnist/results/headline/` were produced by the pure-PyTorch
reference loop BEFORE the `extra_metadata` provenance fields were added
to `fl.server_loop.save_run_outputs`. The methods section of the thesis
claims these files are "fully self-documenting", which is true for runs
produced from now on but false for the existing headline set.

This script adds the missing provenance keys to each file IN PLACE
(safe: only adds keys, never overwrites existing values). Run once.

The fields stamped here describe the headline sweep as actually
executed:
  - partition       = balanced_paired_7_clients (the headline partition)
  - image_size      = 28
  - npz_path        = <repo>/dermamnist_64.npz
  - framework       = pure-pytorch  (reference loop, not Flower)
  - framework_version = "n/a"
  - runner_script   = run_one.py
  - loss_type       = cross_entropy
  - num_classes     = 7 (where missing)
  - provenance_note = backstamped-2026-05-18
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADLINE  = REPO_ROOT / "mnist_dermnist" / "results" / "headline"
NPZ_PATH  = str(REPO_ROOT / "dermamnist_64.npz")

STAMP = {
    "partition": "balanced_paired_7_clients",
    "image_size": 28,
    "npz_path": NPZ_PATH,
    "framework": "pure-pytorch",
    "framework_version": "n/a",
    "runner_script": "run_one.py",
    "loss_type": "cross_entropy",
    "num_classes": 7,
    "provenance_note": f"backstamped-{date.today().isoformat()}",
}


def main() -> int:
    if not HEADLINE.is_dir():
        print(f"ERROR: {HEADLINE} not found", file=sys.stderr)
        return 1

    json_paths = sorted(HEADLINE.glob("test_at_best_*.json"))
    if not json_paths:
        print(f"ERROR: no test_at_best_*.json in {HEADLINE}", file=sys.stderr)
        return 1

    print(f"Back-stamping {len(json_paths)} headline JSON files in {HEADLINE}")
    print(f"Adding: {list(STAMP.keys())}\n")

    n_updated = 0
    n_already_complete = 0
    for fp in json_paths:
        data = json.loads(fp.read_text())
        added = []
        for k, v in STAMP.items():
            if k not in data:
                data[k] = v
                added.append(k)
        if added:
            fp.write_text(json.dumps(data, indent=2))
            n_updated += 1
            print(f"  {fp.name:55s} added: {added}")
        else:
            n_already_complete += 1
            print(f"  {fp.name:55s} already complete")

    print(f"\nDone: {n_updated} updated, {n_already_complete} already complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
