from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.download import load_mnist
from models.simple_cnn import SimpleCNN
from trainers.centralised import CentralisedTrainer
from utils.seed import set_all_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    train_ds, test_ds = load_mnist()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = SimpleCNN(in_channels=1, num_classes=10)
    trainer = CentralisedTrainer(model, train_loader, test_loader, test_loader, device, 10, lr=args.lr)
    result = trainer.train(args.epochs, Path('results/centralised/mnist'), class_names=[str(i) for i in range(10)])
    print(f"Final MNIST test accuracy: {result['test_metrics']['accuracy']:.4f}")


if __name__ == '__main__':
    main()
