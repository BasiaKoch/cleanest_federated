from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import inspect
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl


def get_params(model):
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_params(model, arrays):
    sd = model.state_dict()
    new = {k: torch.as_tensor(v, dtype=sd[k].dtype) for k, v in zip(sd.keys(), arrays)}
    model.load_state_dict(new, strict=True)


class SmokeClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = nn.Linear(2, 2)
        x = torch.randn(100, 2)
        y = (x[:, 0] > 0).long()
        self.loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.loss = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        self.model.train()
        for x, y in self.loader:
            self.opt.zero_grad()
            loss = self.loss(self.model(x), y)
            loss.backward()
            self.opt.step()
        return get_params(self.model), len(self.loader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        self.model.eval()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in self.loader:
                logits = self.model(x)
                total_loss += self.loss(logits, y).item() * y.numel()
                correct += (logits.argmax(1) == y).sum().item()
                total += y.numel()
        return total_loss / total, total, {'accuracy': correct / total}


def main():
    print(f'Flower version: {fl.__version__}')
    print(f'start_simulation signature: {inspect.signature(fl.simulation.start_simulation)}')
    model = nn.Linear(2, 2)
    params = get_params(model)
    model2 = nn.Linear(2, 2)
    set_params(model2, params)
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
        assert k1 == k2 and torch.allclose(v1, v2)

    def client_fn(cid):
        return SmokeClient().to_client()

    def eval_fn(server_round, parameters, config):
        c = SmokeClient()
        loss, n, metrics = c.evaluate(parameters, config)
        return loss, metrics

    strategy = fl.server.strategy.FedAvg(evaluate_fn=eval_fn, min_fit_clients=2, min_available_clients=2, min_evaluate_clients=2)
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
        client_resources={'num_cpus': 1, 'num_gpus': 0.0},
    )
    print(f'Round 1 metrics: {history.metrics_centralized}')
    print('Simulation API used: fl.simulation.start_simulation')
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
