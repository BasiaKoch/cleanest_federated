from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from losses.focal_loss import FocalLoss
from losses.logit_adjustment import post_hoc_logit_adjustment
from losses.weighted_ce import compute_class_weights
from metrics.per_class_tracker import PerClassTracker
from models import get_model, count_parameters
from utils.config_loader import config_to_experiment_name


def test_config_name_includes_local_epochs_and_seed():
    cfg = {'_config_stem': 'fedavg_dir05', 'federation': {'local_epochs': 5}, 'misc': {'seed': 123}}
    assert config_to_experiment_name(cfg) == 'fedavg_dir05_E5_s123'


def test_focal_gamma_zero_matches_cross_entropy():
    logits = torch.randn(6, 4)
    targets = torch.tensor([0, 1, 2, 3, 0, 1])
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, targets), F.cross_entropy(logits, targets))


def test_class_weights_zero_count_is_zero():
    weights = compute_class_weights([10, 0, 5])
    assert weights[1].item() == 0.0


def test_logit_adjustment_tau_zero_is_identity():
    logits = torch.randn(3, 4)
    priors = torch.ones(4) / 4
    assert torch.allclose(post_hoc_logit_adjustment(logits, priors, tau=0.0), logits)


def test_model_factory_smoke():
    model = get_model('simple_cnn', 3, 7, image_size=28)
    assert model(torch.randn(1, 3, 28, 28)).shape == (1, 7)
    assert count_parameters(model)['trainable'] > 0


def test_per_class_tracker_convergence_round(tmp_path):
    tracker = PerClassTracker(2, ['A', 'B'], tmp_path)
    tracker.log_round(1, {'f1_A': 0.1, 'f1_B': 0.2})
    tracker.log_round(2, {'f1_A': 0.4, 'f1_B': np.nan})
    assert tracker.compute_convergence_round(threshold=0.3) == {'A': 2, 'B': None}
