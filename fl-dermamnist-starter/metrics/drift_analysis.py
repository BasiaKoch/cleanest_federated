import numpy as np


def _flatten(params):
    return np.concatenate([np.asarray(p).ravel() for p in params])


def compute_cosine_similarity(params1, params2):
    flat1, flat2 = _flatten(params1), _flatten(params2)
    return float(np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2) + 1e-12))


def compute_l2_distance(params1, params2):
    return float(np.linalg.norm(_flatten(params1) - _flatten(params2)))
