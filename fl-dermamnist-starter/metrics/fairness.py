import numpy as np


def equity_gap(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmax(values) - np.nanmin(values))


def coefficient_of_variation(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanstd(values) / (np.nanmean(values) + 1e-12))


def worst_to_best_ratio(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmin(values) / (np.nanmax(values) + 1e-12))


def minority_macro_f1(per_class_f1, minority_indices=(3, 6)):
    arr = np.asarray(per_class_f1, dtype=float)
    return float(np.nanmean(arr[list(minority_indices)]))


def client_bACC_stdev(per_client_bACCs):
    return float(np.nanstd(np.asarray(per_client_bACCs, dtype=float)))


def worst_client_gap(per_client_bACCs):
    values = np.asarray(per_client_bACCs, dtype=float)
    return float(np.nanmax(values) - np.nanmin(values))


def client_worst_class_recall(per_client_metrics):
    recalls = per_client_metrics['per_class_recall_json'] if hasattr(per_client_metrics, '__getitem__') else per_client_metrics
    out = []
    for item in recalls:
        if isinstance(item, str):
            import json
            item = json.loads(item)
        out.append(float(np.nanmin(np.asarray(item, dtype=float))))
    return out
