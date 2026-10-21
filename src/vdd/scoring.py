"""Normal-validation calibration, never using test labels or test scores."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def calibrate(scores, weight=0.5, quantile=0.95):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 2 or len(scores) == 0:
        raise ValueError("Expected nonempty N x 2 validation scores.")
    if not np.isfinite(scores).all() or not 0 <= weight <= 1 or not 0 < quantile < 1:
        raise ValueError("Invalid calibration inputs.")
    center = scores.mean(0)
    scale = np.maximum(scores.std(0), 1e-8)
    normalized = (scores - center) / scale
    ensemble = weight * normalized[:, 0] + (1 - weight) * normalized[:, 1]
    return {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "weight": weight,
        "quantile": quantile,
        "thresholds": [
            float(np.quantile(normalized[:, 0], quantile)),
            float(np.quantile(normalized[:, 1], quantile)),
            float(np.quantile(ensemble, quantile)),
        ],
    }


def combine(scores, calibration):
    z = (np.asarray(scores) - calibration["center"]) / calibration["scale"]
    w = calibration["weight"]
    return np.column_stack((z, w * z[:, 0] + (1 - w) * z[:, 1]))


def metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    prediction = scores > threshold
    negatives = labels == 0
    return {
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "false_positive_rate": float(prediction[negatives].mean()) if negatives.any() else None,
        "threshold": float(threshold),
        "n_images": len(labels),
    }
