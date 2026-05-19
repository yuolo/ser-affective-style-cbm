# -*- coding: utf-8 -*-
"""Utilities for the disentangled affective-style CBM SER experiments."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# =============================================================================
# LOSSES AND METRICS
# =============================================================================

def batch_correlation_penalty(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.size(0) < 2:
        return torch.tensor(0.0, device=a.device)
    a0 = a - a.mean(dim=0, keepdim=True)
    b0 = b - b.mean(dim=0, keepdim=True)
    a0 = a0 / (a0.std(dim=0, keepdim=True) + 1e-6)
    b0 = b0 / (b0.std(dim=0, keepdim=True) + 1e-6)
    corr = (a0.T @ b0) / max(a.size(0) - 1, 1)
    return (corr ** 2).mean()


def ece_score(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float32)

    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        if np.any(mask):
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].mean()
            ece += np.mean(mask) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    y_pred = probs.argmax(axis=1)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "uar": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ece": float(ece_score(probs, y_true)),
    }


def grl_schedule(epoch: int, num_epochs: int, max_lambda: float) -> float:
    p = epoch / max(num_epochs - 1, 1)
    return float(max_lambda * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0))


def class_weights_from_labels(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)

