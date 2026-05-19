# -*- coding: utf-8 -*-
"""Utilities for the disentangled affective-style CBM SER experiments."""

from __future__ import annotations

import random

import os
import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

# =============================================================================
# REPRODUCIBILITY / DEVICE
# =============================================================================

def is_mps_available() -> bool:
    """Robust MPS availability check that avoids some VSCode/Pylance false positives."""
    try:
        mps_backend = getattr(torch.backends, "mps", None)
        is_available_fn = getattr(mps_backend, "is_available", None)
        return bool(callable(is_available_fn) and is_available_fn())
    except Exception:
        return False


def empty_mps_cache_safely() -> None:
    try:
        mps_mod = getattr(torch, "mps", None)
        empty_cache_fn = getattr(mps_mod, "empty_cache", None)
        if callable(empty_cache_fn):
            empty_cache_fn()
    except Exception:
        pass


def seed_mps_safely(seed: int) -> None:
    try:
        mps_mod = getattr(torch, "mps", None)
        manual_seed_fn = getattr(mps_mod, "manual_seed", None)
        if callable(manual_seed_fn):
            manual_seed_fn(seed)
    except Exception:
        pass


def select_device(device_choice: str = "auto") -> torch.device:
    choice = str(device_choice).lower().strip()
    if choice not in {"", "auto"}:
        return torch.device(choice)
    if is_mps_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_torch_runtime(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    elif device.type == "mps":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        empty_mps_cache_safely()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    seed_mps_safely(seed)
