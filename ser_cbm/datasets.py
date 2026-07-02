"""Utilities for the disentangled affective-style CBM SER experiments."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# =============================================================================
# DATASET
# =============================================================================

class CREMADConceptDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cache: Dict[str, Dict[str, np.ndarray]],
        aff_targets: np.ndarray,
        style_targets: np.ndarray,
        speaker_to_local: Optional[Dict[str, int]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.cache = cache
        self.aff_targets = aff_targets.astype(np.float32)
        self.style_targets = style_targets.astype(np.float32)
        self.speaker_to_local = speaker_to_local or {}

        assert len(self.df) == len(self.aff_targets) == len(self.style_targets)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        path = str(row["path"])
        logmel = self.cache[path]["logmel"]
        x = torch.tensor(logmel[None, :, :], dtype=torch.float32)
        y = torch.tensor(int(row["emotion"]), dtype=torch.long)

        speaker_str = str(row["speaker"])
        speaker_local = self.speaker_to_local.get(speaker_str, -1)
        speaker_local = torch.tensor(speaker_local, dtype=torch.long)

        aff = torch.tensor(self.aff_targets[idx], dtype=torch.float32)
        style = torch.tensor(self.style_targets[idx], dtype=torch.float32)

        return {
            "x": x,
            "y": y,
            "speaker_local": speaker_local,
            "aff_targets": aff,
            "style_targets": style,
        }

