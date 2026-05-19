# -*- coding: utf-8 -*-
"""Utilities for the disentangled affective-style CBM SER experiments."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

# =============================================================================
# MODEL
# =============================================================================

class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverseFn.apply(x, lambd)


class CRNNEncoder(nn.Module):
    def __init__(self, n_mels: int, h_dim: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.gru = nn.GRU(
            input_size=96,
            hidden_size=h_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)              # [B, C, mel', time']
        z = z.mean(dim=2)             # [B, C, time']
        z = z.transpose(1, 2)         # [B, time', C]
        out, _ = self.gru(z)          # [B, time', H]
        h = out.mean(dim=1)           # [B, H]
        h = self.proj(h)
        return h


class DisentangledAffectiveStyleCBM(nn.Module):
    def __init__(
        self,
        n_mels: int,
        h_dim: int,
        n_aff: int,
        n_style: int,
        n_emotions: int,
        n_train_speakers: int,
        dropout: float,
        emotion_head_input: str = "aff",
        use_aff_concept_branch: bool = True,
        use_style_branch: bool = True,
    ):
        super().__init__()
        if emotion_head_input not in {"aff", "encoder"}:
            raise ValueError("emotion_head_input must be 'aff' or 'encoder'.")
        if emotion_head_input == "aff" and not use_aff_concept_branch:
            raise ValueError("Affective concept branch is required when emotion_head_input='aff'.")

        self.n_aff = n_aff
        self.n_style = n_style
        self.emotion_head_input = emotion_head_input
        self.use_aff_concept_branch = use_aff_concept_branch
        self.use_style_branch = use_style_branch
        self.encoder = CRNNEncoder(n_mels=n_mels, h_dim=h_dim, dropout=dropout)

        self.aff_head = nn.Sequential(
            nn.Linear(h_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_aff),
            nn.Sigmoid(),
        )

        self.style_head = nn.Sequential(
            nn.Linear(h_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_style),
            nn.Sigmoid(),
        )

        # Main emotion classifier sees only affective concepts.
        self.emotion_head = nn.Sequential(
            nn.Linear(n_aff, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_emotions),
        )

        self.encoder_emotion_head = nn.Sequential(
            nn.Linear(h_dim, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, n_emotions),
        )

        # Positive speaker classifier: style concepts should absorb speaker/style info.
        self.style_speaker_head = nn.Sequential(
            nn.Linear(n_style, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, n_train_speakers),
        )

        # Adversarial speaker classifier: affective concepts should not reveal speaker.
        self.aff_speaker_adv_head = nn.Sequential(
            nn.Linear(n_aff, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, n_train_speakers),
        )

        # Optional adversarial emotion classifier: style concepts should not reveal emotion.
        self.style_emotion_adv_head = nn.Sequential(
            nn.Linear(n_style, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_emotions),
        )

    def forward(self, x: torch.Tensor, grl_lambda: float = 1.0) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        if self.use_aff_concept_branch:
            c_aff = self.aff_head(h)
        else:
            c_aff = torch.zeros(h.size(0), self.n_aff, dtype=h.dtype, device=h.device)

        if self.use_style_branch:
            c_style = self.style_head(h)
        else:
            c_style = torch.zeros(h.size(0), self.n_style, dtype=h.dtype, device=h.device)

        if self.emotion_head_input == "encoder":
            emotion_logits = self.encoder_emotion_head(h)
        else:
            emotion_logits = self.emotion_head(c_aff)
        style_speaker_logits = self.style_speaker_head(c_style)
        aff_speaker_adv_logits = self.aff_speaker_adv_head(grad_reverse(c_aff, grl_lambda))
        style_emotion_adv_logits = self.style_emotion_adv_head(grad_reverse(c_style, grl_lambda))

        return {
            "h": h,
            "c_aff": c_aff,
            "c_style": c_style,
            "emotion_logits": emotion_logits,
            "style_speaker_logits": style_speaker_logits,
            "aff_speaker_adv_logits": aff_speaker_adv_logits,
            "style_emotion_adv_logits": style_emotion_adv_logits,
        }

