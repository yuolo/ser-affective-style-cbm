"""Utilities for the disentangled affective-style CBM SER experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

from .concepts import (
    baseline_matrix_for_rows,
    build_concepts_from_baseline_and_deviation,
    compute_speaker_baselines,
)
from .config import Config
from .datasets import CREMADConceptDataset
from .metrics import batch_correlation_penalty, compute_metrics

# =============================================================================
# TRAIN / EVAL
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    cfg: Config,
    emotion_weights: torch.Tensor,
    grl_lambda: float,
) -> Dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "emo": 0.0,
        "aff": 0.0,
        "style": 0.0,
        "style_spk": 0.0,
        "aff_spk_adv": 0.0,
        "style_emo_adv": 0.0,
        "orth": 0.0,
    }
    n = 0

    emotion_weights = emotion_weights.to(device)

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        speaker_local = batch["speaker_local"].to(device)
        aff_targets = batch["aff_targets"].to(device)
        style_targets = batch["style_targets"].to(device)

        valid_spk = speaker_local >= 0
        if not valid_spk.all():
            raise RuntimeError("Training batch contains speakers missing from local speaker mapping.")

        out = model(x, grl_lambda=grl_lambda)

        loss_emo = F.cross_entropy(out["emotion_logits"], y, weight=emotion_weights)
        loss_aff = F.smooth_l1_loss(out["c_aff"], aff_targets)
        loss_style = F.smooth_l1_loss(out["c_style"], style_targets)
        loss_style_spk = F.cross_entropy(out["style_speaker_logits"], speaker_local)
        loss_aff_spk_adv = F.cross_entropy(out["aff_speaker_adv_logits"], speaker_local)
        loss_style_emo_adv = F.cross_entropy(out["style_emotion_adv_logits"], y, weight=emotion_weights)
        loss_orth = batch_correlation_penalty(out["c_aff"], out["c_style"])

        loss = loss_emo

        if cfg.USE_AFF_CONCEPT_BRANCH and cfg.USE_AFF_CONCEPT_SUPERVISION:
            loss = loss + cfg.LAMBDA_AFF_CONCEPT * loss_aff

        if cfg.USE_STYLE_BRANCH:
            loss = loss + cfg.LAMBDA_STYLE_CONCEPT * loss_style
            loss = loss + cfg.LAMBDA_STYLE_SPEAKER * loss_style_spk

        if cfg.USE_AFF_CONCEPT_BRANCH and cfg.USE_AFF_SPEAKER_ADVERSARY:
            loss = loss + cfg.LAMBDA_AFF_SPK_ADV * loss_aff_spk_adv

        if cfg.USE_STYLE_EMOTION_ADVERSARY and cfg.USE_STYLE_BRANCH:
            loss = loss + cfg.LAMBDA_STYLE_EMO_ADV * loss_style_emo_adv

        if cfg.USE_ORTHOGONALITY and cfg.USE_STYLE_BRANCH:
            loss = loss + cfg.LAMBDA_ORTH * loss_orth

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = x.size(0)
        n += bs
        totals["loss"] += float(loss.item()) * bs
        totals["emo"] += float(loss_emo.item()) * bs
        totals["aff"] += float(loss_aff.item()) * bs
        totals["style"] += float(loss_style.item()) * bs
        totals["style_spk"] += float(loss_style_spk.item()) * bs
        totals["aff_spk_adv"] += float(loss_aff_spk_adv.item()) * bs
        totals["style_emo_adv"] += float(loss_style_emo_adv.item()) * bs
        totals["orth"] += float(loss_orth.item()) * bs

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    cfg: Optional[Config] = None,
) -> Dict[str, Any]:
    model.eval()
    ys = []
    logits = []
    c_affs = []
    c_styles = []
    aff_targets = []
    style_targets = []

    for batch in loader:
        x = batch["x"].to(device)
        out = model(x, grl_lambda=0.0)
        ys.append(batch["y"].detach().cpu().numpy())
        logits.append(out["emotion_logits"].cpu().numpy())
        c_affs.append(out["c_aff"].cpu().numpy())
        c_styles.append(out["c_style"].cpu().numpy())
        aff_targets.append(batch["aff_targets"].detach().cpu().numpy())
        style_targets.append(batch["style_targets"].detach().cpu().numpy())

    y_true = np.concatenate(ys)
    logit_arr = np.concatenate(logits)
    c_aff = np.concatenate(c_affs)
    c_style = np.concatenate(c_styles)
    aff_t = np.concatenate(aff_targets)
    style_t = np.concatenate(style_targets)

    metrics = compute_metrics(y_true, logit_arr)
    metrics["aff_mae"] = float(np.mean(np.abs(c_aff - aff_t)))
    metrics["style_mae"] = float(np.mean(np.abs(c_style - style_t)))
    if cfg is not None:
        if not (cfg.USE_AFF_CONCEPT_BRANCH and cfg.USE_AFF_CONCEPT_SUPERVISION):
            metrics["aff_mae"] = float("nan")
        if not cfg.USE_STYLE_BRANCH:
            metrics["style_mae"] = float("nan")

    return {
        "metrics": metrics,
        "y_true": y_true,
        "logits": logit_arr,
        "c_aff": c_aff,
        "c_style": c_style,
        "aff_targets": aff_t,
        "style_targets": style_t,
    }


def _feature_matrix_for_df(df: pd.DataFrame, cache: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    return np.stack([cache[str(p)]["concept_features"] for p in df["path"].astype(str).tolist()], axis=0).astype(np.float32)


def _feature_names_from_cache(cache: Dict[str, Dict[str, np.ndarray]]) -> List[str]:
    arr = cache["__concept_feature_names__"]["names"]
    return [str(x) for x in arr.tolist()]


def make_loaders_for_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cache: Dict[str, Dict[str, np.ndarray]],
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[str, np.ndarray]]:
    """Fit feature scaler on train only, then build B+C concepts.

    Main training targets:
        train style = train speaker baseline
        train affective = train utterance deviation from train speaker baseline

    Val/test targets are diagnostic proxies. They are not used for fitting.
    """
    feature_names = _feature_names_from_cache(cache)

    train_feats = _feature_matrix_for_df(train_df, cache)
    val_feats = _feature_matrix_for_df(val_df, cache)
    test_feats = _feature_matrix_for_df(test_df, cache)

    scaler = RobustScaler()
    scaler.fit(train_feats)

    train_z = np.clip(scaler.transform(train_feats), -5.0, 5.0).astype(np.float32)
    val_z = np.clip(scaler.transform(val_feats), -5.0, 5.0).astype(np.float32)
    test_z = np.clip(scaler.transform(test_feats), -5.0, 5.0).astype(np.float32)

    train_speakers_arr = train_df["speaker"].astype(str).to_numpy(dtype=str)
    val_speakers_arr = val_df["speaker"].astype(str).to_numpy(dtype=str)
    test_speakers_arr = test_df["speaker"].astype(str).to_numpy(dtype=str)

    train_baselines = compute_speaker_baselines(train_z, train_speakers_arr)
    train_global_baseline = np.mean(train_z, axis=0).astype(np.float32)
    train_baseline_mat = baseline_matrix_for_rows(
        train_z,
        train_speakers_arr,
        baseline_by_speaker=train_baselines,
        fallback_global=train_global_baseline,
    )

    if cfg.DIAGNOSTIC_LOCAL_BASELINES_FOR_VAL_TEST:
        val_baselines = compute_speaker_baselines(val_z, val_speakers_arr)
        test_baselines = compute_speaker_baselines(test_z, test_speakers_arr)
    else:
        val_baselines = train_baselines
        test_baselines = train_baselines

    val_baseline_mat = baseline_matrix_for_rows(
        val_z,
        val_speakers_arr,
        baseline_by_speaker=val_baselines,
        fallback_global=train_global_baseline,
    )
    test_baseline_mat = baseline_matrix_for_rows(
        test_z,
        test_speakers_arr,
        baseline_by_speaker=test_baselines,
        fallback_global=train_global_baseline,
    )

    train_aff, train_style = build_concepts_from_baseline_and_deviation(train_z, train_baseline_mat, feature_names)
    val_aff, val_style = build_concepts_from_baseline_and_deviation(val_z, val_baseline_mat, feature_names)
    test_aff, test_style = build_concepts_from_baseline_and_deviation(test_z, test_baseline_mat, feature_names)

    train_speakers = sorted(train_df["speaker"].astype(str).unique().tolist())
    speaker_to_local = {spk: i for i, spk in enumerate(train_speakers)}

    train_ds = CREMADConceptDataset(train_df, cache, train_aff, train_style, speaker_to_local=speaker_to_local)
    val_ds = CREMADConceptDataset(val_df, cache, val_aff, val_style, speaker_to_local=speaker_to_local)
    test_ds = CREMADConceptDataset(test_df, cache, test_aff, test_style, speaker_to_local=speaker_to_local)

    device_type = str(cfg.DEVICE).lower()
    pin_memory = device_type.startswith("cuda")
    persistent_workers = cfg.NUM_WORKERS > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    concept_arrays = {
        "train_aff": train_aff,
        "train_style": train_style,
        "val_aff": val_aff,
        "val_style": val_style,
        "test_aff": test_aff,
        "test_style": test_style,
        "train_z": train_z,
        "val_z": val_z,
        "test_z": test_z,
    }

    return train_loader, val_loader, test_loader, speaker_to_local, concept_arrays

