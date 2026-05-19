# -*- coding: utf-8 -*-
"""Utilities for the disentangled affective-style CBM SER experiments."""

from __future__ import annotations

from dataclasses import dataclass

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    # -------------------------------------------------------------------------
    # Paths and dataset selection
    # -------------------------------------------------------------------------
    # DATASET picks between "cremad" (6-way) and "iemocap" (4-way protocol with
    # "exc" folded into "happy"). The corresponding directory below is the one
    # actually read from at run time.
    DATASET: str = "cremad"
    CREMA_D_DIR: str = "AudioWAV"
    IEMOCAP_DIR: str = "iemocap"
    OUT_DIR: str = "si_affcbm_egemaps_bd_outputs"

    # -------------------------------------------------------------------------
    # Concept feature backend
    # -------------------------------------------------------------------------
    # Main paper-grade setting: "opensmile".
    # Fallback ablation/debug setting: "librosa".
    FEATURE_BACKEND: str = "opensmile"
    OPENSMILE_FEATURE_SET: str = "eGeMAPSv02"
    CACHE_CONCEPT_FEATURES_TO_CSV: bool = True
    FORCE_REBUILD_CONCEPT_CACHE: bool = False

    # If True, validation/test targets use local speaker baselines computed within
    # that split. These targets are diagnostic only and are not used for fitting.
    DIAGNOSTIC_LOCAL_BASELINES_FOR_VAL_TEST: bool = True

    # Use concept MAE in checkpoint selection. For strict paper experiments,
    # keep this 0.0 because val/test baseline/deviation targets are diagnostic proxies.
    SELECTION_CONCEPT_PENALTY: float = 0.0

    # -------------------------------------------------------------------------
    # Audio / spectrogram
    # -------------------------------------------------------------------------
    SR: int = 16000
    MAX_SECONDS: float = 3.5
    N_FFT: int = 400
    HOP_LENGTH: int = 160
    WIN_LENGTH: int = 400
    N_MELS: int = 64
    FMIN: int = 50
    FMAX: int = 7600

    # Only used by librosa fallback primitives.
    YIN_FRAME_LENGTH: int = 1024
    YIN_FMIN: int = 50
    YIN_FMAX: int = 500

    # -------------------------------------------------------------------------
    # CV / training
    # -------------------------------------------------------------------------
    N_SPLITS: int = 5
    INNER_VAL_SIZE: float = 0.15
    SEED: int = 42
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 40
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    NUM_WORKERS: int = 0  # MPS/Mac VSCode: keep 0 to avoid multiprocessing issues
    PATIENCE: int = 10

    # -------------------------------------------------------------------------
    # Model dimensions
    # -------------------------------------------------------------------------
    H_DIM: int = 192
    N_AFF_CONCEPTS: int = 6
    N_STYLE_CONCEPTS: int = 6
    DROPOUT: float = 0.25

    # -------------------------------------------------------------------------
    # Loss weights
    # -------------------------------------------------------------------------
    LAMBDA_AFF_CONCEPT: float = 1.50
    LAMBDA_STYLE_CONCEPT: float = 0.50
    LAMBDA_STYLE_SPEAKER: float = 0.50
    LAMBDA_AFF_SPK_ADV: float = 0.35
    LAMBDA_STYLE_EMO_ADV: float = 0.10
    LAMBDA_ORTH: float = 0.05

    # GRL schedule
    GRL_MAX_LAMBDA: float = 1.0

    # -------------------------------------------------------------------------
    # Ablation switches
    # -------------------------------------------------------------------------
    # "aff": emotion is predicted from affective concepts.
    # "encoder": plain SER baseline, emotion is predicted directly from encoder h.
    EMOTION_HEAD_INPUT: str = "aff"
    USE_AFF_CONCEPT_BRANCH: bool = True
    USE_AFF_CONCEPT_SUPERVISION: bool = True
    USE_AFF_SPEAKER_ADVERSARY: bool = True
    USE_STYLE_EMOTION_ADVERSARY: bool = True
    USE_ORTHOGONALITY: bool = True
    USE_STYLE_BRANCH: bool = True

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    SPEAKER_PROBE_REPEATS: int = 5
    SPEAKER_PROBE_TEST_SIZE: float = 0.30

    # -------------------------------------------------------------------------
    # Runtime
    # -------------------------------------------------------------------------
    DEVICE: str = "auto"  # auto -> mps on Apple Silicon, then cuda, then cpu


CFG = Config()


EMOTION_MAP = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5,
}

EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# IEMOCAP categorical labels live in the EmoEvaluation .txt files, not in the
# filenames. We use the canonical 4-class benchmark protocol: angry, happy,
# sad, neutral, with "excited" (exc) merged into "happy" (~5531 utterances).
IEMOCAP_EMOTION_MAP = {
    "ang": 0,
    "hap": 1,
    "exc": 1,  # excited folded into happy per the standard IEMOCAP 4-way setup
    "neu": 2,
    "sad": 3,
}

IEMOCAP_EMOTION_NAMES = ["angry", "happy", "neutral", "sad"]

DATASET_DISPLAY_NAMES = {
    "cremad": "CREMA-D",
    "iemocap": "IEMOCAP",
}

DATASET_ALIASES = {
    "cremad": "cremad",
    "crema-d": "cremad",
    "crema_d": "cremad",
    "crema": "cremad",
    "iemocap": "iemocap",
    "iemo-cap": "iemocap",
    "iemo_cap": "iemocap",
}

AFF_CONCEPT_NAMES = [
    "vocal_arousal",
    "pitch_instability",
    "energy_variability",
    "pause_hesitation",
    "voice_tension",
    "rhythm_irregularity",
]

STYLE_CONCEPT_NAMES = [
    "baseline_pitch_level",
    "habitual_loudness_level",
    "timbre_brightness",
    "spectral_breadth",
    "articulation_sharpness",
    "tempo_tendency",
]

LIBROSA_PRIMITIVE_NAMES = [
    "f0_mean",
    "f0_std",
    "f0_range",
    "rms_mean",
    "rms_std",
    "pause_ratio",
    "centroid_mean",
    "bandwidth_mean",
    "flatness_mean",
    "zcr_mean",
    "onset_rate",
    "onset_interval_cv",
]

