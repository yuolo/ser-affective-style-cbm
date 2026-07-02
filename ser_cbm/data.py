"""Utilities for the disentangled affective-style CBM SER experiments."""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import opensmile
except Exception:
    opensmile = None

from .config import (
    Config,
    DATASET_ALIASES,
    DATASET_DISPLAY_NAMES,
    EMOTION_MAP,
    EMOTION_NAMES,
    IEMOCAP_EMOTION_MAP,
    IEMOCAP_EMOTION_NAMES,
    LIBROSA_PRIMITIVE_NAMES,
)

# =============================================================================
# DATA DISCOVERY
# =============================================================================

def parse_cremad_filename(path: str) -> Optional[Dict[str, Any]]:
    """Parse CREMA-D filename.

    Example:
        1001_DFA_ANG_XX.wav
        actor=1001, sentence=DFA, emotion=ANG, intensity=XX
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    if len(parts) < 4:
        return None

    speaker = parts[0]
    emotion_code = parts[2]
    intensity = parts[3]

    if emotion_code not in EMOTION_MAP:
        return None

    return {
        "path": path,
        "speaker": speaker,
        "emotion_code": emotion_code,
        "emotion": EMOTION_MAP[emotion_code],
        "intensity": intensity,
        "filename": base,
    }


def discover_cremad(cremad_dir: str) -> pd.DataFrame:
    wavs = sorted(glob.glob(os.path.join(cremad_dir, "*.wav")))
    rows = []
    for p in wavs:
        item = parse_cremad_filename(p)
        if item is not None:
            rows.append(item)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise FileNotFoundError(
            f"No valid CREMA-D wav files found in: {cremad_dir}\n"
            "Expected filenames like 1001_DFA_ANG_XX.wav"
        )
    return df.reset_index(drop=True)


IEMOCAP_LABEL_LINE_RE = re.compile(
    r"^\[\d+\.\d+\s*-\s*\d+\.\d+\]\s+(\S+)\s+(\w+)\s+\["
)


def parse_iemocap_eval_file(eval_path: str, session_dir: str) -> List[Dict[str, Any]]:
    """Parse one IEMOCAP EmoEvaluation .txt file into labelled-utterance rows.

    The categorical emotion is the per-turn majority code assigned by the
    annotators. The speaker is the session id plus the gender flag of the
    speaking turn (the F/M in the turn name), giving 10 distinct speakers
    overall. The wav lives at
    <session>/sentences/wav/<dialog>/<turn>.wav.
    """
    rows: List[Dict[str, Any]] = []
    with open(eval_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = IEMOCAP_LABEL_LINE_RE.match(line)
            if match is None:
                continue
            turn_id, emotion_code = match.group(1), match.group(2)
            if emotion_code not in IEMOCAP_EMOTION_MAP:
                continue

            dialog = turn_id.rsplit("_", 1)[0]
            gender = turn_id.rsplit("_", 1)[1][0]
            wav_path = os.path.join(
                session_dir, "sentences", "wav", dialog, turn_id + ".wav"
            )
            if not os.path.exists(wav_path):
                continue

            rows.append(
                {
                    "path": wav_path,
                    "speaker": f"{turn_id[:5]}_{gender}",
                    "emotion_code": emotion_code,
                    "emotion": IEMOCAP_EMOTION_MAP[emotion_code],
                    "session": turn_id[:5],
                    "dialog": dialog,
                    "filename": turn_id + ".wav",
                    "dataset": "iemocap",
                }
            )
    return rows


def discover_iemocap(iemocap_dir: str) -> pd.DataFrame:
    eval_files = sorted(
        glob.glob(
            os.path.join(iemocap_dir, "Session*", "dialog", "EmoEvaluation", "*.txt")
        )
    )
    eval_files = [f for f in eval_files if not os.path.basename(f).startswith(".")]
    rows: List[Dict[str, Any]] = []
    for eval_path in eval_files:
        session_dir = eval_path.split(os.sep + "dialog" + os.sep)[0]
        rows.extend(parse_iemocap_eval_file(eval_path, session_dir))
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise FileNotFoundError(
            f"No valid IEMOCAP utterances found under: {iemocap_dir}\n"
            "Expected Session*/dialog/EmoEvaluation/*.txt label files and "
            "matching Session*/sentences/wav/<dialog>/<turn>.wav audio."
        )
    return df.reset_index(drop=True)


# =============================================================================
# DATASET DISPATCH
# =============================================================================

def normalize_dataset_name(dataset: str) -> str:
    key = str(dataset).strip().lower()
    if key not in DATASET_ALIASES:
        known = ", ".join(sorted(set(DATASET_ALIASES.values())))
        raise ValueError(f"Unknown dataset {dataset!r}. Expected one of: {known}")
    return DATASET_ALIASES[key]


def dataset_display_name(dataset: str) -> str:
    return DATASET_DISPLAY_NAMES[normalize_dataset_name(dataset)]


def emotion_map_for_dataset(dataset: str) -> Dict[str, int]:
    dataset_name = normalize_dataset_name(dataset)
    if dataset_name == "cremad":
        return dict(EMOTION_MAP)
    if dataset_name == "iemocap":
        return dict(IEMOCAP_EMOTION_MAP)
    raise ValueError(f"Unsupported dataset: {dataset}")


def emotion_names_for_dataset(dataset: str) -> List[str]:
    dataset_name = normalize_dataset_name(dataset)
    if dataset_name == "cremad":
        return list(EMOTION_NAMES)
    if dataset_name == "iemocap":
        return list(IEMOCAP_EMOTION_NAMES)
    raise ValueError(f"Unsupported dataset: {dataset}")


def dataset_dir_for_config(cfg: Config) -> str:
    dataset_name = normalize_dataset_name(cfg.DATASET)
    if dataset_name == "cremad":
        return str(cfg.CREMA_D_DIR)
    if dataset_name == "iemocap":
        return str(cfg.IEMOCAP_DIR)
    raise ValueError(f"Unsupported dataset: {cfg.DATASET}")


def discover_dataset(cfg: Config) -> pd.DataFrame:
    """Dataset-aware discovery: dispatches on cfg.DATASET."""
    dataset_name = normalize_dataset_name(cfg.DATASET)
    if dataset_name == "cremad":
        return discover_cremad(cfg.CREMA_D_DIR)
    if dataset_name == "iemocap":
        return discover_iemocap(cfg.IEMOCAP_DIR)
    raise ValueError(f"Unsupported dataset: {cfg.DATASET}")


# =============================================================================
# AUDIO PROCESSING / FEATURE EXTRACTION
# =============================================================================

def load_audio_fixed(
    path: str,
    sr: int,
    max_seconds: float,
    normalize_peak: bool = False,
) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    max_len = int(sr * max_seconds)
    if len(y) > max_len:
        y = y[:max_len]
    elif len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode="constant")
    y = y.astype(np.float32)
    if normalize_peak:
        peak = np.max(np.abs(y)) + 1e-8
        y = y / peak
    return y


def waveform_to_logmel(y: np.ndarray, cfg: Config) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.SR,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel


def safe_nan_to_num(x: np.ndarray, value: float = 0.0) -> np.ndarray:
    return np.nan_to_num(x, nan=value, posinf=value, neginf=value)


def make_opensmile_extractor(cfg: Config):
    if opensmile is None:
        raise ImportError(
            "opensmile is not installed. Install it with:\n"
            "    python -m pip install opensmile\n"
            "or set CFG.FEATURE_BACKEND = 'librosa' for the fallback ablation."
        )
    try:
        feature_set = getattr(opensmile.FeatureSet, cfg.OPENSMILE_FEATURE_SET)
    except Exception as exc:
        available = [x for x in dir(opensmile.FeatureSet) if not x.startswith("_")]
        raise ValueError(
            f"Unknown openSMILE FeatureSet: {cfg.OPENSMILE_FEATURE_SET}.\n"
            f"Available FeatureSet names include: {available}"
        ) from exc
    return opensmile.Smile(
        feature_set=feature_set,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_egemaps_features_from_signal(y: np.ndarray, sr: int, smile) -> Tuple[np.ndarray, List[str]]:
    """Extract eGeMAPS functionals from a fixed-length signal."""
    try:
        df_feat = smile.process_signal(y.astype(np.float32), sr)
    except Exception as exc:
        raise RuntimeError(f"openSMILE failed on an audio signal: {exc}") from exc

    if len(df_feat) == 0:
        raise RuntimeError("openSMILE returned an empty feature frame.")

    df_num = df_feat.apply(pd.to_numeric, errors="coerce")
    arr = df_num.to_numpy(dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] > 1:
        vec = np.nanmean(arr, axis=0)
    else:
        vec = arr.reshape(-1)
    names = [str(c) for c in df_num.columns.tolist()]
    return safe_nan_to_num(vec.astype(np.float32)), names


def extract_librosa_primitives(y: np.ndarray, cfg: Config) -> np.ndarray:
    """Fallback acoustic primitives for ablations/debugging.

    The main paper setting should use openSMILE/eGeMAPS.
    """
    hop = cfg.HOP_LENGTH
    frame_length = cfg.WIN_LENGTH

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    rms = safe_nan_to_num(rms)
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    if np.max(rms) > 1e-8:
        pause_threshold = max(1e-4, 0.10 * float(np.max(rms)))
        pause_ratio = float(np.mean(rms < pause_threshold))
    else:
        pause_ratio = 1.0

    try:
        f0 = librosa.yin(
            y,
            fmin=cfg.YIN_FMIN,
            fmax=cfg.YIN_FMAX,
            sr=cfg.SR,
            frame_length=cfg.YIN_FRAME_LENGTH,
            hop_length=hop,
        )
        f0 = safe_nan_to_num(f0)
        if len(rms) == len(f0) and np.max(rms) > 1e-8:
            voiced = rms > np.percentile(rms, 35)
            f0_voiced = f0[voiced]
        else:
            f0_voiced = f0
        f0_voiced = f0_voiced[(f0_voiced > cfg.YIN_FMIN) & (f0_voiced < cfg.YIN_FMAX)]
        if len(f0_voiced) < 3:
            f0_mean, f0_std, f0_range = 0.0, 0.0, 0.0
        else:
            f0_mean = float(np.mean(f0_voiced))
            f0_std = float(np.std(f0_voiced))
            f0_range = float(np.percentile(f0_voiced, 95) - np.percentile(f0_voiced, 5))
    except Exception:
        f0_mean, f0_std, f0_range = 0.0, 0.0, 0.0

    centroid = librosa.feature.spectral_centroid(y=y, sr=cfg.SR, n_fft=cfg.N_FFT, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=cfg.SR, n_fft=cfg.N_FFT, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=cfg.N_FFT, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop)[0]

    centroid_mean = float(np.mean(safe_nan_to_num(centroid)))
    bandwidth_mean = float(np.mean(safe_nan_to_num(bandwidth)))
    flatness_mean = float(np.mean(safe_nan_to_num(flatness)))
    zcr_mean = float(np.mean(safe_nan_to_num(zcr)))

    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=cfg.SR, hop_length=hop)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=cfg.SR, hop_length=hop)
        duration = len(y) / cfg.SR
        onset_rate = float(len(onset_frames) / max(duration, 1e-6))
        if len(onset_frames) >= 3:
            onset_times = librosa.frames_to_time(onset_frames, sr=cfg.SR, hop_length=hop)
            intervals = np.diff(onset_times)
            onset_interval_cv = float(np.std(intervals) / (np.mean(intervals) + 1e-6))
        else:
            onset_interval_cv = 0.0
    except Exception:
        onset_rate = 0.0
        onset_interval_cv = 0.0

    feats = np.array(
        [
            f0_mean,
            f0_std,
            f0_range,
            rms_mean,
            rms_std,
            pause_ratio,
            centroid_mean,
            bandwidth_mean,
            flatness_mean,
            zcr_mean,
            onset_rate,
            onset_interval_cv,
        ],
        dtype=np.float32,
    )
    return safe_nan_to_num(feats)


def concept_cache_csv_path(cfg: Config) -> str:
    safe_backend = str(cfg.FEATURE_BACKEND).lower().strip()
    safe_set = str(cfg.OPENSMILE_FEATURE_SET).strip()
    return os.path.join(cfg.OUT_DIR, f"concept_feature_cache_{safe_backend}_{safe_set}.csv")


def try_load_concept_feature_cache(
    df: pd.DataFrame,
    cfg: Config
) -> Optional[Tuple[Dict[str, np.ndarray], List[str]]]:
    if not cfg.CACHE_CONCEPT_FEATURES_TO_CSV or cfg.FORCE_REBUILD_CONCEPT_CACHE:
        return None

    path = concept_cache_csv_path(cfg)
    if not os.path.exists(path):
        return None

    try:
        feat_df = pd.read_csv(path)

        if "path" not in feat_df.columns:
            return None

        feat_df["path"] = feat_df["path"].astype(str)
        feat_df = feat_df.drop_duplicates(subset=["path"], keep="last")

        needed_paths = set(df["path"].astype(str).tolist())
        got_paths = set(feat_df["path"].astype(str).tolist())

        if not needed_paths.issubset(got_paths):
            return None

        feature_names = [c for c in feat_df.columns if c not in {"path", "filename"}]

        feat_df = feat_df.set_index("path")

        cache: Dict[str, np.ndarray] = {}

        for p in df["path"].astype(str).tolist():
            vec = (
                feat_df
                .loc[[p], feature_names]
                .to_numpy(dtype=np.float32)
                .reshape(-1)
            )
            cache[p] = safe_nan_to_num(vec.astype(np.float32))

        print(f"Loaded concept feature cache: {path}")
        return cache, feature_names

    except Exception as exc:
        print(f"Could not load concept feature cache; rebuilding. Reason: {exc}")
        return None


def save_concept_feature_cache(
    feature_cache: Dict[str, np.ndarray],
    feature_names: List[str],
    cfg: Config
) -> None:
    if not cfg.CACHE_CONCEPT_FEATURES_TO_CSV:
        return

    path = concept_cache_csv_path(cfg)

    try:
        rows: List[Dict[str, Any]] = []

        for p, vec in feature_cache.items():
            row: Dict[str, Any] = {
                "path": str(p),
                "filename": os.path.basename(str(p)),
            }

            for name, val in zip(feature_names, vec):
                row[str(name)] = float(val)

            rows.append(row)

        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved concept feature cache: {path}")

    except Exception as exc:
        print(f"Could not save concept feature cache: {exc}")


def build_feature_cache(df: pd.DataFrame, cfg: Config) -> Dict[str, Dict[str, np.ndarray]]:
    """Precompute log-mel and concept features.

    logmel is used by the neural encoder.
    concept_features are eGeMAPS functionals by default and are used only to build
    weak concept targets in a fold-specific way.
    """
    backend = str(cfg.FEATURE_BACKEND).lower().strip()
    if backend not in {"opensmile", "librosa"}:
        raise ValueError("CFG.FEATURE_BACKEND must be 'opensmile' or 'librosa'.")

    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    concept_feature_cache: Optional[Dict[str, np.ndarray]] = None
    concept_feature_names: Optional[List[str]] = None

    cached = try_load_concept_feature_cache(df, cfg)
    if cached is not None:
        concept_feature_cache, concept_feature_names = cached

    smile = None
    if concept_feature_cache is None:
        if backend == "opensmile":
            smile = make_opensmile_extractor(cfg)
            print(f"Using openSMILE feature set: {cfg.OPENSMILE_FEATURE_SET}")
        else:
            print("Using librosa fallback primitive features for concepts.")

    cache: Dict[str, Dict[str, np.ndarray]] = {}
    new_concept_feature_cache: Dict[str, np.ndarray] = {}
    final_feature_names: Optional[List[str]] = concept_feature_names

    for path in tqdm(df["path"].astype(str).tolist(), desc="Extracting logmel/concept features"):
        y_raw = load_audio_fixed(path, cfg.SR, cfg.MAX_SECONDS, normalize_peak=False)
        y_model = load_audio_fixed(path, cfg.SR, cfg.MAX_SECONDS, normalize_peak=True)
        logmel = waveform_to_logmel(y_model, cfg)

        if concept_feature_cache is not None:
            concept_vec = concept_feature_cache[path]
        else:
            if backend == "opensmile":
                concept_vec, names = extract_egemaps_features_from_signal(y_raw, cfg.SR, smile)
            else:
                concept_vec = extract_librosa_primitives(y_raw, cfg)
                names = list(LIBROSA_PRIMITIVE_NAMES)

            if final_feature_names is None:
                final_feature_names = names
            elif len(names) != len(final_feature_names):
                raise RuntimeError("Inconsistent concept feature dimensionality across files.")

            new_concept_feature_cache[path] = concept_vec

        cache[path] = {
            "logmel": logmel,
            "concept_features": concept_vec.astype(np.float32),
        }

    if concept_feature_cache is None and final_feature_names is not None:
        save_concept_feature_cache(new_concept_feature_cache, final_feature_names, cfg)

    if final_feature_names is None:
        raise RuntimeError("No concept feature names were extracted.")

    cache["__concept_feature_names__"] = {"names": np.array(final_feature_names, dtype=object)}
    return cache
