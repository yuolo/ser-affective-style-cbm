"""Utilities for the disentangled affective-style CBM SER experiments."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config

# =============================================================================
# BASELINE / DEVIATION CONCEPT TARGETS
# =============================================================================

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _lower_names(feature_names: List[str]) -> List[str]:
    return [str(n).lower() for n in feature_names]


def _match_indices(
    feature_names: List[str],
    any_of: Tuple[str, ...],
    all_of: Tuple[str, ...] = (),
    none_of: Tuple[str, ...] = (),
) -> List[int]:
    names_l = _lower_names(feature_names)
    idxs = []
    any_l = tuple(s.lower() for s in any_of)
    all_l = tuple(s.lower() for s in all_of)
    none_l = tuple(s.lower() for s in none_of)
    for i, name in enumerate(names_l):
        if any_l and not any(s in name for s in any_l):
            continue
        if all_l and not all(s in name for s in all_l):
            continue
        if none_l and any(s in name for s in none_l):
            continue
        idxs.append(i)
    return idxs


def _group_value(z: np.ndarray, idxs: List[int]) -> np.ndarray:
    if len(idxs) == 0:
        return np.zeros(z.shape[0], dtype=np.float32)
    return np.mean(z[:, idxs], axis=1).astype(np.float32)


def _feature_groups(feature_names: List[str]) -> Dict[str, List[int]]:
    """Map eGeMAPS/librosa names into semantic feature groups.

    The matching is intentionally permissive because eGeMAPS column names vary
    slightly across openSMILE versions.
    """
    groups: Dict[str, List[int]] = {}

    groups["f0_mean"] = _match_indices(
        feature_names,
        any_of=("f0", "pitch"),
        none_of=("stddev", "std", "range", "pctl", "percentile"),
    )
    groups["f0_var"] = _match_indices(
        feature_names,
        any_of=("f0", "pitch"),
        none_of=(),
    )
    groups["f0_var"] = [
        i for i in groups["f0_var"]
        if any(tok in feature_names[i].lower() for tok in ("std", "range", "pctl", "percentile"))
    ]

    groups["loud_mean"] = _match_indices(
        feature_names,
        any_of=("loudness", "rms", "energy"),
        none_of=("stddev", "std", "range", "pctl", "percentile"),
    )
    groups["loud_var"] = _match_indices(
        feature_names,
        any_of=("loudness", "rms", "energy"),
    )
    groups["loud_var"] = [
        i for i in groups["loud_var"]
        if any(tok in feature_names[i].lower() for tok in ("std", "range", "pctl", "percentile"))
    ]

    groups["spectral_flux"] = _match_indices(feature_names, any_of=("spectralflux", "spectral_flux"))
    groups["brightness"] = _match_indices(
        feature_names,
        any_of=("alpharatio", "hammarberg", "centroid", "brightness"),
    )
    groups["spectral_breadth"] = _match_indices(
        feature_names,
        any_of=("mfcc", "slope", "bandwidth", "lspfrequency", "spectral"),
    )
    groups["flatness_zcr"] = _match_indices(feature_names, any_of=("flatness", "zcr", "zero_crossing"))

    groups["jitter"] = _match_indices(feature_names, any_of=("jitter",))
    groups["shimmer"] = _match_indices(feature_names, any_of=("shimmer",))
    groups["hnr"] = _match_indices(feature_names, any_of=("hnr", "harmonic"))
    groups["voice_quality"] = sorted(set(groups["jitter"] + groups["shimmer"] + groups["hnr"]))

    groups["voiced_rate"] = _match_indices(feature_names, any_of=("voicedsegmentspersec", "voicedsegment", "onset_rate"))
    groups["segment_lengths"] = _match_indices(
        feature_names,
        any_of=("voicedsegmentlength", "unvoicedsegmentlength", "segmentlength", "onset_interval"),
    )
    groups["pause"] = _match_indices(feature_names, any_of=("unvoiced", "pause", "silence"))

    direct = {name: i for i, name in enumerate(feature_names)}
    for k in ["f0_mean", "f0_std", "f0_range", "rms_mean", "rms_std", "pause_ratio",
              "centroid_mean", "bandwidth_mean", "flatness_mean", "zcr_mean",
              "onset_rate", "onset_interval_cv"]:
        if k in direct:
            idx = direct[k]
            if k == "f0_mean":
                groups["f0_mean"] = sorted(set(groups["f0_mean"] + [idx]))
            elif k in {"f0_std", "f0_range"}:
                groups["f0_var"] = sorted(set(groups["f0_var"] + [idx]))
            elif k == "rms_mean":
                groups["loud_mean"] = sorted(set(groups["loud_mean"] + [idx]))
            elif k == "rms_std":
                groups["loud_var"] = sorted(set(groups["loud_var"] + [idx]))
            elif k == "pause_ratio":
                groups["pause"] = sorted(set(groups["pause"] + [idx]))
            elif k in {"centroid_mean", "bandwidth_mean"}:
                groups["brightness"] = sorted(set(groups["brightness"] + [idx]))
                groups["spectral_breadth"] = sorted(set(groups["spectral_breadth"] + [idx]))
            elif k in {"flatness_mean", "zcr_mean"}:
                groups["flatness_zcr"] = sorted(set(groups["flatness_zcr"] + [idx]))
            elif k == "onset_rate":
                groups["voiced_rate"] = sorted(set(groups["voiced_rate"] + [idx]))
            elif k == "onset_interval_cv":
                groups["segment_lengths"] = sorted(set(groups["segment_lengths"] + [idx]))

    return groups


def write_feature_group_report(feature_names: List[str], cfg: Config) -> None:
    groups = _feature_groups(feature_names)
    rows = []
    for group, idxs in groups.items():
        rows.append({
            "group": group,
            "n_features": len(idxs),
            "features": "; ".join([feature_names[i] for i in idxs[:25]]),
        })
    path = os.path.join(cfg.OUT_DIR, "concept_feature_group_report.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print("Concept feature group report:", path)


def compute_speaker_baselines(z: np.ndarray, speakers: np.ndarray) -> Dict[str, np.ndarray]:
    speakers = np.asarray(speakers, dtype=str)
    baselines: Dict[str, np.ndarray] = {}
    for spk in sorted(np.unique(speakers).tolist()):
        mask = speakers == spk
        if np.any(mask):
            baselines[str(spk)] = np.mean(z[mask], axis=0).astype(np.float32)
    return baselines


def baseline_matrix_for_rows(
    z: np.ndarray,
    speakers: np.ndarray,
    baseline_by_speaker: Dict[str, np.ndarray],
    fallback_global: np.ndarray,
) -> np.ndarray:
    speakers = np.asarray(speakers, dtype=str)
    rows = []
    for spk in speakers:
        rows.append(baseline_by_speaker.get(str(spk), fallback_global))
    return np.stack(rows, axis=0).astype(np.float32)


def build_concepts_from_baseline_and_deviation(
    z: np.ndarray,
    baseline_z: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build B+C weak targets.

    style targets are computed from speaker baseline features.
    affective targets are computed from utterance deviation from speaker baseline.
    """
    dev_z = z - baseline_z
    groups = _feature_groups(feature_names)

    f0_mean_d = _group_value(dev_z, groups["f0_mean"])
    f0_var_d = _group_value(dev_z, groups["f0_var"])
    loud_mean_d = _group_value(dev_z, groups["loud_mean"])
    loud_var_d = _group_value(dev_z, groups["loud_var"])
    spectral_flux_d = _group_value(dev_z, groups["spectral_flux"])
    brightness_d = _group_value(dev_z, groups["brightness"])
    flatness_zcr_d = _group_value(dev_z, groups["flatness_zcr"])
    voice_quality_d = _group_value(dev_z, groups["voice_quality"])
    hnr_d = _group_value(dev_z, groups["hnr"])
    pause_d = _group_value(dev_z, groups["pause"])
    voiced_rate_d = _group_value(dev_z, groups["voiced_rate"])
    segment_len_d = _group_value(dev_z, groups["segment_lengths"])

    vocal_arousal = 0.50 * loud_mean_d + 0.30 * f0_mean_d + 0.20 * spectral_flux_d
    pitch_instability = 0.70 * f0_var_d + 0.30 * voice_quality_d
    energy_variability = 0.80 * loud_var_d + 0.20 * spectral_flux_d
    pause_hesitation = 0.70 * pause_d - 0.30 * voiced_rate_d
    voice_tension = 0.35 * voice_quality_d + 0.25 * brightness_d + 0.20 * flatness_zcr_d - 0.20 * hnr_d
    rhythm_irregularity = 0.60 * segment_len_d + 0.25 * np.abs(voiced_rate_d) + 0.15 * np.abs(loud_var_d)

    aff_raw = np.stack(
        [
            vocal_arousal,
            pitch_instability,
            energy_variability,
            pause_hesitation,
            voice_tension,
            rhythm_irregularity,
        ],
        axis=1,
    )

    f0_mean_b = _group_value(baseline_z, groups["f0_mean"])
    loud_mean_b = _group_value(baseline_z, groups["loud_mean"])
    brightness_b = _group_value(baseline_z, groups["brightness"])
    spectral_breadth_b = _group_value(baseline_z, groups["spectral_breadth"])
    flatness_zcr_b = _group_value(baseline_z, groups["flatness_zcr"])
    spectral_flux_b = _group_value(baseline_z, groups["spectral_flux"])
    voice_quality_b = _group_value(baseline_z, groups["voice_quality"])
    voiced_rate_b = _group_value(baseline_z, groups["voiced_rate"])

    baseline_pitch_level = f0_mean_b
    habitual_loudness_level = loud_mean_b
    timbre_brightness = 0.60 * brightness_b + 0.20 * spectral_flux_b + 0.20 * flatness_zcr_b
    spectral_breadth = spectral_breadth_b
    articulation_sharpness = 0.45 * flatness_zcr_b + 0.35 * spectral_flux_b + 0.20 * voice_quality_b
    tempo_tendency = voiced_rate_b

    style_raw = np.stack(
        [
            baseline_pitch_level,
            habitual_loudness_level,
            timbre_brightness,
            spectral_breadth,
            articulation_sharpness,
            tempo_tendency,
        ],
        axis=1,
    )

    aff_targets = sigmoid_np(np.clip(aff_raw, -4.0, 4.0)).astype(np.float32)
    style_targets = sigmoid_np(np.clip(style_raw, -4.0, 4.0)).astype(np.float32)
    return aff_targets, style_targets

