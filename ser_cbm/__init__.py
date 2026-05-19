# -*- coding: utf-8 -*-
"""Public API for the SER concept bottleneck code release."""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

from .config import (
    AFF_CONCEPT_NAMES,
    CFG,
    DATASET_ALIASES,
    DATASET_DISPLAY_NAMES,
    EMOTION_MAP,
    EMOTION_NAMES,
    IEMOCAP_EMOTION_MAP,
    IEMOCAP_EMOTION_NAMES,
    LIBROSA_PRIMITIVE_NAMES,
    STYLE_CONCEPT_NAMES,
    Config,
)
from .concepts import (
    baseline_matrix_for_rows,
    build_concepts_from_baseline_and_deviation,
    compute_speaker_baselines,
    write_feature_group_report,
)
from .data import (
    build_feature_cache,
    concept_cache_csv_path,
    dataset_dir_for_config,
    dataset_display_name,
    discover_cremad,
    discover_dataset,
    discover_iemocap,
    emotion_map_for_dataset,
    emotion_names_for_dataset,
    extract_egemaps_features_from_signal,
    extract_librosa_primitives,
    load_audio_fixed,
    make_opensmile_extractor,
    normalize_dataset_name,
    parse_cremad_filename,
    parse_iemocap_eval_file,
    safe_nan_to_num,
    save_concept_feature_cache,
    try_load_concept_feature_cache,
    waveform_to_logmel,
)
from .datasets import CREMADConceptDataset
from .diagnostics import (
    build_speaker_probe_chance_summary,
    concept_intervention_diagnostics,
    dataframe_to_markdown_table,
    fit_logistic_regression_probe,
    prefix_metrics,
    speaker_leakage_audit,
    speaker_probe_accuracy,
    write_concept_mae_by_concept,
    write_dataset_statistics,
    write_factorization_audit_figure,
    write_main_claim_summary,
    write_speaker_probe_chance_summary,
)
from .experiment import run_experiment
from .metrics import (
    batch_correlation_penalty,
    class_weights_from_labels,
    compute_metrics,
    ece_score,
    grl_schedule,
)
from .models import CRNNEncoder, DisentangledAffectiveStyleCBM, GradReverseFn, grad_reverse
from .runtime import (
    clear_device_cache,
    configure_torch_runtime,
    empty_mps_cache_safely,
    is_mps_available,
    seed_mps_safely,
    select_device,
    set_seed,
)
from .training import evaluate_model, make_loaders_for_fold, train_one_epoch
