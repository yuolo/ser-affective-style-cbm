# -*- coding: utf-8 -*-
"""
Run paper baselines and collect a single comparison table.

Baselines:
    B1  eGeMAPS + Logistic Regression / Linear SVM
    B3  Plain CRNN SER encoder
    B4  Concept-only CBM
    B5  Dual affect-style, no adversaries
    B6  Full regularized model

The eGeMAPS baselines use the exact same outer GroupKFold speaker-disjoint
train/test indices as the neural experiments in main_egemaps_baseline_deviation_cbm.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

from ser_cbm import (
    Config,
    build_feature_cache,
    dataset_display_name,
    discover_cremad,
    discover_dataset,
    normalize_dataset_name,
    run_experiment,
    write_dataset_statistics,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "baseline_comparison_outputs"
DEFAULT_DATA_DIRS = {
    "cremad": PROJECT_ROOT / "AudioWAV",
    "iemocap": PROJECT_ROOT / "iemocap",
}
DEFAULT_CACHE_SOURCE = (
    PROJECT_ROOT
    / "si_affcbm_egemaps_bd_outputs"
    / "concept_feature_cache_opensmile_eGeMAPSv02.csv"
)


NEURAL_BASELINES: Dict[str, Dict[str, Any]] = {
    "B3_plain_crnn_ser": {
        "label": "B3 Plain CRNN SER",
        "overrides": {
            "EMOTION_HEAD_INPUT": "encoder",
            "USE_AFF_CONCEPT_BRANCH": False,
            "USE_AFF_CONCEPT_SUPERVISION": False,
            "USE_AFF_SPEAKER_ADVERSARY": False,
            "USE_STYLE_BRANCH": False,
            "USE_STYLE_EMOTION_ADVERSARY": False,
            "USE_ORTHOGONALITY": False,
        },
    },
    "B4_concept_only_cbm": {
        "label": "B4 Concept-only CBM",
        "overrides": {
            "EMOTION_HEAD_INPUT": "aff",
            "USE_AFF_CONCEPT_BRANCH": True,
            "USE_AFF_CONCEPT_SUPERVISION": True,
            "USE_AFF_SPEAKER_ADVERSARY": False,
            "USE_STYLE_BRANCH": False,
            "USE_STYLE_EMOTION_ADVERSARY": False,
            "USE_ORTHOGONALITY": False,
        },
    },
    "B5_dual_no_adversaries": {
        "label": "B5 Dual affect-style, no adversaries",
        "overrides": {
            "EMOTION_HEAD_INPUT": "aff",
            "USE_AFF_CONCEPT_BRANCH": True,
            "USE_AFF_CONCEPT_SUPERVISION": True,
            "USE_STYLE_BRANCH": True,
            "USE_AFF_SPEAKER_ADVERSARY": False,
            "USE_STYLE_EMOTION_ADVERSARY": False,
            "USE_ORTHOGONALITY": False,
        },
    },
    "B6_full_regularized": {
        "label": "B6 Full regularized model",
        "overrides": {
            "EMOTION_HEAD_INPUT": "aff",
            "USE_AFF_CONCEPT_BRANCH": True,
            "USE_AFF_CONCEPT_SUPERVISION": True,
            "USE_STYLE_BRANCH": True,
            "USE_AFF_SPEAKER_ADVERSARY": True,
            "USE_STYLE_EMOTION_ADVERSARY": True,
            "USE_ORTHOGONALITY": True,
        },
    },
}


EGEMAPS_BASELINES: Dict[str, Dict[str, Any]] = {
    "B1_egemaps_lr": {
        "label": "B1 eGeMAPS + LR",
        "classifier": "lr",
    },
    "B1_egemaps_svm": {
        "label": "B1 eGeMAPS + Linear SVM",
        "classifier": "svm",
    },
}


DEFAULT_MODELS = [
    "B1_egemaps_lr",
    "B1_egemaps_svm",
    "B3_plain_crnn_ser",
    "B4_concept_only_cbm",
    "B5_dual_no_adversaries",
    "B6_full_regularized",
]


TABLE_COLUMNS = [
    "model",
    "uar_mean",
    "uar_std",
    "macro_f1_mean",
    "macro_f1_std",
    "acc_mean",
    "acc_std",
    "speaker_probe_aff_mean",
    "speaker_probe_aff_uniform_chance_mean",
    "speaker_probe_aff_majority_chance_mean",
    "speaker_probe_style_mean",
    "speaker_probe_style_uniform_chance_mean",
    "speaker_probe_style_majority_chance_mean",
    "style_swap_consistency_mean",
    "aff_swap_sensitivity_mean",
]

MEAN_STD_TABLE_SPECS = [
    ("UAR ↑", "uar_mean", "uar_std"),
    ("Macro-F1 ↑", "macro_f1_mean", "macro_f1_std"),
    ("Accuracy ↑", "acc_mean", "acc_std"),
    ("Speaker probe aff ↓", "speaker_probe_aff_mean", None),
    ("Aff uniform chance", "speaker_probe_aff_uniform_chance_mean", None),
    ("Aff majority chance", "speaker_probe_aff_majority_chance_mean", None),
    ("Speaker probe style ↑", "speaker_probe_style_mean", None),
    ("Style uniform chance", "speaker_probe_style_uniform_chance_mean", None),
    ("Style majority chance", "speaker_probe_style_majority_chance_mean", None),
    ("Style swap ↑", "style_swap_consistency_mean", None),
    ("Affect swap ↑", "aff_swap_sensitivity_mean", None),
]


def _apply_overrides(cfg: Config, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Config has no field named {key!r}")
        setattr(cfg, key, value)


def _apply_dataset(cfg: Config, dataset: str, data_dir: Optional[Path] = None) -> None:
    dataset_name = normalize_dataset_name(dataset)
    cfg.DATASET = dataset_name
    resolved_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIRS[dataset_name]
    if dataset_name == "cremad":
        cfg.CREMA_D_DIR = str(resolved_dir)
    elif dataset_name == "iemocap":
        cfg.IEMOCAP_DIR = str(resolved_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def _default_output_root(dataset: str, requested_output_root: str) -> Path:
    dataset_name = normalize_dataset_name(dataset)
    if requested_output_root == str(DEFAULT_OUTPUT_ROOT) and dataset_name != "cremad":
        return PROJECT_ROOT / f"{dataset_name}_baseline_comparison_outputs"
    return Path(requested_output_root).expanduser().resolve()


def build_neural_config(
    model_key: str,
    output_root: Path,
    quick: bool = False,
    dataset: str = "cremad",
    data_dir: Optional[Path] = None,
) -> Config:
    if model_key not in NEURAL_BASELINES:
        raise KeyError(f"Unknown neural baseline: {model_key}")

    cfg = Config()
    _apply_dataset(cfg, dataset, data_dir=data_dir)
    cfg.OUT_DIR = str(output_root / model_key)
    _apply_overrides(cfg, NEURAL_BASELINES[model_key]["overrides"])

    if quick:
        cfg.N_SPLITS = 2
        cfg.NUM_EPOCHS = 2
        cfg.PATIENCE = 2
        cfg.SPEAKER_PROBE_REPEATS = 2

    return cfg


def build_egemaps_config(
    model_key: str,
    output_root: Path,
    quick: bool = False,
    dataset: str = "cremad",
    data_dir: Optional[Path] = None,
) -> Config:
    if model_key not in EGEMAPS_BASELINES:
        raise KeyError(f"Unknown eGeMAPS baseline: {model_key}")

    cfg = Config()
    _apply_dataset(cfg, dataset, data_dir=data_dir)
    cfg.OUT_DIR = str(output_root / model_key)
    cfg.FEATURE_BACKEND = "opensmile"

    if quick:
        cfg.N_SPLITS = 2

    return cfg


def _copy_concept_cache_if_available(cfg: Config, cache_source: Path, force: bool = False) -> None:
    if str(cfg.FEATURE_BACKEND).lower().strip() != "opensmile":
        return
    if not cfg.CACHE_CONCEPT_FEATURES_TO_CSV:
        return
    if cfg.FORCE_REBUILD_CONCEPT_CACHE:
        return
    if not cache_source.exists():
        return

    out_dir = Path(cfg.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dest = out_dir / f"concept_feature_cache_opensmile_{cfg.OPENSMILE_FEATURE_SET}.csv"
    if cache_dest.exists() and not force:
        return
    shutil.copy2(cache_source, cache_dest)


def write_run_config(cfg: Config, model_key: str, label: str, overrides: Dict[str, Any]) -> None:
    out_dir = Path(cfg.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": model_key,
        "label": label,
        "overrides": overrides,
        "config": asdict(cfg),
    }
    with (out_dir / "baseline_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _feature_columns(feat_df: pd.DataFrame) -> List[str]:
    return [c for c in feat_df.columns if c not in {"path", "filename"}]


def load_egemaps_matrix_from_cache(df: pd.DataFrame, cache_source: Path) -> Optional[Tuple[np.ndarray, List[str]]]:
    if not cache_source.exists():
        return None

    feat_df = pd.read_csv(cache_source)
    if "path" not in feat_df.columns:
        return None

    feat_df["path"] = feat_df["path"].astype(str)
    feat_df = feat_df.drop_duplicates(subset=["path"], keep="last")
    needed_paths = set(df["path"].astype(str).tolist())
    got_paths = set(feat_df["path"].astype(str).tolist())
    if not needed_paths.issubset(got_paths):
        return None

    feature_names = _feature_columns(feat_df)
    feat_df = feat_df.set_index("path")
    X = (
        feat_df
        .loc[df["path"].astype(str).tolist(), feature_names]
        .to_numpy(dtype=np.float32)
    )
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X, feature_names


def load_egemaps_matrix(df: pd.DataFrame, cfg: Config, cache_source: Path) -> Tuple[np.ndarray, List[str]]:
    cached = load_egemaps_matrix_from_cache(df, cache_source)
    if cached is not None:
        return cached

    cache = build_feature_cache(df, cfg)
    feature_names = [str(x) for x in cache["__concept_feature_names__"]["names"].tolist()]  # type: ignore
    X = np.stack(
        [cache[str(p)]["concept_features"] for p in df["path"].astype(str).tolist()],
        axis=0,
    ).astype(np.float32)
    return X, feature_names


def make_classifier(kind: str, seed: int) -> Pipeline:
    if kind == "lr":
        clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear",
                tol=1e-3,
                random_state=seed,
            )
        )
    elif kind == "svm":
        clf = LinearSVC(
            class_weight="balanced",
            random_state=seed,
            max_iter=20000,
            dual=False,
        )
    else:
        raise ValueError(f"Unknown classifier kind: {kind}")

    return Pipeline([
        ("scaler", RobustScaler()),
        ("clf", clf),
    ])


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "uar": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_egemaps_baseline(
    model_key: str,
    output_root: Path,
    cache_source: Path,
    quick: bool = False,
    force_cache_copy: bool = False,
    dataset: str = "cremad",
    data_dir: Optional[Path] = None,
) -> None:
    spec = EGEMAPS_BASELINES[model_key]
    cfg = build_egemaps_config(model_key, output_root, quick=quick, dataset=dataset, data_dir=data_dir)
    out_dir = Path(cfg.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_concept_cache_if_available(cfg, cache_source, force=force_cache_copy)
    write_run_config(cfg, model_key, spec["label"], {"classifier": spec["classifier"]})

    df = discover_dataset(cfg)
    write_dataset_statistics(df, cfg)
    X, feature_names = load_egemaps_matrix(df, cfg, cache_source)
    y = df["emotion"].to_numpy(dtype=np.int64)
    groups = df["speaker"].astype(str).to_numpy(dtype=str)
    idx_all = np.arange(len(df))

    rows = []
    pred_rows = []
    outer = GroupKFold(n_splits=cfg.N_SPLITS)
    for fold, (train_idx, test_idx) in enumerate(outer.split(idx_all, y, groups), start=1):
        model = make_classifier(str(spec["classifier"]), cfg.SEED + fold)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx]).astype(np.int64)
        metrics = metric_row(y[test_idx], pred)

        train_speakers = set(groups[train_idx].tolist())
        test_speakers = set(groups[test_idx].tolist())
        rows.append({
            "fold": fold,
            "test_acc": metrics["acc"],
            "test_uar": metrics["uar"],
            "test_macro_f1": metrics["macro_f1"],
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_train_speakers": int(len(train_speakers)),
            "n_test_speakers": int(len(test_speakers)),
            "n_train_test_speaker_overlap": int(len(train_speakers & test_speakers)),
        })

        fold_pred = df.iloc[test_idx].copy()
        fold_pred["fold"] = fold
        fold_pred["y_true"] = y[test_idx]
        fold_pred["y_pred"] = pred
        pred_rows.append(fold_pred)

    pd.DataFrame(rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.concat(pred_rows, axis=0).reset_index(drop=True).to_csv(out_dir / "test_predictions.csv", index=False)

    feature_report = pd.DataFrame({
        "feature": feature_names,
        "index": np.arange(len(feature_names), dtype=np.int64),
    })
    feature_report.to_csv(out_dir / "egemaps_feature_names.csv", index=False)


def run_neural_baseline(
    model_key: str,
    output_root: Path,
    cache_source: Path,
    quick: bool = False,
    force_cache_copy: bool = False,
    dataset: str = "cremad",
    data_dir: Optional[Path] = None,
) -> None:
    spec = NEURAL_BASELINES[model_key]
    cfg = build_neural_config(model_key, output_root, quick=quick, dataset=dataset, data_dir=data_dir)
    _copy_concept_cache_if_available(cfg, cache_source, force=force_cache_copy)
    write_run_config(cfg, model_key, spec["label"], dict(spec["overrides"]))
    run_experiment(cfg)


def _metric_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").mean(skipna=True))


def summarize_model(model_key: str, output_root: Path) -> Dict[str, Any]:
    if model_key in EGEMAPS_BASELINES:
        label = EGEMAPS_BASELINES[model_key]["label"]
    else:
        label = NEURAL_BASELINES[model_key]["label"]

    metrics_path = output_root / model_key / "fold_metrics.csv"
    row: Dict[str, Any] = {
        "model_key": model_key,
        "model": label,
        "status": "missing",
        "folds": 0,
    }
    if not metrics_path.exists():
        return row

    df = pd.read_csv(metrics_path)
    row.update({
        "status": "ok",
        "folds": int(len(df)),
        "uar_mean": _metric_mean(df, "test_uar"),
        "uar_std": float(pd.to_numeric(df.get("test_uar"), errors="coerce").std(skipna=True)) if "test_uar" in df else float("nan"),
        "macro_f1_mean": _metric_mean(df, "test_macro_f1"),
        "macro_f1_std": float(pd.to_numeric(df.get("test_macro_f1"), errors="coerce").std(skipna=True)) if "test_macro_f1" in df else float("nan"),
        "acc_mean": _metric_mean(df, "test_acc"),
        "acc_std": float(pd.to_numeric(df.get("test_acc"), errors="coerce").std(skipna=True)) if "test_acc" in df else float("nan"),
        "speaker_probe_aff_mean": _metric_mean(df, "speaker_probe_aff_acc"),
        "speaker_probe_aff_uniform_chance_mean": _metric_mean(df, "speaker_probe_aff_probe_chance_uniform"),
        "speaker_probe_aff_majority_chance_mean": _metric_mean(df, "speaker_probe_aff_probe_chance_majority"),
        "speaker_probe_style_mean": _metric_mean(df, "speaker_probe_style_acc"),
        "speaker_probe_style_uniform_chance_mean": _metric_mean(df, "speaker_probe_style_probe_chance_uniform"),
        "speaker_probe_style_majority_chance_mean": _metric_mean(df, "speaker_probe_style_probe_chance_majority"),
        "style_swap_consistency_mean": _metric_mean(df, "style_swap_consistency"),
        "aff_swap_sensitivity_mean": _metric_mean(df, "aff_swap_sensitivity"),
    })

    if model_key == "B3_plain_crnn_ser":
        # Plain CRNN has no c_aff/c_style; concept/swap probes from disabled branches
        # are dummy diagnostics and must not be reported as evidence.
        row["speaker_probe_aff_mean"] = float("nan")
        row["speaker_probe_aff_uniform_chance_mean"] = float("nan")
        row["speaker_probe_aff_majority_chance_mean"] = float("nan")
        row["speaker_probe_style_mean"] = float("nan")
        row["speaker_probe_style_uniform_chance_mean"] = float("nan")
        row["speaker_probe_style_majority_chance_mean"] = float("nan")
        row["style_swap_consistency_mean"] = float("nan")
        row["aff_swap_sensitivity_mean"] = float("nan")

    if model_key == "B4_concept_only_cbm":
        # Concept-only CBM has no style branch; style probes/swaps are not defined.
        row["speaker_probe_style_mean"] = float("nan")
        row["speaker_probe_style_uniform_chance_mean"] = float("nan")
        row["speaker_probe_style_majority_chance_mean"] = float("nan")
        row["style_swap_consistency_mean"] = float("nan")

    return row


def _format_table_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.4f}"
    if pd.isna(value):
        return "NA"
    return str(value)


def _format_mean_std(mean: Any, std: Any = None) -> str:
    try:
        mean_f = float(mean)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(mean_f):
        return "NA"

    if std is None:
        return f"{mean_f:.4f}"

    try:
        std_f = float(std)
    except (TypeError, ValueError):
        return f"{mean_f:.4f}"
    if not np.isfinite(std_f):
        return f"{mean_f:.4f}"
    return f"{mean_f:.4f} ± {std_f:.4f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    body = [[_format_table_value(row[c]) for c in df.columns] for _, row in df.iterrows()]
    widths = [max(len(headers[i]), *(len(row[i]) for row in body)) for i in range(len(headers))]

    def fmt(values: List[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(headers), sep] + [fmt(row) for row in body]) + "\n"


def build_mean_std_display_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        display: Dict[str, str] = {"Model": str(row.get("model", "NA"))}
        for label, mean_col, std_col in MEAN_STD_TABLE_SPECS:
            if mean_col not in summary.columns:
                continue
            display[label] = _format_mean_std(
                row.get(mean_col, float("nan")),
                row.get(std_col, None) if std_col else None,
            )
        rows.append(display)
    return pd.DataFrame(rows)


def paired_model_gap_summary(
    output_root: Path,
    model_a: str = "B3_plain_crnn_ser",
    model_b: str = "B5_dual_no_adversaries",
) -> Tuple[pd.DataFrame, str]:
    path_a = output_root / model_a / "fold_metrics.csv"
    path_b = output_root / model_b / "fold_metrics.csv"
    rows = []

    if not path_a.exists() or not path_b.exists():
        msg = "Paired Plain CRNN vs B5 summary unavailable: missing fold_metrics.csv."
        return pd.DataFrame(rows), msg

    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    merged = a.merge(b, on="fold", suffixes=("_plain_crnn", "_b5"))
    if merged.empty:
        msg = "Paired Plain CRNN vs B5 summary unavailable: no shared folds."
        return pd.DataFrame(rows), msg

    specs = [
        ("UAR", "test_uar"),
        ("Macro-F1", "test_macro_f1"),
        ("Accuracy", "test_acc"),
    ]
    for label, col in specs:
        a_col = f"{col}_plain_crnn"
        b_col = f"{col}_b5"
        if a_col not in merged.columns or b_col not in merged.columns:
            continue
        gap = pd.to_numeric(merged[a_col], errors="coerce") - pd.to_numeric(merged[b_col], errors="coerce")
        valid = gap.dropna()
        rows.append({
            "metric": label,
            "plain_crnn_wins": int((valid > 0).sum()),
            "b5_wins": int((valid < 0).sum()),
            "ties": int((valid == 0).sum()),
            "folds": int(len(valid)),
            "mean_gap_plain_minus_b5": float(valid.mean()) if len(valid) else float("nan"),
            "std_gap": float(valid.std()) if len(valid) else float("nan"),
        })

    paired = pd.DataFrame(rows)
    if paired.empty:
        msg = "Paired Plain CRNN vs B5 summary unavailable: metrics missing."
        return paired, msg

    uar_row = paired[paired["metric"] == "UAR"].iloc[0]
    msg = (
        "Plain CRNN outperformed B5 in "
        f"{int(uar_row['plain_crnn_wins'])}/{int(uar_row['folds'])} folds; "
        f"mean paired UAR gap = {float(uar_row['mean_gap_plain_minus_b5']):.4f} "
        "(Plain CRNN - B5)."
    )
    return paired, msg


def write_comparison_summary(output_root: Path, models: Iterable[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [summarize_model(model_key, output_root) for model_key in models]
    summary = pd.DataFrame(rows)

    summary_path = output_root / "baseline_comparison_summary.csv"
    table_path = output_root / "paper_model_comparison_table.md"
    paired_csv_path = output_root / "paired_plain_crnn_vs_b5.csv"
    paired_txt_path = output_root / "paired_plain_crnn_vs_b5.txt"
    summary.to_csv(summary_path, index=False)

    table = build_mean_std_display_table(summary)
    paired, paired_msg = paired_model_gap_summary(output_root)
    paired.to_csv(paired_csv_path, index=False)

    with table_path.open("w", encoding="utf-8") as f:
        f.write("# Paper Model Comparison\n\n")
        f.write(dataframe_to_markdown(table))
        f.write("\n")
        f.write("## Paired Plain CRNN vs B5\n\n")
        f.write(paired_msg + "\n")
        if not paired.empty:
            f.write("\n")
            f.write(dataframe_to_markdown(paired))

    with paired_txt_path.open("w", encoding="utf-8") as f:
        f.write(paired_msg + "\n")

    print(f"Wrote {summary_path}")
    print(f"Wrote {table_path}")
    print(f"Wrote {paired_csv_path}")
    print(f"Wrote {paired_txt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run eGeMAPS and neural baselines, then write a paper comparison table.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Model keys to run or summarize.",
    )
    parser.add_argument(
        "--dataset",
        default="cremad",
        choices=["cremad", "iemocap"],
        help="Dataset to run with the same speaker-disjoint protocol.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset directory. Defaults to AudioWAV for CREMA-D and iemocap/ for IEMOCAP.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where per-model outputs and comparison tables are written. "
        "Auto-prefixed with the dataset name for non-CREMA-D runs.",
    )
    parser.add_argument(
        "--cache-source",
        default=str(DEFAULT_CACHE_SOURCE),
        help="Existing openSMILE/eGeMAPS feature cache.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")
    parser.add_argument("--quick", action="store_true", help="Use tiny smoke-test settings for neural runs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip models with existing fold_metrics.csv.")
    parser.add_argument("--force-cache-copy", action="store_true", help="Overwrite copied eGeMAPS caches.")
    parser.add_argument("--summarize-only", action="store_true", help="Only write comparison summary tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = normalize_dataset_name(args.dataset)
    output_root = _default_output_root(dataset_name, args.output_root)
    cache_source = Path(args.cache_source).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else None
    models = list(args.models)

    if args.summarize_only:
        write_comparison_summary(output_root, models)
        return

    for model_key in models:
        if model_key in EGEMAPS_BASELINES:
            label = EGEMAPS_BASELINES[model_key]["label"]
            cfg = build_egemaps_config(
                model_key, output_root, quick=args.quick,
                dataset=dataset_name, data_dir=data_dir,
            )
        else:
            label = NEURAL_BASELINES[model_key]["label"]
            cfg = build_neural_config(
                model_key, output_root, quick=args.quick,
                dataset=dataset_name, data_dir=data_dir,
            )

        metrics_path = Path(cfg.OUT_DIR) / "fold_metrics.csv"

        print("\n" + "=" * 100)
        print(f"Baseline: {model_key} | {label}")
        print(f"Dataset: {dataset_display_name(dataset_name)}")
        print(f"Output dir: {cfg.OUT_DIR}")

        if args.dry_run:
            continue

        if args.skip_existing and metrics_path.exists():
            print(f"Skipping existing run: {metrics_path}")
            continue

        if model_key in EGEMAPS_BASELINES:
            run_egemaps_baseline(
                model_key=model_key,
                output_root=output_root,
                cache_source=cache_source,
                quick=args.quick,
                force_cache_copy=args.force_cache_copy,
                dataset=dataset_name,
                data_dir=data_dir,
            )
        else:
            run_neural_baseline(
                model_key=model_key,
                output_root=output_root,
                cache_source=cache_source,
                quick=args.quick,
                force_cache_copy=args.force_cache_copy,
                dataset=dataset_name,
                data_dir=data_dir,
            )

    if not args.dry_run:
        write_comparison_summary(output_root, models)


if __name__ == "__main__":
    main()
