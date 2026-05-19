# -*- coding: utf-8 -*-
"""
Run ablation variants for the eGeMAPS baseline/deviation CBM experiment.

This file intentionally does not modify either main experiment script. It imports
the existing Config and run_experiment entry point, applies per-variant overrides,
and writes each run into its own output directory.

Examples:
    python3 ablation_runner.py --dry-run
    python3 ablation_runner.py --variants plain_ser_encoder full no_style_branch
    python3 ablation_runner.py --quick --variants concept_bottleneck_only full
    python3 ablation_runner.py --summarize-only
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from ser_cbm import Config, dataset_display_name, normalize_dataset_name, run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "ablation_outputs"
DEFAULT_DATA_DIRS = {
    "cremad": PROJECT_ROOT / "AudioWAV",
    "iemocap": PROJECT_ROOT / "iemocap",
}
DEFAULT_CACHE_SOURCE = (
    PROJECT_ROOT
    / "si_affcbm_egemaps_bd_outputs"
    / "concept_feature_cache_opensmile_eGeMAPSv02.csv"
)


ABLATIONS: Dict[str, Dict[str, Any]] = {
    "plain_ser_encoder": {
        "EMOTION_HEAD_INPUT": "encoder",
        "USE_AFF_CONCEPT_BRANCH": False,
        "USE_AFF_CONCEPT_SUPERVISION": False,
        "USE_STYLE_BRANCH": False,
        "USE_AFF_SPEAKER_ADVERSARY": False,
        "USE_STYLE_EMOTION_ADVERSARY": False,
        "USE_ORTHOGONALITY": False,
    },
    "concept_bottleneck_only": {
        "EMOTION_HEAD_INPUT": "aff",
        "USE_AFF_CONCEPT_BRANCH": True,
        "USE_AFF_CONCEPT_SUPERVISION": True,
        "USE_STYLE_BRANCH": False,
        "USE_AFF_SPEAKER_ADVERSARY": False,
        "USE_STYLE_EMOTION_ADVERSARY": False,
        "USE_ORTHOGONALITY": False,
    },
    "dual_branch_no_adversaries": {
        "EMOTION_HEAD_INPUT": "aff",
        "USE_AFF_CONCEPT_BRANCH": True,
        "USE_AFF_CONCEPT_SUPERVISION": True,
        "USE_STYLE_BRANCH": True,
        "USE_AFF_SPEAKER_ADVERSARY": False,
        "USE_STYLE_EMOTION_ADVERSARY": False,
        "USE_ORTHOGONALITY": False,
    },
    "dual_branch_orthogonality": {
        "EMOTION_HEAD_INPUT": "aff",
        "USE_AFF_CONCEPT_BRANCH": True,
        "USE_AFF_CONCEPT_SUPERVISION": True,
        "USE_STYLE_BRANCH": True,
        "USE_AFF_SPEAKER_ADVERSARY": False,
        "USE_STYLE_EMOTION_ADVERSARY": False,
        "USE_ORTHOGONALITY": True,
    },
    "full": {},
    "no_style_branch": {
        "EMOTION_HEAD_INPUT": "aff",
        "USE_AFF_CONCEPT_BRANCH": True,
        "USE_AFF_CONCEPT_SUPERVISION": True,
        "USE_STYLE_BRANCH": False,
        "USE_STYLE_EMOTION_ADVERSARY": False,
        "USE_ORTHOGONALITY": False,
    },
    # Backward-compatible aliases for older experiment folders.
    "no_adversaries": {
        "USE_AFF_SPEAKER_ADVERSARY": False,
        "USE_STYLE_EMOTION_ADVERSARY": False,
    },
    "no_aff_spk_adv": {
        "USE_AFF_SPEAKER_ADVERSARY": False,
    },
    "no_style_emo_adv": {
        "USE_STYLE_EMOTION_ADVERSARY": False,
    },
    "no_orth": {
        "USE_ORTHOGONALITY": False,
    },
    "no_local_val_test_baselines": {
        "DIAGNOSTIC_LOCAL_BASELINES_FOR_VAL_TEST": False,
    },
    "librosa_backend_bd": {
        "FEATURE_BACKEND": "librosa",
    },
}

DEFAULT_VARIANTS = [
    "plain_ser_encoder",
    "concept_bottleneck_only",
    "dual_branch_no_adversaries",
    "dual_branch_orthogonality",
    "full",
    "no_style_branch",
]


SUMMARY_METRICS = [
    "test_acc",
    "test_uar",
    "test_macro_f1",
    "test_ece",
    "test_aff_mae",
    "test_style_mae",
    "speaker_probe_aff_acc",
    "speaker_probe_style_acc",
    "speaker_probe_aff_probe_chance_uniform",
    "speaker_probe_aff_probe_chance_majority",
    "speaker_probe_aff_probe_leakage_index",
    "speaker_probe_style_probe_chance_uniform",
    "speaker_probe_style_probe_chance_majority",
    "speaker_probe_style_probe_leakage_index",
    "style_swap_consistency",
    "style_swap_prob_l1",
    "aff_swap_sensitivity",
    "aff_swap_prob_l1",
    "emotion_probe_aff_uar",
    "emotion_probe_style_uar",
    "emotion_probe_both_uar",
]


def _concept_cache_name(cfg: Config) -> str:
    safe_backend = str(cfg.FEATURE_BACKEND).lower().strip()
    safe_set = str(cfg.OPENSMILE_FEATURE_SET).strip()
    return f"concept_feature_cache_{safe_backend}_{safe_set}.csv"


def _apply_overrides(cfg: Config, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Config has no field named {key!r}")
        setattr(cfg, key, value)


def _copy_concept_cache_if_available(
    cfg: Config,
    cache_source: Path,
    force: bool = False,
) -> None:
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
    cache_dest = out_dir / _concept_cache_name(cfg)

    if cache_dest.exists() and not force:
        return

    shutil.copy2(cache_source, cache_dest)


def build_config(
    variant: str,
    output_root: Path,
    quick: bool = False,
    dataset: str = "cremad",
    data_dir: Optional[Path] = None,
) -> Config:
    if variant not in ABLATIONS:
        raise KeyError(f"Unknown ablation variant: {variant}")

    cfg = Config()
    dataset_name = normalize_dataset_name(dataset)
    cfg.DATASET = dataset_name
    resolved_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIRS[dataset_name]
    if dataset_name == "cremad":
        cfg.CREMA_D_DIR = str(resolved_dir)
    elif dataset_name == "iemocap":
        cfg.IEMOCAP_DIR = str(resolved_dir)
    cfg.OUT_DIR = str(output_root / variant)

    _apply_overrides(cfg, ABLATIONS[variant])

    if quick:
        cfg.N_SPLITS = 2
        cfg.NUM_EPOCHS = 2
        cfg.PATIENCE = 2

    return cfg


def write_run_config(cfg: Config, variant: str, overrides: Dict[str, Any]) -> None:
    out_dir = Path(cfg.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": variant,
        "overrides": overrides,
        "config": asdict(cfg),
    }
    with (out_dir / "ablation_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def summarize_ablation_outputs(
    output_root: Path,
    variants: Iterable[str],
    metrics: List[str] = SUMMARY_METRICS,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for variant in variants:
        metrics_path = output_root / variant / "fold_metrics.csv"
        if not metrics_path.exists():
            rows.append({
                "variant": variant,
                "status": "missing",
                "folds": 0,
            })
            continue

        df = pd.read_csv(metrics_path)
        row: Dict[str, Any] = {
            "variant": variant,
            "status": "ok",
            "folds": int(len(df)),
        }

        for metric in metrics:
            if metric not in df.columns:
                continue
            row[f"{metric}_mean"] = float(df[metric].mean(skipna=True))
            row[f"{metric}_std"] = float(df[metric].std(skipna=True))

        rows.append(row)

    return pd.DataFrame(rows)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No completed ablation outputs found._\n"

    headers = [str(c) for c in df.columns]
    body = []
    for _, row in df.iterrows():
        body.append([str(row[c]) for c in df.columns])

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in body))
        for i in range(len(headers))
    ]

    def fmt_row(values: List[str]) -> str:
        cells = [values[i].ljust(widths[i]) for i in range(len(values))]
        return "| " + " | ".join(cells) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(row) for row in body]) + "\n"


def write_summary_tables(output_root: Path, variants: Iterable[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = summarize_ablation_outputs(output_root, variants)

    csv_path = output_root / "ablation_summary.csv"
    md_path = output_root / "ablation_summary.md"

    summary.to_csv(csv_path, index=False)

    display_cols = [
        "variant",
        "folds",
        "test_uar_mean",
        "test_uar_std",
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "test_aff_mae_mean",
        "test_style_mae_mean",
        "speaker_probe_aff_acc_mean",
        "speaker_probe_style_acc_mean",
        "speaker_probe_aff_probe_chance_uniform_mean",
        "speaker_probe_aff_probe_chance_majority_mean",
        "speaker_probe_aff_probe_leakage_index_mean",
        "speaker_probe_style_probe_chance_uniform_mean",
        "speaker_probe_style_probe_chance_majority_mean",
        "speaker_probe_style_probe_leakage_index_mean",
        "style_swap_consistency_mean",
        "aff_swap_sensitivity_mean",
    ]
    display_cols = [c for c in display_cols if c in summary.columns]
    table = summary[display_cols].copy()

    numeric_cols = table.select_dtypes(include=["number"]).columns
    table[numeric_cols] = table[numeric_cols].round(4)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Ablation Summary\n\n")
        f.write(_dataframe_to_markdown(table))
        f.write("\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and summarize ablations for the eGeMAPS baseline/deviation CBM.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=list(ABLATIONS.keys()),
        help="Ablation variants to run or summarize.",
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
        help="Directory where per-variant outputs are written. "
        "Auto-prefixed with the dataset name for non-CREMA-D runs.",
    )
    parser.add_argument(
        "--cache-source",
        default=str(DEFAULT_CACHE_SOURCE),
        help="Existing openSMILE concept cache to copy into each openSMILE run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned variants and configs without training.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use tiny 2-fold/2-epoch runs for smoke testing.",
    )
    parser.add_argument(
        "--force-cache-copy",
        action="store_true",
        help="Overwrite per-variant copied concept caches when cache-source exists.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip variants whose fold_metrics.csv already exists.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only build ablation_summary.csv/.md from existing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_name = normalize_dataset_name(args.dataset)
    if args.output_root == str(DEFAULT_OUTPUT_ROOT) and dataset_name != "cremad":
        output_root = PROJECT_ROOT / f"{dataset_name}_ablation_outputs"
    else:
        output_root = Path(args.output_root).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else None
    cache_source = Path(args.cache_source).expanduser().resolve()
    variants = list(args.variants)

    if args.summarize_only:
        write_summary_tables(output_root, variants)
        return

    for variant in variants:
        cfg = build_config(
            variant,
            output_root,
            quick=args.quick,
            dataset=dataset_name,
            data_dir=data_dir,
        )
        overrides = dict(ABLATIONS[variant])
        metrics_path = Path(cfg.OUT_DIR) / "fold_metrics.csv"

        print("\n" + "=" * 100)
        print(f"Ablation variant: {variant}")
        print(f"Dataset: {dataset_display_name(dataset_name)}")
        print(f"Output dir: {cfg.OUT_DIR}")
        print(f"Overrides: {overrides if overrides else '{}'}")

        if args.dry_run:
            continue

        if args.skip_existing and metrics_path.exists():
            print(f"Skipping existing run: {metrics_path}")
            continue

        _copy_concept_cache_if_available(
            cfg,
            cache_source=cache_source,
            force=args.force_cache_copy,
        )
        write_run_config(cfg, variant, overrides)
        run_experiment(cfg)

    if not args.dry_run:
        write_summary_tables(output_root, variants)


if __name__ == "__main__":
    main()
