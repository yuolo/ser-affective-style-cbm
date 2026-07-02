"""
Train a Plain CRNN SER baseline and run a post-hoc speaker probe on encoder h.

This script exists because Plain CRNN has no c_aff or c_style. Speaker leakage for
that baseline must be measured on the real encoder representation:

    speech -> CRNN encoder h -> emotion head
    frozen h -> speaker probe

Outputs:
    plain_crnn_encoder_embeddings.csv
    plain_crnn_encoder_speaker_probe_by_fold.csv
    plain_crnn_fold_metrics.csv
    representation_speaker_probe_summary.csv/.md
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ser_cbm import (
    Config,
    DisentangledAffectiveStyleCBM,
    build_feature_cache,
    class_weights_from_labels,
    clear_device_cache,
    configure_torch_runtime,
    dataset_display_name,
    discover_cremad,
    discover_dataset,
    emotion_names_for_dataset,
    evaluate_model,
    grl_schedule,
    make_loaders_for_fold,
    normalize_dataset_name,
    select_device,
    set_seed,
    speaker_leakage_audit,
    train_one_epoch,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plain_encoder_speaker_probe_outputs"
DEFAULT_DATA_DIRS = {
    "cremad": PROJECT_ROOT / "AudioWAV",
    "iemocap": PROJECT_ROOT / "iemocap",
}
DEFAULT_CACHE_SOURCE = (
    PROJECT_ROOT
    / "si_affcbm_egemaps_bd_outputs"
    / "concept_feature_cache_opensmile_eGeMAPSv02.csv"
)
def default_b5_run_dirs(dataset: str = "cremad") -> List[Path]:
    """B5 (dual, no-adversary) run directories to read concept predictions from.

    Output directories are dataset-prefixed for non-CREMA-D runs (see the runner
    scripts), so the auto-discovered B5 predictions must match the probe dataset.
    Otherwise an IEMOCAP probe would fall back to CREMA-D concept predictions and
    write a mismatched summary.
    """
    dataset_name = normalize_dataset_name(dataset)
    prefix = "" if dataset_name == "cremad" else f"{dataset_name}_"
    return [
        PROJECT_ROOT / f"{prefix}baseline_comparison_outputs" / "B5_dual_no_adversaries",
        PROJECT_ROOT / f"{prefix}ablation_outputs" / "no_adversaries",
    ]


def build_plain_crnn_config(
    output_dir: Path,
    quick: bool = False,
    dataset: str = "cremad",
    data_dir: Optional[Path] = None,
) -> Config:
    cfg = Config()
    dataset_name = normalize_dataset_name(dataset)
    cfg.DATASET = dataset_name
    resolved_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIRS[dataset_name]
    if dataset_name == "cremad":
        cfg.CREMA_D_DIR = str(resolved_dir)
    elif dataset_name == "iemocap":
        cfg.IEMOCAP_DIR = str(resolved_dir)
    cfg.OUT_DIR = str(output_dir)
    cfg.EMOTION_HEAD_INPUT = "encoder"
    cfg.USE_AFF_CONCEPT_BRANCH = False
    cfg.USE_AFF_CONCEPT_SUPERVISION = False
    cfg.USE_AFF_SPEAKER_ADVERSARY = False
    cfg.USE_STYLE_BRANCH = False
    cfg.USE_STYLE_EMOTION_ADVERSARY = False
    cfg.USE_ORTHOGONALITY = False
    cfg.SELECTION_CONCEPT_PENALTY = 0.0

    if quick:
        cfg.N_SPLITS = 2
        cfg.NUM_EPOCHS = 2
        cfg.PATIENCE = 2
        cfg.SPEAKER_PROBE_REPEATS = 2

    return cfg


def copy_concept_cache_if_available(cfg: Config, cache_source: Path, force: bool = False) -> None:
    if not cache_source.exists():
        return
    out_dir = Path(cfg.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"concept_feature_cache_opensmile_{cfg.OPENSMILE_FEATURE_SET}.csv"
    if dest.exists() and not force:
        return
    shutil.copy2(cache_source, dest)


@torch.no_grad()
def collect_encoder_h(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str) -> np.ndarray:
    model.eval()
    hs = []
    for batch in loader:
        x = batch["x"].to(device)
        out = model(x, grl_lambda=0.0)
        hs.append(out["h"].detach().cpu().numpy())
    return np.concatenate(hs, axis=0).astype(np.float32)


def embeddings_dataframe(test_df: pd.DataFrame, h: np.ndarray, fold: int) -> pd.DataFrame:
    if len(test_df) != len(h):
        raise RuntimeError(f"Embedding count mismatch for fold {fold}: df={len(test_df)} h={len(h)}")

    out = test_df.copy().reset_index(drop=True)
    out.insert(0, "fold", fold)
    out.insert(1, "sample_id", out["filename"].astype(str))
    h_cols = [f"h_{i:03d}" for i in range(h.shape[1])]
    h_df = pd.DataFrame(h, columns=h_cols)
    return pd.concat([out, h_df], axis=1)


def run_plain_crnn_encoder_probe(
    output_dir: Path,
    cache_source: Path,
    quick: bool = False,
    force_cache_copy: bool = False,
    max_folds: Optional[int] = None,
    dataset: str = "cremad",
    data_dir: Optional[Path] = None,
) -> None:
    cfg = build_plain_crnn_config(output_dir, quick=quick, dataset=dataset, data_dir=data_dir)
    copy_concept_cache_if_available(cfg, cache_source, force=force_cache_copy)

    set_seed(cfg.SEED)
    device = select_device(cfg.DEVICE)
    configure_torch_runtime(device)
    cfg.DEVICE = str(device)
    if device.type == "mps":
        cfg.NUM_WORKERS = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    df = discover_dataset(cfg)
    n_emotions = len(emotion_names_for_dataset(cfg.DATASET))
    cache = build_feature_cache(df, cfg)

    groups = df["speaker"].astype(str).to_numpy(dtype=str)
    y_all = df["emotion"].to_numpy(dtype=np.int64)
    X_all = np.arange(len(df))

    outer = GroupKFold(n_splits=cfg.N_SPLITS)
    metric_rows: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []
    embedding_frames: List[pd.DataFrame] = []

    for fold, (trainval_idx, test_idx) in enumerate(outer.split(X_all, y_all, groups), start=1):
        if max_folds is not None and fold > max_folds:
            break

        print("\n" + "#" * 100)
        print(f"Plain CRNN encoder probe fold {fold}/{cfg.N_SPLITS}")
        print("#" * 100)

        trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        inner_groups = trainval_df["speaker"].astype(str).to_numpy(dtype=str)
        inner_y = trainval_df["emotion"].to_numpy(dtype=np.int64)
        inner_X = np.arange(len(trainval_df))
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.INNER_VAL_SIZE, random_state=cfg.SEED + fold)
        inner_train_idx, inner_val_idx = next(gss.split(inner_X, inner_y, inner_groups))

        train_df = trainval_df.iloc[inner_train_idx].reset_index(drop=True)
        val_df = trainval_df.iloc[inner_val_idx].reset_index(drop=True)

        train_loader, val_loader, test_loader, speaker_to_local, _ = make_loaders_for_fold(
            train_df,
            val_df,
            test_df,
            cache,
            cfg,
        )

        model = DisentangledAffectiveStyleCBM(
            n_mels=cfg.N_MELS,
            h_dim=cfg.H_DIM,
            n_aff=cfg.N_AFF_CONCEPTS,
            n_style=cfg.N_STYLE_CONCEPTS,
            n_emotions=n_emotions,
            n_train_speakers=len(speaker_to_local),
            dropout=cfg.DROPOUT,
            emotion_head_input=cfg.EMOTION_HEAD_INPUT,
            use_aff_concept_branch=cfg.USE_AFF_CONCEPT_BRANCH,
            use_style_branch=cfg.USE_STYLE_BRANCH,
        ).to(cfg.DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)
        emotion_weights = class_weights_from_labels(train_df["emotion"].to_numpy(dtype=np.int64), n_emotions)

        best_score = -1e9
        best_state = None
        best_epoch = 0
        bad_epochs = 0

        for epoch in range(1, cfg.NUM_EPOCHS + 1):
            grl_lambd = grl_schedule(epoch - 1, cfg.NUM_EPOCHS, cfg.GRL_MAX_LAMBDA)
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=cfg.DEVICE,
                cfg=cfg,
                emotion_weights=emotion_weights,
                grl_lambda=grl_lambd,
            )
            scheduler.step()

            val_out = evaluate_model(model, val_loader, cfg.DEVICE, cfg=cfg)
            val_metrics = val_out["metrics"]
            score = val_metrics["uar"] + 0.5 * val_metrics["macro_f1"]

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if epoch == 1 or epoch % 5 == 0 or bad_epochs == 0:
                print(
                    f"[Fold {fold} | E{epoch:03d}] "
                    f"val_UAR={val_metrics['uar']:.4f} "
                    f"val_F1={val_metrics['macro_f1']:.4f}"
                )

            if bad_epochs >= cfg.PATIENCE:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        test_out = evaluate_model(model, test_loader, cfg.DEVICE, cfg=cfg)
        test_metrics = test_out["metrics"]
        h = collect_encoder_h(model, test_loader, cfg.DEVICE)
        fold_embeddings = embeddings_dataframe(test_df, h, fold)
        embedding_frames.append(fold_embeddings)

        speaker_labels = test_df["speaker"].astype(str).to_numpy(dtype=str)
        audit_h = speaker_leakage_audit(
            h,
            speaker_labels,
            seed=cfg.SEED + fold,
            test_size=cfg.SPEAKER_PROBE_TEST_SIZE,
            n_repeats=cfg.SPEAKER_PROBE_REPEATS,
        )

        metric_rows.append({
            "fold": fold,
            "best_epoch": best_epoch,
            "test_acc": test_metrics["acc"],
            "test_uar": test_metrics["uar"],
            "test_macro_f1": test_metrics["macro_f1"],
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_train_speakers": train_df["speaker"].nunique(),
            "n_val_speakers": val_df["speaker"].nunique(),
            "n_test_speakers": test_df["speaker"].nunique(),
            "n_train_test_speaker_overlap": len(set(train_df["speaker"].astype(str)) & set(test_df["speaker"].astype(str))),
        })
        probe_rows.append({
            "fold": fold,
            "representation": "Plain CRNN encoder h",
            "speaker_probe_acc": audit_h["probe_acc_mean"],
            "speaker_probe_acc_std": audit_h["probe_acc_std"],
            "speaker_probe_chance_uniform": audit_h["probe_chance_uniform"],
            "speaker_probe_chance_majority": audit_h["probe_chance_majority"],
            "speaker_probe_leakage_index": audit_h["probe_leakage_index"],
            "n_speakers": audit_h["probe_n_speakers"],
            "n_samples": audit_h["probe_n_samples"],
        })

        print(f"Fold {fold} h speaker probe acc: {audit_h['probe_acc_mean']:.4f}")
        clear_device_cache(torch.device(cfg.DEVICE))

    metrics = pd.DataFrame(metric_rows)
    probes = pd.DataFrame(probe_rows)
    embeddings = pd.concat(embedding_frames, axis=0).reset_index(drop=True)

    metrics.to_csv(output_dir / "plain_crnn_fold_metrics.csv", index=False)
    probes.to_csv(output_dir / "plain_crnn_encoder_speaker_probe_by_fold.csv", index=False)
    embeddings.to_csv(output_dir / "plain_crnn_encoder_embeddings.csv", index=False)


def _mean_std(df: pd.DataFrame, col: str) -> Dict[str, float]:
    values = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
    return {
        "mean": float(values.mean(skipna=True)) if len(values) else float("nan"),
        "std": float(values.std(skipna=True)) if len(values) else float("nan"),
    }


def summarize_plain_h(output_dir: Path) -> Optional[Dict[str, Any]]:
    path = output_dir / "plain_crnn_encoder_speaker_probe_by_fold.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    stats = _mean_std(df, "speaker_probe_acc")
    return {
        "representation": "Plain CRNN encoder h",
        "source": str(path),
        "speaker_probe_accuracy_mean": stats["mean"],
        "speaker_probe_accuracy_std": stats["std"],
        "speaker_probe_chance_uniform_mean": _mean_std(df, "speaker_probe_chance_uniform")["mean"],
        "speaker_probe_chance_majority_mean": _mean_std(df, "speaker_probe_chance_majority")["mean"],
        "speaker_probe_leakage_index_mean": _mean_std(df, "speaker_probe_leakage_index")["mean"],
        "folds": int(len(df)),
    }


def _concept_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(prefix) and not c.startswith(f"target_{prefix}")]


def summarize_b5_from_predictions(path: Path, repeats: int, test_size: float, seed: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    pred = pd.read_csv(path)
    if "fold" not in pred.columns or "speaker" not in pred.columns:
        return []

    aff_cols = _concept_columns(pred, "c_aff_")
    style_cols = _concept_columns(pred, "c_style_")
    specs = [
        ("Affective concepts c_aff from B5", aff_cols),
        ("Style concepts c_style from B5", style_cols),
    ]

    out_rows = []
    for representation, cols in specs:
        if not cols:
            continue
        fold_rows = []
        for fold in sorted(pred["fold"].dropna().unique().tolist()):
            fold_df = pred[pred["fold"] == fold]
            X = fold_df[cols].to_numpy(dtype=np.float32)
            speakers = fold_df["speaker"].astype(str).to_numpy(dtype=str)
            audit = speaker_leakage_audit(
                X,
                speakers,
                seed=seed + int(fold),
                test_size=test_size,
                n_repeats=repeats,
            )
            fold_rows.append(audit)

        accs = np.array([r["probe_acc_mean"] for r in fold_rows], dtype=np.float32)
        uniform_chances = np.array([r["probe_chance_uniform"] for r in fold_rows], dtype=np.float32)
        majority_chances = np.array([r["probe_chance_majority"] for r in fold_rows], dtype=np.float32)
        leakage = np.array([r["probe_leakage_index"] for r in fold_rows], dtype=np.float32)
        out_rows.append({
            "representation": representation,
            "source": str(path),
            "speaker_probe_accuracy_mean": float(np.nanmean(accs)),
            "speaker_probe_accuracy_std": float(np.nanstd(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "speaker_probe_chance_uniform_mean": float(np.nanmean(uniform_chances)),
            "speaker_probe_chance_majority_mean": float(np.nanmean(majority_chances)),
            "speaker_probe_leakage_index_mean": float(np.nanmean(leakage)),
            "folds": int(len(fold_rows)),
        })

    return out_rows


def find_b5_predictions(explicit_run_dir: Optional[Path], dataset: str = "cremad") -> Optional[Path]:
    candidates = []
    if explicit_run_dir is not None:
        candidates.append(explicit_run_dir / "test_predictions_and_concepts.csv")
    candidates.extend(
        [p / "test_predictions_and_concepts.csv" for p in default_b5_run_dirs(dataset)]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.4f}"
    if pd.isna(value):
        return "NA"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    body = [[_fmt(row[c]) for c in df.columns] for _, row in df.iterrows()]
    widths = [max(len(headers[i]), *(len(row[i]) for row in body)) for i in range(len(headers))]

    def fmt_row(values: List[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(row) for row in body]) + "\n"


def write_representation_summary(output_dir: Path, b5_run_dir: Optional[Path], cfg: Config) -> None:
    rows: List[Dict[str, Any]] = []
    plain = summarize_plain_h(output_dir)
    if plain is not None:
        rows.append(plain)

    b5_predictions = find_b5_predictions(b5_run_dir, dataset=cfg.DATASET)
    if b5_predictions is not None:
        rows.extend(
            summarize_b5_from_predictions(
                b5_predictions,
                repeats=cfg.SPEAKER_PROBE_REPEATS,
                test_size=cfg.SPEAKER_PROBE_TEST_SIZE,
                seed=cfg.SEED,
            )
        )

    summary = pd.DataFrame(rows)
    csv_path = output_dir / "representation_speaker_probe_summary.csv"
    md_path = output_dir / "representation_speaker_probe_summary.md"
    summary.to_csv(csv_path, index=False)

    display_cols = [
        "representation",
        "speaker_probe_accuracy_mean",
        "speaker_probe_accuracy_std",
        "speaker_probe_chance_uniform_mean",
        "speaker_probe_chance_majority_mean",
        "speaker_probe_leakage_index_mean",
        "folds",
    ]
    display_cols = [c for c in display_cols if c in summary.columns]
    table = summary[display_cols].rename(columns={
        "representation": "Representation",
        "speaker_probe_accuracy_mean": "Speaker probe accuracy ↓",
        "speaker_probe_accuracy_std": "Std",
        "speaker_probe_chance_uniform_mean": "Uniform chance",
        "speaker_probe_chance_majority_mean": "Majority chance",
        "speaker_probe_leakage_index_mean": "Leakage index",
        "folds": "Folds",
    })

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Representation Speaker Probe\n\n")
        f.write(dataframe_to_markdown(table))

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Plain CRNN and probe speaker identity from encoder h.",
    )
    parser.add_argument("--dataset", default="cremad", choices=["cremad", "iemocap"],
                        help="Dataset for the speaker-disjoint probe.")
    parser.add_argument("--data-dir", default=None,
                        help="Dataset directory. Defaults to AudioWAV for CREMA-D and iemocap/ for IEMOCAP.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory. Auto-prefixed with the dataset name for non-CREMA-D runs.")
    parser.add_argument("--cache-source", default=str(DEFAULT_CACHE_SOURCE), help="openSMILE/eGeMAPS cache source.")
    parser.add_argument("--b5-run-dir", default=None, help="Optional B5 run dir with test_predictions_and_concepts.csv.")
    parser.add_argument("--quick", action="store_true", help="2-fold/2-epoch smoke run.")
    parser.add_argument("--max-folds", type=int, default=None, help="Optional fold cap for debugging.")
    parser.add_argument("--force-cache-copy", action="store_true", help="Overwrite copied concept cache.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned paths without training.")
    parser.add_argument("--summarize-only", action="store_true", help="Only rebuild representation summary from existing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = normalize_dataset_name(args.dataset)
    if args.output_dir == str(DEFAULT_OUTPUT_DIR) and dataset_name != "cremad":
        output_dir = PROJECT_ROOT / f"{dataset_name}_plain_encoder_speaker_probe_outputs"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else None
    cache_source = Path(args.cache_source).expanduser().resolve()
    b5_run_dir = Path(args.b5_run_dir).expanduser().resolve() if args.b5_run_dir else None
    cfg = build_plain_crnn_config(output_dir, quick=args.quick, dataset=dataset_name, data_dir=data_dir)

    print(f"Dataset: {dataset_display_name(dataset_name)}")
    print("Output dir:", output_dir)
    print("Cache source:", cache_source)
    print("B5 run dir:", b5_run_dir or "auto")

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        run_plain_crnn_encoder_probe(
            output_dir=output_dir,
            cache_source=cache_source,
            quick=args.quick,
            force_cache_copy=args.force_cache_copy,
            max_folds=args.max_folds,
            dataset=dataset_name,
            data_dir=data_dir,
        )

    write_representation_summary(output_dir, b5_run_dir, cfg)


if __name__ == "__main__":
    main()
