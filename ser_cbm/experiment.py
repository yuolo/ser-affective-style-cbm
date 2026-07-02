"""Utilities for the disentangled affective-style CBM SER experiments."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .concepts import write_feature_group_report
from .config import AFF_CONCEPT_NAMES, CFG, EMOTION_NAMES, STYLE_CONCEPT_NAMES, Config
from .data import (
    build_feature_cache,
    dataset_dir_for_config,
    dataset_display_name,
    discover_cremad,
    discover_dataset,
    emotion_names_for_dataset,
    normalize_dataset_name,
)
from .diagnostics import (
    concept_intervention_diagnostics,
    prefix_metrics,
    speaker_leakage_audit,
    write_concept_mae_by_concept,
    write_dataset_statistics,
    write_factorization_audit_figure,
    write_main_claim_summary,
    write_speaker_probe_chance_summary,
)
from .metrics import class_weights_from_labels, grl_schedule
from .models import DisentangledAffectiveStyleCBM
from .runtime import (
    clear_device_cache,
    configure_torch_runtime,
    is_mps_available,
    select_device,
    set_seed,
)
from .training import _feature_names_from_cache, evaluate_model, make_loaders_for_fold, train_one_epoch

# =============================================================================
# MAIN CV
# =============================================================================

def run_experiment(cfg: Config) -> None:
    set_seed(cfg.SEED)
    device = select_device(cfg.DEVICE)
    configure_torch_runtime(device)
    cfg.DEVICE = str(device)

    if device.type == "mps":
        cfg.NUM_WORKERS = 0

    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    dataset_name = normalize_dataset_name(cfg.DATASET)
    emotion_names = emotion_names_for_dataset(dataset_name)

    print("=" * 100)
    print("Disentangled Affective--Style CBM with eGeMAPS baseline/deviation concepts")
    print("Device:", device)
    print("MPS available:", is_mps_available())
    print("CUDA available:", torch.cuda.is_available())
    print("DataLoader workers:", cfg.NUM_WORKERS)
    print(f"Dataset: {dataset_display_name(dataset_name)}")
    print("Dataset dir:", dataset_dir_for_config(cfg))
    print("Feature backend:", cfg.FEATURE_BACKEND)
    if str(cfg.FEATURE_BACKEND).lower().strip() == "opensmile":
        print("openSMILE feature set:", cfg.OPENSMILE_FEATURE_SET)
    print("Emotion head input:", cfg.EMOTION_HEAD_INPUT)
    print("Affective concept branch:", cfg.USE_AFF_CONCEPT_BRANCH)
    print("Affective concept supervision:", cfg.USE_AFF_CONCEPT_SUPERVISION)
    print("Style branch:", cfg.USE_STYLE_BRANCH)
    print("Selection concept penalty:", cfg.SELECTION_CONCEPT_PENALTY)
    print("=" * 100)

    df = discover_dataset(cfg)
    print(f"Discovered {len(df)} clips from {df['speaker'].nunique()} speakers")
    print("Emotion counts:")
    print(df["emotion_code"].value_counts().sort_index())
    dataset_stats_csv, dataset_stats_md = write_dataset_statistics(df, cfg)

    cache = build_feature_cache(df, cfg)
    feature_names = _feature_names_from_cache(cache)
    write_feature_group_report(feature_names, cfg)

    groups = df["speaker"].astype(str).to_numpy(dtype=str)
    y_all = df["emotion"].to_numpy(dtype=np.int64)
    X_all = np.arange(len(df))

    outer = GroupKFold(n_splits=cfg.N_SPLITS)
    rows = []
    all_fold_predictions = []

    for fold, (trainval_idx, test_idx) in enumerate(outer.split(X_all, y_all, groups), start=1):
        print("\n" + "#" * 100)
        print(f"Fold {fold}/{cfg.N_SPLITS}")
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

        print(f"Train clips={len(train_df)} | Val clips={len(val_df)} | Test clips={len(test_df)}")
        print(
            f"Train speakers={train_df['speaker'].nunique()} | "
            f"Val speakers={val_df['speaker'].nunique()} | Test speakers={test_df['speaker'].nunique()}"
        )

        train_loader, val_loader, test_loader, speaker_to_local, _ = make_loaders_for_fold(
            train_df, val_df, test_df, cache, cfg
        )
        n_train_speakers = len(speaker_to_local)

        model = DisentangledAffectiveStyleCBM(
            n_mels=cfg.N_MELS,
            h_dim=cfg.H_DIM,
            n_aff=cfg.N_AFF_CONCEPTS,
            n_style=cfg.N_STYLE_CONCEPTS,
            n_emotions=len(emotion_names),
            n_train_speakers=n_train_speakers,
            dropout=cfg.DROPOUT,
            emotion_head_input=cfg.EMOTION_HEAD_INPUT,
            use_aff_concept_branch=cfg.USE_AFF_CONCEPT_BRANCH,
            use_style_branch=cfg.USE_STYLE_BRANCH,
        ).to(cfg.DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)
        emotion_weights = class_weights_from_labels(train_df["emotion"].to_numpy(dtype=np.int64), len(emotion_names))

        best_score = -1e9
        best_state = None
        best_epoch = 0
        bad_epochs = 0

        for epoch in range(1, cfg.NUM_EPOCHS + 1):
            grl_lambd = grl_schedule(epoch - 1, cfg.NUM_EPOCHS, cfg.GRL_MAX_LAMBDA)
            tr_losses = train_one_epoch(
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
            aff_penalty = val_metrics["aff_mae"] if np.isfinite(val_metrics["aff_mae"]) else 0.0

            score = (
                val_metrics["uar"]
                + 0.5 * val_metrics["macro_f1"]
                - float(cfg.SELECTION_CONCEPT_PENALTY) * aff_penalty
            )

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
                    f"loss={tr_losses['loss']:.4f} "
                    f"val_UAR={val_metrics['uar']:.4f} "
                    f"val_F1={val_metrics['macro_f1']:.4f} "
                    f"val_ECE={val_metrics['ece']:.4f} "
                    f"aff_MAE={val_metrics['aff_mae']:.4f} "
                    f"style_MAE={val_metrics['style_mae']:.4f} "
                    f"grl={grl_lambd:.3f}"
                )

            if bad_epochs >= cfg.PATIENCE:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        test_out = evaluate_model(model, test_loader, cfg.DEVICE, cfg=cfg)
        test_metrics = test_out["metrics"]

        test_speakers = test_df["speaker"].astype(str).to_numpy(dtype=str)
        audit_aff = speaker_leakage_audit(
            test_out["c_aff"],
            test_speakers,
            seed=cfg.SEED + fold,
            test_size=cfg.SPEAKER_PROBE_TEST_SIZE,
            n_repeats=cfg.SPEAKER_PROBE_REPEATS,
        )
        audit_style = speaker_leakage_audit(
            test_out["c_style"],
            test_speakers,
            seed=cfg.SEED + 100 + fold,
            test_size=cfg.SPEAKER_PROBE_TEST_SIZE,
            n_repeats=cfg.SPEAKER_PROBE_REPEATS,
        )
        probe_aff = audit_aff["probe_acc_mean"]
        probe_style = audit_style["probe_acc_mean"]

        intervention = concept_intervention_diagnostics(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=cfg.DEVICE,
            seed=cfg.SEED + fold,
        )

        row = {
            "fold": fold,
            "emotion_head_input": cfg.EMOTION_HEAD_INPUT,
            "use_aff_concept_branch": cfg.USE_AFF_CONCEPT_BRANCH,
            "use_aff_concept_supervision": cfg.USE_AFF_CONCEPT_SUPERVISION,
            "use_style_branch": cfg.USE_STYLE_BRANCH,
            "use_aff_speaker_adversary": cfg.USE_AFF_SPEAKER_ADVERSARY,
            "use_style_emotion_adversary": cfg.USE_STYLE_EMOTION_ADVERSARY,
            "use_orthogonality": cfg.USE_ORTHOGONALITY,
            "best_epoch": best_epoch,
            "test_acc": test_metrics["acc"],
            "test_uar": test_metrics["uar"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_ece": test_metrics["ece"],
            "test_aff_mae": test_metrics["aff_mae"],
            "test_style_mae": test_metrics["style_mae"],
            "speaker_probe_aff_acc": probe_aff,
            "speaker_probe_style_acc": probe_style,
            **prefix_metrics("speaker_probe_aff", audit_aff),
            **prefix_metrics("speaker_probe_style", audit_style),
            "swap_consistency": intervention["style_swap_consistency"],
            "style_swap_consistency": intervention["style_swap_consistency"],
            "style_swap_prob_l1": intervention["style_swap_prob_l1"],
            "aff_swap_sensitivity": intervention["aff_swap_sensitivity"],
            "aff_swap_prob_l1": intervention["aff_swap_prob_l1"],
            "emotion_probe_aff_uar": intervention["emotion_probe_aff_uar"],
            "emotion_probe_style_uar": intervention["emotion_probe_style_uar"],
            "emotion_probe_both_uar": intervention["emotion_probe_both_uar"],
            "emotion_probe_aff_macro_f1": intervention["emotion_probe_aff_macro_f1"],
            "emotion_probe_style_macro_f1": intervention["emotion_probe_style_macro_f1"],
            "emotion_probe_both_macro_f1": intervention["emotion_probe_both_macro_f1"],
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_train_speakers": train_df["speaker"].nunique(),
            "n_val_speakers": val_df["speaker"].nunique(),
            "n_test_speakers": test_df["speaker"].nunique(),
            "n_train_val_speaker_overlap": len(set(train_df["speaker"].astype(str)) & set(val_df["speaker"].astype(str))),
            "n_train_test_speaker_overlap": len(set(train_df["speaker"].astype(str)) & set(test_df["speaker"].astype(str))),
            "n_val_test_speaker_overlap": len(set(val_df["speaker"].astype(str)) & set(test_df["speaker"].astype(str))),
        }
        rows.append(row)

        print("\nFold test metrics:")
        for k, v in row.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        probs = torch.softmax(torch.tensor(test_out["logits"]), dim=1).numpy()
        pred = probs.argmax(axis=1)
        fold_pred_df = test_df.copy()
        fold_pred_df["fold"] = fold
        fold_pred_df["y_true"] = test_out["y_true"]
        fold_pred_df["y_pred"] = pred
        for j, name in enumerate(emotion_names):
            fold_pred_df[f"prob_{name}"] = probs[:, j]
        for j, name in enumerate(AFF_CONCEPT_NAMES):
            fold_pred_df[f"c_aff_{name}"] = test_out["c_aff"][:, j]
            fold_pred_df[f"target_aff_{name}"] = test_out["aff_targets"][:, j]
        for j, name in enumerate(STYLE_CONCEPT_NAMES):
            fold_pred_df[f"c_style_{name}"] = test_out["c_style"][:, j]
            fold_pred_df[f"target_style_{name}"] = test_out["style_targets"][:, j]
        all_fold_predictions.append(fold_pred_df)
        clear_device_cache(torch.device(cfg.DEVICE))

    results = pd.DataFrame(rows)
    pred_all = pd.concat(all_fold_predictions, axis=0).reset_index(drop=True)

    results_path = os.path.join(cfg.OUT_DIR, "fold_metrics.csv")
    pred_path = os.path.join(cfg.OUT_DIR, "test_predictions_and_concepts.csv")
    results.to_csv(results_path, index=False)
    pred_all.to_csv(pred_path, index=False)

    leakage_cols = [
        "fold",
        "emotion_head_input",
        "use_aff_concept_branch",
        "use_aff_concept_supervision",
        "use_style_branch",
        "use_aff_speaker_adversary",
        "use_style_emotion_adversary",
        "use_orthogonality",
        "n_train_val_speaker_overlap",
        "n_train_test_speaker_overlap",
        "n_val_test_speaker_overlap",
        "speaker_probe_aff_probe_acc_mean",
        "speaker_probe_aff_probe_acc_std",
        "speaker_probe_aff_probe_chance_uniform",
        "speaker_probe_aff_probe_chance_majority",
        "speaker_probe_aff_probe_leakage_index",
        "speaker_probe_aff_probe_n_speakers",
        "speaker_probe_aff_probe_n_samples",
        "speaker_probe_style_probe_acc_mean",
        "speaker_probe_style_probe_acc_std",
        "speaker_probe_style_probe_chance_uniform",
        "speaker_probe_style_probe_chance_majority",
        "speaker_probe_style_probe_leakage_index",
        "speaker_probe_style_probe_n_speakers",
        "speaker_probe_style_probe_n_samples",
    ]
    leakage_cols = [c for c in leakage_cols if c in results.columns]
    leakage_path = os.path.join(cfg.OUT_DIR, "speaker_leakage_audit.csv")
    results[leakage_cols].to_csv(leakage_path, index=False)
    speaker_chance_csv, speaker_chance_md, speaker_summary = write_speaker_probe_chance_summary(
        results,
        cfg,
        pred_all=pred_all,
    )
    claim_summary_path = write_main_claim_summary(results, cfg, speaker_summary=speaker_summary)
    concept_mae_path = write_concept_mae_by_concept(pred_all, cfg)
    factorization_figure_path = write_factorization_audit_figure(results, cfg)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    metric_cols = [
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
    for col in metric_cols:
        mean = results[col].mean(skipna=True)
        std = results[col].std(skipna=True)
        print(f"{col:28s}: {mean:.4f} ± {std:.4f}")

    print("\nSaved:")
    print("  ", results_path)
    print("  ", pred_path)
    print("  ", dataset_stats_csv)
    print("  ", dataset_stats_md)
    print("  ", leakage_path)
    print("  ", claim_summary_path)
    print("  ", speaker_chance_csv)
    print("  ", speaker_chance_md)
    print("  ", concept_mae_path)
    print("  ", factorization_figure_path)

    cm = confusion_matrix(pred_all["y_true"], pred_all["y_pred"], labels=list(range(len(emotion_names))))
    cm_df = pd.DataFrame(cm, index=emotion_names, columns=emotion_names)
    cm_path = os.path.join(cfg.OUT_DIR, "confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print("  ", cm_path)

