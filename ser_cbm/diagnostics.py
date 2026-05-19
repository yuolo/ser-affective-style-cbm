# -*- coding: utf-8 -*-
"""Utilities for the disentangled affective-style CBM SER experiments."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from .config import AFF_CONCEPT_NAMES, EMOTION_MAP, EMOTION_NAMES, STYLE_CONCEPT_NAMES, Config
from .data import dataset_display_name, emotion_names_for_dataset, normalize_dataset_name

# =============================================================================
# DIAGNOSTIC SPEAKER PROBE AND COUNTERFACTUAL SWAPPING
# =============================================================================

def fit_logistic_regression_probe(X: np.ndarray, y: np.ndarray, seed: int) -> Optional[LogisticRegression]:
    """Fit a robust sklearn probe.

    Newer scikit-learn versions no longer accept multi_class="auto", so this helper
    intentionally leaves multiclass handling to LogisticRegression itself.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    if X.ndim != 2 or len(X) != len(y) or len(np.unique(y)) < 2:
        return None

    try:
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
        clf.fit(X, y)
        return clf
    except Exception:
        return None


def speaker_probe_accuracy(
    concepts: np.ndarray,
    speaker_labels: np.ndarray,
    seed: int,
    test_size: float = 0.30,
) -> float:
    """Diagnostic probe: can speaker ID be recovered from a representation?"""
    audit = speaker_leakage_audit(
        concepts=concepts,
        speaker_labels=speaker_labels,
        seed=seed,
        test_size=test_size,
        n_repeats=1,
    )
    return audit["probe_acc_mean"]


def speaker_leakage_audit(
    concepts: np.ndarray,
    speaker_labels: np.ndarray,
    seed: int,
    test_size: float = 0.30,
    n_repeats: int = 5,
) -> Dict[str, float]:
    """Repeated linear speaker probe with chance baselines.

    The split is utterance-level inside the evaluated fold: this is intentional for
    a leakage audit because the question is whether known speaker identities are
    linearly separable from the frozen representation.
    """
    X_all = np.asarray(concepts, dtype=np.float32)
    speaker_labels = np.asarray(speaker_labels, dtype=str)

    unique, counts = np.unique(speaker_labels, return_counts=True)
    keep_speakers = unique[counts >= 3]
    keep = np.isin(speaker_labels, keep_speakers)
    X = X_all[keep]
    y = speaker_labels[keep]

    unique_kept, kept_counts = np.unique(y, return_counts=True)
    n_classes = int(len(unique_kept))
    n_samples = int(len(y))
    chance_uniform = float(1.0 / n_classes) if n_classes > 0 else float("nan")
    chance_majority = float(np.max(kept_counts) / n_samples) if n_samples > 0 else float("nan")

    empty = {
        "probe_acc_mean": float("nan"),
        "probe_acc_std": float("nan"),
        "probe_chance_uniform": chance_uniform,
        "probe_chance_majority": chance_majority,
        "probe_leakage_index": float("nan"),
        "probe_n_speakers": float(n_classes),
        "probe_n_samples": float(n_samples),
    }

    if n_classes < 2 or n_samples < 20:
        return empty

    splitter = StratifiedShuffleSplit(
        n_splits=max(int(n_repeats), 1),
        test_size=test_size,
        random_state=seed,
    )
    accs = []
    try:
        splits = list(splitter.split(X, y))
    except ValueError:
        return empty

    for split_idx, (tr, te) in enumerate(splits):
        clf = fit_logistic_regression_probe(X[tr], y[tr], seed=seed + split_idx)
        if clf is None:
            continue
        pred = clf.predict(X[te])
        accs.append(float(accuracy_score(y[te], pred)))

    if not accs:
        return empty

    acc_mean = float(np.mean(accs))
    acc_std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    denom = max(1.0 - chance_uniform, 1e-8) if np.isfinite(chance_uniform) else float("nan")
    leakage_index = float((acc_mean - chance_uniform) / denom) if np.isfinite(denom) else float("nan")
    return {
        "probe_acc_mean": acc_mean,
        "probe_acc_std": acc_std,
        "probe_chance_uniform": chance_uniform,
        "probe_chance_majority": chance_majority,
        "probe_leakage_index": leakage_index,
        "probe_n_speakers": float(n_classes),
        "probe_n_samples": float(n_samples),
    }


def prefix_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _mean_metric(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").mean(skipna=True))


def _format_metric(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"

    headers = [str(c) for c in df.columns]
    body = []
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            val = row[c]
            if isinstance(val, (float, np.floating)):
                vals.append(_format_metric(float(val)))
            elif pd.isna(val):
                vals.append("NA")
            else:
                vals.append(str(val))
        body.append(vals)

    widths = [max(len(headers[i]), *(len(row[i]) for row in body)) for i in range(len(headers))]

    def fmt(values: List[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(headers), sep] + [fmt(row) for row in body]) + "\n"


def write_dataset_statistics(df: pd.DataFrame, cfg: Config) -> Tuple[str, str]:
    dataset_name = normalize_dataset_name(cfg.DATASET)
    emotion_names = emotion_names_for_dataset(dataset_name)
    # Count by the final class index, not the raw emotion code, so datasets
    # that fold several codes into one class (e.g. IEMOCAP "exc" -> "happy")
    # are tallied correctly instead of one code silently overwriting another.
    class_counts = (
        df["emotion"]
        .astype(int)
        .value_counts()
        .reindex(range(len(emotion_names)), fill_value=0)
    )

    row: Dict[str, Any] = {
        "dataset": dataset_display_name(dataset_name),
        "utterances": int(len(df)),
        "speakers": int(df["speaker"].astype(str).nunique()),
    }
    for idx, name in enumerate(emotion_names):
        row[f"{name}_count"] = int(class_counts.loc[idx])

    stats = pd.DataFrame([row])
    csv_path = os.path.join(cfg.OUT_DIR, "dataset_statistics.csv")
    md_path = os.path.join(cfg.OUT_DIR, "dataset_statistics.md")
    stats.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Statistics\n\n")
        f.write(dataframe_to_markdown_table(stats))

    return csv_path, md_path


def _speaker_summary_value(
    speaker_summary: Optional[pd.DataFrame],
    representation: str,
    col: str,
) -> float:
    if speaker_summary is None or speaker_summary.empty or col not in speaker_summary.columns:
        return float("nan")
    mask = speaker_summary["representation"].astype(str) == representation
    if not mask.any():
        return float("nan")
    return float(pd.to_numeric(speaker_summary.loc[mask, col], errors="coerce").mean(skipna=True))


def write_main_claim_summary(
    results: pd.DataFrame,
    cfg: Config,
    speaker_summary: Optional[pd.DataFrame] = None,
) -> str:
    path = os.path.join(cfg.OUT_DIR, "main_claim_summary.txt")
    aff_rep = "Affective concepts c_aff"
    style_rep = "Style concepts c_style"
    aff_uniform = _mean_metric(results, "speaker_probe_aff_probe_chance_uniform")
    aff_majority = _mean_metric(results, "speaker_probe_aff_probe_chance_majority")
    style_uniform = _mean_metric(results, "speaker_probe_style_probe_chance_uniform")
    style_majority = _mean_metric(results, "speaker_probe_style_probe_chance_majority")

    if not np.isfinite(aff_uniform):
        aff_uniform = _speaker_summary_value(speaker_summary, aff_rep, "uniform_chance")
    if not np.isfinite(aff_majority):
        aff_majority = _speaker_summary_value(speaker_summary, aff_rep, "majority_chance")
    if not np.isfinite(style_uniform):
        style_uniform = _speaker_summary_value(speaker_summary, style_rep, "uniform_chance")
    if not np.isfinite(style_majority):
        style_majority = _speaker_summary_value(speaker_summary, style_rep, "majority_chance")

    lines = [
        f"UAR = {_format_metric(_mean_metric(results, 'test_uar'))}",
        f"Macro-F1 = {_format_metric(_mean_metric(results, 'test_macro_f1'))}",
        f"speaker_probe_aff = {_format_metric(_mean_metric(results, 'speaker_probe_aff_acc'))}",
        f"speaker_probe_aff_uniform_chance = {_format_metric(aff_uniform)}",
        f"speaker_probe_aff_majority_chance = {_format_metric(aff_majority)}",
        f"speaker_probe_style = {_format_metric(_mean_metric(results, 'speaker_probe_style_acc'))}",
        f"speaker_probe_style_uniform_chance = {_format_metric(style_uniform)}",
        f"speaker_probe_style_majority_chance = {_format_metric(style_majority)}",
        f"style_swap_consistency = {_format_metric(_mean_metric(results, 'style_swap_consistency'))}",
        f"aff_swap_sensitivity = {_format_metric(_mean_metric(results, 'aff_swap_sensitivity'))}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _concept_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def _speaker_probe_summary_from_predictions(pred_all: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if "fold" not in pred_all.columns or "speaker" not in pred_all.columns:
        return pd.DataFrame()

    specs = [
        ("Affective concepts c_aff", _concept_cols(pred_all, "c_aff_")),
        ("Style concepts c_style", _concept_cols(pred_all, "c_style_")),
    ]
    rows = []
    for representation, cols in specs:
        if not cols:
            continue
        fold_audits = []
        for fold in sorted(pred_all["fold"].dropna().unique().tolist()):
            fold_df = pred_all[pred_all["fold"] == fold]
            audit = speaker_leakage_audit(
                fold_df[cols].to_numpy(dtype=np.float32),
                fold_df["speaker"].astype(str).to_numpy(dtype=str),
                seed=cfg.SEED + int(fold),
                test_size=cfg.SPEAKER_PROBE_TEST_SIZE,
                n_repeats=cfg.SPEAKER_PROBE_REPEATS,
            )
            fold_audits.append(audit)

        if not fold_audits:
            continue

        def mean_key(key: str) -> float:
            vals = np.array([a[key] for a in fold_audits], dtype=np.float32)
            return float(np.nanmean(vals))

        def std_key(key: str) -> float:
            vals = np.array([a[key] for a in fold_audits], dtype=np.float32)
            return float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0

        rows.append({
            "representation": representation,
            "speaker_probe_accuracy": mean_key("probe_acc_mean"),
            "speaker_probe_accuracy_std": std_key("probe_acc_mean"),
            "uniform_chance": mean_key("probe_chance_uniform"),
            "majority_chance": mean_key("probe_chance_majority"),
            "leakage_index": mean_key("probe_leakage_index"),
            "n_speakers": mean_key("probe_n_speakers"),
        })

    return pd.DataFrame(rows)


def build_speaker_probe_chance_summary(
    results: pd.DataFrame,
    cfg: Config,
    pred_all: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []
    specs = [
        ("Affective concepts c_aff", "speaker_probe_aff"),
        ("Style concepts c_style", "speaker_probe_style"),
    ]
    for representation, prefix in specs:
        rows.append({
            "representation": representation,
            "speaker_probe_accuracy": _mean_metric(results, f"{prefix}_acc"),
            "speaker_probe_accuracy_std": _mean_metric(results, f"{prefix}_probe_acc_std"),
            "uniform_chance": _mean_metric(results, f"{prefix}_probe_chance_uniform"),
            "majority_chance": _mean_metric(results, f"{prefix}_probe_chance_majority"),
            "leakage_index": _mean_metric(results, f"{prefix}_probe_leakage_index"),
            "n_speakers": _mean_metric(results, f"{prefix}_probe_n_speakers"),
        })

    summary = pd.DataFrame(rows)
    if pred_all is not None:
        needs_prediction_fallback = (
            summary.empty
            or "uniform_chance" not in summary.columns
            or not np.isfinite(pd.to_numeric(summary["uniform_chance"], errors="coerce")).any()
        )
        if needs_prediction_fallback:
            prediction_summary = _speaker_probe_summary_from_predictions(pred_all, cfg)
            if not prediction_summary.empty:
                summary = prediction_summary
                overrides = {
                    "Affective concepts c_aff": "speaker_probe_aff",
                    "Style concepts c_style": "speaker_probe_style",
                }
                for representation, prefix in overrides.items():
                    acc = _mean_metric(results, f"{prefix}_acc")
                    if not np.isfinite(acc) or "representation" not in summary.columns:
                        continue
                    mask = summary["representation"].astype(str) == representation
                    if mask.any():
                        summary.loc[mask, "speaker_probe_accuracy"] = acc
    return summary


def write_speaker_probe_chance_summary(
    results: pd.DataFrame,
    cfg: Config,
    pred_all: Optional[pd.DataFrame] = None,
) -> Tuple[str, str, pd.DataFrame]:
    summary = build_speaker_probe_chance_summary(results, cfg, pred_all=pred_all)
    csv_path = os.path.join(cfg.OUT_DIR, "speaker_probe_chance_summary.csv")
    md_path = os.path.join(cfg.OUT_DIR, "speaker_probe_chance_summary.md")
    summary.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Speaker Probe Chance Baselines\n\n")
        f.write(dataframe_to_markdown_table(summary))

    return csv_path, md_path, summary


def write_concept_mae_by_concept(pred_all: pd.DataFrame, cfg: Config) -> str:
    rows = []
    specs = [
        (
            "affective",
            bool(cfg.USE_AFF_CONCEPT_BRANCH and cfg.USE_AFF_CONCEPT_SUPERVISION),
            AFF_CONCEPT_NAMES,
            "c_aff_",
            "target_aff_",
        ),
        (
            "style",
            bool(cfg.USE_STYLE_BRANCH),
            STYLE_CONCEPT_NAMES,
            "c_style_",
            "target_style_",
        ),
    ]

    for branch, branch_active, names, pred_prefix, target_prefix in specs:
        for concept in names:
            pred_col = f"{pred_prefix}{concept}"
            target_col = f"{target_prefix}{concept}"
            if pred_col not in pred_all.columns or target_col not in pred_all.columns:
                continue

            pred = pd.to_numeric(pred_all[pred_col], errors="coerce")
            target = pd.to_numeric(pred_all[target_col], errors="coerce")
            err = (pred - target).abs()
            valid = err.notna()

            rows.append({
                "branch": branch,
                "concept": concept,
                "branch_active": branch_active,
                "mae": float(err[valid].mean()) if branch_active and valid.any() else float("nan"),
                "abs_error_std": float(err[valid].std()) if branch_active and valid.any() else float("nan"),
                "pred_mean": float(pred[valid].mean()) if valid.any() else float("nan"),
                "target_mean": float(target[valid].mean()) if valid.any() else float("nan"),
                "n": int(valid.sum()),
            })

    path = os.path.join(cfg.OUT_DIR, "concept_mae_by_concept.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _pdf_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_add_text(ops: List[str], x: float, y: float, text: str, size: float = 10.0, bold: bool = False) -> None:
    font = "F2" if bold else "F1"
    ops.append(f"BT /{font} {size:.1f} Tf {x:.1f} {y:.1f} Td ({_pdf_escape(text)}) Tj ET")


def _pdf_add_rect(ops: List[str], x: float, y: float, w: float, h: float, rgb: Tuple[float, float, float]) -> None:
    r, g, b = rgb
    ops.append(f"q {r:.3f} {g:.3f} {b:.3f} rg {x:.1f} {y:.1f} {w:.1f} {h:.1f} re f Q")


def _pdf_add_line(ops: List[str], x1: float, y1: float, x2: float, y2: float, gray: float = 0.75, width: float = 0.8) -> None:
    ops.append(f"q {gray:.3f} {gray:.3f} {gray:.3f} RG {width:.2f} w {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S Q")


def _write_simple_pdf(path: str, width: int, height: int, ops: List[str]) -> None:
    content = ("\n".join(ops) + "\n").encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"endstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )

    with open(path, "wb") as f:
        f.write(pdf)


def write_factorization_audit_figure(results: pd.DataFrame, cfg: Config) -> str:
    path = os.path.join(cfg.OUT_DIR, "factorization_audit_figure.pdf")
    width, height = 612, 420
    ops: List[str] = []

    _pdf_add_text(ops, 44, 388, "Factorization audit", size=16, bold=True)
    _pdf_add_text(
        ops,
        44,
        369,
        "Speaker separability and concept intervention diagnostics averaged across folds.",
        size=9,
    )

    panels = [
        (
            54,
            "Speaker probe accuracy",
            [
                ("affective", _mean_metric(results, "speaker_probe_aff_acc"), (0.180, 0.360, 0.720)),
                ("style", _mean_metric(results, "speaker_probe_style_acc"), (0.760, 0.300, 0.240)),
            ],
        ),
        (
            330,
            "Swap diagnostics",
            [
                ("style-swap", _mean_metric(results, "style_swap_consistency"), (0.200, 0.560, 0.360)),
                ("aff-swap", _mean_metric(results, "aff_swap_sensitivity"), (0.860, 0.570, 0.170)),
            ],
        ),
    ]

    base_y = 112.0
    chart_h = 210.0
    chart_w = 220.0
    bar_w = 55.0

    for panel_x, title, bars in panels:
        _pdf_add_text(ops, panel_x, 340, title, size=11, bold=True)
        _pdf_add_line(ops, panel_x, base_y, panel_x + chart_w, base_y, gray=0.15, width=1.0)
        _pdf_add_line(ops, panel_x, base_y, panel_x, base_y + chart_h, gray=0.15, width=1.0)
        for tick in [0.0, 0.5, 1.0]:
            y = base_y + tick * chart_h
            _pdf_add_line(ops, panel_x, y, panel_x + chart_w, y, gray=0.88, width=0.5)
            _pdf_add_text(ops, panel_x - 23, y - 3, f"{tick:.1f}", size=8)

        for idx, (label, value, color) in enumerate(bars):
            value_for_bar = value if np.isfinite(value) else 0.0
            value_for_bar = float(np.clip(value_for_bar, 0.0, 1.0))
            bar_h = value_for_bar * chart_h
            x = panel_x + 45 + idx * 92
            _pdf_add_rect(ops, x, base_y, bar_w, bar_h, color)
            _pdf_add_text(ops, x + 5, base_y + bar_h + 8, _format_metric(value), size=9, bold=True)
            _pdf_add_text(ops, x - 2, 88, label, size=9)

    _pdf_add_text(ops, 44, 42, "Expected pattern: lower affective speaker probe, higher style speaker probe, stable style-swap,", size=8)
    _pdf_add_text(ops, 44, 29, "and high affect-swap sensitivity.", size=8)

    _write_simple_pdf(path, width, height, ops)
    return path


@torch.no_grad()
def collect_concepts_for_diagnostics(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    ys = []
    c_affs = []
    c_styles = []

    for batch in loader:
        x = batch["x"].to(device)
        out = model(x, grl_lambda=0.0)
        ys.append(batch["y"].detach().cpu().numpy())
        c_affs.append(out["c_aff"].detach().cpu().numpy())
        c_styles.append(out["c_style"].detach().cpu().numpy())

    return {
        "y": np.concatenate(ys).astype(np.int64),
        "c_aff": np.concatenate(c_affs).astype(np.float32),
        "c_style": np.concatenate(c_styles).astype(np.float32),
    }


def _safe_probe_metrics(clf: Optional[LogisticRegression], X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if clf is None:
        return {"acc": float("nan"), "uar": float("nan"), "macro_f1": float("nan")}
    try:
        pred = clf.predict(np.asarray(X, dtype=np.float32))
        return {
            "acc": float(accuracy_score(y, pred)),
            "uar": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        }
    except Exception:
        return {"acc": float("nan"), "uar": float("nan"), "macro_f1": float("nan")}


def _nontrivial_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.arange(n)
    perm = rng.permutation(n)
    if np.all(perm == np.arange(n)):
        perm = np.roll(perm, 1)
    return perm


def concept_intervention_diagnostics(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    seed: int,
) -> Dict[str, float]:
    """Post-hoc concept-level intervention diagnostics.

    We fit a frozen post-hoc probe on train concepts using [c_aff, c_style].
    Then we swap concept subspaces on the test fold.
    """
    rng = np.random.default_rng(seed)

    train = collect_concepts_for_diagnostics(model, train_loader, device)
    test = collect_concepts_for_diagnostics(model, test_loader, device)

    Xtr_aff = train["c_aff"]
    Xte_aff = test["c_aff"]
    Xtr_style = train["c_style"]
    Xte_style = test["c_style"]
    Xtr_both = np.concatenate([Xtr_aff, Xtr_style], axis=1)
    Xte_both = np.concatenate([Xte_aff, Xte_style], axis=1)
    ytr = train["y"]
    yte = test["y"]

    aff_probe = fit_logistic_regression_probe(Xtr_aff, ytr, seed=seed)
    style_probe = fit_logistic_regression_probe(Xtr_style, ytr, seed=seed + 17)
    both_probe = fit_logistic_regression_probe(Xtr_both, ytr, seed=seed + 31)

    aff_m = _safe_probe_metrics(aff_probe, Xte_aff, yte)
    style_m = _safe_probe_metrics(style_probe, Xte_style, yte)
    both_m = _safe_probe_metrics(both_probe, Xte_both, yte)

    out = {
        "emotion_probe_aff_acc": aff_m["acc"],
        "emotion_probe_aff_uar": aff_m["uar"],
        "emotion_probe_aff_macro_f1": aff_m["macro_f1"],
        "emotion_probe_style_acc": style_m["acc"],
        "emotion_probe_style_uar": style_m["uar"],
        "emotion_probe_style_macro_f1": style_m["macro_f1"],
        "emotion_probe_both_acc": both_m["acc"],
        "emotion_probe_both_uar": both_m["uar"],
        "emotion_probe_both_macro_f1": both_m["macro_f1"],
        "style_swap_consistency": float("nan"),
        "style_swap_prob_l1": float("nan"),
        "aff_swap_sensitivity": float("nan"),
        "aff_swap_prob_l1": float("nan"),
    }

    if both_probe is None or len(Xte_both) < 2:
        return out

    try:
        base_pred = both_probe.predict(Xte_both)
        base_prob = both_probe.predict_proba(Xte_both)

        perm_style = _nontrivial_permutation(len(Xte_style), rng)
        Xte_style_swapped = np.concatenate([Xte_aff, Xte_style[perm_style]], axis=1)
        style_swap_pred = both_probe.predict(Xte_style_swapped)
        style_swap_prob = both_probe.predict_proba(Xte_style_swapped)

        perm_aff = _nontrivial_permutation(len(Xte_aff), rng)
        Xte_aff_swapped = np.concatenate([Xte_aff[perm_aff], Xte_style], axis=1)
        aff_swap_pred = both_probe.predict(Xte_aff_swapped)
        aff_swap_prob = both_probe.predict_proba(Xte_aff_swapped)

        out["style_swap_consistency"] = float(np.mean(base_pred == style_swap_pred))
        out["style_swap_prob_l1"] = float(np.mean(np.sum(np.abs(base_prob - style_swap_prob), axis=1)))
        out["aff_swap_sensitivity"] = float(1.0 - np.mean(base_pred == aff_swap_pred))
        out["aff_swap_prob_l1"] = float(np.mean(np.sum(np.abs(base_prob - aff_swap_prob), axis=1)))
    except Exception:
        pass

    return out

