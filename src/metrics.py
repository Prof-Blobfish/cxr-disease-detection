# AUROC, per-class metrics, aggregation

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compute_multilabel_metrics(
        logits: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        class_names: list[str],
) -> dict[str, Any]:
    logits_np = _to_numpy(logits).astype(np.float32, copy=False)
    targets_np = _to_numpy(targets).astype(np.float32, copy=False)

    if logits_np.shape != targets_np.shape:
        raise ValueError(f"Shape mismatch between logits and targets: {logits_np.shape} vs {targets_np.shape}")

    if logits_np.ndim != 2:
        raise ValueError(f"Expected 2D arrays for logits and targets, got shape: {logits_np.shape}")
    
    if len(class_names) != logits_np.shape[1]:
        raise ValueError(f"Number of class names ({len(class_names)}) does not match number of classes in logits ({logits_np.shape[1]})")
    
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))

    per_class_rows: list[dict[str, Any]] = []
    valid_aurocs: list[float] = []
    valid_auprcs: list[float] = []

    for class_idx, class_name in enumerate(class_names):
        y_true = targets_np[:, class_idx]
        y_score = probs_np[:, class_idx]

        positive_count = int(y_true.sum())
        negative_count = int(len(y_true)) - positive_count
        positive_prevalence = float(positive_count / len(y_true))

        valid_for_auroc = positive_count > 0 and negative_count > 0
        if valid_for_auroc:
            auroc = float(roc_auc_score(y_true, y_score))
            valid_aurocs.append(auroc)
        else:
            auroc = None

        valid_for_auprc = positive_count > 0
        if valid_for_auprc:
            auprc = float(average_precision_score(y_true, y_score))
            valid_auprcs.append(auprc)
        else:
            auprc = None

        per_class_rows.append(
            {
                "class_name": class_name,
                "class_index": class_idx,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_prevalence": positive_prevalence,
                "auroc": auroc,
                "auprc": auprc,
                "valid_for_auroc": valid_for_auroc,
                "valid_for_auprc": valid_for_auprc,
            }
        )

    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    macro_auprc = float(np.mean(valid_auprcs)) if valid_auprcs else float("nan")

    return {
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "num_valid_auroc_classes": len(valid_aurocs),
        "num_valid_auprc_classes": len(valid_auprcs),
        "num_total_classes": len(class_names),
        "per_class": per_class_rows,
    }

def per_class_metrics_to_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(metrics["per_class"])

def _prepare_probability_targets(
        probs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        class_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    probs_np = _to_numpy(probs).astype(np.float32, copy=False)
    targets_np = _to_numpy(targets).astype(np.float32, copy=False)

    if probs_np.shape != targets_np.shape:
        raise ValueError(
            f"Shape mismatch between probs and targets: {probs_np.shape} vs {targets_np.shape}"
        )

    if probs_np.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays for probs and targets, got shape: {probs_np.shape}"
        )

    if len(class_names) != probs_np.shape[1]:
        raise ValueError(
            f"Number of class names ({len(class_names)}) does not match "
            f"number of classes in probs ({probs_np.shape[1]})"
        )

    if np.any(~np.isfinite(probs_np)):
        raise ValueError("Probabilities contain non-finite values.")

    if np.any(~np.isfinite(targets_np)):
        raise ValueError("Targets contain non-finite values.")

    if np.any((probs_np < 0.0) | (probs_np > 1.0)):
        raise ValueError("Expected probabilities in [0, 1].")

    return probs_np, targets_np.round().astype(np.int64, copy=False)


def _binary_confusion_counts(
        y_true: np.ndarray,
        y_pred: np.ndarray,
) -> tuple[int, int, int, int]:
    y_true_bool = y_true.astype(bool, copy=False)
    y_pred_bool = y_pred.astype(bool, copy=False)

    tp = int(np.logical_and(y_true_bool, y_pred_bool).sum())
    fp = int(np.logical_and(~y_true_bool, y_pred_bool).sum())
    tn = int(np.logical_and(~y_true_bool, ~y_pred_bool).sum())
    fn = int(np.logical_and(y_true_bool, ~y_pred_bool).sum())

    return tp, fp, tn, fn


def _metrics_from_confusion_counts(
        tp: int,
        fp: int,
        tn: int,
        fn: int,
) -> tuple[float, float | None, float | None, float | None]:
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    if (tp + fn) > 0:
        recall = float(tp / (tp + fn))
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    else:
        recall = None
        f1 = None

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None

    return precision, recall, f1, specificity


def _resolve_threshold_array(
        thresholds: Mapping[str, float] | Sequence[float] | np.ndarray,
        class_names: list[str],
) -> np.ndarray:
    if isinstance(thresholds, Mapping):
        missing = [class_name for class_name in class_names if class_name not in thresholds]
        if missing:
            raise ValueError(f"Missing thresholds for classes: {missing}")

        threshold_array = np.asarray(
            [float(thresholds[class_name]) for class_name in class_names],
            dtype=np.float32,
        )
    else:
        threshold_array = np.asarray(thresholds, dtype=np.float32)

    if threshold_array.shape != (len(class_names),):
        raise ValueError(
            f"Expected threshold array of shape ({len(class_names)},), "
            f"got {threshold_array.shape}"
        )

    if np.any((threshold_array < 0.0) | (threshold_array > 1.0)):
        raise ValueError("Thresholds must be in [0, 1].")

    return threshold_array


def tune_per_class_thresholds(
        probs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        class_names: list[str],
        objective: str = "f1",
        threshold_grid: Sequence[float] | np.ndarray | None = None,
        fallback_threshold: float = 0.5,
) -> dict[str, Any]:
    probs_np, targets_np = _prepare_probability_targets(
        probs=probs,
        targets=targets,
        class_names=class_names,
    )

    objective = str(objective).lower()
    if objective != "f1":
        raise ValueError(f"Unsupported threshold objective: {objective}")

    if not 0.0 <= float(fallback_threshold) <= 1.0:
        raise ValueError(f"fallback_threshold must be in [0, 1], got {fallback_threshold}")

    if threshold_grid is None:
        threshold_grid_np = np.linspace(0.01, 0.99, 99, dtype=np.float32)
    else:
        threshold_grid_np = np.asarray(list(threshold_grid), dtype=np.float32)

    if threshold_grid_np.ndim != 1 or len(threshold_grid_np) == 0:
        raise ValueError("threshold_grid must be a non-empty 1D sequence.")

    if np.any((threshold_grid_np < 0.0) | (threshold_grid_np > 1.0)):
        raise ValueError("All threshold_grid values must be in [0, 1].")

    thresholds: list[float] = []
    per_class_rows: list[dict[str, Any]] = []

    for class_idx, class_name in enumerate(class_names):
        y_true = targets_np[:, class_idx]
        y_score = probs_np[:, class_idx]

        positive_count = int(y_true.sum())
        negative_count = int(len(y_true) - positive_count)
        positive_prevalence = float(positive_count / len(y_true))

        best_threshold = float(fallback_threshold)
        best_score = None
        best_stats = None
        best_distance = float("inf")
        tuned = positive_count > 0

        if tuned:
            current_best_score = -1.0

            for threshold in threshold_grid_np:
                threshold_value = float(threshold)
                y_pred = y_score >= threshold_value

                tp, fp, tn, fn = _binary_confusion_counts(y_true, y_pred)
                precision, recall, f1, specificity = _metrics_from_confusion_counts(
                    tp=tp,
                    fp=fp,
                    tn=tn,
                    fn=fn,
                )

                score = 0.0 if f1 is None else float(f1)
                distance = abs(threshold_value - float(fallback_threshold))

                if (
                    score > current_best_score
                    or (
                        np.isclose(score, current_best_score)
                        and distance < best_distance
                    )
                ):
                    current_best_score = score
                    best_threshold = threshold_value
                    best_score = score
                    best_distance = distance
                    best_stats = {
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "specificity": specificity,
                        "predicted_positive_count": int(y_pred.sum()),
                    }
        else:
            y_pred = y_score >= float(fallback_threshold)
            tp, fp, tn, fn = _binary_confusion_counts(y_true, y_pred)
            precision, recall, f1, specificity = _metrics_from_confusion_counts(
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
            )
            best_stats = {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "predicted_positive_count": int(y_pred.sum()),
            }

        thresholds.append(best_threshold)
        per_class_rows.append(
            {
                "class_name": class_name,
                "class_index": class_idx,
                "threshold": best_threshold,
                "objective": objective,
                "objective_value": best_score,
                "tuned": tuned,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_prevalence": positive_prevalence,
                **best_stats,
            }
        )

    return {
        "objective": objective,
        "fallback_threshold": float(fallback_threshold),
        "thresholds": thresholds,
        "per_class": per_class_rows,
    }


def compute_thresholded_multilabel_metrics(
        probs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        class_names: list[str],
        thresholds: Mapping[str, float] | Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    probs_np, targets_np = _prepare_probability_targets(
        probs=probs,
        targets=targets,
        class_names=class_names,
    )
    threshold_array = _resolve_threshold_array(
        thresholds=thresholds,
        class_names=class_names,
    )

    per_class_rows: list[dict[str, Any]] = []
    valid_precisions: list[float] = []
    valid_recalls: list[float] = []
    valid_f1s: list[float] = []
    valid_specificities: list[float] = []

    for class_idx, class_name in enumerate(class_names):
        y_true = targets_np[:, class_idx]
        y_score = probs_np[:, class_idx]
        threshold = float(threshold_array[class_idx])
        y_pred = y_score >= threshold

        tp, fp, tn, fn = _binary_confusion_counts(y_true, y_pred)
        precision, recall, f1, specificity = _metrics_from_confusion_counts(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )

        positive_count = int(y_true.sum())
        negative_count = int(len(y_true) - positive_count)
        positive_prevalence = float(positive_count / len(y_true))
        predicted_positive_count = int(y_pred.sum())

        valid_for_precision = positive_count > 0
        valid_for_recall = positive_count > 0
        valid_for_f1 = positive_count > 0
        valid_for_specificity = negative_count > 0

        if valid_for_precision:
            valid_precisions.append(float(precision))
        if valid_for_recall and recall is not None:
            valid_recalls.append(float(recall))
        if valid_for_f1 and f1 is not None:
            valid_f1s.append(float(f1))
        if valid_for_specificity and specificity is not None:
            valid_specificities.append(float(specificity))

        per_class_rows.append(
            {
                "class_name": class_name,
                "class_index": class_idx,
                "threshold": threshold,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_prevalence": positive_prevalence,
                "predicted_positive_count": predicted_positive_count,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "valid_for_precision": valid_for_precision,
                "valid_for_recall": valid_for_recall,
                "valid_for_f1": valid_for_f1,
                "valid_for_specificity": valid_for_specificity,
            }
        )

    return {
        "macro_precision": float(np.mean(valid_precisions)) if valid_precisions else float("nan"),
        "macro_recall": float(np.mean(valid_recalls)) if valid_recalls else float("nan"),
        "macro_f1": float(np.mean(valid_f1s)) if valid_f1s else float("nan"),
        "macro_specificity": float(np.mean(valid_specificities)) if valid_specificities else float("nan"),
        "num_valid_precision_classes": len(valid_precisions),
        "num_valid_recall_classes": len(valid_recalls),
        "num_valid_f1_classes": len(valid_f1s),
        "num_valid_specificity_classes": len(valid_specificities),
        "num_total_classes": len(class_names),
        "per_class": per_class_rows,
    }