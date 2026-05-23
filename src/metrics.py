# AUROC, per-class metrics, aggregation

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


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

    for class_idx, class_name in enumerate(class_names):
        y_true = targets_np[:, class_idx]
        y_score = probs_np[:, class_idx]

        positive_count = int(y_true.sum())
        negative_count = int(len(y_true)) - positive_count

        if positive_count == 0 or negative_count == 0:
            auroc = None
            valid_for_auroc = False
        else:
            auroc = float(roc_auc_score(y_true, y_score))
            valid_aurocs.append(auroc)
            valid_for_auroc = True

        per_class_rows.append(
            {
                "class_name": class_name,
                "class_index": class_idx,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "auroc": auroc,
                "valid_for_auroc": valid_for_auroc,
            }
        )

    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")

    return {
        "macro_auroc": macro_auroc,
        "num_valid_auroc_classes": len(valid_aurocs),
        "num_total_classes": len(class_names),
        "per_class": per_class_rows,
    }

def per_class_metrics_to_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(metrics["per_class"])