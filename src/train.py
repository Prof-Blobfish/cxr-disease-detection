# Run: python src/train.py --config experiments/densenet121_seed42_fold0.yaml --overwrite

from pathlib import Path
import shutil
import argparse

from tqdm.auto import tqdm

import yaml
import csv
import json

import math
import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import DataLoader

from config import PROJECT_ROOT, NUM_WORKERS
from data import prepare_data
from datasets import build_datasets
from transforms import build_transforms
from models.factory import build_model, build_parameter_groups
from metrics import compute_multilabel_metrics, per_class_metrics_to_frame

# ====================== Config & Setup ======================

def load_run_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.is_file():
        config_path = PROJECT_ROOT / config_path
    
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    run_dir = (PROJECT_ROOT / cfg["output_dir"] / cfg["run_name"]).resolve()
    cfg["run_dir"] = str(run_dir)

    return cfg

def clear_run_artifacts(run_dir: Path) -> None:
    owned_files = [
        "config.yaml",
        "class_order.json",
        "history.csv",
        "best_val_metrics.json",
        "metrics_test.json",
        "best_val_per_class_metrics.csv",
        "per_class_metrics_test.csv",
        "best_val_predictions.csv",
        "test_predictions.csv",
    ]

    for filename in owned_files:
        artifact_path = run_dir / filename
        if artifact_path.exists():
            artifact_path.unlink()

    for dirname in ["checkpoints", "plots"]:
        directory_path = run_dir / dirname
        if not directory_path.exists():
            continue

        for child in directory_path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

def initialize_run_dir(
        cfg: dict,
        overwrite: bool = False,
) -> Path:
    run_dir = Path(cfg["run_dir"])
    config_out_path = run_dir / "config.yaml"

    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists and is not empty: {run_dir}"
            )
        
        clear_run_artifacts(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    with config_out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    return run_dir

def set_global_seed(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)

def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def make_loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator

# ====================== Builders ======================

def build_criterion(
        cfg: dict,
        pos_weights,
        device: torch.device,
) -> torch.nn.Module:
    loss_type = str(cfg["loss_type"]).lower()

    if loss_type == "bce_logits":
        pos_weight = torch.as_tensor(
            pos_weights,
            dtype=torch.float32,
            device=device,
        )
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    raise ValueError(f"Unsupported loss_type: {loss_type}")

def build_optimizer(
        cfg: dict,
        param_groups: list[dict],
) -> torch.optim.Optimizer:
    optimizer_name = str(cfg["optimizer"]["name"]).lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups)
    
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def build_scheduler(
        cfg: dict,
        optimizer: torch.optim.Optimizer,
):
    scheduler_type = str(cfg.get("scheduler_type", "none")).lower()
    scheduler_cfg = cfg.get("scheduler", {}) or {}

    if scheduler_type in {"none", ""}:
        return None
    
    if scheduler_type == "cosine":
        eta_min = float(scheduler_cfg.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg["num_epochs"]),
            eta_min=eta_min,
        )
    
    raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

# ====================== Training ======================

def train_one_epoch(
        model: torch.nn.Module,
        train_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        max_batches: int | None = None,
        progress_desc: str = "Train",
) -> dict[str, float]:
    model.train()

    running_loss = 0.0
    num_examples = 0
    num_batches = 0

    total_batches = len(train_loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    progress_bar = tqdm(
        train_loader,
        total=total_batches,
        desc=progress_desc,
        leave=False,
    )

    for batch_idx, batch in enumerate(progress_bar):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at batch {batch_idx}: {loss.item()}")
        
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        num_examples += batch_size
        num_batches += 1

        avg_loss = running_loss / max(num_examples, 1)
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        if max_batches is not None and num_batches >= max_batches:
            break

    epoch_loss = running_loss / max(num_examples, 1)

    return {
        "train_loss": epoch_loss,
        "num_batches": float(num_batches),
        "num_examples": float(num_examples),
    }

def evaluate_one_epoch(
        model: torch.nn.Module,
        eval_loader: DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        max_batches: int | None = None,
        progress_desc: str = "Val",
) -> dict[str, object]:
    model.eval()

    running_loss = 0.0
    num_examples = 0
    num_batches = 0

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_image_ids: list[str] = []

    total_batches = len(eval_loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    progress_bar = tqdm(
        eval_loader,
        total=total_batches,
        desc=progress_desc,
        leave=False,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite loss at batch {batch_idx}: {loss.item()}"
                )
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            num_examples += batch_size
            num_batches += 1

            avg_loss = running_loss / max(num_examples, 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_image_ids.extend(batch["image_id"])

            if max_batches is not None and num_batches >= max_batches:
                break

    loss = running_loss / max(num_examples, 1)

    return {
        "loss": loss,
        "num_batches": float(num_batches),
        "num_examples": float(num_examples),
        "logits": torch.cat(all_logits, dim=0),
        "targets": torch.cat(all_targets, dim=0),
        "image_ids": all_image_ids,
    }

# ====================== Artifacts & Logging ======================

def make_history_row(
        epoch: int,
        train_metrics: dict[str, float],
        split_metrics: dict[str, object],
        metric_summary: dict[str, float],
) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "train_loss": float(train_metrics["train_loss"]),
        "val_loss": float(split_metrics["loss"]),
        "val_macro_auroc": float(metric_summary["macro_auroc"]),
        "val_macro_auprc": float(metric_summary["macro_auprc"]),
    }

def save_history_csv(
        history: list[dict[str, float | int]],
        run_dir: str | Path,
) -> Path:
    run_dir = Path(run_dir)
    history_path = run_dir / "history.csv"

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_macro_auroc",
        "val_macro_auprc",
    ]

    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in history:
            writer.writerow(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                    "val_macro_auroc": float(row["val_macro_auroc"]),
                    "val_macro_auprc": float(row["val_macro_auprc"]),
                }
            )
    
    return history_path

def _json_safe_float(value: object) -> float | None:
    if value is None:
        return None
    
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    
    return numeric_value

def save_metrics_json(
        epoch: int,
        split: str,
        split_metrics: dict[str, object],
        metric_summary: dict[str, object],
        run_dir: str | Path,
        output_filename: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    metrics_filename = output_filename or f"metrics_{split}.json"
    metrics_path = run_dir / metrics_filename

    payload = {
        "epoch": int(epoch),
        "split": split,
        f"{split}_loss": _json_safe_float(split_metrics["loss"]),
        f"{split}_macro_auroc": _json_safe_float(metric_summary["macro_auroc"]),
        f"{split}_macro_auprc": _json_safe_float(metric_summary["macro_auprc"]),
        "num_valid_auroc_classes": int(metric_summary["num_valid_auroc_classes"]),
        "num_valid_auprc_classes": int(metric_summary["num_valid_auprc_classes"]),
        "num_total_classes": int(metric_summary["num_total_classes"]),
        "num_batches": int(split_metrics["num_batches"]),
        "num_examples": int(split_metrics["num_examples"]),
    }

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    return metrics_path

def save_per_class_metrics_csv(
        epoch: int,
        split: str,
        metric_summary: dict[str, object],
        run_dir: str | Path,
        output_filename: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    per_class_filename = output_filename or f"per_class_metrics_{split}.csv"
    per_class_path = run_dir / per_class_filename

    per_class_df = per_class_metrics_to_frame(metric_summary).copy()

    if per_class_df.empty:
        raise ValueError("No per-class metrics available to save.")

    per_class_df.insert(0, "split", split)
    per_class_df.insert(0, "epoch", int(epoch))

    column_order = [
        "epoch",
        "split",
        "class_name",
        "class_index",
        "positive_count",
        "negative_count",
        "positive_prevalence",
        "auroc",
        "auprc",
        "valid_for_auroc",
        "valid_for_auprc",
    ]
    per_class_df = per_class_df.loc[:, column_order]

    per_class_df.to_csv(per_class_path, index=False)

    return per_class_path

def save_predictions_csv(
        epoch: int,
        split: str,
        split_metrics: dict[str, object],
        target_columns: list[str],
        run_dir: str | Path,
        output_filename: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    predictions_filename = output_filename or f"{split}_predictions.csv"
    predictions_path = run_dir / predictions_filename

    image_ids = list(split_metrics["image_ids"])
    logits = split_metrics["logits"]
    targets = split_metrics["targets"]

    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError(
            "split_metrics['logits'] and split_metrics['targets'] must be torch.Tensor values."
        )
    
    if logits.shape != targets.shape:
        raise ValueError(
            f"Logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}"
        )
    
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits/targets, got shape {tuple(logits.shape)}")
    
    if len(image_ids) != logits.shape[0]:
        raise ValueError(
            f"image_ids length ({len(image_ids)}) does not match row count ({logits.shape[0]})"
        )
    
    if len(target_columns) != logits.shape[1]:
        raise ValueError(
            f"target_columns length ({len(target_columns)}) does not match class dimension ({logits.shape[1]})"
        )
    
    probs = torch.sigmoid(logits).cpu().numpy()
    targets_np = targets.cpu().numpy()

    pred_columns = [
        f"pred_{target_column.removeprefix('target_')}"
        for target_column in target_columns
    ]

    base_df = pd.DataFrame(
        {
            "epoch": int(epoch),
            "split": split,
            "image_id": image_ids,
        }
    )
    pred_df = pd.DataFrame(probs, columns=pred_columns)
    target_df = pd.DataFrame(targets_np, columns=target_columns).round().astype("int64")

    predictions_df = pd.concat([base_df, pred_df, target_df], axis=1)
    predictions_df.to_csv(predictions_path, index=False)

    return predictions_path

def save_class_order_json(
        class_names: list[str],
        target_columns: list[str],
        run_dir: str | Path,
) -> Path:
    run_dir = Path(run_dir)
    class_order_path = run_dir / "class_order.json"

    if len(class_names) != len(target_columns):
        raise ValueError(
            f"class_names length ({len(class_names)}) does not match "
            f"target_columns length ({len(target_columns)})"
        )
    
    pred_columns = [
        f"pred_{target_column.removeprefix('target_')}"
        for target_column in target_columns
    ]

    payload = {
        "num_classes": len(class_names),
        "class_names": class_names,
        "target_columns": target_columns,
        "pred_columns": pred_columns,
        "classes": [
            {
                "index": index,
                "class_name": class_name,
                "target_column": target_columns[index],
                "pred_column": pred_columns[index],
            }
            for index, class_name in enumerate(class_names)
        ],
    }

    with class_order_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    return class_order_path

def resolve_checkpoint_metric(
        cfg: dict,
        split_metrics: dict[str, object],
        metric_summary: dict[str, object],
) -> float:
    metric_name = str(cfg["checkpoint_metric"]["name"]).lower()

    if metric_name == "val_macro_auroc":
        metric_value = metric_summary["macro_auroc"]
    elif metric_name == "val_loss":
        metric_value = split_metrics["loss"]
    else:
        raise ValueError(f"Unsupported checkpoint metric: {metric_name}")

    metric_value = float(metric_value)
    if not math.isfinite(metric_value):
        raise ValueError(
            f"Checkpoint metric '{metric_name}' value is not finite: {metric_value}"
        )

    return metric_value

def is_better_checkpoint(
        current_value: float,
        best_value: float | None,
        mode: str,
) -> bool:
    mode = str(mode).lower()

    if mode not in {"max", "min"}:
        raise ValueError(f"Unsupported checkpoint mode: {mode}")

    if best_value is None:
        return True

    if mode == "max":
        return current_value > best_value

    return current_value < best_value
    
def save_checkpoint(
        checkpoint_path: str | Path,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        current_metric: float,
        best_metric: float | None,
        cfg: dict,
        scheduler=None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)

    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "current_metric": float(current_metric),
        "best_metric": None if best_metric is None else float(best_metric),
        "checkpoint_metric_name": str(cfg["checkpoint_metric"]["name"]).lower(),
        "checkpoint_metric_mode": str(cfg["checkpoint_metric"]["mode"]).lower(),
    }

    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, checkpoint_path)
    return checkpoint_path

def persist_epoch_state(
        *,
        epoch: int,
        run_dir: str | Path,
        history: list[dict[str, float | int]],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: dict,
        current_metric: float,
        best_metric: float | None,
        scheduler=None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)

    history_path = save_history_csv(history, run_dir)
    last_checkpoint_path = save_checkpoint(
        checkpoint_path=run_dir / "checkpoints" / "last.pt",
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        current_metric=current_metric,
        best_metric=best_metric,
        cfg=cfg,
        scheduler=scheduler,
    )

    return {
        "history": history_path,
        "last_checkpoint": last_checkpoint_path,
    }

def persist_best_validation_artifacts(
        *,
        epoch: int,
        run_dir: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: dict,
        split_metrics: dict[str, object],
        metric_summary: dict[str, object],
        current_metric: float,
        best_metric: float | None,
        target_columns: list[str],
        scheduler=None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)

    best_checkpoint_path = save_checkpoint(
        checkpoint_path=run_dir / "checkpoints" / "best.pt",
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        current_metric=current_metric,
        best_metric=best_metric,
        cfg=cfg,
        scheduler=scheduler,
    )

    best_val_metrics_path = save_metrics_json(
        epoch=epoch,
        split="val",
        split_metrics=split_metrics,
        metric_summary=metric_summary,
        run_dir=run_dir,
        output_filename="best_val_metrics.json",
    )

    best_val_per_class_metrics_path = save_per_class_metrics_csv(
        epoch=epoch,
        split="val",
        metric_summary=metric_summary,
        run_dir=run_dir,
        output_filename="best_val_per_class_metrics.csv",
    )

    best_val_predictions_path = save_predictions_csv(
        epoch=epoch,
        split="val",
        split_metrics=split_metrics,
        target_columns=target_columns,
        run_dir=run_dir,
        output_filename="best_val_predictions.csv",
    )

    return {
        "best_checkpoint": best_checkpoint_path,
        "best_val_metrics": best_val_metrics_path,
        "best_val_per_class_metrics": best_val_per_class_metrics_path,
        "best_val_predictions": best_val_predictions_path,
    }
    
def print_epoch_summary(
        *,
        epoch: int,
        num_epochs: int,
        cfg: dict,
        train_metrics: dict[str, float],
        split_metrics: dict[str, object],
        metric_summary: dict[str, float],
        current_metric: float,
        best_epoch: int,
        epoch_artifacts: dict[str, Path],
        best_artifacts: dict[str, Path] | None,
) -> None:
    print(f"=== Epoch {epoch} / {num_epochs} Summary ===")
    print(f"Train loss: {train_metrics['train_loss']:.6f}")
    print(f"Val loss: {float(split_metrics['loss']):.6f}")
    print(f"Val macro AUROC: {float(metric_summary['macro_auroc']):.6f}")
    print(f"Val macro AUPRC: {float(metric_summary['macro_auprc']):.6f}")
    print(
        f"Checkpoint metric ({cfg['checkpoint_metric']['name']}): "
        f"{current_metric:.6f}"
    )
    print(f"Best epoch so far: {best_epoch}")
    print(f"Saved history to: {epoch_artifacts['history']}")
    print(f"Saved last checkpoint to: {epoch_artifacts['last_checkpoint']}")

    if best_artifacts is not None:
        print(f"Updated best checkpoint: {best_artifacts['best_checkpoint']}")
        print(f"Saved best validation metrics to: {best_artifacts['best_val_metrics']}")
        print(
            f"Saved best per-class validation metrics to: "
            f"{best_artifacts['best_val_per_class_metrics']}"
        )
        print(
            f"Saved best validation predictions to: "
            f"{best_artifacts['best_val_predictions']}"
        )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite known artifacts in an existing run directory.",
    )
    args = parser.parse_args()

    cfg = load_run_config(args.config)

    train_seed = int(cfg["train_seed"])
    split_seed = int(cfg["split_seed"])
    deterministic = bool(cfg["deterministic"])
    
    train_aug_cfg = cfg.get("augmentation", {}).get("train", {}) or {}
    rotation_deg = float(train_aug_cfg.get("rotation_deg", 0.0))

    set_global_seed(train_seed, deterministic=deterministic)
    bundle = prepare_data(split_seed=split_seed)

    print("\n=== Run Config ===")
    print(f"Run: {cfg['run_name']}")
    print(f"Model: {cfg['model_name']}")
    print(f"Backbone LR: {cfg['optimizer']['backbone_lr']}")
    print(f"Rotation Degree: {rotation_deg}")
    print(f"Train seed: {train_seed}")
    print(f"Split seed: {split_seed}")
    print(f"Deterministic: {deterministic}")


    print("\n=== Data Summary ===")
    print(f"Train rows: {len(bundle.train_df)}")
    print(f"Val rows: {len(bundle.val_df)}")
    print(f"Test rows: {len(bundle.test_df)}")
    print(f"Num classes: {len(bundle.class_names)}")
    print(f"First classes: {bundle.class_names[:5]}")
    print(f"Num target columns: {len(bundle.target_columns)}")
    print(f"Pos weight shape: {bundle.pos_weights.shape}")

    run_dir = initialize_run_dir(cfg, overwrite=args.overwrite)

    class_order_path = save_class_order_json(
        class_names=bundle.class_names,
        target_columns=bundle.target_columns,
        run_dir=run_dir,
    )

    print(f"Saved class order to: {class_order_path}")

    transform_dict = build_transforms(cfg)

    datasets = build_datasets(
        bundle,
        train_transform=transform_dict["train"],
        eval_transform=transform_dict["eval"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_cuda = device.type == "cuda"
    
    loader_common = {
        "batch_size": cfg["batch_size"],
        "num_workers": NUM_WORKERS,
        "pin_memory": use_cuda,
        "persistent_workers": NUM_WORKERS > 0,
    }
    if NUM_WORKERS > 0:
        loader_common["prefetch_factor"] = 2

    train_generator = make_loader_generator(train_seed)

    train_loader = DataLoader(
        datasets["train"], 
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
        **loader_common,
    )
    val_loader = DataLoader(
        datasets["val"], 
        shuffle=False,
        **loader_common,
    )
    test_loader = DataLoader(
        datasets["test"], 
        shuffle=False,
        **loader_common,
    )

    model = build_model(cfg, num_classes=len(bundle.class_names)).to(device)
    param_groups = build_parameter_groups(model, cfg)

    criterion = build_criterion(
        cfg=cfg,
        pos_weights=bundle.pos_weights,
        device=device,
    )
    optimizer = build_optimizer(
        cfg=cfg,
        param_groups=param_groups,
    )
    scheduler = build_scheduler(
        cfg=cfg,
        optimizer=optimizer,
    )

    history: list[dict[str, float | int]] = []

    num_epochs = int(cfg["num_epochs"])
    patience = int(cfg["patience"])
    checkpoint_mode = str(cfg["checkpoint_metric"]["mode"]).lower()

    best_metric: float | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    train_max_batches = None
    val_max_batches = None
    for epoch in range(1, num_epochs + 1):
        print(f"=== Epoch {epoch} / {num_epochs} ===")

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=train_max_batches,
            progress_desc=f"Train {epoch}/{num_epochs}",
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            eval_loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=val_max_batches,
            progress_desc=f"Val {epoch}/{num_epochs}",
        )

        metric_summary = compute_multilabel_metrics(
            logits=val_metrics["logits"],
            targets=val_metrics["targets"],
            class_names=bundle.class_names,
        )

        history_row = make_history_row(
            epoch=epoch,
            train_metrics=train_metrics,
            split_metrics=val_metrics,
            metric_summary=metric_summary,
        )
        history.append(history_row)

        current_metric = resolve_checkpoint_metric(
            cfg=cfg,
            split_metrics=val_metrics,
            metric_summary=metric_summary,
        )

        improved = is_better_checkpoint(
            current_value=current_metric,
            best_value=best_metric,
            mode=checkpoint_mode,
        )

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_artifacts = persist_epoch_state(
            epoch=epoch,
            run_dir=run_dir,
            history=history,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            current_metric=current_metric,
            best_metric=best_metric,
            scheduler=scheduler,
        )

        best_artifacts = None
        if improved:
            best_artifacts = persist_best_validation_artifacts(
                epoch=epoch,
                run_dir=run_dir,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                split_metrics=val_metrics,
                metric_summary=metric_summary,
                current_metric=current_metric,
                best_metric=best_metric,
                target_columns=bundle.target_columns,
                scheduler=scheduler,
            )

        print_epoch_summary(
            epoch=epoch,
            num_epochs=num_epochs,
            cfg=cfg,
            train_metrics=train_metrics,
            split_metrics=val_metrics,
            metric_summary=metric_summary,
            current_metric=current_metric,
            best_epoch=best_epoch,
            epoch_artifacts=epoch_artifacts,
            best_artifacts=best_artifacts,
        )

        current_lrs = [group["lr"] for group in optimizer.param_groups]
        print(f"Current LRs: {current_lrs}")

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch} after "
                f"{epochs_without_improvement} epochs without improvement."
            )
            break

        if scheduler is not None:
            scheduler.step()

    # Training metrics

    best_checkpoint_path = run_dir / "checkpoints" / "best.pt"
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)

    model.load_state_dict(best_checkpoint["model_state_dict"])

    print(f"Loaded best checkpoint from: {best_checkpoint_path}")
    print(f"Evaluting best checkpoint from epoch: {best_checkpoint['epoch']}")

    test_metrics = evaluate_one_epoch(
        model=model,
        eval_loader=test_loader,
        criterion=criterion,
        device=device,
        max_batches=None,
        progress_desc="Test",
    )

    test_metric_summary = compute_multilabel_metrics(
        logits=test_metrics["logits"],
        targets=test_metrics["targets"],
        class_names=bundle.class_names,
    )

    metrics_test_path = save_metrics_json(
        epoch=best_checkpoint["epoch"],
        split="test",
        split_metrics=test_metrics,
        metric_summary=test_metric_summary,
        run_dir=run_dir,
    )

    per_class_metrics_test_path = save_per_class_metrics_csv(
        epoch=int(best_checkpoint["epoch"]),
        split="test",
        metric_summary=test_metric_summary,
        run_dir=run_dir,
    )

    test_predictions_path = save_predictions_csv(
        epoch=int(best_checkpoint["epoch"]),
        split="test",
        split_metrics=test_metrics,
        target_columns=bundle.target_columns,
        run_dir=run_dir,
    )

    print("=== Test Summary ===")
    print(f"Test loss: {float(test_metrics['loss']):.6f}")
    print(f"Test macro AUROC: {float(test_metric_summary['macro_auroc']):.6f}")
    print(f"Test macro AUPRC: {float(test_metric_summary['macro_auprc']):.6f}")
    print(f"Saved test metrics to: {metrics_test_path}")
    print(f"Saved per-class test metrics to: {per_class_metrics_test_path}")
    print(f"Saved test predictions to: {test_predictions_path}")

    print("=== Training Complete ===")
    print(f"Best epoch: {best_epoch}")
    if best_metric is not None:
        print(
            f"Best {cfg['checkpoint_metric']['name']}: "
            f"{best_metric:.6f}"
        )

if __name__ == "__main__":
    main()