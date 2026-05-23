from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATHS = [
    "experiments/densenet121_seed42.yaml",
    "experiments/densenet121_seed44.yaml",
    "experiments/densenet121_seed46.yaml",
]

COMMON_FLAGS = ["--overwrite"]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")

    return data


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")

    return data


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def run_one_config(config_rel_path: str) -> dict:
    config_path = resolve_project_path(config_rel_path)
    cfg = load_yaml(config_path)

    run_name = str(cfg["run_name"])
    output_dir = resolve_project_path(str(cfg["output_dir"]))
    run_dir = output_dir / run_name

    command = [
        sys.executable,
        "src/train.py",
        "--config",
        str(config_path),
        *COMMON_FLAGS,
    ]

    print()
    print("=" * 80)
    print(f"Running config: {config_rel_path}")
    print(f"Run name: {run_name}")
    print(f"Train seed: {cfg['train_seed']}")
    print(f"Split seed: {cfg['split_seed']}")
    print("Command:")
    print(" ".join(command))
    print("=" * 80)

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Run failed for {config_rel_path} with return code {completed.returncode}"
        )

    best_val_metrics_path = run_dir / "best_val_metrics.json"
    test_metrics_path = run_dir / "metrics_test.json"

    if not best_val_metrics_path.is_file():
        raise FileNotFoundError(f"Missing file: {best_val_metrics_path}")

    if not test_metrics_path.is_file():
        raise FileNotFoundError(f"Missing file: {test_metrics_path}")

    best_val_metrics = load_json(best_val_metrics_path)
    test_metrics = load_json(test_metrics_path)

    return {
        "config_path": config_rel_path,
        "run_name": run_name,
        "train_seed": int(cfg["train_seed"]),
        "split_seed": int(cfg["split_seed"]),
        "best_epoch": int(best_val_metrics["epoch"]),
        "val_macro_auroc": float(best_val_metrics["val_macro_auroc"]),
        "val_macro_auprc": float(best_val_metrics["val_macro_auprc"]),
        "test_macro_auroc": float(test_metrics["test_macro_auroc"]),
        "test_macro_auprc": float(test_metrics["test_macro_auprc"]),
    }


def main() -> None:
    results: list[dict] = []

    for config_rel_path in CONFIG_PATHS:
        result = run_one_config(config_rel_path)
        results.append(result)

    print()
    print("=== Finished ===")
    for result in results:
        print(
            f"{result['run_name']} | "
            f"train_seed={result['train_seed']} | "
            f"split_seed={result['split_seed']} | "
            f"best_epoch={result['best_epoch']} | "
            f"val_auroc={result['val_macro_auroc']:.4f} | "
            f"val_auprc={result['val_macro_auprc']:.4f} | "
            f"test_auroc={result['test_macro_auroc']:.4f} | "
            f"test_auprc={result['test_macro_auprc']:.4f}"
        )


if __name__ == "__main__":
    main()