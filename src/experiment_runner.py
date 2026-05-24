from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
import math

import yaml

CONFIG_PATHS = [
    "experiments/convnext_tiny_v1_seed42.yaml",
    "experiments/densenet121_v1_seed42.yaml",
    "experiments/resnet50_v1_seed42.yaml",
]
TRAIN_MODE = "resume"    # "overwrite" | "resume"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOT_SCRIPT_PATH = PROJECT_ROOT / "src" / "plot.py"
THRESHOLD_SCRIPT_PATH = PROJECT_ROOT / "src" / "tune_thresholds.py"

SUMMARY_CSV_PATH = (
    PROJECT_ROOT / "outputs" / "experiment_summaries" / "experiment_runner_summary.csv"
)

RUN_THRESHOLD_TUNING = True
RUN_PLOTS = True
PLOT_COMMON_FLAGS = ["--top-k-pr-classes", "5"]
THRESHOLD_COMMON_FLAGS: list[str] = []

SUMMARY_METRIC_KEYS = [
    "val_macro_auprc",
    "test_macro_auprc",
    "tuned_val_macro_f1",
    "tuned_val_macro_recall",
    "tuned_val_macro_specificity",
    "tuned_test_macro_f1",
    "tuned_test_macro_recall",
    "tuned_test_macro_specificity",
]


def build_train_flags() -> list[str]:
    if TRAIN_MODE == "overwrite":
        return ["--overwrite"]
    if TRAIN_MODE == "resume":
        return ["--resume"]
    raise ValueError(f"Unsupported TRAIN_MODE: {TRAIN_MODE}")


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


def maybe_metric(payload: dict, *keys: str) -> float | None:
    current: object = payload

    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    if current is None:
        return None

    numeric_value = float(current)
    if not math.isfinite(numeric_value):
        return None

    return numeric_value


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def build_result_row(
        *,
        config_path: str,
        cfg: dict,
        run_dir: Path,
        run_summary: dict,
        thresholds_created: bool,
        plots_created: bool,
) -> dict:
    return {
        "config_path": config_path,
        "run_name": str(run_summary["run_name"]),
        "run_dir": str(run_dir),
        "train_seed": int(cfg["train_seed"]),
        "split_seed": int(cfg["split_seed"]),
        "best_epoch": int(run_summary["best_epoch"]),
        "stop_reason": str(run_summary.get("stop_reason", "unknown")),
        "thresholds_created": bool(thresholds_created),
        "plots_created": bool(plots_created),
        "val_macro_auprc": maybe_metric(run_summary, "ranking_metrics", "val", "macro_auprc"),
        "test_macro_auprc": maybe_metric(run_summary, "ranking_metrics", "test", "macro_auprc"),
        "tuned_val_macro_f1": maybe_metric(run_summary, "threshold_metrics", "tuned", "val", "macro_f1"),
        "tuned_val_macro_recall": maybe_metric(run_summary, "threshold_metrics", "tuned", "val", "macro_recall"),
        "tuned_val_macro_specificity": maybe_metric(run_summary, "threshold_metrics", "tuned", "val", "macro_specificity"),
        "tuned_test_macro_f1": maybe_metric(run_summary, "threshold_metrics", "tuned", "test", "macro_f1"),
        "tuned_test_macro_recall": maybe_metric(run_summary, "threshold_metrics", "tuned", "test", "macro_recall"),
        "tuned_test_macro_specificity": maybe_metric(run_summary, "threshold_metrics", "tuned", "test", "macro_specificity"),
    }


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def metric_mean_std(results: list[dict], key: str) -> tuple[float | None, float | None]:
    values = [float(result[key]) for result in results if result.get(key) is not None]
    if not values:
        return None, None

    metric_mean = mean(values)
    metric_std = stdev(values) if len(values) > 1 else 0.0
    return metric_mean, metric_std


def save_summary_csv(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "row_type",
        "config_path",
        "run_name",
        "run_dir",
        "train_seed",
        "split_seed",
        "best_epoch",
        "stop_reason",
        "thresholds_created",
        "plots_created",
        "val_macro_auprc",
        "test_macro_auprc",
        "tuned_val_macro_f1",
        "tuned_val_macro_recall",
        "tuned_val_macro_specificity",
        "tuned_test_macro_f1",
        "tuned_test_macro_recall",
        "tuned_test_macro_specificity",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "row_type": "run",
                    "config_path": result["config_path"],
                    "run_name": result["run_name"],
                    "run_dir": result["run_dir"],
                    "train_seed": int(result["train_seed"]),
                    "split_seed": int(result["split_seed"]),
                    "best_epoch": int(result["best_epoch"]),
                    "stop_reason": result["stop_reason"],
                    "thresholds_created": bool(result["thresholds_created"]),
                    "plots_created": bool(result["plots_created"]),
                    "val_macro_auprc": result["val_macro_auprc"],
                    "test_macro_auprc": result["test_macro_auprc"],
                    "tuned_val_macro_f1": result["tuned_val_macro_f1"],
                    "tuned_val_macro_recall": result["tuned_val_macro_recall"],
                    "tuned_val_macro_specificity": result["tuned_val_macro_specificity"],
                    "tuned_test_macro_f1": result["tuned_test_macro_f1"],
                    "tuned_test_macro_recall": result["tuned_test_macro_recall"],
                    "tuned_test_macro_specificity": result["tuned_test_macro_specificity"],
                }
            )

        if results:
            for row_type in ["mean", "std"]:
                aggregate_row = {"row_type": row_type}
                for metric_key in SUMMARY_METRIC_KEYS:
                    metric_mean, metric_std = metric_mean_std(results, metric_key)
                    aggregate_row[metric_key] = (
                        metric_mean if row_type == "mean" else metric_std
                    )
                writer.writerow(aggregate_row)


def run_plot_generation(run_dir: Path) -> bool:
    if not PLOT_SCRIPT_PATH.is_file():
        raise FileNotFoundError(f"Plot script not found: {PLOT_SCRIPT_PATH}")

    command = [
        sys.executable,
        str(PLOT_SCRIPT_PATH),
        "--run-dir",
        str(run_dir),
        *PLOT_COMMON_FLAGS,
    ]

    print()
    print("-" * 80)
    print(f"Generating plots for: {run_dir.name}")
    print("Command:")
    print(" ".join(command))
    print("-" * 80)

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        print(
            f"Warning: plot generation failed for {run_dir.name} "
            f"with return code {completed.returncode}"
        )
        return False

    return True


def run_threshold_tuning(run_dir: Path) -> bool:
    if not THRESHOLD_SCRIPT_PATH.is_file():
        raise FileNotFoundError(f"Threshold script not found: {THRESHOLD_SCRIPT_PATH}")

    command = [
        sys.executable,
        str(THRESHOLD_SCRIPT_PATH),
        "--run-dir",
        str(run_dir),
        *THRESHOLD_COMMON_FLAGS,
    ]

    print()
    print("-" * 80)
    print(f"Tuning thresholds for: {run_dir.name}")
    print("Command:")
    print(" ".join(command))
    print("-" * 80)

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        print(
            f"Warning: threshold tuning failed for {run_dir.name} "
            f"with return code {completed.returncode}"
        )
        return False

    return True


def run_one_config(config_rel_path: str) -> dict:
    config_path = resolve_project_path(config_rel_path)
    cfg = load_yaml(config_path)

    run_name = config_path.stem
    output_dir = resolve_project_path(str(cfg["output_dir"]))
    run_dir = output_dir / run_name

    command = [
        sys.executable,
        "-u",
        "src/train.py",
        "--config",
        str(config_path),
        *build_train_flags(),
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

    thresholds_created = False
    if RUN_THRESHOLD_TUNING:
        thresholds_created = run_threshold_tuning(run_dir)

    plots_created = False
    if RUN_PLOTS:
        plots_created = run_plot_generation(run_dir)

    run_summary_path = run_dir / "run_summary.json"
    if not run_summary_path.is_file():
        raise FileNotFoundError(f"Missing file: {run_summary_path}")

    run_summary = load_json(run_summary_path)

    return build_result_row(
        config_path=config_rel_path,
        cfg=cfg,
        run_dir=run_dir,
        run_summary=run_summary,
        thresholds_created=thresholds_created,
        plots_created=plots_created,
    )


def print_final_summary(results: list[dict]) -> None:
    print()
    print("=== Finished ===")
    for result in results:
        print(
            f"{result['run_name']} | "
            f"train_seed={result['train_seed']} | "
            f"split_seed={result['split_seed']} | "
            f"best_epoch={result['best_epoch']} | "
            f"stop_reason={result['stop_reason']} | "
            f"thresholds_created={result['thresholds_created']} | "
            f"plots_created={result['plots_created']} | "
            f"val_auprc={format_metric(result['val_macro_auprc'])} | "
            f"test_auprc={format_metric(result['test_macro_auprc'])} | "
            f"tuned_test_f1={format_metric(result['tuned_test_macro_f1'])} | "
            f"tuned_test_recall={format_metric(result['tuned_test_macro_recall'])} | "
            f"tuned_test_specificity={format_metric(result['tuned_test_macro_specificity'])}"
        )

    if results:
        val_auprc_mean, val_auprc_std = metric_mean_std(results, "val_macro_auprc")
        test_auprc_mean, test_auprc_std = metric_mean_std(results, "test_macro_auprc")
        tuned_test_f1_mean, tuned_test_f1_std = metric_mean_std(results, "tuned_test_macro_f1")
        tuned_test_recall_mean, tuned_test_recall_std = metric_mean_std(results, "tuned_test_macro_recall")
        tuned_test_specificity_mean, tuned_test_specificity_std = metric_mean_std(results, "tuned_test_macro_specificity")

        print()
        print("=== Aggregate Summary ===")
        print(f"Val macro AUPRC: {format_metric(val_auprc_mean)} +/- {format_metric(val_auprc_std)}")
        print(f"Test macro AUPRC: {format_metric(test_auprc_mean)} +/- {format_metric(test_auprc_std)}")
        print(f"Tuned test macro F1: {format_metric(tuned_test_f1_mean)} +/- {format_metric(tuned_test_f1_std)}")
        print(f"Tuned test macro Recall: {format_metric(tuned_test_recall_mean)} +/- {format_metric(tuned_test_recall_std)}")
        print(f"Tuned test macro Specificity: {format_metric(tuned_test_specificity_mean)} +/- {format_metric(tuned_test_specificity_std)}")
        print(f"Saved summary CSV to: {SUMMARY_CSV_PATH}")


def main() -> None:
    results: list[dict] = []

    try:
        for config_rel_path in CONFIG_PATHS:
            result = run_one_config(config_rel_path)
            results.append(result)
            save_summary_csv(results, SUMMARY_CSV_PATH)
    except Exception:
        if results:
            save_summary_csv(results, SUMMARY_CSV_PATH)
        raise

    print_final_summary(results)


if __name__ == "__main__":
    main()