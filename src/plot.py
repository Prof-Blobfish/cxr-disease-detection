from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


VAL_COLOR = "#1f77b4"
TEST_COLOR = "#ff7f0e"
BAR_COLOR = "#2a6f97"



def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path

def load_json(path: Path) -> dict:
    with require_file(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    
    return data


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_file(path))


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output_df = df.copy()
    for column in columns:
        if column in output_df.columns:
            output_df[column] = pd.to_numeric(output_df[column], errors="coerce")
    return output_df


def build_class_map(class_order_payload: dict) -> dict[str, dict[str, str]]:
    classes = class_order_payload.get("classes")
    if not isinstance(classes, list):
        raise ValueError("class_order.json must contain a 'classes' list")
    
    class_map: dict[str, dict[str, str]] = {}
    for row in classes:
        if not isinstance(row, dict):
            continue

        class_name = row.get("class_name")
        pred_column = row.get("pred_column")
        target_column = row.get("target_column")

        if not class_name or not pred_column or not target_column:
            continue

        class_map[str(class_name)] = {
            "pred_column": str(pred_column),
            "target_column": str(target_column),
        }

    if not class_map:
        raise ValueError("No valid class mappings found in class_order.json")
    
    return class_map


def select_pr_classes(
        val_per_class_df: pd.DataFrame,
        class_map: dict[str, dict[str, str]],
        requested_classes: list[str] | None,
        top_k: int,
) -> list[str]:
    available_classes = set(class_map)

    if requested_classes:
        missing = [class_name for class_name in requested_classes if class_name not in available_classes]
        if missing:
            raise ValueError(f"Unknown class names in --pr-classes: {missing}.")
        return requested_classes
    
    ranked_df = val_per_class_df.copy()
    ranked_df = coerce_numeric_columns(
        ranked_df, 
        ["positive_count", "positive_prevalence", "auprc"]
    )
    ranked_df = ranked_df[ranked_df["positive_count"] > 0]
    ranked_df = ranked_df.sort_values(
        ["positive_prevalence", "auprc"],
        ascending=[False, False],
    )

    selected_classes = [
        class_name
        for class_name in ranked_df["class_name"].astype(str).tolist()
        if class_name in available_classes
    ][:top_k]

    if not selected_classes:
        raise ValueError("Could not select any valid classes for PR curves.")

    return selected_classes


def save_history_plot(history_df: pd.DataFrame, best_epoch: int, output_path: Path) -> None:
    history_df = coerce_numeric_columns(
        history_df,
        ["epoch", "train_loss", "val_loss", "val_macro_auroc", "val_macro_auprc"],
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    panels = [
        ("train_loss", "Train Loss", "#2a9d8f", axes[0, 0]),
        ("val_loss", "Validation Loss", "#e76f51", axes[0, 1]),
        ("val_macro_auroc", "Validation Macro AUROC", "#264653", axes[1, 0]),
        ("val_macro_auprc", "Validation Macro AUPRC", "#f4a261", axes[1, 1]),
    ]

    for index, (column, title, color, axis) in enumerate(panels):
        axis.plot(
            history_df["epoch"],
            history_df[column],
            marker="o",
            linewidth=2,
            color=color,
        )
        axis.axvline(
            best_epoch,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="Best epoch" if index == 0 else None,
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.25)

        if column in {"val_macro_auroc", "val_macro_auprc"}:
            axis.set_ylim(0.0, 1.0)

    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Score")
    axes[1, 1].set_ylabel("Score")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle("Training History", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_per_class_auprc_bar(
    per_class_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    plot_df = per_class_df.copy()
    plot_df = coerce_numeric_columns(plot_df, ["auprc", "positive_prevalence"])
    plot_df = plot_df.dropna(subset=["auprc"]).sort_values("auprc", ascending=True)

    if plot_df.empty:
        raise ValueError(f"No valid AUPRC values available for plot: {title}")

    figure_height = max(6, 0.45 * len(plot_df))
    fig, axis = plt.subplots(figsize=(10, figure_height))

    axis.barh(plot_df["class_name"], plot_df["auprc"], color=BAR_COLOR, alpha=0.9)

    for index, (_, row) in enumerate(plot_df.iterrows()):
        axis.text(
            float(row["auprc"]) + 0.01,
            index,
            f"{float(row['auprc']):.3f}",
            va="center",
            fontsize=9,
        )

    axis.set_xlim(0.0, 1.05)
    axis.set_xlabel("AUPRC")
    axis.set_title(title)
    axis.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def annotate_scatter(axis: plt.Axes, plot_df: pd.DataFrame) -> None:
    for _, row in plot_df.iterrows():
        axis.annotate(
            str(row["class_name"]),
            (float(row["positive_prevalence"]), float(row["auprc"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )


def save_prevalence_vs_auprc_plot(
    val_per_class_df: pd.DataFrame,
    test_per_class_df: pd.DataFrame,
    output_path: Path,
) -> None:
    val_df = coerce_numeric_columns(val_per_class_df, ["positive_prevalence", "auprc"])
    test_df = coerce_numeric_columns(test_per_class_df, ["positive_prevalence", "auprc"])

    val_df = val_df.dropna(subset=["positive_prevalence", "auprc"]).copy()
    test_df = test_df.dropna(subset=["positive_prevalence", "auprc"]).copy()

    if val_df.empty or test_df.empty:
        raise ValueError("Need valid prevalence and AUPRC values for both val and test scatter plots.")

    max_prevalence = max(
        float(val_df["positive_prevalence"].max()),
        float(test_df["positive_prevalence"].max()),
    )
    x_max = min(1.0, max_prevalence * 1.10 if max_prevalence > 0 else 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    panels = [
        (axes[0], val_df, "Validation: Prevalence vs AUPRC", VAL_COLOR),
        (axes[1], test_df, "Test: Prevalence vs AUPRC", TEST_COLOR),
    ]

    for axis, plot_df, title, color in panels:
        axis.scatter(
            plot_df["positive_prevalence"],
            plot_df["auprc"],
            s=70,
            color=color,
            alpha=0.85,
        )
        annotate_scatter(axis, plot_df)
        axis.set_title(title)
        axis.set_xlabel("Positive Prevalence")
        axis.set_xlim(0.0, x_max)
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, alpha=0.25)

    axes[0].set_ylabel("AUPRC")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_val_pr_curves_plot(
    val_predictions_df: pd.DataFrame,
    val_per_class_df: pd.DataFrame,
    class_map: dict[str, dict[str, str]],
    selected_classes: list[str],
    output_path: Path,
) -> None:
    prevalence_lookup = (
        coerce_numeric_columns(val_per_class_df, ["positive_prevalence"])
        .set_index("class_name")["positive_prevalence"]
        .to_dict()
    )

    fig, axis = plt.subplots(figsize=(10, 8))
    plotted_any = False

    for class_name in selected_classes:
        mapping = class_map[class_name]
        pred_column = mapping["pred_column"]
        target_column = mapping["target_column"]

        if pred_column not in val_predictions_df.columns:
            raise KeyError(f"Missing prediction column '{pred_column}' in best_val_predictions.csv")
        if target_column not in val_predictions_df.columns:
            raise KeyError(f"Missing target column '{target_column}' in best_val_predictions.csv")

        y_true = pd.to_numeric(val_predictions_df[target_column], errors="coerce").fillna(0).astype(int).to_numpy()
        y_score = pd.to_numeric(val_predictions_df[pred_column], errors="coerce").fillna(0.0).astype(float).to_numpy()

        if y_true.sum() == 0:
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        average_precision = average_precision_score(y_true, y_score)
        prevalence = float(prevalence_lookup.get(class_name, y_true.mean()))

        axis.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{class_name} (AP={average_precision:.3f}, prev={prevalence:.3f})",
        )
        plotted_any = True

    if not plotted_any:
        raise ValueError("No valid PR curves could be generated for the selected classes.")

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-Recall Curves (Best Validation Checkpoint)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def build_threshold_delta_frame(
    default_df: pd.DataFrame,
    tuned_df: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = ["f1", "recall", "specificity"]

    default_df = coerce_numeric_columns(
        default_df,
        ["class_index", *metric_columns],
    ).copy()
    tuned_df = coerce_numeric_columns(
        tuned_df,
        ["class_index", *metric_columns],
    ).copy()

    merged = default_df.merge(
        tuned_df,
        on=["class_name", "class_index"],
        how="inner",
        suffixes=("_default", "_tuned"),
    )

    if merged.empty:
        raise ValueError("No overlapping classes found between default and tuned threshold metrics.")

    for metric_name in metric_columns:
        merged[f"delta_{metric_name}"] = (
            merged[f"{metric_name}_tuned"] - merged[f"{metric_name}_default"]
        )

    delta_columns = [f"delta_{metric_name}" for metric_name in metric_columns]
    merged[delta_columns] = merged[delta_columns].fillna(0.0)

    return merged.sort_values("delta_f1", ascending=False)


def save_threshold_macro_comparison_plot(
    default_metrics: dict,
    tuned_metrics: dict,
    split: str,
    output_path: Path,
) -> None:
    metric_columns = ["macro_f1", "macro_recall", "macro_specificity"]
    metric_labels = ["F1", "Recall", "Specificity"]

    default_values = [float(default_metrics[column]) for column in metric_columns]
    tuned_values = [float(tuned_metrics[column]) for column in metric_columns]

    x = np.arange(len(metric_labels))
    width = 0.34

    fig, axis = plt.subplots(figsize=(9, 5.5))

    default_bars = axis.bar(
        x - width / 2,
        default_values,
        width=width,
        label="Default 0.5",
        color="#8d99ae",
    )
    tuned_bars = axis.bar(
        x + width / 2,
        tuned_values,
        width=width,
        label="Tuned",
        color="#2a6f97",
    )

    for bars in (default_bars, tuned_bars):
        for bar in bars:
            value = float(bar.get_height())
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axis.set_xticks(x)
    axis.set_xticklabels(metric_labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title(f"Default 0.5 vs Tuned Threshold Metrics ({split.title()})")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_threshold_delta_heatmap(
    default_df: pd.DataFrame,
    tuned_df: pd.DataFrame,
    split: str,
    output_path: Path,
) -> None:
    delta_df = build_threshold_delta_frame(default_df, tuned_df)

    value_columns = ["delta_f1", "delta_recall", "delta_specificity"]
    metric_labels = ["F1", "Recall", "Specificity"]
    heatmap_values = delta_df[value_columns].to_numpy(dtype=float)

    figure_height = max(6, 0.45 * len(delta_df))
    fig, axis = plt.subplots(figsize=(8, figure_height))

    max_abs = float(np.nanmax(np.abs(heatmap_values))) if heatmap_values.size else 1.0
    max_abs = max(max_abs, 1e-6)

    image = axis.imshow(
        heatmap_values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
    )

    axis.set_xticks(np.arange(len(metric_labels)))
    axis.set_xticklabels(metric_labels)
    axis.set_yticks(np.arange(len(delta_df)))
    axis.set_yticklabels(delta_df["class_name"].astype(str).tolist())
    axis.set_title(f"Tuned - Default 0.5 Metric Delta ({split.title()})")

    for row_idx in range(heatmap_values.shape[0]):
        for col_idx in range(heatmap_values.shape[1]):
            value = float(heatmap_values[row_idx, col_idx])
            axis.text(
                col_idx,
                row_idx,
                f"{value:+.03f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Metric delta")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def select_split_rows(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if "split" not in df.columns:
        raise KeyError("Expected a 'split' column in merged per-class metrics.")

    filtered_df = df[df["split"].astype(str) == split].copy()
    if filtered_df.empty:
        raise ValueError(f"No rows found for split '{split}'.")

    return filtered_df


def select_threshold_rows(
    df: pd.DataFrame,
    *,
    split: str,
    threshold_scheme: str,
) -> pd.DataFrame:
    if "threshold_scheme" not in df.columns:
        raise KeyError("Expected a 'threshold_scheme' column in per_class_threshold_metrics.csv.")

    filtered_df = select_split_rows(df, split)
    filtered_df = filtered_df[
        filtered_df["threshold_scheme"].astype(str) == threshold_scheme
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No rows found for split '{split}' and threshold_scheme '{threshold_scheme}'."
        )

    return filtered_df

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a single training run directory under outputs/runs.",
    )
    parser.add_argument(
        "--pr-classes",
        nargs="*",
        default=None,
        help="Optional explicit class names for the validation PR-curve plot.",
    )
    parser.add_argument(
        "--top-k-pr-classes",
        type=int,
        default=5,
        help="Number of most prevalent validation classes to use when --pr-classes is omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory not found: {run_dir}")

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    history_df = load_csv(run_dir / "history.csv")
    run_summary = load_json(run_dir / "run_summary.json")
    per_class_ranking_df = load_csv(run_dir / "per_class_ranking_metrics.csv")
    val_per_class_df = select_split_rows(per_class_ranking_df, "val")
    test_per_class_df = select_split_rows(per_class_ranking_df, "test")
    val_predictions_df = load_csv(run_dir / "best_val_predictions.csv")
    class_order_payload = load_json(run_dir / "class_order.json")

    threshold_summary = run_summary.get("threshold_metrics")
    threshold_per_class_path = run_dir / "per_class_threshold_metrics.csv"

    best_epoch = int(run_summary["best_epoch"])
    class_map = build_class_map(class_order_payload)
    selected_classes = select_pr_classes(
        val_per_class_df=val_per_class_df,
        class_map=class_map,
        requested_classes=args.pr_classes,
        top_k=args.top_k_pr_classes,
    )

    output_paths = {
        "history": plot_dir / "history.png",
        "per_class_val": plot_dir / "per_class_auprc_val.png",
        "per_class_test": plot_dir / "per_class_auprc_test.png",
        "prevalence_scatter": plot_dir / "prevalence_vs_auprc.png",
        "pr_curves_val": plot_dir / "pr_curves_selected_classes_val.png",
    }

    save_history_plot(
        history_df=history_df,
        best_epoch=best_epoch,
        output_path=output_paths["history"],
    )

    save_per_class_auprc_bar(
        per_class_df=val_per_class_df,
        title="Per-Class AUPRC (Best Validation Checkpoint)",
        output_path=output_paths["per_class_val"],
    )

    save_per_class_auprc_bar(
        per_class_df=test_per_class_df,
        title="Per-Class AUPRC (Test Set)",
        output_path=output_paths["per_class_test"],
    )

    save_prevalence_vs_auprc_plot(
        val_per_class_df=val_per_class_df,
        test_per_class_df=test_per_class_df,
        output_path=output_paths["prevalence_scatter"],
    )

    save_val_pr_curves_plot(
        val_predictions_df=val_predictions_df,
        val_per_class_df=val_per_class_df,
        class_map=class_map,
        selected_classes=selected_classes,
        output_path=output_paths["pr_curves_val"],
    )

    threshold_comparison_ready = (
        isinstance(threshold_summary, dict)
        and threshold_per_class_path.is_file()
    )

    if threshold_comparison_ready:
        threshold_per_class_df = load_csv(threshold_per_class_path)

        default_val_metrics = threshold_summary["default_0.5"]["val"]
        default_test_metrics = threshold_summary["default_0.5"]["test"]
        tuned_val_metrics = threshold_summary["tuned"]["val"]
        tuned_test_metrics = threshold_summary["tuned"]["test"]

        default_val_per_class_df = select_threshold_rows(
            threshold_per_class_df,
            split="val",
            threshold_scheme="default_0.5",
        )
        default_test_per_class_df = select_threshold_rows(
            threshold_per_class_df,
            split="test",
            threshold_scheme="default_0.5",
        )
        tuned_val_per_class_df = select_threshold_rows(
            threshold_per_class_df,
            split="val",
            threshold_scheme="tuned",
        )
        tuned_test_per_class_df = select_threshold_rows(
            threshold_per_class_df,
            split="test",
            threshold_scheme="tuned",
        )

        threshold_output_paths = {
            "threshold_macro_val": plot_dir / "threshold_macro_comparison_val.png",
            "threshold_macro_test": plot_dir / "threshold_macro_comparison_test.png",
            "threshold_delta_val": plot_dir / "threshold_delta_heatmap_val.png",
            "threshold_delta_test": plot_dir / "threshold_delta_heatmap_test.png",
        }

        save_threshold_macro_comparison_plot(
            default_metrics=default_val_metrics,
            tuned_metrics=tuned_val_metrics,
            split="val",
            output_path=threshold_output_paths["threshold_macro_val"],
        )
        save_threshold_macro_comparison_plot(
            default_metrics=default_test_metrics,
            tuned_metrics=tuned_test_metrics,
            split="test",
            output_path=threshold_output_paths["threshold_macro_test"],
        )

        save_threshold_delta_heatmap(
            default_df=default_val_per_class_df,
            tuned_df=tuned_val_per_class_df,
            split="val",
            output_path=threshold_output_paths["threshold_delta_val"],
        )
        save_threshold_delta_heatmap(
            default_df=default_test_per_class_df,
            tuned_df=tuned_test_per_class_df,
            split="test",
            output_path=threshold_output_paths["threshold_delta_test"],
        )

        output_paths.update(threshold_output_paths)

    print("Saved plots:")
    for output_path in output_paths.values():
        print(f"- {output_path}")


if __name__ == "__main__":
    main()