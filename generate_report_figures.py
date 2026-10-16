import argparse
import csv
import math
import os
import tempfile
from itertools import product
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_DIR = PROJECT_ROOT / "CSVs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Figures" / "report"

# Matplotlib otherwise tries to use the user's read-only configuration directory
# in restricted execution environments.
_MPL_CACHE = Path(tempfile.gettempdir()) / "prj_line_detection_matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle


MODELS = ("2a", "2b", "2c", "2d")
MODES = ("strict", "moderate", "loose")
SPLITS = ("york", "wireframe")
SYNTHETIC_FAMILIES = (
    "single_luma",
    "single_chroma",
    "single_noisy",
    "parallel_luma",
    "parallel_chroma",
    "rectangle_luma",
    "rectangle_chroma",
    "triangle",
    "crossing",
    "rotated_grid",
)
CHROMATIC_FAMILIES = {
    "single_chroma",
    "parallel_chroma",
    "rectangle_chroma",
}

MODEL_LABELS = {
    "2a": "2a  Structure grid",
    "2b": "2b  Color-EMD grid",
    "2c": "2c  Structure refinement",
    "2d": "2d  Color-EMD refinement",
}
SHORT_MODEL_LABELS = {model: model for model in MODELS}
SPLIT_LABELS = {"york": "YorkUrban", "wireframe": "Wireframe"}
MODE_LABELS = {"strict": "Strict", "moderate": "Moderate", "loose": "Loose"}
MODE_ABBREVIATIONS = {"strict": "S", "moderate": "M", "loose": "L"}

# Okabe-Ito-derived colors remain distinguishable in common color-vision deficits.
MODEL_COLORS = {
    "2a": "#0072B2",
    "2b": "#E69F00",
    "2c": "#009E73",
    "2d": "#CC79A7",
}
MODE_COLORS = {
    "strict": "#4C78A8",
    "moderate": "#F58518",
    "loose": "#54A24B",
}
MODE_MARKERS = {"strict": "o", "moderate": "s", "loose": "^"}
METRIC_COLORS = {
    "precision": "#0072B2",
    "recall": "#E69F00",
    "f1": "#009E73",
}
REFERENCE_COLOR = "#0072B2"
COMPARISON_COLOR = "#D55E00"
PAIRED_F1_COMPARISONS = (
    ("2b - 2a", "2b", "2a", "#E69F00"),
    ("2c - 2a", "2c", "2a", "#009E73"),
    ("2d - 2b", "2d", "2b", "#CC79A7"),
    ("2d - 2c", "2d", "2c", "#7A5195"),
)

MAIN_AVERAGE_COLUMNS = {
    "split",
    "model",
    "mode",
    "image_count",
    "mean_precision",
    "mean_recall",
    "mean_f1",
    "mean_location_error_px",
    "mean_orientation_error_deg",
    "mean_runtime_seconds",
    "mean_raw_candidates",
    "mean_retained_candidates",
    "mean_detected_segments",
    "mean_gt_segments",
}
MAIN_RESULT_COLUMNS = {
    "image",
    "split",
    "model",
    "mode",
    "status",
    "precision",
    "recall",
    "f1",
    "location_error_px",
    "orientation_error_deg",
    "runtime_seconds",
    "raw_candidates",
    "retained_candidates",
    "segments",
    "gt_segments",
    "error",
}
K_COLUMNS = {
    "split",
    "cohort",
    "model",
    "mode",
    "image_count",
    "reference_k",
    "comparison_k",
    "k8_mean_f1",
    "k4_mean_f1",
    "k8_mean_runtime_seconds",
    "k4_mean_runtime_seconds",
}
PADDING_COLUMNS = {
    "split",
    "cohort",
    "model",
    "mode",
    "image_count",
    "default_padding_px",
    "comparison_padding_px",
    "default_mean_precision",
    "zero_mean_precision",
    "default_mean_recall",
    "zero_mean_recall",
    "default_mean_f1",
    "zero_mean_f1",
}
SYNTHETIC_FAMILY_COLUMNS = {
    "dataset",
    "split",
    "family",
    "model",
    "evaluator",
    "mode",
    "image_count",
    "valid_gt_image_count",
    "valid_gt_mean_f1",
    "source_config_hash",
}


def configure_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#4A4A4A",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#202020",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path, required_columns):
    path = Path(path)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = required_columns - columns

        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"input table is empty: {path}")

    return rows


def as_float(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {field} value: {row.get(field)!r}") from error

    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} value: {value}")

    return value


def as_int(row, field):
    value = as_float(row, field)

    if not value.is_integer():
        raise ValueError(f"expected integer {field}, found {value}")

    return int(value)


def unique_index(rows, fields, source_name):
    index = {}

    for row in rows:
        key = tuple(row[field] for field in fields)

        if key in index:
            raise ValueError(f"duplicate {source_name} key: {key}")

        index[key] = row

    return index


def validate_sources(main_averages, main_results, k_rows, padding_rows):
    main_index = unique_index(main_averages, ("split", "model", "mode"), "main average")
    expected_main_keys = set(product(SPLITS, MODELS, MODES))
    observed_main_keys = {
        key for key in main_index if key[0] in SPLITS
    }

    if observed_main_keys != expected_main_keys:
        raise ValueError("main average table does not contain every requested combination")

    result_index = unique_index(
        main_results, ("image", "model", "mode"), "main result"
    )

    if len(result_index) != len(main_results):
        raise ValueError("main results contain duplicate jobs")

    failed = [
        row
        for row in main_results
        if row["status"] != "ok" or row["error"].strip()
    ]

    if failed:
        raise ValueError(f"main result table contains {len(failed)} failed rows")

    for split in SPLITS:
        split_rows = [row for row in main_results if row["split"] == split]
        image_sets = []

        for model, mode in product(MODELS, MODES):
            group = [
                row
                for row in split_rows
                if row["model"] == model and row["mode"] == mode
            ]
            image_ids = {row["image"] for row in group}

            if len(image_ids) != 51:
                raise ValueError(
                    f"expected 51 {split} images for {(model, mode)}, "
                    f"found {len(image_ids)}"
                )

            image_sets.append(image_ids)

            average_row = main_index[(split, model, mode)]

            if as_int(average_row, "image_count") != 51:
                raise ValueError(f"incorrect image count for {(split, model, mode)}")

            for output_field, input_field in (
                ("mean_precision", "precision"),
                ("mean_recall", "recall"),
                ("mean_f1", "f1"),
                ("mean_location_error_px", "location_error_px"),
                ("mean_orientation_error_deg", "orientation_error_deg"),
                ("mean_runtime_seconds", "runtime_seconds"),
                ("mean_raw_candidates", "raw_candidates"),
                ("mean_retained_candidates", "retained_candidates"),
                ("mean_detected_segments", "segments"),
                ("mean_gt_segments", "gt_segments"),
            ):
                measured = float(np.mean([as_float(row, input_field) for row in group]))
                expected = as_float(average_row, output_field)

                if not math.isclose(measured, expected, abs_tol=1e-6):
                    raise ValueError(
                        f"{output_field} does not reproduce for {(split, model, mode)}"
                    )

        if any(image_ids != image_sets[0] for image_ids in image_sets[1:]):
            raise ValueError(f"models do not share the same {split} image cohort")

    k_index = unique_index(k_rows, ("split", "model", "mode"), "k ablation")
    expected_k_keys = set(product(SPLITS, ("2b", "2d"), MODES))
    observed_k_keys = {key for key in k_index if key[0] in SPLITS}

    if observed_k_keys != expected_k_keys:
        raise ValueError("k ablation table does not contain every requested combination")

    for key in expected_k_keys:
        row = k_index[key]

        if as_int(row, "image_count") != 5:
            raise ValueError(f"k ablation group is not a five-image cohort: {key}")

        if row["reference_k"] != "8" or row["comparison_k"] != "4":
            raise ValueError(f"unexpected k comparison for {key}")

    padding_index = unique_index(
        padding_rows, ("split", "model", "mode"), "padding ablation"
    )
    expected_padding_keys = set(product(SPLITS, ("2b", "2c", "2d"), MODES))
    observed_padding_keys = {key for key in padding_index if key[0] in SPLITS}

    if observed_padding_keys != expected_padding_keys:
        raise ValueError(
            "padding ablation table does not contain every requested combination"
        )

    expected_default_padding = {"2b": 8.0, "2c": 4.0, "2d": 8.0}

    for key in expected_padding_keys:
        row = padding_index[key]

        if as_int(row, "image_count") != 5:
            raise ValueError(f"padding group is not a five-image cohort: {key}")

        if as_float(row, "default_padding_px") != expected_default_padding[key[1]]:
            raise ValueError(f"unexpected default padding for {key}")

        if as_float(row, "comparison_padding_px") != 0.0:
            raise ValueError(f"comparison padding is not zero for {key}")

    return main_index, k_index, padding_index


def validate_synthetic_family_rows(rows):
    index = unique_index(
        rows,
        ("family", "model", "mode"),
        "synthetic-family summary",
    )
    expected_keys = set(product(SYNTHETIC_FAMILIES, MODELS, MODES))

    if set(index) != expected_keys:
        missing = sorted(expected_keys - set(index))
        extra = sorted(set(index) - expected_keys)
        raise ValueError(
            "synthetic-family table does not contain the expected matrix; "
            f"missing={missing}, extra={extra}"
        )

    config_hashes = {row["source_config_hash"] for row in rows}

    if len(config_hashes) != 1:
        raise ValueError("synthetic-family rows mix multiple configurations")

    for key, row in index.items():
        if row["dataset"] != "synthetic_100" or row["split"] != "synthetic":
            raise ValueError(f"unexpected synthetic dataset identity for {key}")

        if row["evaluator"] != "matlab":
            raise ValueError(f"synthetic row was not evaluated by MATLAB: {key}")

        if as_int(row, "image_count") != 5:
            raise ValueError(f"synthetic family does not contain five images: {key}")

        valid_gt_count = as_int(row, "valid_gt_image_count")

        if not 0 <= valid_gt_count <= 5:
            raise ValueError(f"invalid synthetic GT image count for {key}")

        if valid_gt_count == 0:
            if row["valid_gt_mean_f1"].strip():
                raise ValueError(f"invalid family has a valid-GT F1 value: {key}")
        else:
            f1_value = as_float(row, "valid_gt_mean_f1")

            if not 0.0 <= f1_value <= 1.0:
                raise ValueError(f"synthetic F1 is outside [0, 1] for {key}")

    for family in SYNTHETIC_FAMILIES:
        valid_counts = {
            as_int(index[(family, model, mode)], "valid_gt_image_count")
            for model, mode in product(MODELS, MODES)
        }

        if len(valid_counts) != 1:
            raise ValueError(f"synthetic GT validity differs within {family}")

    return index


def save_figure(fig, output_dir, stem, title):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    metadata = {
        "Title": title,
        "Author": "Line Segment Detection Project",
        "Subject": "Report figure generated from benchmark CSV files",
    }

    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.12,
        metadata=metadata,
    )
    plt.close(fig)
    return png_path, pdf_path


def annotate_bars(ax, bars, precision=3, fontsize=7.5):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.{precision}f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#333333",
        )


def add_source_note(fig, text):
    fig.text(0.5, -0.015, text, ha="center", va="top", fontsize=8, color="#555555")


def draw_pipeline(output_dir):
    fig, ax = plt.subplots(figsize=(17, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.015, 0.34, 0.105, 0.32, "Input", "Color image", "#E8EAED"),
        (
            0.155,
            0.25,
            0.185,
            0.50,
            "Tasks 1 and 2",
            "Candidate construction\n\n2a/2c: structure tensor\n2b/2d: Lab color EMD",
            "#DCEAF7",
        ),
        (
            0.375,
            0.34,
            0.115,
            0.32,
            "Candidates",
            r"$(r_j, n_j, s_j)$",
            "#FFF1CC",
        ),
        (
            0.525,
            0.30,
            0.120,
            0.40,
            "Task 3",
            "Pairwise line\nsimilarity  " + r"$p_{j,k}$",
            "#DDF2E8",
        ),
        (
            0.680,
            0.25,
            0.135,
            0.50,
            "Task 4",
            "Weighted complete graph\n\nClique partition:\nGAEC + KLj",
            "#FCE5CD",
        ),
        (
            0.850,
            0.30,
            0.105,
            0.40,
            "Segments",
            "Cluster averaging\nand endpoints",
            "#D9EAD3",
        ),
    ]

    for x, y, width, height, heading, body, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.1,
            edgecolor="#4A4A4A",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2.0,
            y + height * 0.72,
            heading,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            x + width / 2.0,
            y + height * 0.39,
            body,
            ha="center",
            va="center",
            fontsize=9.5,
            linespacing=1.25,
        )

    evaluation = FancyBboxPatch(
        (0.835, 0.02),
        0.135,
        0.16,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.1,
        edgecolor="#4A4A4A",
        facecolor="#EADCF2",
    )
    ax.add_patch(evaluation)
    ax.text(
        0.9025,
        0.10,
        "[3] evaluation\nStrict / moderate / loose",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
    )

    for start_x, end_x in (
        (0.120, 0.155),
        (0.340, 0.375),
        (0.490, 0.525),
        (0.645, 0.680),
        (0.815, 0.850),
    ):
        ax.add_patch(
            FancyArrowPatch(
                (start_x + 0.004, 0.50),
                (end_x - 0.004, 0.50),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color="#444444",
            )
        )

    ax.add_patch(
        FancyArrowPatch(
            (0.9025, 0.30),
            (0.9025, 0.18),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#444444",
        )
    )
    ax.text(
        0.5,
        0.91,
        "Line-segment detection and evaluation pipeline",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.84,
        "Each of Algorithms 2a-2d produces a separate candidate set and follows the same downstream pipeline.",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )

    return save_figure(
        fig,
        output_dir,
        "01_pipeline",
        "Line-segment detection and evaluation pipeline",
    )


def draw_main_f1(main_index, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)
    x = np.arange(len(MODELS))
    width = 0.23
    offsets = (-width, 0.0, width)
    maximum = max(
        as_float(main_index[(split, model, mode)], "mean_f1")
        for split, model, mode in product(SPLITS, MODELS, MODES)
    )

    for ax, split in zip(axes, SPLITS):
        for mode, offset in zip(MODES, offsets):
            values = [
                as_float(main_index[(split, model, mode)], "mean_f1")
                for model in MODELS
            ]
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=MODE_LABELS[mode],
                color=MODE_COLORS[mode],
                edgecolor="white",
                linewidth=0.7,
            )
            annotate_bars(ax, bars)

        ax.set_title(f"{SPLIT_LABELS[split]} (n=51)")
        ax.set_xticks(x, [SHORT_MODEL_LABELS[model] for model in MODELS])
        ax.set_xlabel("Candidate-generation algorithm")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Mean F1")
    axes[0].set_ylim(0.0, maximum * 1.23)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
    )
    fig.suptitle(
        "Main F1 comparison by evaluation tolerance",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    add_source_note(
        fig,
        "Macro mean over the frozen 51-image split; official [3] MATLAB evaluation.",
    )
    fig.subplots_adjust(top=0.76, bottom=0.16, wspace=0.12)

    return save_figure(
        fig,
        output_dir,
        "02_main_f1_by_dataset",
        "Main F1 comparison by dataset and evaluation tolerance",
    )


def draw_precision_recall(main_index, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), sharex=True, sharey=True)
    maximum_recall = max(
        as_float(main_index[(split, model, mode)], "mean_recall")
        for split, model, mode in product(SPLITS, MODELS, MODES)
    )
    maximum_precision = max(
        as_float(main_index[(split, model, mode)], "mean_precision")
        for split, model, mode in product(SPLITS, MODELS, MODES)
    )

    for ax, split in zip(axes, SPLITS):
        for model in MODELS:
            recalls = [
                as_float(main_index[(split, model, mode)], "mean_recall")
                for mode in MODES
            ]
            precisions = [
                as_float(main_index[(split, model, mode)], "mean_precision")
                for mode in MODES
            ]
            ax.plot(
                recalls,
                precisions,
                color=MODEL_COLORS[model],
                linewidth=1.8,
                alpha=0.85,
            )

            for mode, recall, precision in zip(MODES, recalls, precisions):
                ax.scatter(
                    recall,
                    precision,
                    s=58,
                    marker=MODE_MARKERS[mode],
                    color=MODEL_COLORS[model],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
                ax.annotate(
                    MODE_ABBREVIATIONS[mode],
                    (recall, precision),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=MODEL_COLORS[model],
                )

        ax.set_title(f"{SPLIT_LABELS[split]} (n=51)")
        ax.set_xlabel("Mean recall")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Mean precision")
    axes[0].set_xlim(0.0, maximum_recall * 1.08)
    axes[0].set_ylim(0.0, maximum_precision * 1.08)
    algorithm_handles = [
        Line2D([0], [0], color=MODEL_COLORS[model], lw=2, label=MODEL_LABELS[model])
        for model in MODELS
    ]
    mode_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker=MODE_MARKERS[mode],
            linestyle="None",
            markersize=6,
            label=MODE_LABELS[mode],
        )
        for mode in MODES
    ]
    fig.legend(
        handles=algorithm_handles + mode_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=7,
        fontsize=8.5,
    )
    fig.suptitle(
        "Precision-recall behavior across [3] evaluation tolerances",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    add_source_note(
        fig,
        "S/M/L denote strict, moderate and loose matching tolerances; values are macro image means.",
    )
    fig.subplots_adjust(top=0.75, bottom=0.15, wspace=0.12)

    return save_figure(
        fig,
        output_dir,
        "03_precision_recall_tradeoff",
        "Precision-recall behavior across evaluation tolerances",
    )


def draw_errors(main_index, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), sharex="col")
    x = np.arange(len(MODELS))
    width = 0.23
    offsets = (-width, 0.0, width)
    metric_rows = (
        ("mean_location_error_px", "Mean location error (pixels)"),
        ("mean_orientation_error_deg", "Mean orientation error (degrees)"),
    )

    for column, split in enumerate(SPLITS):
        for row_number, (field, ylabel) in enumerate(metric_rows):
            ax = axes[row_number, column]

            for mode, offset in zip(MODES, offsets):
                values = [
                    as_float(main_index[(split, model, mode)], field)
                    for model in MODELS
                ]
                ax.bar(
                    x + offset,
                    values,
                    width,
                    color=MODE_COLORS[mode],
                    edgecolor="white",
                    linewidth=0.6,
                )

            if row_number == 0:
                ax.set_title(f"{SPLIT_LABELS[split]} (n=51)")

            ax.set_ylabel(ylabel)
            ax.set_xticks(x, [SHORT_MODEL_LABELS[model] for model in MODELS])
            ax.spines[["top", "right"]].set_visible(False)

        axes[1, column].set_xlabel("Candidate-generation algorithm")

    mode_handles = [
        Patch(facecolor=MODE_COLORS[mode], label=MODE_LABELS[mode]) for mode in MODES
    ]
    fig.legend(
        handles=mode_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
    )
    fig.suptitle(
        "Localization and orientation errors of correctly matched pairs",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    add_source_note(
        fig,
        "Mean evaluator-reported per-image errors; official [3] MATLAB evaluation.",
    )
    fig.subplots_adjust(top=0.83, bottom=0.09, hspace=0.22, wspace=0.19)

    return save_figure(
        fig,
        output_dir,
        "04_location_orientation_error",
        "Location and orientation error comparison",
    )


def draw_runtime_tradeoff(main_index, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), sharey=True)
    maximum_f1 = max(
        as_float(main_index[(split, model, mode)], "mean_f1")
        for split, model, mode in product(SPLITS, MODELS, MODES)
    )

    for ax, split in zip(axes, SPLITS):
        for model in MODELS:
            runtime_values = [
                as_float(main_index[(split, model, mode)], "mean_runtime_seconds")
                for mode in MODES
            ]
            f1_values = [
                as_float(main_index[(split, model, mode)], "mean_f1")
                for mode in MODES
            ]
            ax.plot(
                runtime_values,
                f1_values,
                color=MODEL_COLORS[model],
                linewidth=1.1,
                alpha=0.55,
            )

            for mode, runtime, f1_value in zip(MODES, runtime_values, f1_values):
                ax.scatter(
                    runtime,
                    f1_value,
                    s=65,
                    marker=MODE_MARKERS[mode],
                    color=MODEL_COLORS[model],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )

        ax.set_xscale("log")
        ax.set_title(f"{SPLIT_LABELS[split]} (n=51)")
        ax.set_xlabel("Mean pipeline runtime per image (seconds, log scale)")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Mean F1")
    axes[0].set_ylim(0.0, maximum_f1 * 1.08)
    algorithm_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=MODEL_COLORS[model],
            linewidth=1.5,
            label=MODEL_LABELS[model],
        )
        for model in MODELS
    ]
    mode_handles = [
        Line2D(
            [0],
            [0],
            marker=MODE_MARKERS[mode],
            color="#555555",
            linestyle="None",
            label=MODE_LABELS[mode],
        )
        for mode in MODES
    ]
    fig.legend(
        handles=algorithm_handles + mode_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=7,
        fontsize=8.5,
    )
    fig.suptitle(
        "Accuracy-runtime trade-off",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    add_source_note(
        fig,
        "Runtime is independent of matching tolerance; three vertically aligned markers show strict/moderate/loose F1.",
    )
    fig.subplots_adjust(top=0.75, bottom=0.17, wspace=0.12)

    return save_figure(
        fig,
        output_dir,
        "05_f1_runtime_tradeoff",
        "F1 and runtime trade-off",
    )


def draw_per_image_f1(main_results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), sharey=True)
    centers = np.arange(len(MODELS)) * 1.35
    offsets = {"strict": -0.27, "moderate": 0.0, "loose": 0.27}

    all_f1 = [as_float(row, "f1") for row in main_results]
    upper_limit = max(all_f1) * 1.06 if all_f1 else 1.0

    for ax, split in zip(axes, SPLITS):
        for mode in MODES:
            datasets = []
            positions = []

            for center, model in zip(centers, MODELS):
                values = [
                    as_float(row, "f1")
                    for row in main_results
                    if row["split"] == split
                    and row["model"] == model
                    and row["mode"] == mode
                ]

                if len(values) != 51:
                    raise ValueError(f"unexpected boxplot sample size: {(split, model, mode)}")

                datasets.append(values)
                positions.append(center + offsets[mode])

            boxplot = ax.boxplot(
                datasets,
                positions=positions,
                widths=0.22,
                patch_artist=True,
                showfliers=True,
                manage_ticks=False,
                medianprops={"color": "#202020", "linewidth": 1.2},
                boxprops={"linewidth": 0.8, "edgecolor": "#555555"},
                whiskerprops={"linewidth": 0.8, "color": "#555555"},
                capprops={"linewidth": 0.8, "color": "#555555"},
                flierprops={
                    "marker": ".",
                    "markersize": 3,
                    "markerfacecolor": MODE_COLORS[mode],
                    "markeredgecolor": MODE_COLORS[mode],
                    "alpha": 0.55,
                },
            )

            for box in boxplot["boxes"]:
                box.set_facecolor(MODE_COLORS[mode])
                box.set_alpha(0.78)

        ax.set_title(f"{SPLIT_LABELS[split]} (n=51)")
        ax.set_xticks(centers, [SHORT_MODEL_LABELS[model] for model in MODELS])
        ax.set_xlabel("Candidate-generation algorithm")
        ax.set_ylim(0.0, upper_limit)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Per-image F1")
    handles = [
        Patch(facecolor=MODE_COLORS[mode], label=MODE_LABELS[mode], alpha=0.78)
        for mode in MODES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
    )
    fig.suptitle(
        "Per-image F1 variability",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    add_source_note(
        fig,
        "Boxes show the interquartile range, center line the median, whiskers 1.5 x IQR and points the outliers.",
    )
    fig.subplots_adjust(top=0.77, bottom=0.16, wspace=0.12)

    return save_figure(
        fig,
        output_dir,
        "06_per_image_f1_variability",
        "Per-image F1 distributions",
    )


def draw_paired_f1_differences(main_results, output_dir):
    result_index = unique_index(
        main_results,
        ("split", "image", "model", "mode"),
        "paired F1 result",
    )
    image_ids = {
        split: sorted(
            {
                row["image"]
                for row in main_results
                if row["split"] == split
            }
        )
        for split in SPLITS
    }
    paired_values = {}

    for split, mode in product(SPLITS, MODES):
        for label, first_model, second_model, _color in PAIRED_F1_COMPARISONS:
            values = np.asarray(
                [
                    as_float(
                        result_index[(split, image_id, first_model, mode)],
                        "f1",
                    )
                    - as_float(
                        result_index[(split, image_id, second_model, mode)],
                        "f1",
                    )
                    for image_id in image_ids[split]
                ],
                dtype=np.float64,
            )

            if len(values) != 51:
                raise ValueError(
                    f"unexpected paired F1 sample size: {(split, mode, label)}"
                )

            paired_values[(split, mode, label)] = values

    all_values = np.concatenate(list(paired_values.values()))
    value_span = max(float(np.ptp(all_values)), 0.05)
    lower_limit = min(float(np.min(all_values)) - value_span * 0.08, -0.02)
    annotation_y = float(np.max(all_values)) + value_span * 0.08
    upper_limit = annotation_y + value_span * 0.11

    fig, axes = plt.subplots(
        len(SPLITS),
        len(MODES),
        figsize=(15.5, 8.8),
        sharex=True,
        sharey=True,
    )
    positions = np.arange(len(PAIRED_F1_COMPARISONS))
    comparison_labels = [item[0] for item in PAIRED_F1_COMPARISONS]

    for row_number, split in enumerate(SPLITS):
        for column_number, mode in enumerate(MODES):
            ax = axes[row_number, column_number]
            datasets = [
                paired_values[(split, mode, label)]
                for label in comparison_labels
            ]
            boxplot = ax.boxplot(
                datasets,
                positions=positions,
                widths=0.52,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                medianprops={"color": "#202020", "linewidth": 1.3},
                boxprops={"linewidth": 0.9, "edgecolor": "#555555"},
                whiskerprops={"linewidth": 0.9, "color": "#555555"},
                capprops={"linewidth": 0.9, "color": "#555555"},
            )

            for comparison_number, (box, comparison) in enumerate(
                zip(boxplot["boxes"], PAIRED_F1_COMPARISONS)
            ):
                label, _first_model, _second_model, color = comparison
                values = paired_values[(split, mode, label)]
                box.set_facecolor(color)
                box.set_alpha(0.68)

                jitter = np.linspace(-0.15, 0.15, len(values))
                jitter = np.roll(
                    jitter,
                    7 * comparison_number + 11 * row_number + 3 * column_number,
                )
                ax.scatter(
                    comparison_number + jitter,
                    values,
                    s=11,
                    color=color,
                    edgecolor="white",
                    linewidth=0.25,
                    alpha=0.58,
                    zorder=3,
                )

                ties = int(np.count_nonzero(np.isclose(values, 0.0, atol=1e-12)))
                wins = int(np.count_nonzero(values > 1e-12))
                losses = len(values) - wins - ties
                ax.text(
                    comparison_number,
                    annotation_y,
                    f"{wins}/{ties}/{losses}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#333333",
                )

            ax.axhline(0.0, color="#202020", linewidth=1.0, linestyle="--")
            ax.set_title(
                f"{SPLIT_LABELS[split]} - {MODE_LABELS[mode]}",
                fontsize=11,
            )
            ax.set_xticks(positions, comparison_labels, rotation=15)
            ax.set_ylim(lower_limit, upper_limit)
            ax.spines[["top", "right"]].set_visible(False)

            if column_number == 0:
                ax.set_ylabel("Paired F1 difference")

            if row_number == len(SPLITS) - 1:
                ax.set_xlabel("Algorithm comparison")

    fig.suptitle(
        "Paired per-image F1 differences",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.945,
        "Positive values favor the first algorithm; labels above boxes show wins / ties / losses.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )
    add_source_note(
        fig,
        "Each distribution contains the same 51 paired images; official [3] MATLAB evaluation.",
    )
    fig.subplots_adjust(
        top=0.87,
        bottom=0.12,
        hspace=0.28,
        wspace=0.12,
    )

    return save_figure(
        fig,
        output_dir,
        "16_paired_per_image_f1_differences",
        "Paired per-image F1 differences",
    )


def draw_candidate_diagnostics(main_index, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    x = np.arange(len(MODELS))
    width = 0.34

    for column, split in enumerate(SPLITS):
        representative_rows = [main_index[(split, model, "moderate")] for model in MODELS]
        raw = [as_float(row, "mean_raw_candidates") for row in representative_rows]
        retained = [
            as_float(row, "mean_retained_candidates") for row in representative_rows
        ]
        output_segments = [
            as_float(row, "mean_detected_segments") for row in representative_rows
        ]
        gt_segments = [as_float(row, "mean_gt_segments") for row in representative_rows]

        candidate_ax = axes[0, column]
        raw_bars = candidate_ax.bar(
            x - width / 2.0,
            raw,
            width,
            label="Raw candidates",
            color="#9ECAE1",
            edgecolor="white",
        )
        retained_bars = candidate_ax.bar(
            x + width / 2.0,
            retained,
            width,
            label="Retained for Task 4",
            color="#3182BD",
            edgecolor="white",
        )
        annotate_bars(candidate_ax, raw_bars, precision=0, fontsize=7)
        annotate_bars(candidate_ax, retained_bars, precision=0, fontsize=7)
        candidate_ax.set_title(f"{SPLIT_LABELS[split]}: candidates")
        candidate_ax.set_ylabel("Mean candidates per image")
        candidate_ax.set_xticks(x, [SHORT_MODEL_LABELS[model] for model in MODELS])
        candidate_ax.spines[["top", "right"]].set_visible(False)

        segment_ax = axes[1, column]
        detected_bars = segment_ax.bar(
            x - width / 2.0,
            output_segments,
            width,
            label="Detected segments",
            color="#F28E2B",
            edgecolor="white",
        )
        gt_bars = segment_ax.bar(
            x + width / 2.0,
            gt_segments,
            width,
            label="Ground-truth segments",
            color="#6B6B6B",
            edgecolor="white",
        )
        annotate_bars(segment_ax, detected_bars, precision=0, fontsize=7)
        annotate_bars(segment_ax, gt_bars, precision=0, fontsize=7)
        segment_ax.set_title(f"{SPLIT_LABELS[split]}: final segments")
        segment_ax.set_ylabel("Mean segments per image")
        segment_ax.set_xlabel("Candidate-generation algorithm")
        segment_ax.set_xticks(x, [SHORT_MODEL_LABELS[model] for model in MODELS])
        segment_ax.spines[["top", "right"]].set_visible(False)

    candidate_handles = [
        Patch(facecolor="#9ECAE1", label="Raw candidates"),
        Patch(facecolor="#3182BD", label="Retained for Task 4"),
        Patch(facecolor="#F28E2B", label="Detected segments"),
        Patch(facecolor="#6B6B6B", label="Ground-truth segments"),
    ]
    fig.legend(
        handles=candidate_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
    )
    fig.suptitle(
        "Candidate retention and segment-count diagnostics",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    add_source_note(
        fig,
        "Means over 51 images per split; candidate statistics are independent of evaluation tolerance.",
    )
    fig.subplots_adjust(top=0.82, bottom=0.09, hspace=0.30, wspace=0.17)

    return save_figure(
        fig,
        output_dir,
        "07_candidate_segment_diagnostics",
        "Candidate and segment diagnostics",
    )


def draw_k_ablation(k_index, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.6))
    condition_width = 0.34
    f1_labels = [
        f"{model}\n{MODE_ABBREVIATIONS[mode]}"
        for model in ("2b", "2d")
        for mode in MODES
    ]
    f1_x = np.arange(len(f1_labels))
    runtime_x = np.arange(2)

    all_f1_values = []

    for split in SPLITS:
        for model, mode in product(("2b", "2d"), MODES):
            row = k_index[(split, model, mode)]
            all_f1_values.extend(
                [as_float(row, "k8_mean_f1"), as_float(row, "k4_mean_f1")]
            )

    f1_upper = max(all_f1_values) * 1.20

    for column, split in enumerate(SPLITS):
        rows = [
            k_index[(split, model, mode)]
            for model in ("2b", "2d")
            for mode in MODES
        ]
        k8_f1 = [as_float(row, "k8_mean_f1") for row in rows]
        k4_f1 = [as_float(row, "k4_mean_f1") for row in rows]
        top_ax = axes[0, column]
        top_ax.bar(
            f1_x - condition_width / 2.0,
            k8_f1,
            condition_width,
            color=REFERENCE_COLOR,
            edgecolor="white",
            label="k=8 (default)",
        )
        top_ax.bar(
            f1_x + condition_width / 2.0,
            k4_f1,
            condition_width,
            color=COMPARISON_COLOR,
            edgecolor="white",
            label="k=4",
        )
        top_ax.axvline(2.5, color="#B0B0B0", linewidth=0.8)
        top_ax.set_title(f"{SPLIT_LABELS[split]} (n=5): F1")
        top_ax.set_ylabel("Mean F1")
        top_ax.set_xticks(f1_x, f1_labels)
        top_ax.set_ylim(0.0, f1_upper)
        top_ax.spines[["top", "right"]].set_visible(False)

        runtime_rows = [k_index[(split, model, "strict")] for model in ("2b", "2d")]
        k8_runtime = [as_float(row, "k8_mean_runtime_seconds") for row in runtime_rows]
        k4_runtime = [as_float(row, "k4_mean_runtime_seconds") for row in runtime_rows]
        bottom_ax = axes[1, column]
        k8_bars = bottom_ax.bar(
            runtime_x - condition_width / 2.0,
            k8_runtime,
            condition_width,
            color=REFERENCE_COLOR,
            edgecolor="white",
        )
        k4_bars = bottom_ax.bar(
            runtime_x + condition_width / 2.0,
            k4_runtime,
            condition_width,
            color=COMPARISON_COLOR,
            edgecolor="white",
        )
        annotate_bars(bottom_ax, k8_bars, precision=1)
        annotate_bars(bottom_ax, k4_bars, precision=1)
        bottom_ax.set_title(f"{SPLIT_LABELS[split]} (n=5): runtime")
        bottom_ax.set_ylabel("Mean runtime per image (seconds)")
        bottom_ax.set_xlabel("Color-EMD algorithm")
        bottom_ax.set_xticks(runtime_x, ["2b", "2d"])
        bottom_ax.spines[["top", "right"]].set_visible(False)

    condition_handles = [
        Patch(facecolor=REFERENCE_COLOR, label="k=8 (default)"),
        Patch(facecolor=COMPARISON_COLOR, label="k=4"),
    ]
    fig.legend(
        handles=condition_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
    )
    fig.suptitle(
        "Exact-EMD direction-shortlist ablation",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    add_source_note(
        fig,
        "Paired 5+5 cohort. All 64 directions are proxy-ranked; k controls how many receive exact EMD evaluation.",
    )
    fig.subplots_adjust(top=0.82, bottom=0.10, hspace=0.32, wspace=0.17)

    return save_figure(
        fig,
        output_dir,
        "08_emd_shortlist_k_ablation",
        "Exact-EMD direction-shortlist ablation",
    )


def draw_padding_ablation(padding_index, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.8))
    condition_width = 0.34
    models = ("2b", "2c", "2d")
    labels = [
        f"{model}\n{MODE_ABBREVIATIONS[mode]}"
        for model in models
        for mode in MODES
    ]
    x = np.arange(len(labels))

    all_f1 = []
    all_deltas = []

    for split, model, mode in product(SPLITS, models, MODES):
        row = padding_index[(split, model, mode)]
        all_f1.extend(
            [as_float(row, "default_mean_f1"), as_float(row, "zero_mean_f1")]
        )

        for metric in ("precision", "recall", "f1"):
            all_deltas.append(
                as_float(row, f"zero_mean_{metric}")
                - as_float(row, f"default_mean_{metric}")
            )

    f1_upper = max(all_f1) * 1.20
    delta_limit = max(abs(value) for value in all_deltas) * 1.18

    for column, split in enumerate(SPLITS):
        rows = [
            padding_index[(split, model, mode)]
            for model in models
            for mode in MODES
        ]
        default_f1 = [as_float(row, "default_mean_f1") for row in rows]
        zero_f1 = [as_float(row, "zero_mean_f1") for row in rows]
        top_ax = axes[0, column]
        top_ax.bar(
            x - condition_width / 2.0,
            default_f1,
            condition_width,
            color=REFERENCE_COLOR,
            edgecolor="white",
        )
        top_ax.bar(
            x + condition_width / 2.0,
            zero_f1,
            condition_width,
            color=COMPARISON_COLOR,
            edgecolor="white",
        )
        top_ax.axvline(2.5, color="#B0B0B0", linewidth=0.8)
        top_ax.axvline(5.5, color="#B0B0B0", linewidth=0.8)
        top_ax.set_title(f"{SPLIT_LABELS[split]} (n=5): absolute F1")
        top_ax.set_ylabel("Mean F1")
        top_ax.set_xticks(x, labels)
        top_ax.set_ylim(0.0, f1_upper)
        top_ax.spines[["top", "right"]].set_visible(False)

        delta_ax = axes[1, column]
        metric_width = 0.24
        metric_offsets = (-metric_width, 0.0, metric_width)

        for metric, offset in zip(("precision", "recall", "f1"), metric_offsets):
            deltas = [
                as_float(row, f"zero_mean_{metric}")
                - as_float(row, f"default_mean_{metric}")
                for row in rows
            ]
            delta_ax.bar(
                x + offset,
                deltas,
                metric_width,
                color=METRIC_COLORS[metric],
                edgecolor="white",
                label=metric.capitalize(),
            )

        delta_ax.axhline(0.0, color="#333333", linewidth=0.9)
        delta_ax.axvline(2.5, color="#B0B0B0", linewidth=0.8)
        delta_ax.axvline(5.5, color="#B0B0B0", linewidth=0.8)
        delta_ax.set_title(f"{SPLIT_LABELS[split]}: zero minus default")
        delta_ax.set_ylabel("Change in mean metric")
        delta_ax.set_xlabel("Algorithm and evaluation tolerance")
        delta_ax.set_xticks(x, labels)
        delta_ax.set_ylim(-delta_limit, delta_limit)
        delta_ax.spines[["top", "right"]].set_visible(False)

    condition_handles = [
        Patch(facecolor=REFERENCE_COLOR, label="Default padding (2b=8, 2c=4, 2d=8 px)"),
        Patch(facecolor=COMPARISON_COLOR, label="Zero padding"),
    ]
    metric_handles = [
        Patch(facecolor=METRIC_COLORS[metric], label=f"Delta {metric.capitalize()}")
        for metric in ("precision", "recall", "f1")
    ]
    fig.legend(
        handles=condition_handles + metric_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=5,
        fontsize=8.5,
    )
    fig.suptitle(
        "Endpoint-padding sensitivity",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    add_source_note(
        fig,
        "Paired 5+5 cohort. Positive deltas indicate that zero padding produced a larger metric.",
    )
    fig.subplots_adjust(top=0.82, bottom=0.11, hspace=0.32, wspace=0.16)

    return save_figure(
        fig,
        output_dir,
        "09_endpoint_padding_ablation",
        "Endpoint-padding sensitivity",
    )


def draw_synthetic_family_heatmap(synthetic_index, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 8.4), sharey=True)
    image = None
    valid_counts = {
        family: as_int(
            synthetic_index[(family, MODELS[0], MODES[0])],
            "valid_gt_image_count",
        )
        for family in SYNTHETIC_FAMILIES
    }
    color_map = plt.get_cmap("cividis").copy()
    color_map.set_bad("#E6E6E6")

    for ax, mode in zip(axes, MODES):
        matrix = np.asarray(
            [
                [
                    (
                        as_float(
                            synthetic_index[(family, model, mode)],
                            "valid_gt_mean_f1",
                        )
                        if valid_counts[family] > 0
                        else np.nan
                    )
                    for model in MODELS
                ]
                for family in SYNTHETIC_FAMILIES
            ],
            dtype=np.float64,
        )
        image = ax.imshow(
            matrix,
            cmap=color_map,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="auto",
        )

        for row_index, column_index in np.ndindex(matrix.shape):
            value = matrix[row_index, column_index]
            label = (
                "N/A"
                if not np.isfinite(value)
                else "0" if value < 0.0005 else f"{value:.2f}"
            )
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight=(
                    "bold" if np.isfinite(value) and value >= 0.5 else "normal"
                ),
                color=(
                    "black"
                    if not np.isfinite(value) or value >= 0.55
                    else "white"
                ),
            )

        for family in CHROMATIC_FAMILIES:
            row_index = SYNTHETIC_FAMILIES.index(family)
            ax.add_patch(
                Rectangle(
                    (-0.49, row_index - 0.49),
                    len(MODELS) - 0.02,
                    0.98,
                    fill=False,
                    edgecolor="#D55E00",
                    linewidth=2.0,
                    clip_on=False,
                )
            )

        ax.set_title(MODE_LABELS[mode], fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(np.arange(len(MODELS)), MODELS)
        ax.set_xlabel("Algorithm")
        family_labels = [
            f"{family} (n={valid_counts[family]}/5)"
            for family in SYNTHETIC_FAMILIES
        ]
        ax.set_yticks(np.arange(len(SYNTHETIC_FAMILIES)), family_labels)
        ax.tick_params(axis="both", which="both", length=0)
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylabel("Synthetic family")

    for label, family in zip(axes[0].get_yticklabels(), SYNTHETIC_FAMILIES):
        if family in CHROMATIC_FAMILIES:
            label.set_color("#B43C12")
            label.set_fontweight("bold")

    colorbar_axis = fig.add_axes([0.925, 0.16, 0.016, 0.67])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Mean F1", rotation=90, labelpad=10)
    colorbar.outline.set_linewidth(0.7)
    fig.suptitle(
        "Synthetic-family F1 by algorithm and evaluation tolerance",
        y=0.985,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        "Cells average only images retaining GT after official preprocessing; orange outlines mark chromatic families.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#B43C12",
    )
    add_source_note(
        fig,
        "n gives valid images out of five; N/A means no processed GT remained; one shared F1 scale for all panels.",
    )
    fig.subplots_adjust(
        left=0.16,
        right=0.90,
        top=0.87,
        bottom=0.10,
        wspace=0.10,
    )

    return save_figure(
        fig,
        output_dir,
        "15_synthetic_family_f1_heatmap",
        "Synthetic-family F1 heatmap",
    )


def build_figures(csv_dir, output_dir):
    main_average_path = csv_dir / "task4_final_102_averages.csv"
    main_result_path = csv_dir / "task4_final_102_results.csv"
    k_path = csv_dir / "report_k_ablation_comparison.csv"
    padding_path = csv_dir / "report_padding_ablation_comparison.csv"
    synthetic_family_path = csv_dir / "report_synthetic_family_summary.csv"

    main_averages = read_csv(main_average_path, MAIN_AVERAGE_COLUMNS)
    main_results = read_csv(main_result_path, MAIN_RESULT_COLUMNS)
    k_rows = read_csv(k_path, K_COLUMNS)
    padding_rows = read_csv(padding_path, PADDING_COLUMNS)
    synthetic_rows = read_csv(
        synthetic_family_path,
        SYNTHETIC_FAMILY_COLUMNS,
    )
    main_index, k_index, padding_index = validate_sources(
        main_averages, main_results, k_rows, padding_rows
    )
    synthetic_index = validate_synthetic_family_rows(synthetic_rows)

    configure_style()
    generated = []
    generated.extend(draw_pipeline(output_dir))
    generated.extend(draw_main_f1(main_index, output_dir))
    generated.extend(draw_precision_recall(main_index, output_dir))
    generated.extend(draw_errors(main_index, output_dir))
    generated.extend(draw_runtime_tradeoff(main_index, output_dir))
    generated.extend(draw_per_image_f1(main_results, output_dir))
    generated.extend(draw_candidate_diagnostics(main_index, output_dir))
    generated.extend(draw_k_ablation(k_index, output_dir))
    generated.extend(draw_padding_ablation(padding_index, output_dir))
    generated.extend(draw_synthetic_family_heatmap(synthetic_index, output_dir))
    generated.extend(draw_paired_f1_differences(main_results, output_dir))
    return generated


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate report-ready figures from validated benchmark CSV files."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help=f"directory containing report CSV files (default: {DEFAULT_CSV_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"figure output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generated = build_figures(args.csv_dir.resolve(), args.output_dir.resolve())
    print(f"Generated {len(generated)} report assets in {args.output_dir.resolve()}")

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
