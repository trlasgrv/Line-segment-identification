import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from benchmark_checkpoint import atomic_write_csv, validate_unique_rows


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_DIR = PROJECT_ROOT / "CSVs"

MODEL_ORDER = ("2a", "2b", "2c", "2d")
MODE_ORDER = ("strict", "moderate", "loose")
SPLIT_ORDER = ("york", "wireframe", "all")
SYNTHETIC_FAMILY_ORDER = (
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

RESULT_KEY_FIELDS = ("image", "model", "mode")
REQUIRED_RESULT_COLUMNS = {
    "image",
    "split",
    "model",
    "algorithm",
    "metric",
    "evaluator",
    "mode",
    "status",
    "config_hash",
    "runtime_seconds",
    "raw_candidates",
    "retained_candidates",
    "cap_activated",
    "clusters",
    "segments",
    "gt_segments",
    "matched",
    "precision",
    "recall",
    "f1",
    "mean_iou",
    "location_error_px",
    "orientation_error_deg",
    "error",
}

MEAN_FIELDS = {
    "mean_precision": "precision",
    "mean_recall": "recall",
    "mean_f1": "f1",
    "mean_iou": "mean_iou",
    "mean_location_error_px": "location_error_px",
    "mean_orientation_error_deg": "orientation_error_deg",
    "mean_runtime_seconds": "runtime_seconds",
    "mean_raw_candidates": "raw_candidates",
    "mean_retained_candidates": "retained_candidates",
    "mean_clusters": "clusters",
    "mean_detected_segments": "segments",
    "mean_gt_segments": "gt_segments",
    "mean_matched_pairs": "matched",
}

AGGREGATE_COLUMNS = (
    "image_count",
    "valid_gt_image_count",
    "valid_gt_image_rate",
    "valid_gt_mean_precision",
    "valid_gt_mean_recall",
    "valid_gt_mean_f1",
    "mean_precision",
    "mean_recall",
    "mean_f1",
    "pooled_precision",
    "pooled_recall",
    "pooled_f1",
    "mean_iou",
    "mean_location_error_px",
    "mean_orientation_error_deg",
    "mean_runtime_seconds",
    "total_runtime_seconds",
    "mean_raw_candidates",
    "mean_retained_candidates",
    "cap_activation_count",
    "cap_activation_rate",
    "mean_clusters",
    "mean_detected_segments",
    "mean_gt_segments",
    "mean_matched_pairs",
)

FAMILY_COLUMNS = (
    "dataset",
    "split",
    "sample_selection",
    "family",
    "model",
    "algorithm",
    "metric",
    "evaluator",
    "mode",
    "source_config_hash",
) + AGGREGATE_COLUMNS

SYNTHETIC_VALID_SUMMARY_COLUMNS = (
    "dataset",
    "split",
    "sample_selection",
    "validity_filter",
    "model",
    "algorithm",
    "metric",
    "evaluator",
    "mode",
    "source_config_hash",
) + AGGREGATE_COLUMNS

EVALUATION_COMPARISON_METRICS = (
    "mean_precision",
    "mean_recall",
    "mean_f1",
    "pooled_precision",
    "pooled_recall",
    "pooled_f1",
    "mean_iou",
    "mean_location_error_px",
    "mean_orientation_error_deg",
    "mean_detected_segments",
    "mean_gt_segments",
    "mean_matched_pairs",
)

K_COMPARISON_METRICS = EVALUATION_COMPARISON_METRICS + (
    "mean_runtime_seconds",
    "mean_raw_candidates",
    "mean_retained_candidates",
)

K_PREFIX_COLUMNS = tuple(
    column
    for metric in K_COMPARISON_METRICS
    for column in (
        f"k8_{metric}",
        f"k4_{metric}",
        f"k4_minus_k8_{metric}",
    )
)

K_COMPARISON_COLUMNS = (
    "split",
    "cohort",
    "model",
    "algorithm",
    "metric",
    "evaluator",
    "mode",
    "image_count",
    "reference_config_hash",
    "comparison_config_hash",
    "reference_k",
    "comparison_k",
) + K_PREFIX_COLUMNS + ("runtime_change_percent",)

PADDING_PREFIX_COLUMNS = tuple(
    column
    for metric in EVALUATION_COMPARISON_METRICS
    for column in (
        f"default_{metric}",
        f"zero_{metric}",
        f"zero_minus_default_{metric}",
    )
)

PADDING_COMPARISON_COLUMNS = (
    "split",
    "cohort",
    "model",
    "algorithm",
    "metric",
    "evaluator",
    "mode",
    "image_count",
    "reference_config_hash",
    "comparison_config_hash",
    "default_padding_px",
    "comparison_padding_px",
) + PADDING_PREFIX_COLUMNS


def synthetic_family_from_image_id(image_id):
    match = re.fullmatch(r"syn_\d{3}_(.+)", image_id)

    if match is None or match.group(1) not in SYNTHETIC_FAMILY_ORDER:
        raise ValueError(f"unrecognized synthetic image id: {image_id}")

    return match.group(1)


def _float(row, field):
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric value for {field}: {row.get(field)!r}") from error


def _int(row, field):
    value = _float(row, field)

    if not value.is_integer():
        raise ValueError(f"expected integer value for {field}: {value}")

    return int(value)


def _boolean(row, field):
    value = row.get(field, "").strip().lower()

    if value == "true":
        return True

    if value == "false":
        return False

    raise ValueError(f"invalid boolean value for {field}: {row.get(field)!r}")


def load_result_rows(path):
    path = Path(path)

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_RESULT_COLUMNS - columns

        if missing:
            raise ValueError(f"missing columns in {path}: {sorted(missing)}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"result CSV is empty: {path}")

    validate_unique_rows(rows, RESULT_KEY_FIELDS)

    failed = [row for row in rows if row["status"] != "ok" or row["error"]]

    if failed:
        raise ValueError(f"{path} contains {len(failed)} failed result rows")

    return rows


def load_complete_metadata(result_path):
    metadata_path = Path(str(result_path) + ".metadata.json")

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    progress = metadata.get("progress", {})

    if metadata.get("status") != "complete":
        raise ValueError(f"benchmark metadata is not complete: {metadata_path}")

    if progress.get("failed_jobs") != 0 or progress.get("incomplete_jobs") != 0:
        raise ValueError(f"benchmark metadata records unfinished jobs: {metadata_path}")

    return metadata


def validate_rows_against_metadata(rows, metadata, source_name):
    hashes = {row["config_hash"] for row in rows}
    expected_hash = metadata.get("configuration_hash")

    if hashes != {expected_hash}:
        raise ValueError(
            f"configuration hashes in {source_name} do not match its metadata"
        )


def aggregate_rows(rows):
    if not rows:
        raise ValueError("cannot aggregate an empty row collection")

    image_ids = [row["image"] for row in rows]

    if len(image_ids) != len(set(image_ids)):
        raise ValueError("aggregate contains duplicate image rows")

    count = len(rows)
    detected = sum(_int(row, "segments") for row in rows)
    ground_truth = sum(_int(row, "gt_segments") for row in rows)
    matched = sum(_int(row, "matched") for row in rows)
    denominator = detected + ground_truth
    cap_count = sum(_boolean(row, "cap_activated") for row in rows)
    valid_gt_rows = [row for row in rows if _int(row, "gt_segments") > 0]
    valid_gt_count = len(valid_gt_rows)
    result = {
        "image_count": count,
        "valid_gt_image_count": valid_gt_count,
        "valid_gt_image_rate": valid_gt_count / count,
        "valid_gt_mean_precision": (
            sum(_float(row, "precision") for row in valid_gt_rows) / valid_gt_count
            if valid_gt_rows
            else ""
        ),
        "valid_gt_mean_recall": (
            sum(_float(row, "recall") for row in valid_gt_rows) / valid_gt_count
            if valid_gt_rows
            else ""
        ),
        "valid_gt_mean_f1": (
            sum(_float(row, "f1") for row in valid_gt_rows) / valid_gt_count
            if valid_gt_rows
            else ""
        ),
        "pooled_precision": matched / detected if detected else 0.0,
        "pooled_recall": matched / ground_truth if ground_truth else 0.0,
        "pooled_f1": 2.0 * matched / denominator if denominator else 0.0,
        "total_runtime_seconds": sum(
            _float(row, "runtime_seconds") for row in rows
        ),
        "cap_activation_count": cap_count,
        "cap_activation_rate": cap_count / count,
    }

    for output_field, input_field in MEAN_FIELDS.items():
        result[output_field] = sum(
            _float(row, input_field) for row in rows
        ) / count

    return result


def _single_value(rows, field):
    values = {row[field] for row in rows}

    if len(values) != 1:
        raise ValueError(f"expected one {field} value, found {sorted(values)}")

    return next(iter(values))


def build_synthetic_family_rows(rows, source_config_hash):
    groups = defaultdict(list)

    for row in rows:
        if row["split"] != "synthetic":
            raise ValueError("synthetic summary source contains a non-synthetic row")

        family = synthetic_family_from_image_id(row["image"])
        groups[(family, row["model"], row["mode"])].append(row)

    output = []

    for family in SYNTHETIC_FAMILY_ORDER:
        for model in MODEL_ORDER:
            for mode in MODE_ORDER:
                group = groups.get((family, model, mode), [])

                if len(group) != 5:
                    raise ValueError(
                        f"expected five images for {(family, model, mode)}, "
                        f"found {len(group)}"
                    )

                output.append(
                    {
                        "dataset": "synthetic_100",
                        "split": "synthetic",
                        "sample_selection": (
                            "stratified_even_indices_0_2_4_6_8_per_family"
                        ),
                        "family": family,
                        "model": model,
                        "algorithm": _single_value(group, "algorithm"),
                        "metric": _single_value(group, "metric"),
                        "evaluator": _single_value(group, "evaluator"),
                        "mode": mode,
                        "source_config_hash": source_config_hash,
                        **aggregate_rows(group),
                    }
                )

    return output


def build_synthetic_valid_summary_rows(rows, source_config_hash):
    output = []

    for model in MODEL_ORDER:
        for mode in MODE_ORDER:
            group = [
                row
                for row in rows
                if row["model"] == model
                and row["mode"] == mode
                and _int(row, "gt_segments") > 0
            ]

            if len(group) != 20:
                raise ValueError(
                    f"expected 20 valid-GT synthetic images for {(model, mode)}, "
                    f"found {len(group)}"
                )

            output.append(
                {
                    "dataset": "synthetic_100",
                    "split": "synthetic",
                    "sample_selection": (
                        "stratified_even_indices_0_2_4_6_8_per_family"
                    ),
                    "validity_filter": "official_processed_gt_segments_gt_0",
                    "model": model,
                    "algorithm": _single_value(group, "algorithm"),
                    "metric": _single_value(group, "metric"),
                    "evaluator": _single_value(group, "evaluator"),
                    "mode": mode,
                    "source_config_hash": source_config_hash,
                    **aggregate_rows(group),
                }
            )

    return output


def _comparison_groups(reference_rows, comparison_rows, models):
    comparison_rows = [row for row in comparison_rows if row["model"] in models]
    comparison_keys = {
        (row["image"], row["model"], row["mode"])
        for row in comparison_rows
    }
    reference_rows = [
        row
        for row in reference_rows
        if (row["image"], row["model"], row["mode"]) in comparison_keys
    ]
    reference_keys = {
        (row["image"], row["model"], row["mode"])
        for row in reference_rows
    }

    if reference_keys != comparison_keys:
        missing = sorted(comparison_keys - reference_keys)
        extra = sorted(reference_keys - comparison_keys)
        raise ValueError(
            f"comparison cohorts are not paired; missing={missing}, extra={extra}"
        )

    result = []

    for split in SPLIT_ORDER:
        for model in models:
            for mode in MODE_ORDER:
                reference_group = [
                    row
                    for row in reference_rows
                    if row["model"] == model
                    and row["mode"] == mode
                    and (split == "all" or row["split"] == split)
                ]
                comparison_group = [
                    row
                    for row in comparison_rows
                    if row["model"] == model
                    and row["mode"] == mode
                    and (split == "all" or row["split"] == split)
                ]
                reference_ids = {row["image"] for row in reference_group}
                comparison_ids = {row["image"] for row in comparison_group}

                if reference_ids != comparison_ids or not reference_ids:
                    raise ValueError(
                        f"unpaired comparison group: {(split, model, mode)}"
                    )

                result.append(
                    (
                        split,
                        model,
                        mode,
                        reference_group,
                        comparison_group,
                    )
                )

    return result


def _comparison_identity(reference_group, comparison_group):
    for field in ("algorithm", "metric", "evaluator"):
        reference_value = _single_value(reference_group, field)
        comparison_value = _single_value(comparison_group, field)

        if reference_value != comparison_value:
            raise ValueError(
                f"comparison changes {field}: "
                f"{reference_value!r} != {comparison_value!r}"
            )

    return {
        "algorithm": reference_group[0]["algorithm"],
        "metric": reference_group[0]["metric"],
        "evaluator": reference_group[0]["evaluator"],
    }


def build_k_comparison_rows(
    reference_rows,
    comparison_rows,
    reference_config_hash,
    comparison_config_hash,
):
    output = []

    for split, model, mode, reference_group, comparison_group in _comparison_groups(
        reference_rows,
        comparison_rows,
        ("2b", "2d"),
    ):
        reference = aggregate_rows(reference_group)
        comparison = aggregate_rows(comparison_group)
        item = {
            "split": split,
            "cohort": "shared_5_york_5_wireframe_from_final_102",
            "model": model,
            **_comparison_identity(reference_group, comparison_group),
            "mode": mode,
            "image_count": len(reference_group),
            "reference_config_hash": reference_config_hash,
            "comparison_config_hash": comparison_config_hash,
            "reference_k": 8,
            "comparison_k": 4,
        }

        for metric in K_COMPARISON_METRICS:
            item[f"k8_{metric}"] = reference[metric]
            item[f"k4_{metric}"] = comparison[metric]
            item[f"k4_minus_k8_{metric}"] = comparison[metric] - reference[metric]

        runtime = reference["mean_runtime_seconds"]
        item["runtime_change_percent"] = (
            100.0 * (comparison["mean_runtime_seconds"] - runtime) / runtime
            if runtime
            else 0.0
        )
        output.append(item)

    return output


def build_padding_comparison_rows(
    reference_rows,
    comparison_rows,
    reference_config_hash,
    comparison_config_hash,
    default_padding,
):
    output = []

    for split, model, mode, reference_group, comparison_group in _comparison_groups(
        reference_rows,
        comparison_rows,
        ("2b", "2c", "2d"),
    ):
        reference = aggregate_rows(reference_group)
        comparison = aggregate_rows(comparison_group)
        item = {
            "split": split,
            "cohort": "shared_5_york_5_wireframe_from_final_102",
            "model": model,
            **_comparison_identity(reference_group, comparison_group),
            "mode": mode,
            "image_count": len(reference_group),
            "reference_config_hash": reference_config_hash,
            "comparison_config_hash": comparison_config_hash,
            "default_padding_px": default_padding[model],
            "comparison_padding_px": 0.0,
        }

        for metric in EVALUATION_COMPARISON_METRICS:
            item[f"default_{metric}"] = reference[metric]
            item[f"zero_{metric}"] = comparison[metric]
            item[f"zero_minus_default_{metric}"] = (
                comparison[metric] - reference[metric]
            )

        output.append(item)

    return output


def _model_parameter(metadata, model, category, parameter):
    try:
        return metadata["configuration"]["models"][model][category][parameter]
    except KeyError as error:
        raise ValueError(
            f"metadata is missing {model}.{category}.{parameter}"
        ) from error


def generate_report_tables(csv_dir=DEFAULT_CSV_DIR):
    csv_dir = Path(csv_dir)
    main_path = csv_dir / "task4_final_102_results.csv"
    synthetic_path = csv_dir / "task4_synthetic_50_results.csv"
    k4_path = csv_dir / "task4_emd_shortlist_k4_5york_5wireframe_results.csv"
    zero_path = csv_dir / "task4_zero_padding_5york_5wireframe_results.csv"

    main_rows = load_result_rows(main_path)
    synthetic_rows = load_result_rows(synthetic_path)
    k4_rows = load_result_rows(k4_path)
    zero_rows = load_result_rows(zero_path)
    main_metadata = load_complete_metadata(main_path)
    synthetic_metadata = load_complete_metadata(synthetic_path)
    k4_metadata = load_complete_metadata(k4_path)
    zero_metadata = load_complete_metadata(zero_path)

    for rows, metadata, name in (
        (main_rows, main_metadata, main_path.name),
        (synthetic_rows, synthetic_metadata, synthetic_path.name),
        (k4_rows, k4_metadata, k4_path.name),
        (zero_rows, zero_metadata, zero_path.name),
    ):
        validate_rows_against_metadata(rows, metadata, name)

    for model in ("2b", "2d"):
        if _model_parameter(
            main_metadata,
            model,
            "detector_parameters",
            "exact_top_k",
        ) != 8:
            raise ValueError(f"main result does not use k=8 for {model}")

        if _model_parameter(
            k4_metadata,
            model,
            "detector_parameters",
            "exact_top_k",
        ) != 4:
            raise ValueError(f"k ablation does not use k=4 for {model}")

    default_padding = {"2b": 8.0, "2c": 4.0, "2d": 8.0}

    for model, padding in default_padding.items():
        if _model_parameter(
            main_metadata,
            model,
            "segment_parameters",
            "endpoint_padding",
        ) != padding:
            raise ValueError(f"unexpected main endpoint padding for {model}")

        if _model_parameter(
            zero_metadata,
            model,
            "segment_parameters",
            "endpoint_padding",
        ) != 0.0:
            raise ValueError(f"zero-padding run is not zero for {model}")

    family_rows = build_synthetic_family_rows(
        synthetic_rows,
        synthetic_metadata["configuration_hash"],
    )
    synthetic_valid_rows = build_synthetic_valid_summary_rows(
        synthetic_rows,
        synthetic_metadata["configuration_hash"],
    )
    k_rows = build_k_comparison_rows(
        main_rows,
        k4_rows,
        main_metadata["configuration_hash"],
        k4_metadata["configuration_hash"],
    )
    padding_rows = build_padding_comparison_rows(
        main_rows,
        zero_rows,
        main_metadata["configuration_hash"],
        zero_metadata["configuration_hash"],
        default_padding,
    )

    outputs = (
        (
            csv_dir / "report_synthetic_valid_gt_summary.csv",
            SYNTHETIC_VALID_SUMMARY_COLUMNS,
            synthetic_valid_rows,
        ),
        (
            csv_dir / "report_synthetic_family_summary.csv",
            FAMILY_COLUMNS,
            family_rows,
        ),
        (
            csv_dir / "report_k_ablation_comparison.csv",
            K_COMPARISON_COLUMNS,
            k_rows,
        ),
        (
            csv_dir / "report_padding_ablation_comparison.csv",
            PADDING_COMPARISON_COLUMNS,
            padding_rows,
        ),
    )

    for path, columns, rows in outputs:
        atomic_write_csv(path, columns, rows)
        print(f"Wrote {len(rows)} rows to {path}")

    return {path.name: rows for path, _, rows in outputs}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate report-only aggregate and ablation CSV tables."
    )
    parser.add_argument(
        "--csv-dir",
        default=str(DEFAULT_CSV_DIR),
        help="Directory containing authoritative benchmark CSVs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    generate_report_tables(arguments.csv_dir)
