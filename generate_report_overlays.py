import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "dataset" / "report_benchmark" / "manifest.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "CSVs" / "task4_final_102_results.csv"
DEFAULT_SEGMENTS_ROOT = PROJECT_ROOT / "task4_final_102_results_segments"
DEFAULT_SYNTHETIC_MANIFEST = PROJECT_ROOT / "dataset" / "synthetic_100" / "manifest.csv"
DEFAULT_SYNTHETIC_RESULTS = PROJECT_ROOT / "CSVs" / "task4_synthetic_50_results.csv"
DEFAULT_SYNTHETIC_SEGMENTS_ROOT = PROJECT_ROOT / "task4_synthetic_50_results_segments"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Figures" / "report" / "overlays"
DEFAULT_EVALUATOR_DIR = PROJECT_ROOT / "external" / "line_segment_review" / "eva_survey_p"
MATLAB_BRIDGE_DIR = Path(__file__).resolve().parent / "matlab"

_MPL_CACHE = Path(tempfile.gettempdir()) / "prj_line_detection_matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image
from scipy.io import loadmat, savemat

from paper3_matlab_evaluation import (
    find_matlab_executable,
    image_to_matlab_uint8,
    matlab_environment,
)


MODELS = ("2a", "2b", "2c", "2d")
MODEL_COLORS = {
    "2a": "#00A6ED",
    "2b": "#FFB000",
    "2c": "#00C27A",
    "2d": "#F065A6",
}
GT_COLOR = "#00E5FF"


@dataclass(frozen=True)
class OverlayCase:
    image_id: str
    report_label: str
    filename: str


OVERLAY_CASES = (
    OverlayCase(
        "P1030004",
        "Representative YorkUrban image",
        "10_overlay_P1030004_representative_york",
    ),
    OverlayCase(
        "00066618",
        "Representative Wireframe image",
        "11_overlay_00066618_representative_wireframe",
    ),
    OverlayCase(
        "P1040783",
        "Strong 2d result",
        "12_overlay_P1040783_strong_2d",
    ),
    OverlayCase(
        "00037266",
        "Strong 2c result",
        "13_overlay_00037266_strong_2c",
    ),
    OverlayCase(
        "00377104",
        "Shared failure case",
        "14_overlay_00377104_shared_failure",
    ),
)

SYNTHETIC_OVERLAY_CASES = (
    OverlayCase(
        "syn_062_rectangle_chroma",
        "Chromatic rectangle: color EMD succeeds",
        "17_overlay_syn_062_rectangle_chroma_emd_success",
    ),
    OverlayCase(
        "syn_050_rectangle_luma",
        "Luminance rectangle: broad detector success",
        "18_overlay_syn_050_rectangle_luma_success",
    ),
    OverlayCase(
        "syn_048_parallel_chroma",
        "Parallel chromatic edges: shared failure",
        "19_overlay_syn_048_parallel_chroma_failure",
    ),
)


def matlab_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {"id", "split", "image", "gt", "width", "height"}

    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid report manifest: {path}")

    index = {}

    for row in rows:
        if row["id"] in index:
            raise ValueError(f"duplicate manifest image id: {row['id']}")

        index[row["id"]] = row

    return index


def read_loose_results(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "image",
        "split",
        "model",
        "mode",
        "status",
        "config_hash",
        "segments",
        "gt_segments",
        "precision",
        "recall",
        "f1",
        "error",
    }

    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid final result table: {path}")

    index = {}

    for row in rows:
        if row["mode"] != "loose":
            continue

        key = (row["image"], row["model"])

        if key in index:
            raise ValueError(f"duplicate loose result: {key}")

        if row["status"] != "ok" or row["error"].strip():
            raise ValueError(f"failed final result cannot be overlaid: {key}")

        index[key] = row

    return index


def load_json_segments(path):
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError(f"invalid segment JSON: {path}")

    values = np.asarray(data["segments"], dtype=np.float64)

    if values.size == 0:
        return np.zeros((0, 4), dtype=np.float64)

    values = values.reshape(-1, 4)

    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite segment coordinates in {path}")

    return values


def load_detection_artifact(path, image_id, model, config_hash):
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)

    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported segment artifact schema: {path}")

    expected = {
        "image": image_id,
        "model": model,
        "config_hash": config_hash,
    }

    for field, value in expected.items():
        if data.get(field) != value:
            raise ValueError(
                f"segment artifact {field} mismatch in {path}: "
                f"{data.get(field)!r} != {value!r}"
            )

    segments = []

    for segment in data.get("segments", []):
        start = np.asarray(segment.get("start"), dtype=np.float64)
        end = np.asarray(segment.get("end"), dtype=np.float64)

        if start.shape != (2,) or end.shape != (2,):
            raise ValueError(f"invalid endpoint shape in {path}")

        values = np.concatenate([start, end])

        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite endpoint in {path}")

        segments.append(values)

    if not segments:
        return np.zeros((0, 4), dtype=np.float64)

    return np.asarray(segments, dtype=np.float64).reshape(-1, 4)


def load_cases(
    manifest_path,
    result_path,
    segments_root,
    overlay_cases=OVERLAY_CASES,
):
    manifest = read_manifest(manifest_path)
    loose_results = read_loose_results(result_path)
    manifest_root = Path(manifest_path).resolve().parent
    cases = []

    for case in overlay_cases:
        if case.image_id not in manifest:
            raise ValueError(f"requested overlay image is absent: {case.image_id}")

        row = manifest[case.image_id]
        image_path = (manifest_root / row["image"]).resolve()
        gt_path = (manifest_root / row["gt"]).resolve()

        with Image.open(image_path) as source_image:
            image = np.asarray(source_image.convert("RGB"))

        expected_shape = (int(row["height"]), int(row["width"]))

        if image.shape[:2] != expected_shape:
            raise ValueError(
                f"image size mismatch for {case.image_id}: "
                f"{image.shape[:2]} != {expected_shape}"
            )

        gt_segments = load_json_segments(gt_path)
        detections = {}
        metrics = {}

        for model in MODELS:
            result_key = (case.image_id, model)

            if result_key not in loose_results:
                raise ValueError(f"missing loose benchmark result: {result_key}")

            result = loose_results[result_key]

            if result["split"] != row["split"]:
                raise ValueError(f"split mismatch for {result_key}")

            artifact_path = (
                Path(segments_root)
                / row["split"]
                / case.image_id
                / f"{model}.json"
            )
            detections[model] = load_detection_artifact(
                artifact_path,
                case.image_id,
                model,
                result["config_hash"],
            )
            metrics[model] = result

        cases.append(
            {
                "case": case,
                "split": row["split"],
                "image": image,
                "ground_truth": gt_segments,
                "detections": detections,
                "metrics": metrics,
            }
        )

    return cases


def normalized_mat_lines(mat_data, field):
    values = np.asarray(mat_data[field], dtype=np.float64)

    if values.size == 0:
        return np.zeros((0, 4), dtype=np.float64)

    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError(
            f"official preprocessing returned an invalid {field} shape: "
            f"{values.shape}"
        )

    # process_results can append matcher metadata after the four endpoint
    # coordinates. Rows remain lines; only the first four columns are visual.
    values = values[:, :4]

    if not np.all(np.isfinite(values)):
        raise ValueError(f"official preprocessing returned non-finite {field}")

    return values


def process_cases_with_matlab(cases, matlab_bin, evaluator_dir, timeout_seconds):
    matlab_bin = find_matlab_executable(matlab_bin)
    evaluator_dir = Path(evaluator_dir).resolve()

    if not (evaluator_dir / "process_results.p").is_file():
        raise FileNotFoundError(
            f"official [3] process_results.p not found in {evaluator_dir}"
        )

    with tempfile.TemporaryDirectory(
        prefix="paper3_overlay_",
        dir="/tmp",
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        for case_data in cases:
            image_id = case_data["case"].image_id
            savemat(
                input_dir / f"{image_id}.mat",
                {
                    "image": image_to_matlab_uint8(case_data["image"]),
                    "gt_lines": case_data["ground_truth"],
                    **{
                        f"test_{model}": case_data["detections"][model]
                        for model in MODELS
                    },
                },
            )

        command_text = (
            "addpath("
            + matlab_string(MATLAB_BRIDGE_DIR)
            + "); paper3_process_overlay_batch("
            + matlab_string(input_dir)
            + ", "
            + matlab_string(output_dir)
            + ", "
            + matlab_string(evaluator_dir)
            + ");"
        )

        try:
            completed = subprocess.run(
                [matlab_bin, "-batch", command_text],
                env=matlab_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"MATLAB overlay preprocessing exceeded {timeout_seconds} seconds"
            ) from error

        if completed.returncode != 0:
            raise RuntimeError(
                "MATLAB overlay preprocessing failed with exit code "
                f"{completed.returncode}:\n{completed.stdout}"
            )

        for case_data in cases:
            image_id = case_data["case"].image_id
            output_path = output_dir / f"{image_id}.mat"

            if not output_path.is_file():
                raise RuntimeError(f"MATLAB did not produce {output_path}")

            mat_data = loadmat(output_path)
            case_data["ground_truth"] = normalized_mat_lines(
                mat_data, "gt_processed"
            )
            case_data["detections"] = {
                model: normalized_mat_lines(mat_data, f"test_{model}_processed")
                for model in MODELS
            }


def validate_processed_counts(cases):
    for case_data in cases:
        image_id = case_data["case"].image_id
        gt_count = len(case_data["ground_truth"])
        expected_gt_counts = {
            int(float(case_data["metrics"][model]["gt_segments"]))
            for model in MODELS
        }

        if expected_gt_counts != {gt_count}:
            raise ValueError(
                f"official GT count mismatch for {image_id}: "
                f"processed={gt_count}, results={sorted(expected_gt_counts)}"
            )

        for model in MODELS:
            detected_count = len(case_data["detections"][model])
            expected = int(float(case_data["metrics"][model]["segments"]))

            if detected_count != expected:
                raise ValueError(
                    f"official detection count mismatch for {(image_id, model)}: "
                    f"processed={detected_count}, results={expected}"
                )


def line_pairs(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1, 4)
    return values.reshape(-1, 2, 2)


def add_lines(ax, values, color, dense=False):
    pairs = line_pairs(values)

    if not len(pairs):
        return

    shadow_width = 1.35 if dense else 2.5
    color_width = 0.72 if dense else 1.25
    ax.add_collection(
        LineCollection(
            pairs,
            colors="#101010",
            linewidths=shadow_width,
            alpha=0.72,
            capstyle="round",
            joinstyle="round",
        )
    )
    ax.add_collection(
        LineCollection(
            pairs,
            colors=color,
            linewidths=color_width,
            alpha=0.92,
            capstyle="round",
            joinstyle="round",
        )
    )


def configure_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": "#202020",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def render_case(case_data, output_dir):
    case = case_data["case"]
    image = case_data["image"]
    height, width = image.shape[:2]
    aspect = width / height
    figure_width = max(16.0, 5.0 * 3.15 * aspect)
    figure_height = 4.65
    fig, axes = plt.subplots(1, 5, figsize=(figure_width, figure_height))
    panel_data = [("Ground truth", case_data["ground_truth"], GT_COLOR, None)]

    for model in MODELS:
        metric = case_data["metrics"][model]
        panel_data.append(
            (
                model,
                case_data["detections"][model],
                MODEL_COLORS[model],
                float(metric["f1"]),
            )
        )

    for ax, (label, segments, color, f1_value) in zip(axes, panel_data):
        ax.imshow(image, interpolation="nearest")
        add_lines(ax, segments, color, dense=(label == "Ground truth"))
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        if f1_value is None:
            title = f"GT\nn={len(segments)}"
        else:
            title = f"{label}\nn={len(segments)}  |  loose F1={f1_value:.3f}"

        ax.set_title(title, fontsize=10.5, fontweight="bold", color=color, pad=7)

    split_labels = {
        "york": "YorkUrban",
        "wireframe": "Wireframe",
        "synthetic": "Synthetic",
    }
    split_label = split_labels.get(case_data["split"], case_data["split"].title())
    fig.suptitle(
        f"{case.report_label}: {case.image_id} ({split_label})",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.025,
        "Official [3] preprocessing; final report configuration; identical image scale in all panels.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.006, right=0.994, top=0.79, bottom=0.09, wspace=0.018)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{case.filename}.png"
    pdf_path = output_dir / f"{case.filename}.pdf"
    metadata = {
        "Title": f"{case.report_label}: {case.image_id}",
        "Author": "Line Segment Detection Project",
        "Subject": "Ground-truth and final line-segment overlays",
    }
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.08,
        metadata=metadata,
    )
    plt.close(fig)
    return png_path, pdf_path


def generate_overlays(
    manifest_path,
    result_path,
    segments_root,
    output_dir,
    matlab_bin=None,
    evaluator_dir=DEFAULT_EVALUATOR_DIR,
    timeout_seconds=300,
    overlay_cases=OVERLAY_CASES,
):
    cases = load_cases(
        manifest_path,
        result_path,
        segments_root,
        overlay_cases=overlay_cases,
    )
    process_cases_with_matlab(cases, matlab_bin, evaluator_dir, timeout_seconds)
    validate_processed_counts(cases)
    configure_style()
    generated = []

    for case_data in cases:
        generated.extend(render_case(case_data, output_dir))

    return generated


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate report overlays from saved final segment artifacts."
    )
    parser.add_argument(
        "--profile",
        choices=("real", "synthetic"),
        default="real",
        help="select the predefined real or synthetic report cases",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--segments-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--matlab-bin")
    parser.add_argument("--evaluator-dir", type=Path, default=DEFAULT_EVALUATOR_DIR)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")

    if args.profile == "synthetic":
        manifest_path = args.manifest or DEFAULT_SYNTHETIC_MANIFEST
        result_path = args.results or DEFAULT_SYNTHETIC_RESULTS
        segments_root = args.segments_root or DEFAULT_SYNTHETIC_SEGMENTS_ROOT
        overlay_cases = SYNTHETIC_OVERLAY_CASES
    else:
        manifest_path = args.manifest or DEFAULT_MANIFEST
        result_path = args.results or DEFAULT_RESULTS
        segments_root = args.segments_root or DEFAULT_SEGMENTS_ROOT
        overlay_cases = OVERLAY_CASES

    generated = generate_overlays(
        manifest_path=manifest_path.resolve(),
        result_path=result_path.resolve(),
        segments_root=segments_root.resolve(),
        output_dir=args.output_dir.resolve(),
        matlab_bin=args.matlab_bin,
        evaluator_dir=args.evaluator_dir.resolve(),
        timeout_seconds=args.timeout_seconds,
        overlay_cases=overlay_cases,
    )
    print(f"Generated {len(generated)} overlay assets in {args.output_dir.resolve()}")

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
