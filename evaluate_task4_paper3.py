import argparse
import json
import multiprocessing
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from benchmark_checkpoint import (
    CsvCheckpoint,
    atomic_write_json,
    utc_timestamp,
)
from benchmark_metadata import (
    DEFAULT_WORKERS,
    build_run_configuration as _build_run_configuration,
    configuration_hash,
)
from line_pipeline import (
    detect_line_segments,
    line_segment_to_dict,
    read_manifest,
    select_manifest_rows,
)
from paper3_evaluation import (
    PAPER3_EVALUATION_SETTINGS,
    evaluate_detection_with_gt,
    load_segment_collection,
    pipeline_segments_to_collection,
)
from paper3_matlab_evaluation import (
    evaluate_detection_all_settings_matlab,
)


CSV_COLUMNS = (
    "image",
    "split",
    "model",
    "algorithm",
    "metric",
    "evaluator",
    "mode",
    "status",
    "failure_stage",
    "config_hash",
    "runtime_seconds",
    "raw_candidates",
    "retained_candidates",
    "candidate_cap",
    "cap_activated",
    "clusters",
    "objective",
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
)

ROW_KEY_FIELDS = ("image", "split", "model", "mode")
JOB_KEY_FIELDS = ("image", "split", "model")
RUN_SCHEMA_VERSION = 5
SEGMENT_ARTIFACT_SCHEMA_VERSION = 1
SAFE_ARTIFACT_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class ModelConfig:
    algorithm: str
    metric: str = "none"


@dataclass
class JobOutcome:
    rows: list
    segment_artifact: dict | None


MODEL_CONFIGS = {
    "2a": ModelConfig(
        algorithm="structure_tensor_grid",
    ),
    "2b": ModelConfig(
        algorithm="color_distribution_grid",
        metric="emd",
    ),
    "2c": ModelConfig(
        algorithm="structure_tensor_refined",
    ),
    "2d": ModelConfig(
        algorithm="color_distribution_refined",
        metric="emd",
    ),
}


DEFAULT_MODELS = (
    "2a",
    "2b",
    "2c",
    "2d",
)

MAX_CANDIDATES = 2000


def _experiment_overrides(args):
    detector_overrides = {}
    segment_overrides = {}
    emd_shortlist_k = getattr(args, "emd_shortlist_k", None)
    endpoint_padding = getattr(args, "endpoint_padding", None)

    if emd_shortlist_k is not None:
        for model_name in ("2b", "2d"):
            if model_name in args.models:
                detector_overrides[model_name] = {
                    "exact_top_k": int(emd_shortlist_k),
                }

    if endpoint_padding is not None:
        for model_name in args.models:
            segment_overrides[model_name] = {
                "endpoint_padding": float(endpoint_padding),
            }

    return detector_overrides, segment_overrides


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed report configuration for task 4 and evaluate "
            "the final line segments with the [3] protocol."
        )
    )
    parser.add_argument("--test-pack", default="test_pack")
    parser.add_argument("--output-csv", default="task4_paper3_results.csv")
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("synthetic", "york", "wireframe"),
        default=None,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of image/model jobs to execute in separate processes. "
            "The parent process remains the only checkpoint writer."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--emd-shortlist-k",
        type=int,
        default=None,
        help=(
            "Ablation override for the number of proxy-ranked directions "
            "evaluated with exact EMD in 2b and 2d."
        ),
    )
    parser.add_argument(
        "--endpoint-padding",
        type=float,
        default=None,
        help=(
            "Ablation override for endpoint padding in every selected model. "
            "Omit it to use the frozen report defaults."
        ),
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        "--append",
        dest="resume",
        action="store_true",
        help=(
            "Resume a compatible checkpoint, skip completed jobs, and retry "
            "failed or incomplete jobs. --append is retained as a safe alias."
        ),
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing CSV and metadata checkpoint.",
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Metadata sidecar path. Defaults to <output-csv>.metadata.json.",
    )
    parser.add_argument(
        "--segments-dir",
        default=None,
        help=(
            "Directory for one atomic final-segment JSON file per image/model. "
            "Defaults to <output-csv-stem>_segments."
        ),
    )
    parser.add_argument(
        "--evaluator",
        choices=("python", "matlab"),
        default="matlab",
        help=(
            "python uses the local reconstructed [3]-style evaluator; "
            "matlab uses the official [3] MATLAB/p-code evaluator."
        ),
    )
    parser.add_argument(
        "--matlab-bin",
        default=None,
        help="Path to MATLAB executable. Defaults to MATLAB_BIN, PATH, or the local R2026a install.",
    )
    parser.add_argument(
        "--paper3-evaluator-dir",
        default="external/line_segment_review/eva_survey_p",
        help="Path to the official [3] eva_survey_p directory.",
    )
    parser.add_argument(
        "--matlab-timeout-seconds",
        type=float,
        default=180.0,
        help="Timeout for each official MATLAB evaluation job.",
    )
    return parser.parse_args(argv)


def load_image(path):
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    return np.asarray(image, dtype=np.float64) / 255.0


def make_metric_rows(
    row,
    model_name,
    model,
    result,
    runtime,
    ground_truth,
    detected,
    evaluator="python",
    image=None,
    matlab_bin=None,
    paper3_evaluator_dir=None,
    matlab_timeout_seconds=180.0,
    config_hash="",
):
    rows = []

    if evaluator == "matlab":
        if image is None:
            raise ValueError("image is required for MATLAB [3] evaluation")

        metrics_by_mode = evaluate_detection_all_settings_matlab(
            ground_truth,
            detected,
            image,
            settings=PAPER3_EVALUATION_SETTINGS,
            matlab_bin=matlab_bin,
            evaluator_dir=paper3_evaluator_dir,
            timeout_seconds=matlab_timeout_seconds,
        )
    elif evaluator == "python":
        metrics_by_mode = {
            setting.name: evaluate_detection_with_gt(
                ground_truth,
                detected,
                setting,
            )
            for setting in PAPER3_EVALUATION_SETTINGS
        }
    else:
        raise ValueError("evaluator must be 'python' or 'matlab'")

    for setting in PAPER3_EVALUATION_SETTINGS:
        metrics = metrics_by_mode[setting.name]
        rows.append(
            {
                "image": row["id"],
                "split": row["split"],
                "model": model_name,
                "algorithm": model.algorithm,
                "metric": model.metric,
                "evaluator": evaluator,
                "mode": setting.name,
                "status": "ok",
                "failure_stage": "",
                "config_hash": config_hash,
                "runtime_seconds": runtime,
                "raw_candidates": result.raw_candidate_count,
                "retained_candidates": result.retained_candidate_count,
                "candidate_cap": result.candidate_cap,
                "cap_activated": result.candidate_cap_activated,
                "clusters": len(result.clusters),
                "objective": result.objective,
                "segments": metrics.detected_count,
                "gt_segments": metrics.gt_count,
                "matched": metrics.matched_pairs,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "mean_iou": metrics.mean_iou,
                "location_error_px": metrics.location_error_px,
                "orientation_error_deg": metrics.orientation_error_deg,
                "error": "",
            }
        )

    return rows


def error_rows(
    row,
    model_name,
    model,
    error,
    evaluator,
    config_hash,
    failure_stage,
    runtime="",
    result=None,
    ground_truth=None,
):
    rows = []
    raw_candidate_count = result.raw_candidate_count if result is not None else ""
    retained_candidate_count = (
        result.retained_candidate_count if result is not None else ""
    )
    candidate_cap = result.candidate_cap if result is not None else ""
    cap_activated = result.candidate_cap_activated if result is not None else ""
    cluster_count = len(result.clusters) if result is not None else 0
    segment_count = len(result.segments) if result is not None else 0
    objective = result.objective if result is not None else ""
    gt_count = len(ground_truth.values) if ground_truth is not None else 0

    for setting in PAPER3_EVALUATION_SETTINGS:
        rows.append(
            {
                "image": row["id"],
                "split": row["split"],
                "model": model_name,
                "algorithm": model.algorithm,
                "metric": model.metric,
                "evaluator": evaluator,
                "mode": setting.name,
                "status": "failed",
                "failure_stage": failure_stage,
                "config_hash": config_hash,
                "runtime_seconds": runtime,
                "raw_candidates": raw_candidate_count,
                "retained_candidates": retained_candidate_count,
                "candidate_cap": candidate_cap,
                "cap_activated": cap_activated,
                "clusters": cluster_count,
                "objective": objective,
                "segments": segment_count,
                "gt_segments": gt_count,
                "matched": 0,
                "precision": "",
                "recall": "",
                "f1": "",
                "mean_iou": "",
                "location_error_px": "",
                "orientation_error_deg": "",
                "error": error,
            }
        )

    return rows


def build_run_configuration(args, test_pack, manifest_rows=None):
    if manifest_rows is None:
        manifest_rows = read_manifest(test_pack)

    configuration = _build_run_configuration(
        args,
        test_pack,
        manifest_rows=manifest_rows,
        model_configs=MODEL_CONFIGS,
        max_candidates=MAX_CANDIDATES,
        evaluation_settings=PAPER3_EVALUATION_SETTINGS,
        schema_version=RUN_SCHEMA_VERSION,
    )
    detector_overrides, segment_overrides = _experiment_overrides(args)

    for model_name, overrides in detector_overrides.items():
        configuration["models"][model_name]["detector_parameters"].update(
            overrides
        )

    for model_name, overrides in segment_overrides.items():
        configuration["models"][model_name]["segment_parameters"].update(
            overrides
        )

    configuration["experiment_overrides"] = {
        "detector": detector_overrides,
        "segment": segment_overrides,
    }
    return configuration


def _metadata_path(args):
    if args.metadata_json:
        return Path(args.metadata_json)

    output_path = Path(args.output_csv)
    return output_path.with_suffix(output_path.suffix + ".metadata.json")


def _segments_directory(args):
    if getattr(args, "segments_dir", None):
        return Path(args.segments_dir)

    output_path = Path(args.output_csv)
    return output_path.with_name(f"{output_path.stem}_segments")


def _artifact_component(value, field):
    value = str(value)

    if not SAFE_ARTIFACT_COMPONENT.fullmatch(value):
        raise ValueError(
            f"Unsafe {field} value for segment artifact path: {value!r}"
        )

    return value


def _segment_artifact_path(segments_directory, row, model_name):
    split = _artifact_component(row["split"], "split")
    image_id = _artifact_component(row["id"], "image ID")
    model_name = _artifact_component(model_name, "model")
    return Path(segments_directory) / split / image_id / f"{model_name}.json"


def _prepare_segments_directory(path, resume=False, overwrite=False):
    path = Path(path)

    if path.exists() and not path.is_dir():
        raise ValueError(f"Segment artifact path is not a directory: {path}")

    if overwrite and path.exists():
        shutil.rmtree(path)

    if resume and not path.is_dir():
        raise ValueError(
            "Cannot safely resume because the segment artifact directory "
            f"is missing: {path}"
        )

    if not resume and not overwrite and path.exists():
        raise FileExistsError(
            "Segment artifact directory already exists. Use --resume or "
            f"--overwrite: {path}"
        )

    path.mkdir(parents=True, exist_ok=True)


def _segment_artifact_matches(path, row, model_name, config_hash):
    path = Path(path)

    if not path.is_file():
        return False

    try:
        with path.open() as file:
            artifact = json.load(file)
    except (OSError, ValueError):
        return False

    return (
        artifact.get("schema_version") == SEGMENT_ARTIFACT_SCHEMA_VERSION
        and artifact.get("config_hash") == config_hash
        and artifact.get("image") == row["id"]
        and artifact.get("split") == row["split"]
        and artifact.get("model") == model_name
        and isinstance(artifact.get("segments"), list)
    )


def _segment_artifact_data(
    row,
    model_name,
    model,
    result,
    runtime,
    config_hash,
):
    return {
        "schema_version": SEGMENT_ARTIFACT_SCHEMA_VERSION,
        "config_hash": config_hash,
        "image": row["id"],
        "split": row["split"],
        "model": model_name,
        "algorithm": model.algorithm,
        "metric": model.metric,
        "runtime_seconds": float(runtime),
        "raw_candidate_count": int(result.raw_candidate_count),
        "retained_candidate_count": int(result.retained_candidate_count),
        "candidate_cap": result.candidate_cap,
        "candidate_cap_activated": bool(result.candidate_cap_activated),
        "cluster_count": len(result.clusters),
        "objective": float(result.objective),
        "segments": [
            line_segment_to_dict(segment)
            for segment in result.segments
        ],
    }


def _write_segment_artifact(path, artifact):
    atomic_write_json(path, artifact)


def _result_sort_key(row):
    model_order = {name: index for index, name in enumerate(DEFAULT_MODELS)}
    mode_order = {
        setting.name: index
        for index, setting in enumerate(PAPER3_EVALUATION_SETTINGS)
    }
    return (
        row.get("split", ""),
        row.get("image", ""),
        model_order.get(row.get("model", ""), len(model_order)),
        mode_order.get(row.get("mode", ""), len(mode_order)),
    )


def _error_message(error, limit=4000):
    message = f"{type(error).__name__}: {error}"

    if len(message) > limit:
        return message[:limit] + " ... [truncated]"

    return message


def _execute_benchmark_job(
    row,
    model_name,
    test_pack,
    evaluator,
    matlab_bin,
    paper3_evaluator_dir,
    matlab_timeout_seconds,
    config_hash,
    detector_overrides,
    segment_overrides,
):
    model = MODEL_CONFIGS[model_name]
    test_pack = Path(test_pack)
    stage = "load_image"
    result = None
    ground_truth = None
    runtime = ""
    detection_start = None
    segment_artifact = None

    try:
        image = load_image(test_pack / row["image"])
        stage = "load_ground_truth"
        ground_truth = load_segment_collection(test_pack / row["gt"])
        stage = "detect"
        detection_start = time.perf_counter()
        result = detect_line_segments(
            image,
            algorithm=model.algorithm,
            detector_kwargs=detector_overrides.get(model_name),
            max_candidates=MAX_CANDIDATES,
            segment_kwargs=segment_overrides.get(model_name),
        )
        runtime = time.perf_counter() - detection_start
        stage = "serialize_segments"
        segment_artifact = _segment_artifact_data(
            row,
            model_name,
            model,
            result,
            runtime,
            config_hash,
        )
        stage = "evaluate"
        detected = pipeline_segments_to_collection(result.segments)
        rows = make_metric_rows(
            row,
            model_name,
            model,
            result,
            runtime,
            ground_truth,
            detected,
            evaluator=evaluator,
            image=image,
            matlab_bin=matlab_bin,
            paper3_evaluator_dir=paper3_evaluator_dir,
            matlab_timeout_seconds=matlab_timeout_seconds,
            config_hash=config_hash,
        )
    except Exception as error:
        if detection_start is not None and runtime == "":
            runtime = time.perf_counter() - detection_start

        rows = error_rows(
            row,
            model_name,
            model,
            _error_message(error),
            evaluator=evaluator,
            config_hash=config_hash,
            failure_stage=stage,
            runtime=runtime,
            result=result,
            ground_truth=ground_truth,
        )

    return JobOutcome(rows=rows, segment_artifact=segment_artifact)


def _worker_failure_outcome(
    row,
    model_name,
    error,
    evaluator,
    config_hash,
):
    return JobOutcome(
        rows=error_rows(
            row,
            model_name,
            MODEL_CONFIGS[model_name],
            _error_message(error),
            evaluator=evaluator,
            config_hash=config_hash,
            failure_stage="worker_process",
        ),
        segment_artifact=None,
    )


def _failed_rows_from_completed_rows(rows, error, failure_stage):
    message = _error_message(error)
    failed_rows = []

    for source in rows:
        row = dict(source)
        row.update(
            {
                "status": "failed",
                "failure_stage": failure_stage,
                "matched": 0,
                "precision": "",
                "recall": "",
                "f1": "",
                "mean_iou": "",
                "location_error_px": "",
                "orientation_error_deg": "",
                "error": message,
            }
        )
        failed_rows.append(row)

    return failed_rows


def _outcome_succeeded(outcome):
    return bool(outcome.rows) and all(
        row.get("status") == "ok" and not row.get("error")
        for row in outcome.rows
    )


def _checkpoint_outcome(
    checkpoint,
    segments_directory,
    row,
    model_name,
    outcome,
):
    artifact_path = _segment_artifact_path(
        segments_directory,
        row,
        model_name,
    )

    if _outcome_succeeded(outcome) and outcome.segment_artifact is None:
        outcome = JobOutcome(
            rows=_failed_rows_from_completed_rows(
                outcome.rows,
                RuntimeError("worker returned no final-segment artifact"),
                "worker_process",
            ),
            segment_artifact=None,
        )

    if outcome.segment_artifact is not None:
        try:
            _write_segment_artifact(
                artifact_path,
                outcome.segment_artifact,
            )
        except Exception as error:
            outcome = JobOutcome(
                rows=_failed_rows_from_completed_rows(
                    outcome.rows,
                    error,
                    "save_segments",
                ),
                segment_artifact=None,
            )

    checkpoint.checkpoint_job(outcome.rows)
    return outcome


def _print_outcome(model_name, row, outcome):
    if _outcome_succeeded(outcome):
        moderate = next(
            item
            for item in outcome.rows
            if item["mode"] == "moderate"
        )
        print(
            f"Completed {model_name} on {row['id']}: "
            f"candidates={moderate['raw_candidates']}"
            f"->{moderate['retained_candidates']} "
            f"segments={moderate['segments']} "
            f"matched={moderate['matched']} "
            f"moderate_f1={moderate['f1']:.6f} "
            f"runtime={moderate['runtime_seconds']:.2f}s",
            flush=True,
        )
        return

    failed = next(
        (item for item in outcome.rows if item.get("status") != "ok"),
        outcome.rows[0],
    )
    print(
        f"Failed {model_name} on {row['id']} during "
        f"{failed.get('failure_stage', 'unknown')}: "
        f"{failed.get('error', 'unknown error')}",
        flush=True,
    )


def _terminate_executor(executor):
    terminate_workers = getattr(executor, "terminate_workers", None)

    if terminate_workers is not None:
        terminate_workers()
    else:
        executor.shutdown(wait=False, cancel_futures=True)


def run_benchmark(args):
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    if args.matlab_timeout_seconds <= 0:
        raise ValueError("--matlab-timeout-seconds must be positive")

    if args.emd_shortlist_k is not None and args.emd_shortlist_k <= 0:
        raise ValueError("--emd-shortlist-k must be positive")

    if args.endpoint_padding is not None and args.endpoint_padding < 0.0:
        raise ValueError("--endpoint-padding must be non-negative")

    workers = int(getattr(args, "workers", DEFAULT_WORKERS))

    if workers <= 0:
        raise ValueError("--workers must be positive")

    args.workers = workers

    test_pack = Path(args.test_pack).resolve()
    all_manifest_rows = read_manifest(test_pack)
    manifest_rows = select_manifest_rows(
        all_manifest_rows,
        ids=args.ids,
        splits=args.splits,
        limit=args.limit,
    )

    if not manifest_rows:
        raise ValueError("No test-pack rows matched the requested filters.")

    configuration = build_run_configuration(args, test_pack, all_manifest_rows)
    config_hash = configuration_hash(configuration)
    detector_overrides, segment_overrides = _experiment_overrides(args)
    segments_directory = _segments_directory(args).resolve()
    checkpoint = CsvCheckpoint.open(
        output_path=args.output_csv,
        metadata_path=_metadata_path(args),
        columns=CSV_COLUMNS,
        row_key_fields=ROW_KEY_FIELDS,
        job_key_fields=JOB_KEY_FIELDS,
        expected_modes=tuple(
            setting.name
            for setting in PAPER3_EVALUATION_SETTINGS
        ),
        sort_key=_result_sort_key,
        schema_version=RUN_SCHEMA_VERSION,
        configuration=configuration,
        config_hash=config_hash,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    _prepare_segments_directory(
        segments_directory,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    checkpoint.metadata["artifacts"] = {
        "final_segments": {
            "directory": str(segments_directory),
            "format": "one JSON file per image/model",
            "schema_version": SEGMENT_ARTIFACT_SCHEMA_VERSION,
        }
    }
    requested_jobs = [
        (row, model_name)
        for model_name in args.models
        for row in manifest_rows
    ]
    invocation = {
        "started_at": utc_timestamp(),
        "finished_at": None,
        "status": "running",
        "command": [sys.executable, *sys.argv],
        "selection": {
            "ids": args.ids,
            "splits": args.splits,
            "limit": args.limit,
            "models": list(args.models),
        },
        "execution": {
            "workers": workers,
            "mode": "sequential" if workers == 1 else "process_pool",
            "job_unit": "one image/model pair",
        },
        "requested_jobs": len(requested_jobs),
        "skipped_complete_jobs": 0,
        "completed_jobs": 0,
        "failed_jobs": 0,
    }
    checkpoint.metadata.setdefault("invocations", []).append(invocation)
    checkpoint.save_metadata()

    pending_jobs = []

    for row, model_name in requested_jobs:
        job_key = (row["id"], row["split"], model_name)
        previous_status = checkpoint.job_status(job_key)
        artifact_path = _segment_artifact_path(
            segments_directory,
            row,
            model_name,
        )

        if previous_status == "complete" and _segment_artifact_matches(
            artifact_path,
            row,
            model_name,
            config_hash,
        ):
            invocation["skipped_complete_jobs"] += 1
            continue

        pending_jobs.append((row, model_name))

    print(
        f"Prepared {len(pending_jobs)} pending job(s); "
        f"skipped {invocation['skipped_complete_jobs']} completed job(s); "
        f"workers={workers}.",
        flush=True,
    )

    executor = None

    def commit_outcome(row, model_name, outcome):
        outcome = _checkpoint_outcome(
            checkpoint,
            segments_directory,
            row,
            model_name,
            outcome,
        )
        _print_outcome(model_name, row, outcome)

        if _outcome_succeeded(outcome):
            invocation["completed_jobs"] += 1
        else:
            invocation["failed_jobs"] += 1

    worker_arguments = {
        "test_pack": str(test_pack),
        "evaluator": args.evaluator,
        "matlab_bin": args.matlab_bin,
        "paper3_evaluator_dir": str(Path(args.paper3_evaluator_dir).resolve()),
        "matlab_timeout_seconds": args.matlab_timeout_seconds,
        "config_hash": config_hash,
        "detector_overrides": detector_overrides,
        "segment_overrides": segment_overrides,
    }

    try:
        if workers == 1:
            for row, model_name in pending_jobs:
                print(f"Running {model_name} on {row['id']} ...", flush=True)
                outcome = _execute_benchmark_job(
                    row,
                    model_name,
                    **worker_arguments,
                )
                commit_outcome(row, model_name, outcome)
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            future_jobs = {
                executor.submit(
                    _execute_benchmark_job,
                    row,
                    model_name,
                    **worker_arguments,
                ): (row, model_name)
                for row, model_name in pending_jobs
            }

            for future in as_completed(future_jobs):
                row, model_name = future_jobs[future]

                try:
                    outcome = future.result()
                except Exception as error:
                    outcome = _worker_failure_outcome(
                        row,
                        model_name,
                        error,
                        args.evaluator,
                        config_hash,
                    )

                commit_outcome(row, model_name, outcome)

            executor.shutdown(wait=True)
            executor = None
    except KeyboardInterrupt:
        if executor is not None:
            _terminate_executor(executor)

        invocation["status"] = "interrupted"
        invocation["finished_at"] = utc_timestamp()
        checkpoint.metadata["status"] = "interrupted"
        checkpoint.save_metadata()
        print("Interrupted; completed jobs remain checkpointed.", flush=True)
        return 130
    except BaseException:
        if executor is not None:
            _terminate_executor(executor)

        invocation["status"] = "aborted"
        invocation["finished_at"] = utc_timestamp()
        checkpoint.metadata["status"] = "aborted"
        checkpoint.save_metadata()
        raise

    failed_requested_jobs = 0

    for row, model_name in requested_jobs:
        job_key = (row["id"], row["split"], model_name)
        artifact_path = _segment_artifact_path(
            segments_directory,
            row,
            model_name,
        )

        if (
            checkpoint.job_status(job_key) != "complete"
            or not _segment_artifact_matches(
                artifact_path,
                row,
                model_name,
                config_hash,
            )
        ):
            failed_requested_jobs += 1

    invocation["finished_at"] = utc_timestamp()
    invocation["status"] = (
        "complete" if failed_requested_jobs == 0 else "complete_with_failures"
    )
    checkpoint.metadata["status"] = (
        "complete"
        if checkpoint.progress()["failed_jobs"] == 0
        else "complete_with_failures"
    )
    checkpoint.save_metadata()
    print(f"Checkpoint CSV: {checkpoint.output_path}")
    print(f"Run metadata: {checkpoint.metadata_path}")
    print(f"Final segment artifacts: {segments_directory}")

    if failed_requested_jobs:
        print(
            f"Finished with {failed_requested_jobs} failed requested job(s).",
            flush=True,
        )
        return 1

    print("All requested jobs completed successfully.", flush=True)
    return 0


def main(argv=None):
    try:
        return run_benchmark(parse_args(argv))
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
