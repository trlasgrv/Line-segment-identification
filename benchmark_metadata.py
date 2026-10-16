import hashlib
import inspect
import json
import platform
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import PIL

import color_distribution
import color_refinement
import structure_refinement
from clique_partitioning import solve_gaec_klj
from detectors import get_detector
from line_pipeline import (
    build_clique_partition_instance,
    segment_kwargs_with_defaults,
)
from paper3_matlab_evaluation import find_matlab_executable


DEFAULT_WORKERS = 3


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]

    return value


def _callable_defaults(function):
    return {
        name: _json_value(parameter.default)
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _file_sha256(path):
    path = Path(path)

    if not path.is_file():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _file_record(path):
    path = Path(path).resolve()
    record = {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _file_sha256(path),
    }

    if path.exists():
        stat = path.stat()
        record["size_bytes"] = stat.st_size
        record["modified_ns"] = stat.st_mtime_ns

    return record


def _package_version(distribution_name):
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _source_file_records():
    root = Path(__file__).resolve().parent
    relative_paths = (
        "benchmark_checkpoint.py",
        "benchmark_metadata.py",
        "clique_partitioning.py",
        "color_distribution.py",
        "color_grid_candidates.py",
        "color_refinement.py",
        "detectors.py",
        "evaluate_task4_paper3.py",
        "geometry.py",
        "grid_candidates.py",
        "line_pipeline.py",
        "paper3_evaluation.py",
        "paper3_matlab_evaluation.py",
        "similarity.py",
        "structure_refinement.py",
        "structure_tensor.py",
        "matlab/paper3_official_eval_bridge.m",
        "matlab/RotatedRect_.m",
    )
    return {
        relative_path: _file_record(root / relative_path)
        for relative_path in relative_paths
    }


def _detector_runtime_overrides(algorithm):
    if algorithm == "structure_tensor_grid":
        return {"return_scores": True}

    if algorithm == "color_distribution_grid":
        return {"return_arrays": False, "return_scores": True}

    if algorithm in (
        "structure_tensor_refined",
        "color_distribution_refined",
    ):
        return {"return_scores": True}

    return {}


def _model_configuration(model_configs):
    models = {}

    for model_name, model in model_configs.items():
        detector_parameters = _callable_defaults(get_detector(model.algorithm))
        detector_parameters.update(_detector_runtime_overrides(model.algorithm))
        models[model_name] = {
            "algorithm": model.algorithm,
            "metric": model.metric,
            "detector_parameters": detector_parameters,
            "segment_parameters": segment_kwargs_with_defaults(
                model.algorithm,
                detector_kwargs={},
                segment_kwargs={},
            ),
        }

    return _json_value(models)


def _evaluator_configuration(args):
    if args.evaluator == "python":
        return {
            "name": "python",
            "implementation": "local reconstructed [3]-style evaluator",
            "module": _file_record(Path(__file__).resolve().parent / "paper3_evaluation.py"),
        }

    matlab_executable = Path(find_matlab_executable(args.matlab_bin)).resolve()
    evaluator_dir = Path(args.paper3_evaluator_dir).resolve()
    evaluator_files = (
        "eva_detection_w_gt.m",
        "eva_detection_w_gt.p",
        "eva_unit.p",
        "process_results.p",
        "process_results_desc.p",
        "RotatedRect_.mexw64",
    )

    return {
        "name": "matlab",
        "implementation": "official [3] MATLAB/p-code evaluator",
        "matlab_executable": _file_record(matlab_executable),
        "evaluator_directory": str(evaluator_dir),
        "evaluator_files": {
            name: _file_record(evaluator_dir / name)
            for name in evaluator_files
        },
        "bridge_files": {
            name: _file_record(Path(__file__).resolve().parent / "matlab" / name)
            for name in ("paper3_official_eval_bridge.m", "RotatedRect_.m")
        },
        "timeout_seconds": args.matlab_timeout_seconds,
    }


def build_run_configuration(
    args,
    test_pack,
    manifest_rows,
    model_configs,
    max_candidates,
    evaluation_settings,
    schema_version,
):
    test_pack = Path(test_pack).resolve()
    manifest_path = test_pack / "manifest.csv"
    workers = int(getattr(args, "workers", DEFAULT_WORKERS))
    opencv_version = getattr(color_distribution._cv2, "__version__", None)
    skimage_available = find_spec("skimage") is not None
    configuration = {
        "schema_version": schema_version,
        "dataset": {
            "test_pack": str(test_pack),
            "manifest": _file_record(manifest_path),
            "entries": [
                {
                    "id": row["id"],
                    "split": row["split"],
                    "image": _file_record(test_pack / row["image"]),
                    "ground_truth": _file_record(test_pack / row["gt"]),
                }
                for row in manifest_rows
            ],
        },
        "models": _model_configuration(model_configs),
        "pipeline": {
            "candidate_cap": {
                "maximum": max_candidates,
                "selection": "highest native detector confidence",
                "tie_break": "stable detector output order",
            },
            "similarity_parameters": _callable_defaults(
                build_clique_partition_instance
            ),
            "partition_solver": "GAEC + KLj from [2]",
            "partition_parameters": _callable_defaults(solve_gaec_klj),
        },
        "evaluation": {
            "settings": [asdict(setting) for setting in evaluation_settings],
            "backend": _evaluator_configuration(args),
        },
        "execution": {
            "workers": workers,
            "mode": (
                "sequential" if workers == 1 else "process_pool"
            ),
            "job_unit": "one image/model pair",
            "runtime_measurement": (
                "per-job detector wall time under the configured worker load"
            ),
        },
        "numerical_backends": {
            "emd": {
                "primary": (
                    "opencv.EMD"
                    if color_distribution.HAS_OPENCV_EMD
                    else "scipy.optimize.linprog"
                ),
                "fallback": "scipy.optimize.linprog",
                "opencv_available": color_distribution.HAS_OPENCV_EMD,
                "opencv_version": opencv_version,
            },
            "lab_conversion": {
                "selected": (
                    "skimage.color.rgb2lab"
                    if skimage_available
                    else "internal CIE Lab fallback"
                ),
                "scikit_image_available": skimage_available,
            },
            "spatial_nms": {
                "2c": (
                    "scipy.spatial.cKDTree"
                    if structure_refinement.cKDTree is not None
                    else "NumPy fallback"
                ),
                "2d": (
                    "scipy.spatial.cKDTree"
                    if color_refinement.cKDTree is not None
                    else "NumPy fallback"
                ),
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                "numpy": np.__version__,
                "Pillow": PIL.__version__,
                "scipy": _package_version("scipy"),
                "scikit-image": _package_version("scikit-image"),
                "opencv-python": (
                    _package_version("opencv-python")
                    or _package_version("opencv-python-headless")
                ),
            },
        },
        "source_files": _source_file_records(),
    }
    return _json_value(configuration)


def configuration_hash(configuration):
    canonical = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
