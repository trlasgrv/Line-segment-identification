import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from paper3_evaluation import (
    PAPER3_EVALUATION_SETTINGS,
    PAPER3_EVALUATION_SETTINGS_BY_NAME,
    Paper3EvaluationResult,
)


DEFAULT_LOCAL_MATLAB = Path(
    "/home/vanilaski/Applications/MATLAB/R2026a/bin/matlab"
)
DEFAULT_EVALUATOR_DIR = Path(
    "external/line_segment_review/eva_survey_p"
)
BRIDGE_SCRIPT_DIR = Path(__file__).resolve().parent / "matlab"


def find_matlab_executable(matlab_bin=None):
    if matlab_bin:
        return str(Path(matlab_bin).expanduser())

    env_value = os.environ.get("MATLAB_BIN")

    if env_value:
        return str(Path(env_value).expanduser())

    found = shutil.which("matlab")

    if found:
        return found

    if DEFAULT_LOCAL_MATLAB.exists():
        return str(DEFAULT_LOCAL_MATLAB)

    raise FileNotFoundError(
        "Could not find MATLAB. Set MATLAB_BIN or pass --matlab-bin."
    )


def matlab_environment():
    env = os.environ.copy()
    env.setdefault("MATLAB_DISABLE_SERVICEHOST", "true")
    env.setdefault("QT_QPA_PLATFORM", "xcb")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    return env


def image_to_matlab_uint8(image):
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    if np.issubdtype(image.dtype, np.integer):
        return np.clip(image, 0, 255).astype(np.uint8, copy=False)

    image_float = image.astype(np.float64, copy=False)

    if image_float.size and float(np.nanmax(image_float)) <= 1.0:
        image_float = image_float * 255.0

    return np.clip(np.rint(image_float), 0, 255).astype(np.uint8)


def _matlab_string(value):
    value = str(value).replace("'", "''")
    return "'" + value + "'"


def _line_values(collection):
    values = np.asarray(collection.values, dtype=np.float64)

    if values.size == 0:
        return np.zeros((0, 4), dtype=np.float64)

    return values.reshape(-1, 4)


def _as_float_vector(mat_data, key):
    return np.ravel(np.asarray(mat_data[key], dtype=np.float64))


def _index_pairs_at(index_pairs_cell, index):
    value = index_pairs_cell[index]

    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.float64)
    else:
        array = np.asarray(value.item(), dtype=np.float64)

    if array.size == 0:
        return ()

    array = array.reshape(-1, array.shape[-1])

    return tuple(
        tuple(float(item) for item in row)
        for row in array
    )


def evaluate_detection_all_settings_matlab(
    ground_truth,
    detected,
    image,
    settings=PAPER3_EVALUATION_SETTINGS,
    matlab_bin=None,
    evaluator_dir=None,
    timeout_seconds=180,
    keep_temp=False,
):
    """Evaluate with the official [3] MATLAB implementation.

    This calls process_results and eva_detection_w_gt from the official
    line_segment_review/eva_survey_p directory. The p-code eva_unit helper is
    therefore used directly by MATLAB instead of the local Python
    reconstruction.
    """

    matlab_bin = find_matlab_executable(matlab_bin)
    evaluator_dir = Path(evaluator_dir or DEFAULT_EVALUATOR_DIR).resolve()

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    if not evaluator_dir.exists():
        raise FileNotFoundError(
            "Official [3] evaluator directory not found: "
            + str(evaluator_dir)
        )

    if not (evaluator_dir / "eva_unit.p").exists():
        raise FileNotFoundError(
            "Official eva_unit.p not found in " + str(evaluator_dir)
        )

    if not (evaluator_dir / "eva_detection_w_gt.m").exists():
        raise FileNotFoundError(
            "Official eva_detection_w_gt.m not found in " + str(evaluator_dir)
        )

    with tempfile.TemporaryDirectory(
        prefix="paper3_matlab_eval_",
        dir="/tmp",
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / "input.mat"
        output_path = temp_dir / "output.mat"

        savemat(
            input_path,
            {
                "image": image_to_matlab_uint8(image),
                "gt_lines": _line_values(ground_truth),
                "test_lines": _line_values(detected),
                "thres_dist": np.asarray(
                    [setting.distance_threshold_px for setting in settings],
                    dtype=np.float64,
                ),
                "thres_ang": np.asarray(
                    [setting.angle_threshold_deg for setting in settings],
                    dtype=np.float64,
                ),
                "thres_length_ratio": np.asarray(
                    [setting.length_ratio_threshold for setting in settings],
                    dtype=np.float64,
                ),
            },
        )

        command_text = (
            "addpath("
            + _matlab_string(BRIDGE_SCRIPT_DIR)
            + "); paper3_official_eval_bridge("
            + _matlab_string(input_path)
            + ", "
            + _matlab_string(output_path)
            + ", "
            + _matlab_string(evaluator_dir)
            + ");"
        )
        try:
            completed = subprocess.run(
                [
                    matlab_bin,
                    "-batch",
                    command_text,
                ],
                env=matlab_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "MATLAB official [3] evaluation exceeded "
                + str(timeout_seconds)
                + " seconds"
            ) from error

        if completed.returncode != 0:
            raise RuntimeError(
                "MATLAB official [3] evaluation failed with exit code "
                + str(completed.returncode)
                + ":\n"
                + completed.stdout
            )

        mat_data = loadmat(output_path, squeeze_me=False)

        if keep_temp:
            print("MATLAB evaluation temp dir:", temp_dir)

        matched = _as_float_vector(mat_data, "matched")
        precision = _as_float_vector(mat_data, "precision")
        recall = _as_float_vector(mat_data, "recall")
        f1 = _as_float_vector(mat_data, "f1")
        iou = _as_float_vector(mat_data, "iou")
        le = _as_float_vector(mat_data, "le")
        le_a = _as_float_vector(mat_data, "le_a")
        gt_count = int(np.asarray(mat_data["gt_count"]).reshape(-1)[0])
        detected_count = int(
            np.asarray(mat_data["detected_count"]).reshape(-1)[0]
        )
        index_pairs_cell = np.ravel(mat_data["index_pairs"])

        return {
            setting.name: Paper3EvaluationResult(
                matched_pairs=int(round(matched[index])),
                detected_count=detected_count,
                gt_count=gt_count,
                precision=float(precision[index]),
                recall=float(recall[index]),
                f1=float(f1[index]),
                mean_iou=float(iou[index]),
                location_error_px=float(le[index]),
                orientation_error_deg=float(le_a[index]),
                index_pairs=_index_pairs_at(index_pairs_cell, index),
            )
            for index, setting in enumerate(settings)
        }


def evaluate_detection_with_gt_matlab(
    ground_truth,
    detected,
    image,
    setting,
    matlab_bin=None,
    evaluator_dir=None,
    timeout_seconds=180,
):
    if isinstance(setting, str):
        setting = PAPER3_EVALUATION_SETTINGS_BY_NAME[setting]

    metrics = evaluate_detection_all_settings_matlab(
        ground_truth,
        detected,
        image,
        settings=(setting,),
        matlab_bin=matlab_bin,
        evaluator_dir=evaluator_dir,
        timeout_seconds=timeout_seconds,
    )

    return metrics[setting.name]
