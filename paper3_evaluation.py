import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class Paper3EvaluationSetting:
    name: str
    distance_threshold_px: float
    angle_threshold_deg: float
    length_ratio_threshold: float


@dataclass(frozen=True)
class SegmentCollection:
    values: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    midpoints: np.ndarray
    tangents: np.ndarray
    lengths: np.ndarray


@dataclass(frozen=True)
class Paper3EvaluationResult:
    matched_pairs: int
    detected_count: int
    gt_count: int
    precision: float
    recall: float
    f1: float
    mean_iou: float
    location_error_px: float
    orientation_error_deg: float
    index_pairs: tuple


PAPER3_EVALUATION_SETTINGS = (
    Paper3EvaluationSetting(
        name="strict",
        distance_threshold_px=1.0,
        angle_threshold_deg=5.0,
        length_ratio_threshold=0.75,
    ),
    Paper3EvaluationSetting(
        name="moderate",
        distance_threshold_px=3.0,
        angle_threshold_deg=10.0,
        length_ratio_threshold=0.75,
    ),
    Paper3EvaluationSetting(
        name="loose",
        distance_threshold_px=5.0,
        angle_threshold_deg=20.0,
        length_ratio_threshold=0.50,
    ),
)

PAPER3_EVALUATION_SETTINGS_BY_NAME = {
    setting.name: setting
    for setting in PAPER3_EVALUATION_SETTINGS
}


def make_segment_collection(segments):
    values = np.asarray(segments, dtype=np.float64)

    if values.size == 0:
        empty_points = np.empty((0, 2), dtype=np.float64)
        return SegmentCollection(
            values=np.empty((0, 4), dtype=np.float64),
            starts=empty_points,
            ends=empty_points,
            midpoints=empty_points,
            tangents=empty_points,
            lengths=np.empty(0, dtype=np.float64),
        )

    values = values.reshape(-1, 4)
    starts = values[:, :2]
    ends = values[:, 2:4]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    keep = lengths > EPS

    values = values[keep]
    starts = starts[keep]
    ends = ends[keep]
    vectors = vectors[keep]
    lengths = lengths[keep]

    if len(values) == 0:
        empty_points = np.empty((0, 2), dtype=np.float64)
        return SegmentCollection(
            values=np.empty((0, 4), dtype=np.float64),
            starts=empty_points,
            ends=empty_points,
            midpoints=empty_points,
            tangents=empty_points,
            lengths=np.empty(0, dtype=np.float64),
        )

    tangents = vectors / lengths[:, None]
    midpoints = 0.5 * (starts + ends)

    return SegmentCollection(
        values=values,
        starts=starts,
        ends=ends,
        midpoints=midpoints,
        tangents=tangents,
        lengths=lengths,
    )


def load_segment_collection(path, scale_x=1.0, scale_y=1.0):
    path = Path(path)

    with path.open("r") as file:
        data = json.load(file)

    values = np.asarray(data.get("segments", []), dtype=np.float64)

    if values.size:
        values = values.reshape(-1, 4)
        values[:, [0, 2]] *= scale_x
        values[:, [1, 3]] *= scale_y

    return make_segment_collection(values)


def pipeline_segments_to_collection(segments):
    return make_segment_collection(
        [
            [
                float(segment.start[0]),
                float(segment.start[1]),
                float(segment.end[0]),
                float(segment.end[1]),
            ]
            for segment in segments
        ]
    )


def _angle_errors_deg(ref, test):
    dots = np.abs(ref.tangents @ test.tangents.T)
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _directional_midpoint_perpendicular_distances(ref, test):
    distances = np.zeros((len(ref.values), len(test.values)), dtype=np.float64)

    for test_index in range(len(test.values)):
        tangent = test.tangents[test_index]
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        delta = ref.midpoints - test.midpoints[test_index][None, :]
        distances[:, test_index] = np.abs(delta @ normal)

    return distances


def _directional_overlap_ratios(ref, test):
    ratios = np.zeros((len(ref.values), len(test.values)), dtype=np.float64)

    for ref_index in range(len(ref.values)):
        axis = ref.tangents[ref_index]
        origin = ref.starts[ref_index]
        ref_min = 0.0
        ref_max = ref.lengths[ref_index]

        test_start_projection = (test.starts - origin[None, :]) @ axis
        test_end_projection = (test.ends - origin[None, :]) @ axis
        test_min = np.minimum(test_start_projection, test_end_projection)
        test_max = np.maximum(test_start_projection, test_end_projection)
        overlap = np.maximum(
            0.0,
            np.minimum(ref_max, test_max) - np.maximum(ref_min, test_min),
        )
        ratios[ref_index, :] = overlap / max(ref.lengths[ref_index], EPS)

    return ratios


def _eva_unit(ref, test):
    shape = (len(ref.values), len(test.values))

    if shape[0] == 0 or shape[1] == 0:
        zeros = np.zeros(shape, dtype=np.float64)
        return zeros, zeros, zeros

    return (
        _directional_midpoint_perpendicular_distances(ref, test),
        _angle_errors_deg(ref, test),
        _directional_overlap_ratios(ref, test),
    )


def evaluate_detection_with_gt(ground_truth, detected, setting):
    """Evaluate detected line segments with a close port of [3].

    The public [3] code exposes the matching wrapper in eva_detection_w_gt.m:
    bidirectional eva_unit, average distance/angle/IoU matrices, mutual
    maximum-IoU pairs, then threshold filtering. The underlying eva_unit helper
    is released only as MATLAB p-code, so this Python version reconstructs its
    distance/angle/overlap matrices from the paper description and helper names.
    """

    if isinstance(setting, str):
        setting = PAPER3_EVALUATION_SETTINGS_BY_NAME[setting]

    gt_count = len(ground_truth.values)
    detected_count = len(detected.values)

    if gt_count == 0 or detected_count == 0:
        return Paper3EvaluationResult(
            matched_pairs=0,
            detected_count=detected_count,
            gt_count=gt_count,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            mean_iou=setting.length_ratio_threshold,
            location_error_px=setting.distance_threshold_px,
            orientation_error_deg=setting.angle_threshold_deg,
            index_pairs=(),
        )

    mat_dist_1, mat_angle_1, mat_iou_1 = _eva_unit(ground_truth, detected)
    mat_dist_2, mat_angle_2, mat_iou_2 = _eva_unit(detected, ground_truth)

    mat_dist = (mat_dist_1 + mat_dist_2.T) / 2.0
    mat_angle = (mat_angle_1 + mat_angle_2.T) / 2.0
    mat_iou = (mat_iou_1 + mat_iou_2.T) / 2.0

    best_gt_for_detection = np.argmax(mat_iou, axis=0)
    best_detection_for_gt = np.argmax(mat_iou, axis=1)
    index_pairs = []

    for detected_index, gt_index in enumerate(best_gt_for_detection):
        gt_index = int(gt_index)

        if int(best_detection_for_gt[gt_index]) != detected_index:
            continue

        iou = float(mat_iou[gt_index, detected_index])
        distance = float(mat_dist[gt_index, detected_index])
        angle = float(mat_angle[gt_index, detected_index])

        if iou < setting.length_ratio_threshold:
            continue

        if distance > setting.distance_threshold_px:
            continue

        if angle > setting.angle_threshold_deg:
            continue

        index_pairs.append(
            (
                gt_index,
                int(detected_index),
                iou,
                distance,
                angle,
            )
        )

    matched_pairs = len(index_pairs)
    precision = matched_pairs / detected_count if detected_count else 0.0
    recall = matched_pairs / gt_count if gt_count else 0.0

    if precision + recall > 0.0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    if matched_pairs:
        mean_iou = float(np.mean([pair[2] for pair in index_pairs]))
        location_error_px = float(np.mean([pair[3] for pair in index_pairs]))
        orientation_error_deg = float(np.mean([pair[4] for pair in index_pairs]))
    else:
        mean_iou = setting.length_ratio_threshold
        location_error_px = setting.distance_threshold_px
        orientation_error_deg = setting.angle_threshold_deg

    return Paper3EvaluationResult(
        matched_pairs=matched_pairs,
        detected_count=detected_count,
        gt_count=gt_count,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=mean_iou,
        location_error_px=location_error_px,
        orientation_error_deg=orientation_error_deg,
        index_pairs=tuple(index_pairs),
    )


def evaluate_detection_all_settings(ground_truth, detected):
    return {
        setting.name: evaluate_detection_with_gt(
            ground_truth,
            detected,
            setting,
        )
        for setting in PAPER3_EVALUATION_SETTINGS
    }
