import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from clique_partitioning import (
    labels_to_clusters,
    partition_objective,
    solve_gaec_klj,
)
from detectors import available_detectors, cap_candidates, run_candidate_detector
from geometry import (
    EPS,
    candidates_to_arrays,
    normalize_vector,
)
from similarity import clique_partition_weights


@dataclass(frozen=True)
class CliquePartitionInstance:
    weights: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True)
class LineSegment:
    start: np.ndarray
    end: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    length: float
    score: float
    support: int
    cluster_index: int


@dataclass(frozen=True)
class Task4Result:
    candidates: list
    candidate_scores: np.ndarray | None
    raw_candidate_count: int
    retained_candidate_count: int
    candidate_cap: int | None
    candidate_cap_activated: bool
    labels: np.ndarray
    clusters: list
    segments: list
    objective: float
    weights: np.ndarray | None = None
    probabilities: np.ndarray | None = None


def build_clique_partition_instance(
    candidates,
    distance_scale=2.5,
    angle_scale=np.deg2rad(12.0),
    tangent_scale=None,
    eps=1e-6,
):
    weights, probabilities = clique_partition_weights(
        candidates,
        distance_scale=distance_scale,
        angle_scale=angle_scale,
        tangent_scale=tangent_scale,
        eps=eps,
        return_probabilities=True,
    )

    return CliquePartitionInstance(
        weights=weights,
        probabilities=probabilities,
    )


def _cluster_weights(candidate_scores, indices):
    if candidate_scores is None:
        return np.ones(len(indices), dtype=np.float64)

    scores = np.asarray(candidate_scores, dtype=np.float64)[indices]
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.clip(scores, 0.0, None)

    if float(np.sum(scores)) <= EPS:
        return np.ones(len(indices), dtype=np.float64)

    return scores


def _clip_t_interval_to_image(origin, tangent, t_min, t_max, image_shape):
    if image_shape is None:
        return t_min, t_max

    height, width = image_shape[:2]
    bounds = ((0.0, width - 1.0), (0.0, height - 1.0))

    for axis, (low, high) in enumerate(bounds):
        direction = float(tangent[axis])
        coordinate = float(origin[axis])

        if abs(direction) <= EPS:
            if coordinate < low - EPS or coordinate > high + EPS:
                return None

            continue

        t0 = (low - coordinate) / direction
        t1 = (high - coordinate) / direction

        if t0 > t1:
            t0, t1 = t1, t0

        t_min = max(t_min, t0)
        t_max = min(t_max, t1)

        if t_max < t_min:
            return None

    return t_min, t_max


def cluster_to_line_segment(
    positions,
    normals,
    cluster,
    cluster_index,
    candidate_scores=None,
    image_shape=None,
    min_segment_length=6.0,
    endpoint_quantile=0.0,
    endpoint_padding=0.0,
):
    if min_segment_length < 0.0:
        raise ValueError("min_segment_length must be non-negative")

    if not (0.0 <= endpoint_quantile < 0.5):
        raise ValueError("endpoint_quantile must be in [0, 0.5)")

    if endpoint_padding < 0.0:
        raise ValueError("endpoint_padding must be non-negative")

    cluster = np.asarray(cluster, dtype=np.int64)
    support = len(cluster)

    if support == 0:
        return None

    cluster_positions = positions[cluster]
    cluster_normals = normals[cluster].copy()
    weights = _cluster_weights(candidate_scores, cluster)
    reference_normal = cluster_normals[0]
    signs = np.where(cluster_normals @ reference_normal < 0.0, -1.0, 1.0)
    cluster_normals *= signs[:, None]

    weight_total = float(np.sum(weights))
    average_normal = np.sum(cluster_normals * weights[:, None], axis=0)

    if np.linalg.norm(average_normal) <= EPS:
        average_normal = reference_normal

    normal = normalize_vector(average_normal)
    origin = np.sum(cluster_positions * weights[:, None], axis=0) / weight_total
    tangent = np.array([-normal[1], normal[0]], dtype=np.float64)
    projections = (cluster_positions - origin[None, :]) @ tangent

    if endpoint_quantile > 0.0 and support >= 4:
        t_min = float(np.quantile(projections, endpoint_quantile))
        t_max = float(np.quantile(projections, 1.0 - endpoint_quantile))
    else:
        t_min = float(np.min(projections))
        t_max = float(np.max(projections))

    t_min -= endpoint_padding
    t_max += endpoint_padding

    clipped = _clip_t_interval_to_image(
        origin,
        tangent,
        t_min,
        t_max,
        image_shape,
    )

    if clipped is None:
        return None

    t_min, t_max = clipped
    length = float(t_max - t_min)

    if length < min_segment_length:
        return None

    start = origin + t_min * tangent
    end = origin + t_max * tangent
    residuals = np.abs((cluster_positions - origin[None, :]) @ normal)
    localization_score = float(np.exp(-np.mean(residuals)))
    score = float(np.mean(weights) * localization_score)

    return LineSegment(
        start=start,
        end=end,
        normal=normal,
        tangent=tangent,
        length=length,
        score=score,
        support=int(support),
        cluster_index=int(cluster_index),
    )


def clusters_to_line_segments(
    candidates,
    clusters,
    candidate_scores=None,
    image_shape=None,
    min_cluster_size=3,
    min_segment_length=6.0,
    min_segment_score=0.0,
    endpoint_quantile=0.0,
    endpoint_padding=0.0,
    max_segments=None,
):
    if min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be positive")

    if min_segment_score < 0.0:
        raise ValueError("min_segment_score must be non-negative")

    if max_segments is not None and max_segments <= 0:
        raise ValueError("max_segments must be positive or None")

    positions, normals = candidates_to_arrays(candidates)
    segments = []

    for cluster_index, cluster in enumerate(clusters):
        if len(cluster) < min_cluster_size:
            continue

        segment = cluster_to_line_segment(
            positions,
            normals,
            cluster,
            cluster_index,
            candidate_scores=candidate_scores,
            image_shape=image_shape,
            min_segment_length=min_segment_length,
            endpoint_quantile=endpoint_quantile,
            endpoint_padding=endpoint_padding,
        )

        if segment is not None:
            if segment.score < min_segment_score:
                continue

            segments.append(segment)

    segments.sort(
        key=lambda segment: (
            segment.score * np.sqrt(segment.support),
            segment.length,
        ),
        reverse=True,
    )

    if max_segments is not None:
        segments = segments[:int(max_segments)]

    return segments


DEFAULT_SEGMENT_KWARGS = {
    "min_cluster_size": 3,
    "min_segment_length": 15.0,
    "min_segment_score": 0.0,
    "endpoint_quantile": 0.0,
    "endpoint_padding": 0.0,
    "max_segments": None,
}

DEFAULT_ENDPOINT_PADDING_BY_ALGORITHM = {
    "structure_tensor_grid": 0.0,
    "color_distribution_grid": 8.0,
    "structure_tensor_refined": 4.0,
    "color_distribution_refined": 8.0,
}


def segment_kwargs_with_defaults(
    algorithm,
    detector_kwargs,
    segment_kwargs,
):
    resolved = dict(DEFAULT_SEGMENT_KWARGS)
    resolved["endpoint_padding"] = DEFAULT_ENDPOINT_PADDING_BY_ALGORITHM.get(
        algorithm,
        resolved["endpoint_padding"],
    )

    for key, value in dict(segment_kwargs or {}).items():
        if value is not None:
            resolved[key] = value

    return resolved


def detect_line_segments(
    image,
    algorithm="color_distribution_refined",
    detector_kwargs=None,
    max_candidates=2000,
    similarity_kwargs=None,
    partition_kwargs=None,
    segment_kwargs=None,
    return_instance=False,
):
    detector_kwargs = dict(detector_kwargs or {})
    candidates, scores = run_candidate_detector(
        algorithm,
        image,
        detector_kwargs=detector_kwargs,
    )
    raw_candidate_count = len(candidates)
    candidates, scores = cap_candidates(
        candidates,
        scores=scores,
        max_candidates=max_candidates,
    )
    retained_candidate_count = len(candidates)
    candidate_cap_activated = retained_candidate_count < raw_candidate_count

    similarity_kwargs = dict(similarity_kwargs or {})
    partition_kwargs = dict(partition_kwargs or {})
    segment_kwargs = segment_kwargs_with_defaults(
        algorithm,
        detector_kwargs,
        segment_kwargs,
    )

    instance = build_clique_partition_instance(
        candidates,
        **similarity_kwargs,
    )
    labels = solve_gaec_klj(
        instance.weights,
        **partition_kwargs,
    )
    clusters = labels_to_clusters(labels)
    segments = clusters_to_line_segments(
        candidates,
        clusters,
        candidate_scores=scores,
        image_shape=image.shape,
        **segment_kwargs,
    )

    objective = partition_objective(instance.weights, labels)

    if return_instance:
        return Task4Result(
            candidates=candidates,
            candidate_scores=scores,
            raw_candidate_count=raw_candidate_count,
            retained_candidate_count=retained_candidate_count,
            candidate_cap=max_candidates,
            candidate_cap_activated=candidate_cap_activated,
            labels=labels,
            clusters=clusters,
            segments=segments,
            objective=objective,
            weights=instance.weights,
            probabilities=instance.probabilities,
        )

    return Task4Result(
        candidates=candidates,
        candidate_scores=scores,
        raw_candidate_count=raw_candidate_count,
        retained_candidate_count=retained_candidate_count,
        candidate_cap=max_candidates,
        candidate_cap_activated=candidate_cap_activated,
        labels=labels,
        clusters=clusters,
        segments=segments,
        objective=objective,
    )


def line_segment_to_dict(segment):
    return {
        "start": [float(segment.start[0]), float(segment.start[1])],
        "end": [float(segment.end[0]), float(segment.end[1])],
        "normal": [float(segment.normal[0]), float(segment.normal[1])],
        "length": float(segment.length),
        "score": float(segment.score),
        "support": int(segment.support),
        "cluster_index": int(segment.cluster_index),
    }


def load_image(path):
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    return np.asarray(image, dtype=np.float64) / 255.0


def read_manifest(test_pack):
    manifest_path = Path(test_pack) / "manifest.csv"

    with manifest_path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"id", "split", "image", "gt"}
        missing_columns = required_columns - set(reader.fieldnames or ())

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Manifest {manifest_path} is missing required columns: {missing}"
            )

        rows = list(reader)

    seen = set()

    for row in rows:
        key = (row["id"], row["split"])

        if key in seen:
            raise ValueError(f"Duplicate manifest row for id/split: {key}")

        seen.add(key)

    return rows


def select_manifest_rows(rows, ids=None, splits=None, limit=None):
    selected = []
    requested_ids = list(ids) if ids is not None else None
    ids = set(requested_ids) if requested_ids is not None else None
    splits = set(splits) if splits else None

    if requested_ids is not None:
        if len(requested_ids) != len(ids):
            raise ValueError("Requested manifest IDs contain duplicates")

        available_ids = {row["id"] for row in rows}
        missing_ids = sorted(ids - available_ids)

        if missing_ids:
            raise ValueError(
                "Requested manifest IDs were not found: "
                + ", ".join(missing_ids)
            )

    for row in rows:
        if ids is not None and row["id"] not in ids:
            continue

        if splits is not None and row["split"] not in splits:
            continue

        selected.append(row)

    if ids is not None and splits is not None:
        selected_ids = {row["id"] for row in selected}
        excluded_ids = sorted(ids - selected_ids)

        if excluded_ids:
            raise ValueError(
                "Requested manifest IDs were excluded by the split filter: "
                + ", ".join(excluded_ids)
            )

    if limit is not None:
        selected = selected[:limit]

    return selected


def _json_line_for_result(image_id, algorithm, result, runtime):
    return {
        "image_id": image_id,
        "algorithm": algorithm,
        "raw_candidate_count": result.raw_candidate_count,
        "retained_candidate_count": result.retained_candidate_count,
        "candidate_cap": result.candidate_cap,
        "candidate_cap_activated": result.candidate_cap_activated,
        "cluster_count": len(result.clusters),
        "segment_count": len(result.segments),
        "objective": float(result.objective),
        "runtime_seconds": float(runtime),
        "segments": [
            line_segment_to_dict(segment)
            for segment in result.segments
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Task 4: detect candidates, build clique-partition weights, "
            "cluster candidates, and output averaged line segments."
        )
    )
    parser.add_argument("--image", default=None, help="Run one image path.")
    parser.add_argument("--test-pack", default=None, help="Run test_pack manifest.")
    parser.add_argument("--output-jsonl", default="task4_segments.jsonl")
    parser.add_argument(
        "--algorithm",
        choices=available_detectors(),
        default="color_distribution_refined",
    )
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("synthetic", "york", "wireframe"),
        default=None,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.image is None and args.test_pack is None:
        raise ValueError("Provide either --image or --test-pack.")

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = []

    if args.image is not None:
        jobs.append(("image", Path(args.image)))

    if args.test_pack is not None:
        test_pack = Path(args.test_pack)
        rows = select_manifest_rows(
            read_manifest(test_pack),
            ids=args.ids,
            splits=args.splits,
            limit=args.limit,
        )

        for row in rows:
            jobs.append((row["id"], test_pack / row["image"]))

    with output_path.open("w") as file:
        for image_id, path in jobs:
            print(f"Running task 4 on {image_id} with {args.algorithm} ...", flush=True)
            image = load_image(path)
            start = time.perf_counter()
            result = detect_line_segments(
                image,
                algorithm=args.algorithm,
                max_candidates=args.max_candidates,
            )
            runtime = time.perf_counter() - start
            record = _json_line_for_result(
                image_id,
                args.algorithm,
                result,
                runtime,
            )
            file.write(json.dumps(record) + "\n")
            print(
                f"  candidates={record['raw_candidate_count']}"
                f"->{record['retained_candidate_count']} "
                f"clusters={record['cluster_count']} "
                f"segments={record['segment_count']} "
                f"runtime={record['runtime_seconds']:.2f}s",
                flush=True,
            )

    print(f"Saved task-4 segments to {output_path}")


if __name__ == "__main__":
    main()
