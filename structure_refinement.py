import numpy as np

from structure_tensor import (
    prepare_structure_tensor_image,
    structure_tensor_scores_from_tensor,
)
from geometry import (
    angle_to_direction,
    direction_to_angle,
    is_inside_image,
    orientation_distance,
)
from grid_candidates import (
    candidate_directions,
)

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def bilinear_interpolate(image, position):
    x, y = position

    height, width = image.shape

    if not is_inside_image(position, image.shape):
        return None

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))

    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)

    dx = x - x0
    dy = y - y0

    value = (
        (1.0 - dx) * (1.0 - dy) * image[y0, x0]
        + dx * (1.0 - dy) * image[y0, x1]
        + (1.0 - dx) * dy * image[y1, x0]
        + dx * dy * image[y1, x1]
    )

    return float(value)


def largest_eigenvalue_scalar(Txx_value, Txy_value, Tyy_value):
    trace_half = 0.5 * (Txx_value + Tyy_value)
    diff_half = 0.5 * (Txx_value - Tyy_value)

    lambda_max = trace_half + np.sqrt(
        diff_half * diff_half + Txy_value * Txy_value
    )

    return float(lambda_max)


def structure_tensor_score_fractional(
    Txx,
    Txy,
    Tyy,
    position,
    direction,
    eps=1e-12
):
    position = np.asarray(position, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)

    if not is_inside_image(position, Txx.shape):
        return 0.0, 0.0

    norm = np.linalg.norm(direction)

    if norm == 0:
        raise ValueError("direction vector cannot be zero")

    direction = direction / norm

    nx = direction[0]
    ny = direction[1]

    Txx_value = bilinear_interpolate(Txx, position)
    Txy_value = bilinear_interpolate(Txy, position)
    Tyy_value = bilinear_interpolate(Tyy, position)

    strength = largest_eigenvalue_scalar(
        Txx_value,
        Txy_value,
        Tyy_value
    )

    if strength <= eps:
        return 0.0, strength

    score_raw = (
        Txx_value * nx * nx
        + 2.0 * Txy_value * nx * ny
        + Tyy_value * ny * ny
    )

    score_raw = max(score_raw, 0.0)

    score = score_raw / strength
    score = np.clip(score, 0.0, 1.0)

    return float(score), float(strength)


def refinement_objective(score, strength, strength_threshold):
    if strength < strength_threshold:
        return -np.inf

    return score * np.log1p(strength)


def _non_maximum_suppression_loop(
    candidates,
    qualities,
    position_distance=1.25,
    angle_distance=np.pi / 36.0
):
    if len(candidates) <= 1:
        return list(range(len(candidates)))

    if position_distance <= 0 or angle_distance <= 0:
        return list(range(len(candidates)))

    positions = [np.asarray(candidate[0], dtype=np.float64) for candidate in candidates]
    directions = [np.asarray(candidate[1], dtype=np.float64) for candidate in candidates]
    qualities = np.asarray(qualities, dtype=np.float64)

    order = np.argsort(-qualities)
    keep = []

    for index in order:
        if not np.isfinite(qualities[index]):
            continue

        should_keep = True

        for kept_index in keep:
            position_close = (
                np.linalg.norm(positions[index] - positions[kept_index])
                <= position_distance
            )

            if not position_close:
                continue

            angle_close = (
                orientation_distance(directions[index], directions[kept_index])
                <= angle_distance
            )

            if angle_close:
                should_keep = False
                break

        if should_keep:
            keep.append(index)

    return sorted(keep)


def non_maximum_suppression(
    candidates,
    qualities,
    position_distance=1.25,
    angle_distance=np.pi / 36.0
):
    if len(candidates) <= 1:
        return list(range(len(candidates)))

    if position_distance <= 0 or angle_distance <= 0:
        return list(range(len(candidates)))

    if cKDTree is None:
        return _non_maximum_suppression_loop(
            candidates,
            qualities,
            position_distance=position_distance,
            angle_distance=angle_distance,
        )

    positions = np.vstack([
        np.asarray(candidate[0], dtype=np.float64)
        for candidate in candidates
    ])
    directions = np.vstack([
        np.asarray(candidate[1], dtype=np.float64)
        for candidate in candidates
    ])
    qualities = np.asarray(qualities, dtype=np.float64)
    norms = np.linalg.norm(directions, axis=1)
    valid_direction = norms > 0.0
    directions[valid_direction] /= norms[valid_direction, None]

    order = np.argsort(-qualities, kind="mergesort")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order), dtype=np.int64)
    suppressed = np.zeros(len(candidates), dtype=bool)
    keep = []
    tree = cKDTree(positions)
    cos_min = float(np.cos(angle_distance))

    for index in order:
        if (
            suppressed[index]
            or not np.isfinite(qualities[index])
            or not valid_direction[index]
        ):
            continue

        keep.append(index)
        neighbors = np.asarray(
            tree.query_ball_point(positions[index], position_distance),
            dtype=np.int64,
        )

        if len(neighbors) == 0:
            continue

        neighbors = neighbors[
            (neighbors != index)
            & (rank[neighbors] > rank[index])
            & (~suppressed[neighbors])
            & valid_direction[neighbors]
        ]

        if len(neighbors) == 0:
            continue

        close_angle = np.abs(directions[neighbors] @ directions[index]) >= cos_min
        suppressed[neighbors[close_angle]] = True

    return sorted(keep)


def _top_quality_indices(qualities, max_candidates):
    if max_candidates is None or len(qualities) <= max_candidates:
        return np.argsort(-qualities, kind="mergesort")

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive or None")

    order = np.argpartition(-qualities, max_candidates - 1)[:max_candidates]
    return order[np.argsort(-qualities[order], kind="mergesort")]


def _spatially_balanced_seed_indices(
    xs,
    ys,
    qualities,
    max_candidates,
    cell_size,
    per_cell,
):
    if max_candidates is None or len(qualities) <= max_candidates:
        return np.argsort(-qualities, kind="mergesort")

    if cell_size <= 0.0:
        raise ValueError("spatial_seed_cell_size must be positive")

    if per_cell <= 0:
        raise ValueError("spatial_seed_per_cell must be positive")

    max_candidates = int(max_candidates)
    per_cell = int(per_cell)
    order = np.argsort(-qualities, kind="mergesort")
    cell_counts = {}
    selected = []
    selected_mask = np.zeros(len(qualities), dtype=bool)

    for index in order:
        cell = (
            int(np.floor(float(xs[index]) / cell_size)),
            int(np.floor(float(ys[index]) / cell_size)),
        )
        count = cell_counts.get(cell, 0)

        if count >= per_cell:
            continue

        selected.append(index)
        selected_mask[index] = True
        cell_counts[cell] = count + 1

        if len(selected) >= max_candidates:
            break

    if len(selected) < max_candidates:
        for index in order:
            if selected_mask[index]:
                continue

            selected.append(index)

            if len(selected) >= max_candidates:
                break

    return np.asarray(selected, dtype=np.int64)


def _select_starting_candidates(
    scores,
    strength,
    directions,
    score_threshold,
    strength_threshold,
    seed_stride,
    max_seed_candidates,
    spatial_seed_cap,
    spatial_seed_cell_size,
    spatial_seed_per_cell,
):
    if seed_stride <= 0:
        raise ValueError("seed_stride must be positive")

    valid_pixels = strength >= strength_threshold

    if seed_stride > 1:
        stride_mask = np.zeros(strength.shape, dtype=bool)
        stride_mask[::seed_stride, ::seed_stride] = True
        valid_pixels &= stride_mask

    direction_indices = np.argmax(scores, axis=2)
    best_scores = np.max(scores, axis=2)
    mask = (best_scores >= score_threshold) & valid_pixels
    ys, xs = np.nonzero(mask)
    ks = direction_indices[ys, xs]
    candidate_scores = best_scores[ys, xs]

    if len(xs) == 0:
        return [], np.empty(0, dtype=np.float64)

    qualities = candidate_scores * np.log1p(strength[ys, xs])

    if spatial_seed_cap:
        keep = _spatially_balanced_seed_indices(
            xs,
            ys,
            qualities,
            max_seed_candidates,
            spatial_seed_cell_size,
            spatial_seed_per_cell,
        )
    else:
        keep = _top_quality_indices(qualities, max_seed_candidates)

    xs = xs[keep]
    ys = ys[keep]
    ks = ks[keep]
    qualities = qualities[keep]
    candidates = [
        (
            np.array([x, y], dtype=np.float64),
            directions[k].copy(),
        )
        for y, x, k in zip(ys, xs, ks)
    ]

    return candidates, qualities


def refine_candidate(
    Txx,
    Txy,
    Tyy,
    candidate,
    strength_threshold=0.8,
    eps=1e-12,
    max_iterations=20,
    position_step=0.5,
    angle_step=np.pi / 180.0,
    min_position_step=0.05,
    min_angle_step=np.pi / 720.0,
    min_improvement=1e-4
):
    position, direction = candidate

    current_position = np.asarray(position, dtype=np.float64)
    current_angle = direction_to_angle(direction)
    current_direction = angle_to_direction(current_angle)

    current_score, current_strength = structure_tensor_score_fractional(
        Txx,
        Txy,
        Tyy,
        current_position,
        current_direction,
        eps=eps
    )

    current_objective = refinement_objective(
        current_score,
        current_strength,
        strength_threshold
    )

    for _ in range(max_iterations):
        best_position = current_position.copy()
        best_angle = current_angle
        best_score = current_score
        best_strength = current_strength
        best_objective = current_objective

        for dx in (-position_step, 0.0, position_step):
            for dy in (-position_step, 0.0, position_step):
                for da in (-angle_step, 0.0, angle_step):

                    if dx == 0.0 and dy == 0.0 and da == 0.0:
                        continue

                    new_position = current_position + np.array(
                        [dx, dy],
                        dtype=np.float64
                    )

                    if not is_inside_image(new_position, Txx.shape):
                        continue

                    new_angle = (current_angle + da) % np.pi
                    new_direction = angle_to_direction(new_angle)

                    new_score, new_strength = structure_tensor_score_fractional(
                        Txx,
                        Txy,
                        Tyy,
                        new_position,
                        new_direction,
                        eps=eps
                    )

                    new_objective = refinement_objective(
                        new_score,
                        new_strength,
                        strength_threshold
                    )

                    if new_objective > best_objective + min_improvement:
                        best_position = new_position
                        best_angle = new_angle
                        best_score = new_score
                        best_strength = new_strength
                        best_objective = new_objective

        if best_objective > current_objective + min_improvement:
            current_position = best_position
            current_angle = best_angle
            current_score = best_score
            current_strength = best_strength
            current_objective = best_objective

        else:
            position_step *= 0.5
            angle_step *= 0.5

            if position_step < min_position_step and angle_step < min_angle_step:
                break

    final_direction = angle_to_direction(current_angle)

    return current_position, final_direction, current_score, current_strength


def structure_tensor_refined_candidates(
    image,
    num_directions=64,
    sigma=1.0,
    initial_score_threshold=0.6,
    final_score_threshold=0.6,
    strength_threshold=0.6,
    eps=1e-12,
    max_iterations=20,
    position_step=0.5,
    angle_step=np.pi / 180.0,
    min_position_step=0.05,
    min_angle_step=np.pi / 720.0,
    min_improvement=1e-4,
    use_nms=True,
    nms_position_distance=3.0,
    nms_angle_distance=np.pi / 36.0,
    max_seed_candidates=2000,
    seed_stride=2,
    spatial_seed_cap=True,
    spatial_seed_cell_size=32.0,
    spatial_seed_per_cell=8,
    adaptive_sigma=True,
    adaptive_sigma_candidate_threshold=500,
    adaptive_sigma_value=0.7,
    return_scores=False,
    verbose=False,
):
    if adaptive_sigma_candidate_threshold is not None:
        if adaptive_sigma_candidate_threshold <= 0:
            raise ValueError(
                "adaptive_sigma_candidate_threshold must be positive or None"
            )

    if adaptive_sigma_value <= 0.0:
        raise ValueError("adaptive_sigma_value must be positive")

    directions = candidate_directions(num_directions)

    def run_single_scale(sigma_value):
        Txx, Txy, Tyy = prepare_structure_tensor_image(
            image,
            sigma=sigma_value
        )

        scores, strength = structure_tensor_scores_from_tensor(
            Txx,
            Txy,
            Tyy,
            directions,
            eps=eps
        )

        starting_candidates, starting_qualities = _select_starting_candidates(
            scores,
            strength,
            directions,
            initial_score_threshold,
            strength_threshold,
            seed_stride,
            max_seed_candidates,
            spatial_seed_cap,
            spatial_seed_cell_size,
            spatial_seed_per_cell,
        )

        if verbose:
            print(
                "Starting candidates at sigma",
                sigma_value,
                ":",
                len(starting_candidates),
                flush=True,
            )

        if use_nms:
            kept_indices = non_maximum_suppression(
                starting_candidates,
                starting_qualities,
                position_distance=nms_position_distance,
                angle_distance=nms_angle_distance
            )

            starting_candidates = [
                starting_candidates[index]
                for index in kept_indices
            ]
            starting_qualities = starting_qualities[kept_indices]

            if verbose:
                print(
                    "Starting candidates after NMS at sigma",
                    sigma_value,
                    ":",
                    len(starting_candidates),
                    flush=True,
                )

        def refine_one(candidate):
            return refine_candidate(
                Txx,
                Txy,
                Tyy,
                candidate,
                strength_threshold=strength_threshold,
                eps=eps,
                max_iterations=max_iterations,
                position_step=position_step,
                angle_step=angle_step,
                min_position_step=min_position_step,
                min_angle_step=min_angle_step,
                min_improvement=min_improvement
            )

        refined_results = [
            refine_one(candidate)
            for candidate in starting_candidates
        ]

        kept_results = []

        for position, direction, score, candidate_strength in refined_results:
            if (
                score >= final_score_threshold
                and candidate_strength >= strength_threshold
            ):
                kept_results.append(
                    (
                        position.copy(),
                        direction.copy(),
                        score,
                        candidate_strength
                    )
                )

        if use_nms:
            result_candidates = [
                (position, direction)
                for position, direction, _, _ in kept_results
            ]
            result_qualities = [
                refinement_objective(
                    score,
                    candidate_strength,
                    strength_threshold
                )
                for _, _, score, candidate_strength in kept_results
            ]
            kept_indices = non_maximum_suppression(
                result_candidates,
                result_qualities,
                position_distance=nms_position_distance,
                angle_distance=nms_angle_distance
            )
            kept_results = [
                kept_results[index]
                for index in kept_indices
            ]

        return kept_results

    kept_results = run_single_scale(sigma)

    if (
        adaptive_sigma
        and adaptive_sigma_candidate_threshold is not None
        and len(kept_results) < adaptive_sigma_candidate_threshold
        and abs(float(sigma) - float(adaptive_sigma_value)) > eps
    ):
        adaptive_results = run_single_scale(adaptive_sigma_value)

        if len(adaptive_results) > len(kept_results):
            kept_results = adaptive_results

    candidates = [
        (
            position.copy(),
            direction.copy()
        )
        for position, direction, _, _ in kept_results
    ]

    if return_scores:
        final_scores = np.asarray(
            [
                refinement_objective(
                    score,
                    candidate_strength,
                    strength_threshold,
                )
                for _, _, score, candidate_strength in kept_results
            ],
            dtype=np.float64,
        )
        return candidates, final_scores

    return candidates
