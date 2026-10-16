import numpy as np

from color_distribution import color_distribution_score, make_support, prepare_lab_image
from geometry import (
    angle_to_direction,
    direction_to_angle,
    is_inside_image,
    normalize_vectors,
)
from color_grid_candidates import color_distribution_grid_candidates

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def _empty_candidate_arrays():
    return (
        np.empty((0, 2), dtype=np.float64),
        np.empty((0, 2), dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )


def _candidate_arrays_to_list(positions, directions):
    return [
        (
            positions[index].copy(),
            directions[index].copy(),
        )
        for index in range(len(positions))
    ]


def _candidate_list_to_arrays(candidates):
    if len(candidates) == 0:
        positions, directions, _ = _empty_candidate_arrays()
        return positions, directions

    positions = np.asarray(
        [
            candidate[0]
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    directions = np.asarray(
        [
            candidate[1]
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    return positions, directions


def _normalize_direction_array(directions):
    return normalize_vectors(directions)


def _cap_candidate_arrays(positions, directions, scores, max_candidates):
    if max_candidates is None or len(scores) <= max_candidates:
        order = np.argsort(-scores, kind="mergesort")
    else:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive or None")

        order = np.argpartition(-scores, max_candidates - 1)[:max_candidates]
        order = order[np.argsort(-scores[order], kind="mergesort")]

    return positions[order], directions[order], scores[order]


def _spatially_balanced_cap_candidate_arrays(
    positions,
    directions,
    scores,
    max_candidates,
    cell_size,
    per_cell,
):
    if max_candidates is None or len(scores) <= max_candidates:
        return _cap_candidate_arrays(positions, directions, scores, max_candidates)

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive or None")

    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive")

    if per_cell <= 0:
        raise ValueError("per_cell must be positive")

    max_candidates = int(max_candidates)
    per_cell = int(per_cell)
    order = np.argsort(-scores, kind="mergesort")
    cell_counts = {}
    selected = []
    selected_mask = np.zeros(len(scores), dtype=bool)

    for index in order:
        x, y = positions[index]
        cell = (int(np.floor(x / cell_size)), int(np.floor(y / cell_size)))
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

    selected = np.asarray(selected, dtype=np.int64)

    return positions[selected], directions[selected], scores[selected]


def _anisotropic_non_maximum_suppression_loop(
    positions,
    directions,
    qualities,
    normal_distance,
    tangent_distance,
    angle_distance,
):
    normalized, valid_direction = _normalize_direction_array(directions)
    qualities = np.asarray(qualities, dtype=np.float64)
    order = np.argsort(-qualities, kind="mergesort")
    keep = []

    for index in order:
        if not np.isfinite(qualities[index]) or not valid_direction[index]:
            continue

        should_keep = True

        for kept_index in keep:
            kept_normal = normalized[kept_index]
            kept_tangent = np.array(
                [-kept_normal[1], kept_normal[0]],
                dtype=np.float64
            )

            delta = positions[index] - positions[kept_index]
            close_normal = abs(np.dot(delta, kept_normal)) <= normal_distance
            close_tangent = abs(np.dot(delta, kept_tangent)) <= tangent_distance

            if not (close_normal and close_tangent):
                continue

            close_angle = (
                abs(np.dot(normalized[index], kept_normal))
                >= np.cos(angle_distance)
            )

            if close_angle:
                should_keep = False
                break

        if should_keep:
            keep.append(index)

    return np.asarray(keep, dtype=np.int64)


def anisotropic_non_maximum_suppression_arrays(
    positions,
    directions,
    qualities,
    normal_distance=1.5,
    tangent_distance=1.25,
    angle_distance=np.pi / 24.0
):
    positions = np.asarray(positions, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    qualities = np.asarray(qualities, dtype=np.float64)

    if len(positions) <= 1:
        return np.arange(len(positions), dtype=np.int64)

    if normal_distance <= 0 or tangent_distance <= 0 or angle_distance <= 0:
        return np.arange(len(positions), dtype=np.int64)

    if cKDTree is None:
        return _anisotropic_non_maximum_suppression_loop(
            positions,
            directions,
            qualities,
            normal_distance,
            tangent_distance,
            angle_distance,
        )

    normalized, valid_direction = _normalize_direction_array(directions)
    order = np.argsort(-qualities, kind="mergesort")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order), dtype=np.int64)
    suppressed = np.zeros(len(positions), dtype=bool)
    keep = []
    tree = cKDTree(positions)
    radius = float(np.hypot(normal_distance, tangent_distance))
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
            tree.query_ball_point(positions[index], radius),
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

        normal = normalized[index]
        tangent = np.array([-normal[1], normal[0]], dtype=np.float64)
        delta = positions[neighbors] - positions[index]
        close_normal = np.abs(delta @ normal) <= normal_distance
        close_tangent = np.abs(delta @ tangent) <= tangent_distance
        close_angle = np.abs(normalized[neighbors] @ normal) >= cos_min

        suppressed[neighbors[close_normal & close_tangent & close_angle]] = True

    return np.asarray(keep, dtype=np.int64)


def anisotropic_non_maximum_suppression(
    candidates,
    qualities,
    normal_distance=1.5,
    tangent_distance=1.25,
    angle_distance=np.pi / 24.0
):
    positions, directions = _candidate_list_to_arrays(candidates)

    return anisotropic_non_maximum_suppression_arrays(
        positions,
        directions,
        qualities,
        normal_distance=normal_distance,
        tangent_distance=tangent_distance,
        angle_distance=angle_distance,
    ).tolist()


def color_distribution_score_fractional(
    lab_image,
    position,
    direction,
    half_length=7.0,
    half_width=3.0,
    dead_zone=0.5,
    sigma_across=None,
    sigma_along=None,
    min_samples_per_side=5,
    max_samples_per_side=24,
    tau=0.15,
    alpha=0.5,
    eps=1e-12,
    support=None,
):
    if not is_inside_image(position, lab_image.shape):
        return 0.0

    return color_distribution_score(
        lab_image,
        position,
        direction,
        half_length=half_length,
        half_width=half_width,
        dead_zone=dead_zone,
        sigma_across=sigma_across,
        sigma_along=sigma_along,
        min_samples_per_side=min_samples_per_side,
        max_samples_per_side=max_samples_per_side,
        tau=tau,
        alpha=alpha,
        eps=eps,
        score_mode="final",
        support=support,
    )


def refine_candidate(
    lab_image,
    candidate,
    half_length=7.0,
    half_width=3.0,
    dead_zone=0.5,
    sigma_across=None,
    sigma_along=None,
    min_samples_per_side=5,
    max_samples_per_side=24,
    tau=0.15,
    alpha=0.5,
    eps=1e-12,
    max_iterations=3,
    position_step=0.5,
    angle_step=np.pi / 180.0,
    min_position_step=0.05,
    min_angle_step=np.pi / 720.0,
    min_improvement=1e-4,
    initial_score=None,
    support_cache=None
):
    position, direction = candidate

    current_position = np.asarray(position, dtype=np.float64)
    current_angle = direction_to_angle(direction)

    if support_cache is None:
        support_cache = {}

    def support_for_angle(angle):
        angle_value = float(angle % np.pi)
        angle_key = round(angle_value, 12)

        if angle_key not in support_cache:
            support_cache[angle_key] = make_support(
                angle_to_direction(angle_value),
                half_length=half_length,
                half_width=half_width,
                dead_zone=dead_zone,
                sigma_across=sigma_across,
                sigma_along=sigma_along,
            )

        return support_cache[angle_key]

    def score_at(position_to_score, angle_to_score):
        direction_to_score = angle_to_direction(angle_to_score)

        return color_distribution_score_fractional(
            lab_image,
            position_to_score,
            direction_to_score,
            half_length=half_length,
            half_width=half_width,
            dead_zone=dead_zone,
            sigma_across=sigma_across,
            sigma_along=sigma_along,
            min_samples_per_side=min_samples_per_side,
            max_samples_per_side=max_samples_per_side,
            tau=tau,
            alpha=alpha,
            eps=eps,
            support=support_for_angle(angle_to_score),
        )

    if initial_score is None:
        current_score = score_at(current_position, current_angle)
    else:
        current_score = float(initial_score)

    for _ in range(max_iterations):
        best_position = current_position.copy()
        best_angle = current_angle
        best_score = current_score
        normal = angle_to_direction(current_angle)

        for offset_sign in (-1.0, 0.0, 1.0):
            for angle_sign in (-1.0, 0.0, 1.0):
                if offset_sign == 0.0 and angle_sign == 0.0:
                    continue

                new_position = (
                    current_position
                    + offset_sign * position_step * normal
                )

                if not is_inside_image(new_position, lab_image.shape):
                    continue

                new_angle = (
                    current_angle + angle_sign * angle_step
                ) % np.pi
                new_score = score_at(new_position, new_angle)

                if new_score > best_score + min_improvement:
                    best_position = new_position
                    best_angle = new_angle
                    best_score = new_score

        if best_score > current_score + min_improvement:
            current_position = best_position
            current_angle = best_angle
            current_score = best_score
        else:
            position_step *= 0.5
            angle_step *= 0.5

            if position_step < min_position_step and angle_step < min_angle_step:
                break

    final_direction = angle_to_direction(current_angle)

    return current_position, final_direction, float(current_score)


def color_distribution_refined_candidates(
    image,
    num_directions=64,
    initial_score_threshold=0.4,
    final_score_threshold=0.6,
    half_length=7.0,
    half_width=3.0,
    dead_zone=0.5,
    sigma_across=None,
    sigma_along=None,
    min_samples_per_side=5,
    max_samples_per_side=24,
    tau=0.15,
    alpha=0.5,
    eps=1e-12,
    max_iterations=3,
    position_step=0.5,
    angle_step=np.pi / 180.0,
    min_position_step=0.05,
    min_angle_step=np.pi / 720.0,
    min_improvement=1e-4,
    max_starting_candidates=2000,
    max_seed_candidates=20000,
    seed_stride=2,
    exact_top_k=8,
    cheap_score_threshold=0.01,
    directions_per_position=1,
    seed_nms_normal_distance=2.0,
    seed_nms_tangent_distance=1.25,
    seed_nms_angle_distance=np.pi / 18.0,
    spatial_seed_cap=True,
    spatial_seed_cell_size=32.0,
    spatial_seed_per_cell=16,
    final_nms_normal_distance=1.5,
    final_nms_tangent_distance=1.25,
    final_nms_angle_distance=np.pi / 24.0,
    return_scores=False,
    verbose=False,
):
    if not (0.0 <= initial_score_threshold <= 1.0):
        raise ValueError("initial_score_threshold must be in [0, 1]")

    if not (0.0 <= final_score_threshold <= 1.0):
        raise ValueError("final_score_threshold must be in [0, 1]")

    if max_starting_candidates is not None and max_starting_candidates <= 0:
        raise ValueError("max_starting_candidates must be positive or None")

    if max_seed_candidates is not None and max_seed_candidates <= 0:
        raise ValueError("max_seed_candidates must be positive or None")

    if seed_stride <= 0:
        raise ValueError("seed_stride must be positive")

    if cheap_score_threshold < 0:
        raise ValueError("cheap_score_threshold must be non-negative")

    if directions_per_position <= 0:
        raise ValueError("directions_per_position must be positive")

    if spatial_seed_cell_size <= 0.0:
        raise ValueError("spatial_seed_cell_size must be positive")

    if spatial_seed_per_cell <= 0:
        raise ValueError("spatial_seed_per_cell must be positive")

    lab_image = prepare_lab_image(image)

    if exact_top_k <= 0:
        raise ValueError("exact_top_k must be positive")

    starting_positions, starting_normals, starting_scores = color_distribution_grid_candidates(
        image,
        num_directions=num_directions,
        score_threshold=initial_score_threshold,
        half_length=half_length,
        half_width=half_width,
        dead_zone=dead_zone,
        sigma_across=sigma_across,
        sigma_along=sigma_along,
        min_samples_per_side=min_samples_per_side,
        max_samples_per_side=max_samples_per_side,
        tau=tau,
        alpha=alpha,
        eps=eps,
        use_nms=False,
        seed_stride=seed_stride,
        max_candidates=max_seed_candidates,
        exact_top_k=exact_top_k,
        cheap_score_threshold=cheap_score_threshold,
        directions_per_position=directions_per_position,
        lab_image=lab_image,
        return_arrays=True,
        return_scores=True,
        verbose=verbose,
    )
    starting_scores = np.asarray(starting_scores, dtype=np.float64)

    if verbose:
        print("Starting candidates:", len(starting_scores), flush=True)

    if len(starting_scores) == 0:
        if return_scores:
            return [], np.empty(0, dtype=np.float64)

        return []

    kept_indices = anisotropic_non_maximum_suppression_arrays(
        starting_positions,
        starting_normals,
        starting_scores,
        normal_distance=seed_nms_normal_distance,
        tangent_distance=seed_nms_tangent_distance,
        angle_distance=seed_nms_angle_distance,
    )

    starting_positions = starting_positions[kept_indices]
    starting_normals = starting_normals[kept_indices]
    starting_scores = starting_scores[kept_indices]

    if spatial_seed_cap:
        starting_positions, starting_normals, starting_scores = (
            _spatially_balanced_cap_candidate_arrays(
                starting_positions,
                starting_normals,
                starting_scores,
                max_starting_candidates,
                spatial_seed_cell_size,
                spatial_seed_per_cell,
            )
        )
    else:
        starting_positions, starting_normals, starting_scores = _cap_candidate_arrays(
            starting_positions,
            starting_normals,
            starting_scores,
            max_starting_candidates,
        )
    starting_candidates = _candidate_arrays_to_list(
        starting_positions,
        starting_normals,
    )

    if verbose:
        print(
            "Starting candidates after seed NMS/cap:",
            len(starting_candidates),
            flush=True
        )

    shared_support_cache = {}
    candidate_score_pairs = list(zip(starting_candidates, starting_scores))

    def refine_one(candidate_score_pair):
        candidate, initial_score = candidate_score_pair

        return refine_candidate(
            lab_image,
            candidate,
            half_length=half_length,
            half_width=half_width,
            dead_zone=dead_zone,
            sigma_across=sigma_across,
            sigma_along=sigma_along,
            min_samples_per_side=min_samples_per_side,
            max_samples_per_side=max_samples_per_side,
            tau=tau,
            alpha=alpha,
            eps=eps,
            max_iterations=max_iterations,
            position_step=position_step,
            angle_step=angle_step,
            min_position_step=min_position_step,
            min_angle_step=min_angle_step,
            min_improvement=min_improvement,
            initial_score=initial_score,
            support_cache=shared_support_cache,
        )

    refined_results = [
        refine_one(candidate_score_pair)
        for candidate_score_pair in candidate_score_pairs
    ]

    kept_results = []

    for position, direction, score in refined_results:
        if score >= final_score_threshold:
            kept_results.append(
                (
                    position.copy(),
                    direction.copy(),
                    float(score)
                )
            )

    if len(kept_results) > 1:
        result_positions = np.asarray(
            [
                position
                for position, _, _ in kept_results
            ],
            dtype=np.float64,
        )
        result_directions = np.asarray(
            [
                direction
                for _, direction, _ in kept_results
            ],
            dtype=np.float64,
        )
        result_scores = np.asarray(
            [
                score
                for _, _, score in kept_results
            ],
            dtype=np.float64,
        )

        kept_indices = anisotropic_non_maximum_suppression_arrays(
            result_positions,
            result_directions,
            result_scores,
            normal_distance=final_nms_normal_distance,
            tangent_distance=final_nms_tangent_distance,
            angle_distance=final_nms_angle_distance,
        )
        kept_results = [
            kept_results[index]
            for index in kept_indices
        ]

    candidates = [
        (
            position.copy(),
            direction.copy()
        )
        for position, direction, _ in kept_results
    ]
    scores = np.asarray(
        [
            score
            for _, _, score in kept_results
        ],
        dtype=np.float64
    )

    if return_scores:
        return candidates, scores

    return candidates
