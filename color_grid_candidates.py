import numpy as np

from color_distribution import (
    color_distribution_score_from_samples,
    make_support,
    prepare_lab_image,
)
from geometry import orientation_distance
from grid_candidates import candidate_directions

SQRT_THREE = np.sqrt(3.0)


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

    positions = [
        np.asarray(candidate[0], dtype=np.float64)
        for candidate in candidates
    ]
    directions = [
        np.asarray(candidate[1], dtype=np.float64)
        for candidate in candidates
    ]
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


def _cap_candidate_arrays(positions, directions, scores, max_candidates):
    if max_candidates is None or len(scores) <= max_candidates:
        return positions, directions, scores

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive or None")

    keep = np.argpartition(-scores, max_candidates - 1)[:max_candidates]
    keep = keep[np.argsort(-scores[keep], kind="mergesort")]

    return positions[keep], directions[keep], scores[keep]


def _score_samples_batch(
    sampled_values,
    inside,
    support_sides,
    support_weights,
    min_samples_per_side,
    max_samples_per_side,
    tau,
    alpha,
    eps,
):
    scores = np.zeros(len(sampled_values), dtype=np.float64)

    if len(sampled_values) == 0:
        return scores

    for output_index in range(len(sampled_values)):
        inside_x = inside[output_index]
        scores[output_index] = color_distribution_score_from_samples(
            sampled_values[output_index, inside_x],
            support_sides[inside_x],
            support_weights[inside_x],
            min_samples_per_side=min_samples_per_side,
            max_samples_per_side=max_samples_per_side,
            tau=tau,
            alpha=alpha,
            eps=eps,
            score_mode="final",
        )

    return scores


def _cheap_proxy_scores(
    sampled_values,
    inside,
    support_sides,
    support_weights,
    min_samples_per_side,
    eps,
):
    if len(sampled_values) == 0:
        return np.empty(0, dtype=np.float64)

    positive = (support_sides > 0)[None, :] & inside
    negative = (support_sides < 0)[None, :] & inside

    positive_count = np.sum(positive, axis=1)
    negative_count = np.sum(negative, axis=1)

    positive_weights = support_weights[None, :] * positive
    negative_weights = support_weights[None, :] * negative
    positive_total = np.sum(positive_weights, axis=1)
    negative_total = np.sum(negative_weights, axis=1)

    valid = (
        (positive_count >= min_samples_per_side)
        & (negative_count >= min_samples_per_side)
        & (positive_total > eps)
        & (negative_total > eps)
    )
    scores = np.full(len(sampled_values), -np.inf, dtype=np.float64)

    if not np.any(valid):
        return scores

    positive_sum = np.einsum(
        "nsc,ns->nc",
        sampled_values,
        positive_weights,
        optimize=True,
    )
    negative_sum = np.einsum(
        "nsc,ns->nc",
        sampled_values,
        negative_weights,
        optimize=True,
    )
    positive_mean = positive_sum[valid] / positive_total[valid, None]
    negative_mean = negative_sum[valid] / negative_total[valid, None]

    positive_second = np.einsum(
        "nsc,ns->nc",
        sampled_values * sampled_values,
        positive_weights,
        optimize=True,
    )
    negative_second = np.einsum(
        "nsc,ns->nc",
        sampled_values * sampled_values,
        negative_weights,
        optimize=True,
    )
    positive_variance = np.maximum(
        positive_second[valid] / positive_total[valid, None]
        - positive_mean * positive_mean,
        0.0,
    )
    negative_variance = np.maximum(
        negative_second[valid] / negative_total[valid, None]
        - negative_mean * negative_mean,
        0.0,
    )
    positive_std = np.sqrt(positive_variance)
    negative_std = np.sqrt(negative_variance)

    mean_difference = (
        np.linalg.norm(positive_mean - negative_mean, axis=1)
        / SQRT_THREE
    )
    spread_difference = (
        np.linalg.norm(positive_std - negative_std, axis=1)
        / SQRT_THREE
    )
    scores[valid] = mean_difference + 0.5 * spread_difference

    return scores


def _sample_row_with_support(lab_image, y, support, x_indices=None):
    height, width = lab_image.shape[:2]
    offsets = support["offsets"]

    if x_indices is None:
        x_indices = np.arange(width, dtype=np.float64)
    else:
        x_indices = np.asarray(x_indices, dtype=np.float64)

    if len(offsets) == 0:
        return (
            np.empty((len(x_indices), 0, lab_image.shape[2]), dtype=np.float64),
            np.zeros((len(x_indices), 0), dtype=bool),
        )

    x_centers = x_indices[:, None]
    x = x_centers + offsets[None, :, 0]
    y_positions = float(y) + offsets[None, :, 1]
    y_positions = np.broadcast_to(y_positions, x.shape)

    inside = (
        (0.0 <= x)
        & (x <= width - 1)
        & (0.0 <= y_positions)
        & (y_positions <= height - 1)
    )

    x_clipped = np.clip(x, 0.0, width - 1)
    y_clipped = np.clip(y_positions, 0.0, height - 1)

    x0 = np.floor(x_clipped).astype(np.int64)
    y0 = np.floor(y_clipped).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    dx = x_clipped - x0
    dy = y_clipped - y0

    values = (
        ((1.0 - dx) * (1.0 - dy))[:, :, None] * lab_image[y0, x0]
        + (dx * (1.0 - dy))[:, :, None] * lab_image[y0, x1]
        + ((1.0 - dx) * dy)[:, :, None] * lab_image[y1, x0]
        + (dx * dy)[:, :, None] * lab_image[y1, x1]
    )

    return values, inside


def _select_top_proxy_indices(proxy_scores, exact_top_k, cheap_score_threshold):
    num_directions, num_positions = proxy_scores.shape
    exact_top_k = min(int(exact_top_k), num_directions)

    if exact_top_k <= 0:
        raise ValueError("exact_top_k must be positive or None")

    best_proxy = np.max(proxy_scores, axis=0)
    valid_positions = (
        np.isfinite(best_proxy)
        & (best_proxy >= cheap_score_threshold)
    )

    if not np.any(valid_positions):
        return [
            np.empty(0, dtype=np.int64)
            for _ in range(num_directions)
        ]

    top_indices = np.argpartition(
        -proxy_scores,
        exact_top_k - 1,
        axis=0,
    )[:exact_top_k]

    positions_by_direction = []

    for direction_index in range(num_directions):
        selected = np.any(top_indices == direction_index, axis=0)
        positions_by_direction.append(
            np.flatnonzero(selected & valid_positions)
        )

    return positions_by_direction


def _score_row(
    y,
    x_indices,
    lab_image,
    directions,
    supports,
    score_threshold,
    half_length,
    half_width,
    dead_zone,
    sigma_across,
    sigma_along,
    min_samples_per_side,
    max_samples_per_side,
    tau,
    alpha,
    eps,
    exact_top_k,
    cheap_score_threshold,
    directions_per_position,
):
    if directions_per_position <= 0:
        raise ValueError("directions_per_position must be positive")

    keep_multiple_directions = directions_per_position > 1
    best_scores = np.full(len(x_indices), -np.inf, dtype=np.float64)
    best_indices = np.zeros(len(x_indices), dtype=np.int64)
    candidate_position_indices = []
    candidate_direction_indices = []
    candidate_scores = []

    if exact_top_k is not None and exact_top_k < len(supports):
        proxy_scores = np.empty(
            (len(supports), len(x_indices)),
            dtype=np.float64,
        )
        sampled_values_by_direction = []
        inside_by_direction = []

        for k, support in enumerate(supports):
            sampled_values, inside = _sample_row_with_support(
                lab_image,
                y,
                support,
                x_indices=x_indices,
            )
            sampled_values_by_direction.append(sampled_values)
            inside_by_direction.append(inside)
            proxy_scores[k] = _cheap_proxy_scores(
                sampled_values,
                inside,
                support["sides"],
                support["weights"],
                min_samples_per_side,
                eps,
            )

        positions_by_direction = _select_top_proxy_indices(
            proxy_scores,
            exact_top_k,
            cheap_score_threshold,
        )
    else:
        sampled_values_by_direction = None
        inside_by_direction = None
        positions_by_direction = [
            np.arange(len(x_indices), dtype=np.int64)
            for _ in supports
        ]

    for k, support in enumerate(supports):
        selected_positions = positions_by_direction[k]

        if len(selected_positions) == 0:
            continue

        support_sides = support["sides"]
        support_weights = support["weights"]

        if sampled_values_by_direction is None:
            sampled_values, inside = _sample_row_with_support(
                lab_image,
                y,
                support,
                x_indices=x_indices[selected_positions],
            )
        else:
            sampled_values = sampled_values_by_direction[k][selected_positions]
            inside = inside_by_direction[k][selected_positions]

        scores = _score_samples_batch(
            sampled_values,
            inside,
            support_sides,
            support_weights,
            min_samples_per_side,
            max_samples_per_side,
            tau,
            alpha,
            eps,
        )

        if keep_multiple_directions:
            keep_scores = scores >= score_threshold

            if np.any(keep_scores):
                kept_positions = selected_positions[keep_scores]
                candidate_position_indices.append(kept_positions)
                candidate_direction_indices.append(
                    np.full(len(kept_positions), k, dtype=np.int64)
                )
                candidate_scores.append(scores[keep_scores])
        else:
            better = scores > best_scores[selected_positions]
            best_scores[selected_positions[better]] = scores[better]
            best_indices[selected_positions[better]] = k

    if keep_multiple_directions:
        if not candidate_scores:
            return _empty_candidate_arrays()

        candidate_position_indices = np.concatenate(candidate_position_indices)
        candidate_direction_indices = np.concatenate(candidate_direction_indices)
        candidate_scores = np.concatenate(candidate_scores)
        order = np.argsort(-candidate_scores, kind="mergesort")
        counts_by_position = np.zeros(len(x_indices), dtype=np.int64)
        keep_indices = []

        for index in order:
            position_index = int(candidate_position_indices[index])

            if counts_by_position[position_index] >= directions_per_position:
                continue

            keep_indices.append(index)
            counts_by_position[position_index] += 1

        keep_indices = np.asarray(keep_indices, dtype=np.int64)
        selected_position_indices = candidate_position_indices[keep_indices]
        row_positions = np.column_stack(
            (
                x_indices[selected_position_indices].astype(np.float64, copy=False),
                np.full(len(keep_indices), float(y), dtype=np.float64),
            )
        )
        row_directions = directions[candidate_direction_indices[keep_indices]].copy()
        row_scores = candidate_scores[keep_indices].copy()

        return row_positions, row_directions, row_scores

    keep = best_scores >= score_threshold

    if not np.any(keep):
        return _empty_candidate_arrays()

    row_positions = np.column_stack(
        (
            x_indices[keep].astype(np.float64, copy=False),
            np.full(np.count_nonzero(keep), float(y), dtype=np.float64),
        )
    )
    row_directions = directions[best_indices[keep]].copy()
    row_scores = best_scores[keep].copy()

    return row_positions, row_directions, row_scores


def color_distribution_grid_candidates(
    image,
    num_directions=64,
    score_threshold=0.6,
    half_length=7.0,
    half_width=3.0,
    dead_zone=0.5,
    sigma_across=None,
    sigma_along=None,
    min_samples_per_side=5,
    max_samples_per_side=12,
    tau=0.15,
    alpha=0.5,
    eps=1e-12,
    use_nms=True,
    nms_position_distance=1.25,
    nms_angle_distance=np.pi / 36.0,
    seed_stride=4,
    max_candidates=None,
    exact_top_k=8,
    cheap_score_threshold=0.01,
    directions_per_position=1,
    lab_image=None,
    return_arrays=False,
    return_scores=False,
    verbose=False,
):
    if not (0.0 <= score_threshold <= 1.0):
        raise ValueError("score_threshold must be in [0, 1]")

    if seed_stride <= 0:
        raise ValueError("seed_stride must be positive")

    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive or None")

    if exact_top_k is not None and exact_top_k <= 0:
        raise ValueError("exact_top_k must be positive or None")

    if cheap_score_threshold < 0:
        raise ValueError("cheap_score_threshold must be non-negative")

    if directions_per_position <= 0:
        raise ValueError("directions_per_position must be positive")

    directions = candidate_directions(num_directions)
    if lab_image is None:
        lab_image = prepare_lab_image(image)
    else:
        lab_image = np.asarray(lab_image, dtype=np.float64)

    height, width = lab_image.shape[:2]
    seed_stride = int(seed_stride)
    x_indices = np.arange(0, width, seed_stride, dtype=np.int64)
    y_indices = np.arange(0, height, seed_stride, dtype=np.int64)

    supports = [
        make_support(
            direction,
            half_length=half_length,
            half_width=half_width,
            dead_zone=dead_zone,
            sigma_across=sigma_across,
            sigma_along=sigma_along,
        )
        for direction in directions
    ]

    def score_one_row(y):
        return _score_row(
            y,
            x_indices,
            lab_image,
            directions,
            supports,
            score_threshold,
            half_length,
            half_width,
            dead_zone,
            sigma_across,
            sigma_along,
            min_samples_per_side,
            max_samples_per_side,
            tau,
            alpha,
            eps,
            exact_top_k,
            cheap_score_threshold,
            int(directions_per_position),
        )

    row_results = []

    for y in y_indices:
        if verbose and y % 10 == 0:
            print("color distribution grid row", y, "of", height, flush=True)

        row_results.append(score_one_row(y))

    row_positions = []
    row_directions = []
    row_scores = []

    for positions, row_normals, scores in row_results:
        if len(scores) == 0:
            continue

        row_positions.append(positions)
        row_directions.append(row_normals)
        row_scores.append(scores)

    if row_scores:
        positions = np.vstack(row_positions)
        normals = np.vstack(row_directions)
        candidate_scores = np.concatenate(row_scores)
    else:
        positions, normals, candidate_scores = _empty_candidate_arrays()

    if verbose:
        print("Candidates before NMS:", len(candidate_scores), flush=True)

    positions, normals, candidate_scores = _cap_candidate_arrays(
        positions,
        normals,
        candidate_scores,
        max_candidates,
    )

    if use_nms and len(candidate_scores) > 1:
        candidates = _candidate_arrays_to_list(positions, normals)
        kept_indices = non_maximum_suppression(
            candidates,
            candidate_scores,
            position_distance=nms_position_distance,
            angle_distance=nms_angle_distance,
        )

        positions = positions[kept_indices]
        normals = normals[kept_indices]
        candidate_scores = candidate_scores[kept_indices]

        if verbose:
            print("Candidates after NMS:", len(candidate_scores), flush=True)

    if return_arrays:
        if return_scores:
            return positions, normals, candidate_scores

        return positions, normals

    if return_scores:
        candidates = _candidate_arrays_to_list(positions, normals)
        return candidates, np.asarray(candidate_scores, dtype=np.float64)

    candidates = _candidate_arrays_to_list(positions, normals)
    return candidates
