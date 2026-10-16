import numpy as np

from structure_tensor import structure_tensor_scores


def candidate_directions(num_directions):
    # Directions are sampled from [0, pi), because n and -n
    # describe the same line orientation.

    if num_directions <= 0:
        raise ValueError("num_directions must be positive")

    angles = np.linspace(
        0.0,
        np.pi,
        num_directions,
        endpoint=False
    )

    directions = np.stack(
        [np.cos(angles), np.sin(angles)],
        axis=1
    )

    return directions


def select_best_above_threshold(
    scores,
    strength,
    directions,
    score_threshold=0.8,
    strength_threshold=0.9,
    seed_stride=1,
    return_scores=False,
):

    if not (0.0 <= score_threshold <= 1.0):
        raise ValueError("score_threshold must be in [0, 1]")

    if strength_threshold < 0:
        raise ValueError("strength_threshold must be non-negative")

    if seed_stride <= 0:
        raise ValueError("seed_stride must be positive")

    valid_pixels = strength >= strength_threshold

    if seed_stride > 1:
        stride_mask = np.zeros(strength.shape, dtype=bool)
        stride_mask[::seed_stride, ::seed_stride] = True
        valid_pixels &= stride_mask

    best_direction_indices = np.argmax(scores, axis=2)
    best_scores = np.max(scores, axis=2)

    mask = (best_scores >= score_threshold) & valid_pixels

    ys, xs = np.nonzero(mask)

    candidates = [
        (
            np.array([x, y], dtype=np.float64),
            directions[best_direction_indices[y, x]].copy()
        )
        for y, x in zip(ys, xs)
    ]

    if return_scores:
        candidate_scores = (
            best_scores[ys, xs]
            * np.log1p(np.maximum(strength[ys, xs], 0.0))
        )
        return candidates, np.asarray(candidate_scores, dtype=np.float64)

    return candidates


def structure_tensor_grid_candidates(
    image,
    num_directions=64,
    sigma=1.0,
    score_threshold=0.8,
    strength_threshold=0.9,
    eps=1e-12,
    seed_stride=4,
    return_scores=False,
):

    directions = candidate_directions(num_directions)

    scores, strength = structure_tensor_scores(
        image,
        directions,
        sigma=sigma,
        eps=eps
    )

    return select_best_above_threshold(
        scores,
        strength,
        directions,
        score_threshold=score_threshold,
        strength_threshold=strength_threshold,
        seed_stride=seed_stride,
        return_scores=return_scores,
    )
