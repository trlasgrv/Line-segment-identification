import numpy as np

from geometry import (
    EPS,
    candidate_to_arrays,
    candidates_to_arrays,
    normal_angle,
    normalize_vector,
)


def candidate_line_offset(candidate_a, candidate_b):
    position_a, normal_a = candidate_to_arrays(candidate_a)
    position_b, normal_b = candidate_to_arrays(candidate_b)
    delta = position_b - position_a
    distance_to_a = abs(float(np.dot(delta, normal_a)))
    distance_to_b = abs(float(np.dot(delta, normal_b)))

    return max(distance_to_a, distance_to_b)


def candidate_similarity(
    candidate_a,
    candidate_b,
    distance_scale=2.0,
    angle_scale=np.deg2rad(8.0),
    tangent_scale=None,
):
    if distance_scale <= 0.0:
        raise ValueError("distance_scale must be positive")

    if angle_scale <= 0.0:
        raise ValueError("angle_scale must be positive")

    if tangent_scale is not None and tangent_scale <= 0.0:
        raise ValueError("tangent_scale must be positive or None")

    position_a, normal_a = candidate_to_arrays(candidate_a)
    position_b, normal_b = candidate_to_arrays(candidate_b)
    delta = position_b - position_a

    dot = float(np.dot(normal_a, normal_b))
    aligned_normal_b = normal_b if dot >= 0.0 else -normal_b
    angle = float(np.arccos(np.clip(abs(dot), -1.0, 1.0)))

    distance_to_a = abs(float(np.dot(delta, normal_a)))
    distance_to_b = abs(float(np.dot(delta, normal_b)))
    line_offset = max(distance_to_a, distance_to_b)

    exponent = -((angle / angle_scale) ** 2)
    exponent -= (line_offset / distance_scale) ** 2

    if tangent_scale is not None:
        average_normal = normal_a + aligned_normal_b

        if np.linalg.norm(average_normal) <= EPS:
            average_normal = normal_a
        else:
            average_normal = average_normal / np.linalg.norm(average_normal)

        tangent = np.array(
            [-average_normal[1], average_normal[0]],
            dtype=np.float64,
        )
        tangent_distance = abs(float(np.dot(delta, tangent)))
        exponent -= (tangent_distance / tangent_scale) ** 2

    return float(np.clip(np.exp(exponent), 0.0, 1.0))


def pairwise_candidate_probabilities(
    candidates,
    distance_scale=2.0,
    angle_scale=np.deg2rad(8.0),
    tangent_scale=None,
):
    positions, normals = candidates_to_arrays(candidates)
    count = len(positions)
    probabilities = np.eye(count, dtype=np.float64)

    if count <= 1:
        return probabilities

    if distance_scale <= 0.0:
        raise ValueError("distance_scale must be positive")

    if angle_scale <= 0.0:
        raise ValueError("angle_scale must be positive")

    if tangent_scale is not None and tangent_scale <= 0.0:
        raise ValueError("tangent_scale must be positive or None")

    for index in range(count - 1):
        delta = positions[index + 1:] - positions[index]
        dots = normals[index + 1:] @ normals[index]
        angles = np.arccos(np.clip(np.abs(dots), -1.0, 1.0))
        distance_to_index = np.abs(delta @ normals[index])
        distance_to_other = np.abs(np.sum(delta * normals[index + 1:], axis=1))
        offsets = np.maximum(distance_to_index, distance_to_other)
        exponent = -((angles / angle_scale) ** 2)
        exponent -= (offsets / distance_scale) ** 2

        if tangent_scale is not None:
            aligned_normals = normals[index + 1:].copy()
            aligned_normals[dots < 0.0] *= -1.0
            average_normals = aligned_normals + normals[index][None, :]
            lengths = np.linalg.norm(average_normals, axis=1)
            valid = lengths > EPS
            average_normals[valid] /= lengths[valid, None]
            average_normals[~valid] = normals[index]
            tangents = np.column_stack(
                (-average_normals[:, 1], average_normals[:, 0])
            )
            tangent_distances = np.abs(np.sum(delta * tangents, axis=1))
            exponent -= (tangent_distances / tangent_scale) ** 2

        values = np.clip(np.exp(exponent), 0.0, 1.0)
        probabilities[index, index + 1:] = values
        probabilities[index + 1:, index] = values

    return probabilities


def probabilities_to_logit_weights(probabilities, eps=1e-6):
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.ndim != 2 or probabilities.shape[0] != probabilities.shape[1]:
        raise ValueError("probabilities must be a square matrix")

    if not (0.0 < eps < 0.5):
        raise ValueError("eps must be in (0, 0.5)")

    clipped = np.clip(probabilities, eps, 1.0 - eps)
    weights = np.log(clipped / (1.0 - clipped))
    np.fill_diagonal(weights, 0.0)

    return weights


def clique_partition_weights(
    candidates,
    distance_scale=2.0,
    angle_scale=np.deg2rad(8.0),
    tangent_scale=None,
    eps=1e-6,
    return_probabilities=False,
):
    probabilities = pairwise_candidate_probabilities(
        candidates,
        distance_scale=distance_scale,
        angle_scale=angle_scale,
        tangent_scale=tangent_scale,
    )
    weights = probabilities_to_logit_weights(probabilities, eps=eps)

    if return_probabilities:
        return weights, probabilities

    return weights
