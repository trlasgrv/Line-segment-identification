import numpy as np


EPS = 1e-12


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float64)

    if vector.shape != (2,):
        raise ValueError("vector must have shape (2,)")

    norm = float(np.linalg.norm(vector))

    if norm <= EPS:
        raise ValueError("zero vector cannot be normalized")

    return vector / norm


def normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)

    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("vectors must have shape (N, 2)")

    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > EPS
    normalized = np.zeros_like(vectors, dtype=np.float64)
    normalized[valid] = vectors[valid] / lengths[valid, None]

    return normalized, valid


def candidate_to_arrays(candidate):
    if len(candidate) != 2:
        raise ValueError("candidate must be a pair (position, normal)")

    position = np.asarray(candidate[0], dtype=np.float64)

    if position.shape != (2,):
        raise ValueError("candidate position must have shape (2,)")

    return position, normalize_vector(candidate[1])


def candidates_to_arrays(candidates):
    if len(candidates) == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )

    positions = []
    normals = []

    for candidate in candidates:
        position, normal = candidate_to_arrays(candidate)
        positions.append(position)
        normals.append(normal)

    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(normals, dtype=np.float64),
    )


def normal_angle(normal_a, normal_b):
    normal_a = normalize_vector(normal_a)
    normal_b = normalize_vector(normal_b)
    dot = float(np.abs(np.dot(normal_a, normal_b)))
    dot = float(np.clip(dot, -1.0, 1.0))

    return float(np.arccos(dot))


def orientation_distance(direction_a, direction_b):
    return normal_angle(direction_a, direction_b)


def direction_to_angle(direction):
    direction = normalize_vector(direction)
    angle = np.arctan2(direction[1], direction[0])

    return float(angle % np.pi)


def angle_to_direction(angle):
    return np.array(
        [np.cos(angle), np.sin(angle)],
        dtype=np.float64,
    )


def tangent_from_normal(normal):
    normal = normalize_vector(normal)

    return np.array([-normal[1], normal[0]], dtype=np.float64)


def is_inside_image(position, image_shape):
    x, y = np.asarray(position, dtype=np.float64)
    height, width = image_shape[:2]

    return 0.0 <= x <= width - 1 and 0.0 <= y <= height - 1
