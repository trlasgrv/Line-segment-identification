"""Clique-partition heuristics described in reference [2]."""

import heapq

import numpy as np


def _validate_weights(weights, copy=False):
    weights = np.asarray(weights, dtype=np.float64)

    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be a square matrix")

    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values")

    if not np.allclose(weights, weights.T, atol=1e-8):
        raise ValueError("weights must be symmetric")

    if copy:
        weights = weights.copy()

    return weights


def _validate_labels(labels, count):
    labels = np.asarray(labels, dtype=np.int64)

    if labels.ndim != 1 or len(labels) != count:
        raise ValueError("labels must have one entry per candidate")

    return labels


def compact_labels(labels):
    labels = np.asarray(labels, dtype=np.int64)

    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")

    if len(labels) == 0:
        return labels.copy()

    _, compact = np.unique(labels, return_inverse=True)
    return compact.astype(np.int64, copy=False)


def labels_to_clusters(labels, min_size=1):
    if min_size <= 0:
        raise ValueError("min_size must be positive")

    labels = compact_labels(labels)

    return [
        indices
        for label in range(int(labels.max()) + 1 if len(labels) else 0)
        if len(indices := np.flatnonzero(labels == label)) >= min_size
    ]


def _partition_objective_validated(weights, labels):
    objective = 0.0

    for cluster in labels_to_clusters(labels, min_size=2):
        submatrix = weights[np.ix_(cluster, cluster)]
        objective += float(np.sum(np.triu(submatrix, k=1)))

    return objective


def partition_objective(weights, labels):
    weights = _validate_weights(weights)
    labels = _validate_labels(labels, weights.shape[0])
    return _partition_objective_validated(weights, labels)


def solve_gaec(weights, epsilon=1e-9, return_history=False):
    """Run Greedy Additive Edge Contraction from [2]."""
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")

    weights = _validate_weights(weights, copy=True)
    np.fill_diagonal(weights, 0.0)
    count = weights.shape[0]

    if count == 0:
        labels = np.empty(0, dtype=np.int64)
        return (labels, [0.0]) if return_history else labels

    active = np.ones(count, dtype=bool)
    versions = np.zeros(count, dtype=np.int64)
    members = [[index] for index in range(count)]
    gains = weights.copy()
    np.fill_diagonal(gains, -np.inf)

    rows, columns = np.triu_indices(count, k=1)
    positive = gains[rows, columns] > epsilon
    heap = [
        (
            -float(gains[row, column]),
            int(row),
            int(column),
            0,
            0,
        )
        for row, column in zip(rows[positive], columns[positive])
    ]
    heapq.heapify(heap)

    objective = 0.0
    history = [objective]

    while heap:
        negative_gain, first, second, first_version, second_version = heapq.heappop(heap)

        if not active[first] or not active[second]:
            continue

        if versions[first] != first_version or versions[second] != second_version:
            continue

        gain = -negative_gain

        if gain <= epsilon:
            break

        other_clusters = np.flatnonzero(active)
        other_clusters = other_clusters[
            (other_clusters != first) & (other_clusters != second)
        ]
        updated_gains = gains[first, other_clusters] + gains[second, other_clusters]
        gains[first, other_clusters] = updated_gains
        gains[other_clusters, first] = updated_gains

        members[first].extend(members[second])
        members[second].clear()
        active[second] = False
        versions[first] += 1
        versions[second] += 1
        gains[second, :] = -np.inf
        gains[:, second] = -np.inf
        gains[first, first] = -np.inf

        for other, updated_gain in zip(other_clusters, updated_gains):
            if updated_gain <= epsilon:
                continue

            low = min(first, int(other))
            high = max(first, int(other))
            heapq.heappush(
                heap,
                (
                    -float(updated_gain),
                    low,
                    high,
                    int(versions[low]),
                    int(versions[high]),
                ),
            )

        objective += gain
        history.append(objective)

    labels = np.empty(count, dtype=np.int64)

    for label, cluster_index in enumerate(np.flatnonzero(active)):
        labels[members[int(cluster_index)]] = label

    if return_history:
        return labels, history

    return labels


def _update_small_bipartition(
    weights,
    cluster_a,
    cluster_b,
    epsilon,
    max_inner_iterations,
):
    vertices = [int(vertex) for vertex in cluster_a]
    vertices.extend(int(vertex) for vertex in cluster_b)
    size_a = len(cluster_a)
    starts_in_b = [index >= size_a for index in range(len(vertices))]
    differences = []
    gain_from_merging = 0.0

    for first in cluster_a:
        for second in cluster_b:
            gain_from_merging += float(weights[int(first), int(second)])

    for index, vertex in enumerate(vertices):
        difference = 0.0

        for other_index, other_vertex in enumerate(vertices):
            if other_index == index:
                continue

            value = float(weights[vertex, other_vertex])

            if starts_in_b[index] == starts_in_b[other_index]:
                difference -= value
            else:
                difference += value

        differences.append(difference)

    inner_limit = len(vertices)

    if max_inner_iterations is not None:
        inner_limit = min(inner_limit, max_inner_iterations)

    moved = [False] * len(vertices)
    moves = []
    cumulative_gain = 0.0
    best_move_gain = -np.inf
    best_prefix_length = 0

    for _ in range(inner_limit):
        move_index = -1
        move_gain = -np.inf

        for index, difference in enumerate(differences):
            if not moved[index] and difference > move_gain:
                move_index = index
                move_gain = difference

        if move_index < 0:
            break

        moves.append(move_index)
        cumulative_gain += move_gain

        if cumulative_gain > best_move_gain:
            best_move_gain = cumulative_gain
            best_prefix_length = len(moves)

        moved[move_index] = True

        for index, vertex in enumerate(vertices):
            if moved[index]:
                continue

            value = 2.0 * float(weights[vertex, vertices[move_index]])

            if starts_in_b[index] == starts_in_b[move_index]:
                differences[index] += value
            else:
                differences[index] -= value

    if gain_from_merging > best_move_gain and gain_from_merging > epsilon:
        return (
            np.concatenate((cluster_a, cluster_b)),
            np.empty(0, dtype=np.int64),
            gain_from_merging,
        )

    if best_move_gain <= epsilon:
        return cluster_a, cluster_b, 0.0

    moved_in_prefix = [False] * len(vertices)

    for move_index in moves[:best_prefix_length]:
        moved_in_prefix[move_index] = True

    new_a = [
        vertex
        for index, vertex in enumerate(vertices)
        if (not starts_in_b[index] and not moved_in_prefix[index])
        or (starts_in_b[index] and moved_in_prefix[index])
    ]
    new_b = [
        vertex
        for index, vertex in enumerate(vertices)
        if (starts_in_b[index] and not moved_in_prefix[index])
        or (not starts_in_b[index] and moved_in_prefix[index])
    ]

    return (
        np.asarray(new_a, dtype=np.int64),
        np.asarray(new_b, dtype=np.int64),
        best_move_gain,
    )


def _update_bipartition(
    weights,
    cluster_a,
    cluster_b,
    epsilon,
    max_inner_iterations,
):
    cluster_a = np.asarray(cluster_a, dtype=np.int64)
    cluster_b = np.asarray(cluster_b, dtype=np.int64)

    if len(cluster_a) == 0:
        return cluster_a, cluster_b, 0.0

    if len(cluster_a) == 1 and len(cluster_b) == 0:
        return cluster_a, cluster_b, 0.0

    if len(cluster_a) == 1 and len(cluster_b) == 1:
        pair_gain = float(weights[cluster_a[0], cluster_b[0]])

        if pair_gain <= epsilon:
            return cluster_a, cluster_b, 0.0

    if len(cluster_a) + len(cluster_b) <= 24:
        return _update_small_bipartition(
            weights,
            cluster_a,
            cluster_b,
            epsilon,
            max_inner_iterations,
        )

    a_to_a = weights[np.ix_(cluster_a, cluster_a)]
    a_to_b = weights[np.ix_(cluster_a, cluster_b)]
    differences_a = np.sum(a_to_b, axis=1) - np.sum(a_to_a, axis=1)

    if len(cluster_b):
        b_to_b = weights[np.ix_(cluster_b, cluster_b)]
        differences_b = np.sum(a_to_b, axis=0) - np.sum(b_to_b, axis=1)
    else:
        differences_b = np.empty(0, dtype=np.float64)

    gain_from_merging = float(np.sum(a_to_b))
    vertices = np.concatenate((cluster_a, cluster_b))
    starts_in_b = np.arange(len(vertices)) >= len(cluster_a)
    differences = np.concatenate((differences_a, differences_b)).astype(
        np.float64,
        copy=False,
    )
    moved = np.zeros(len(vertices), dtype=bool)
    moves = np.empty(len(vertices), dtype=np.int64)
    inner_limit = len(vertices)

    if max_inner_iterations is not None:
        inner_limit = min(inner_limit, max_inner_iterations)

    cumulative_gain = 0.0
    best_move_gain = -np.inf
    best_prefix_length = 0
    move_count = 0

    for _ in range(inner_limit):
        move_index = int(np.argmax(differences))
        move_gain = float(differences[move_index])

        if not np.isfinite(move_gain):
            break

        moves[move_count] = move_index
        move_count += 1
        cumulative_gain += move_gain

        if cumulative_gain > best_move_gain:
            best_move_gain = cumulative_gain
            best_prefix_length = move_count

        active = ~moved
        same_origin = starts_in_b == starts_in_b[move_index]
        incident_weights = weights[vertices, vertices[move_index]]
        differences[active & same_origin] += 2.0 * incident_weights[active & same_origin]
        differences[active & ~same_origin] -= 2.0 * incident_weights[active & ~same_origin]
        moved[move_index] = True
        differences[move_index] = -np.inf

    if gain_from_merging > best_move_gain and gain_from_merging > epsilon:
        return np.concatenate((cluster_a, cluster_b)), np.empty(0, dtype=np.int64), gain_from_merging

    if best_move_gain <= epsilon:
        return cluster_a, cluster_b, 0.0

    moved_in_prefix = np.zeros(len(vertices), dtype=bool)
    moved_in_prefix[moves[:best_prefix_length]] = True
    moved_from_a = moved_in_prefix[:len(cluster_a)]
    moved_from_b = moved_in_prefix[len(cluster_a):]
    new_a = np.concatenate((cluster_a[~moved_from_a], cluster_b[moved_from_b]))
    new_b = np.concatenate((cluster_b[~moved_from_b], cluster_a[moved_from_a]))

    return new_a, new_b, best_move_gain


def refine_kernighan_lin_joins(
    weights,
    labels,
    max_outer_iterations=100,
    max_inner_iterations=None,
    epsilon=1e-6,
    return_history=False,
):
    """Refine a partition with the Kernighan-Lin algorithm with joins from [2]."""
    if max_outer_iterations <= 0:
        raise ValueError("max_outer_iterations must be positive")

    if max_inner_iterations is not None and max_inner_iterations <= 0:
        raise ValueError("max_inner_iterations must be positive or None")

    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")

    weights = _validate_weights(weights, copy=True)
    np.fill_diagonal(weights, 0.0)
    labels = compact_labels(_validate_labels(labels, weights.shape[0]))

    if len(labels) == 0:
        return (labels, [0.0]) if return_history else labels

    partitions = labels_to_clusters(labels)
    partition_ids = list(range(len(partitions)))
    partition_versions = [0] * len(partitions)
    next_partition_id = len(partitions)
    unchanged_pairs = {}
    history = [_partition_objective_validated(weights, labels)]

    for _ in range(max_outer_iterations):
        total_gain = 0.0

        for first in range(len(partitions) - 1):
            for second in range(first + 1, len(partitions)):
                if len(partitions[second]) == 0:
                    continue

                pair_key = (partition_ids[first], partition_ids[second])
                pair_versions = (
                    partition_versions[first],
                    partition_versions[second],
                )

                if unchanged_pairs.get(pair_key) == pair_versions:
                    continue

                new_first, new_second, gain = _update_bipartition(
                    weights,
                    partitions[first],
                    partitions[second],
                    epsilon,
                    max_inner_iterations,
                )
                partitions[first] = new_first
                partitions[second] = new_second
                total_gain += gain

                if gain > 0.0:
                    partition_versions[first] += 1
                    partition_versions[second] += 1
                else:
                    unchanged_pairs[pair_key] = pair_versions

        nonempty = [
            index
            for index, partition in enumerate(partitions)
            if len(partition)
        ]
        partitions = [partitions[index] for index in nonempty]
        partition_ids = [partition_ids[index] for index in nonempty]
        partition_versions = [partition_versions[index] for index in nonempty]
        existing_partition_count = len(partitions)

        for index in range(existing_partition_count):
            while True:
                remainder, new_partition, gain = _update_bipartition(
                    weights,
                    partitions[index],
                    np.empty(0, dtype=np.int64),
                    epsilon,
                    max_inner_iterations,
                )
                partitions[index] = remainder
                total_gain += gain

                if len(new_partition) == 0:
                    break

                partition_versions[index] += 1
                partitions.append(new_partition)
                partition_ids.append(next_partition_id)
                partition_versions.append(0)
                next_partition_id += 1

        if total_gain == 0.0:
            break

        labels = np.empty(weights.shape[0], dtype=np.int64)

        for label, partition in enumerate(partitions):
            labels[partition] = label

        history.append(_partition_objective_validated(weights, labels))

    labels = np.empty(weights.shape[0], dtype=np.int64)

    for label, partition in enumerate(partitions):
        labels[partition] = label

    if return_history:
        return labels, history

    return labels


def solve_gaec_klj(
    weights,
    gaec_epsilon=1e-9,
    klj_epsilon=1e-6,
    max_outer_iterations=100,
    max_inner_iterations=None,
    return_history=False,
):
    """Initialize with GAEC and refine the result with KLj, as evaluated in [2]."""
    gaec_result = solve_gaec(
        weights,
        epsilon=gaec_epsilon,
        return_history=return_history,
    )

    if return_history:
        gaec_labels, gaec_history = gaec_result
    else:
        gaec_labels = gaec_result

    klj_result = refine_kernighan_lin_joins(
        weights,
        gaec_labels,
        max_outer_iterations=max_outer_iterations,
        max_inner_iterations=max_inner_iterations,
        epsilon=klj_epsilon,
        return_history=return_history,
    )

    if return_history:
        labels, klj_history = klj_result
        return labels, {"gaec": gaec_history, "klj": klj_history}

    return klj_result
