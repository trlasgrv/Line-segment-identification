import numpy as np

from structure_tensor import normalize_directions


SQRT_THREE = np.sqrt(3.0)


try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None

HAS_OPENCV_EMD = _cv2 is not None and hasattr(_cv2, "EMD")


def _rgb_to_lab_fallback(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)

    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )

    rgb_to_xyz = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64
    )

    xyz = np.einsum("...c,dc->...d", linear, rgb_to_xyz)
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    xyz = xyz / white

    delta = 6.0 / 29.0
    delta3 = delta ** 3

    f = np.where(
        xyz > delta3,
        np.cbrt(xyz),
        xyz / (3.0 * delta * delta) + 4.0 / 29.0
    )

    L = 116.0 * f[:, :, 1] - 16.0
    a = 500.0 * (f[:, :, 0] - f[:, :, 1])
    b = 200.0 * (f[:, :, 1] - f[:, :, 2])

    return np.stack([L, a, b], axis=2)


def normalize_lab(lab):
    lab = np.asarray(lab, dtype=np.float64)

    if lab.ndim != 3 or lab.shape[2] != 3:
        raise ValueError("Lab image must have shape (H, W, 3).")

    lab_norm = np.empty_like(lab, dtype=np.float64)
    lab_norm[:, :, 0] = lab[:, :, 0] / 100.0
    lab_norm[:, :, 1] = (lab[:, :, 1] + 128.0) / 255.0
    lab_norm[:, :, 2] = (lab[:, :, 2] + 128.0) / 255.0

    return np.clip(lab_norm, 0.0, 1.0)


def prepare_lab_image(image):
    image_array = np.asarray(image)

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Input image must have shape (H, W, 3).")

    image_float = image_array.astype(np.float64)

    if np.issubdtype(image_array.dtype, np.integer):
        image_float = image_float / 255.0
    elif image_float.size and image_float.max() > 1.0:
        image_float = image_float / 255.0

    image_float = np.clip(image_float, 0.0, 1.0)

    try:
        from skimage import color

        lab = color.rgb2lab(image_float)
    except ImportError:
        lab = _rgb_to_lab_fallback(image_float)

    return normalize_lab(lab)


def _validate_geometry_parameters(
    half_length,
    half_width,
    dead_zone,
    sigma_across,
    sigma_along,
):
    if half_length <= 0:
        raise ValueError("half_length must be positive")

    if half_width <= 0:
        raise ValueError("half_width must be positive")

    if dead_zone < 0:
        raise ValueError("dead_zone must be non-negative")

    if sigma_across is None:
        sigma_across = half_width / 2.0

    if sigma_along is None:
        sigma_along = half_length / 2.0

    if sigma_across <= 0:
        raise ValueError("sigma_across must be positive")

    if sigma_along <= 0:
        raise ValueError("sigma_along must be positive")

    return float(sigma_across), float(sigma_along)


def make_support(
    direction,
    half_length=7.0,
    half_width=3.0,
    dead_zone=0.5,
    sigma_across=None,
    sigma_along=None,
):
    sigma_across, sigma_along = _validate_geometry_parameters(
        half_length,
        half_width,
        dead_zone,
        sigma_across,
        sigma_along,
    )

    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)

    if norm == 0:
        raise ValueError("direction vector cannot be zero")

    normal = direction / norm
    tangent = np.array([-normal[1], normal[0]], dtype=np.float64)

    u_values = np.arange(
        np.ceil(-half_width),
        np.floor(half_width) + 1.0,
        dtype=np.float64
    )
    v_values = np.arange(
        np.ceil(-half_length),
        np.floor(half_length) + 1.0,
        dtype=np.float64
    )

    offsets = []
    sides = []
    weights = []

    for u in u_values:
        if abs(u) <= dead_zone:
            continue

        side = 1 if u > 0.0 else -1

        for v in v_values:
            offset = u * normal + v * tangent
            weight = np.exp(
                - (u * u) / (2.0 * sigma_across * sigma_across)
                - (v * v) / (2.0 * sigma_along * sigma_along)
            )

            offsets.append(offset)
            sides.append(side)
            weights.append(weight)

    if len(offsets) == 0:
        return {
            "offsets": np.empty((0, 2), dtype=np.float64),
            "sides": np.empty(0, dtype=np.int8),
            "weights": np.empty(0, dtype=np.float64),
        }

    return {
        "offsets": np.asarray(offsets, dtype=np.float64),
        "sides": np.asarray(sides, dtype=np.int8),
        "weights": np.asarray(weights, dtype=np.float64),
    }


def _bilinear_sample_many(image, positions):
    height, width = image.shape[:2]

    x = positions[:, 0]
    y = positions[:, 1]

    inside = (
        (0.0 <= x)
        & (x <= width - 1)
        & (0.0 <= y)
        & (y <= height - 1)
    )

    if not np.any(inside):
        return np.empty((0, image.shape[2]), dtype=np.float64), inside

    x_valid = x[inside]
    y_valid = y[inside]

    x0 = np.floor(x_valid).astype(np.int64)
    y0 = np.floor(y_valid).astype(np.int64)

    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    dx = x_valid - x0
    dy = y_valid - y0

    values = (
        ((1.0 - dx) * (1.0 - dy))[:, None] * image[y0, x0]
        + (dx * (1.0 - dy))[:, None] * image[y0, x1]
        + ((1.0 - dx) * dy)[:, None] * image[y1, x0]
        + (dx * dy)[:, None] * image[y1, x1]
    )

    return values, inside


def _normalize_weights(weights, eps):
    total = float(np.sum(weights))

    if total <= eps:
        return None

    return weights / total


def _limit_samples(colors, weights, max_samples_per_side, eps):
    if max_samples_per_side is None or len(colors) <= max_samples_per_side:
        return colors, weights

    if max_samples_per_side <= 0:
        raise ValueError("max_samples_per_side must be positive or None")

    order = np.argsort(-weights, kind="mergesort")
    keep = order[:max_samples_per_side]

    colors = colors[keep]
    weights = weights[keep]
    weights = _normalize_weights(weights, eps)

    if weights is None:
        return colors[:0], np.empty(0, dtype=np.float64)

    return colors, weights


def _limited_support_indices(support, max_samples_per_side):
    if max_samples_per_side is None:
        return None

    max_samples_per_side = int(max_samples_per_side)
    cache = support.setdefault("_limited_indices", {})

    if max_samples_per_side in cache:
        return cache[max_samples_per_side]

    sides = support["sides"]
    weights = support["weights"]
    selected = []

    for side in (1, -1):
        indices = np.flatnonzero(sides == side)

        if len(indices) > max_samples_per_side:
            order = np.argsort(-weights[indices], kind="mergesort")
            indices = indices[order[:max_samples_per_side]]

        selected.append(indices)

    indices = np.concatenate(selected)
    cache[max_samples_per_side] = indices

    return indices


def _weighted_spread(colors, weights):
    mean = np.sum(colors * weights[:, None], axis=0)
    distances = np.linalg.norm(colors - mean[None, :], axis=1) / SQRT_THREE

    return float(np.sum(weights * distances))


def earth_movers_distance(
    colors_positive,
    weights_positive,
    colors_negative,
    weights_negative,
    prefer_opencv=True,
):
    colors_positive = np.asarray(colors_positive, dtype=np.float64)
    colors_negative = np.asarray(colors_negative, dtype=np.float64)
    weights_positive = np.asarray(weights_positive, dtype=np.float64)
    weights_negative = np.asarray(weights_negative, dtype=np.float64)

    if len(colors_positive) == 0 or len(colors_negative) == 0:
        return 0.0

    if len(colors_positive) == 1 and len(colors_negative) == 1:
        distance = np.linalg.norm(colors_positive[0] - colors_negative[0])
        return float(distance / SQRT_THREE)

    if prefer_opencv and HAS_OPENCV_EMD:
        signature_positive = _make_emd_signature(
            colors_positive,
            weights_positive,
        )
        signature_negative = _make_emd_signature(
            colors_negative,
            weights_negative,
        )

        try:
            distance, _, _ = _cv2.EMD(
                signature_positive,
                signature_negative,
                _cv2.DIST_L2
            )
            return float(distance / SQRT_THREE)
        except Exception:
            pass

    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    num_positive = len(colors_positive)
    num_negative = len(colors_negative)

    costs = (
        np.linalg.norm(
            colors_positive[:, None, :] - colors_negative[None, :, :],
            axis=2
        )
        / SQRT_THREE
    )

    num_variables = num_positive * num_negative
    num_constraints = num_positive + num_negative

    A_eq = lil_matrix(
        (num_constraints, num_variables),
        dtype=np.float64
    )

    for i in range(num_positive):
        start = i * num_negative
        A_eq[i, start:start + num_negative] = 1.0

    for j in range(num_negative):
        A_eq[num_positive + j, j:num_variables:num_negative] = 1.0

    b_eq = np.concatenate([weights_positive, weights_negative])

    result = linprog(
        costs.reshape(-1),
        A_eq=A_eq.tocsr(),
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs"
    )

    if not result.success:
        raise RuntimeError("EMD linear program failed: " + result.message)

    return float(result.fun)


def _make_emd_signature(colors, weights):
    signature = np.empty((len(colors), 4), dtype=np.float32)
    signature[:, 0] = weights.astype(np.float32, copy=False)
    signature[:, 1:] = colors.astype(np.float32, copy=False)

    return signature


def _zero_score(return_debug, score_mode, num_positive=0, num_negative=0):
    if score_mode not in ("final", "raw"):
        raise ValueError("score_mode must be either 'final' or 'raw'")

    if not return_debug:
        return 0.0

    debug = {
        "D": 0.0,
        "s_raw": 0.0,
        "S": 0.0,
        "consistency": 0.0,
        "s_final": 0.0,
        "num_samples_positive": int(num_positive),
        "num_samples_negative": int(num_negative),
        "num_samples_used_positive": 0,
        "num_samples_used_negative": 0,
    }

    return 0.0, debug


def color_distribution_score_from_samples(
    colors,
    sides,
    weights,
    min_samples_per_side=5,
    max_samples_per_side=64,
    tau=0.15,
    alpha=0.5,
    eps=1e-12,
    score_mode="final",
    return_debug=False,
):
    if score_mode not in ("final", "raw"):
        raise ValueError("score_mode must be either 'final' or 'raw'")

    if tau <= 0:
        raise ValueError("tau must be positive")

    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    if min_samples_per_side <= 0:
        raise ValueError("min_samples_per_side must be positive")

    if max_samples_per_side is not None and max_samples_per_side <= 0:
        raise ValueError("max_samples_per_side must be positive or None")

    colors = np.asarray(colors, dtype=np.float64)
    sides = np.asarray(sides, dtype=np.int8)
    weights = np.asarray(weights, dtype=np.float64)

    if len(colors) == 0:
        return _zero_score(return_debug, score_mode)

    positive_mask = sides > 0
    negative_mask = sides < 0

    num_positive = int(np.sum(positive_mask))
    num_negative = int(np.sum(negative_mask))

    if (
        num_positive < min_samples_per_side
        or num_negative < min_samples_per_side
    ):
        return _zero_score(
            return_debug,
            score_mode,
            num_positive=num_positive,
            num_negative=num_negative,
        )

    colors_positive = colors[positive_mask]
    colors_negative = colors[negative_mask]

    weights_positive = _normalize_weights(weights[positive_mask], eps)
    weights_negative = _normalize_weights(weights[negative_mask], eps)

    if weights_positive is None or weights_negative is None:
        return _zero_score(
            return_debug,
            score_mode,
            num_positive=num_positive,
            num_negative=num_negative,
        )

    colors_positive, weights_positive = _limit_samples(
        colors_positive,
        weights_positive,
        max_samples_per_side,
        eps,
    )
    colors_negative, weights_negative = _limit_samples(
        colors_negative,
        weights_negative,
        max_samples_per_side,
        eps,
    )

    if len(colors_positive) == 0 or len(colors_negative) == 0:
        return _zero_score(
            return_debug,
            score_mode,
            num_positive=num_positive,
            num_negative=num_negative,
        )

    D = earth_movers_distance(
        colors_positive,
        weights_positive,
        colors_negative,
        weights_negative,
    )
    D = float(np.clip(D, 0.0, 1.0))

    spread_positive = _weighted_spread(colors_positive, weights_positive)
    spread_negative = _weighted_spread(colors_negative, weights_negative)
    S = 0.5 * (spread_positive + spread_negative)

    s_raw = 1.0 - np.exp(-D / tau)
    consistency = D / (D + alpha * S + eps)
    s_final = float(np.clip(s_raw * consistency, 0.0, 1.0))

    if score_mode == "raw":
        score = float(np.clip(s_raw, 0.0, 1.0))
    else:
        score = s_final

    if not return_debug:
        return score

    debug = {
        "D": D,
        "distribution_distance": "emd",
        "s_raw": float(np.clip(s_raw, 0.0, 1.0)),
        "S": float(S),
        "consistency": float(consistency),
        "s_final": s_final,
        "num_samples_positive": num_positive,
        "num_samples_negative": num_negative,
        "num_samples_used_positive": int(len(colors_positive)),
        "num_samples_used_negative": int(len(colors_negative)),
    }

    return score, debug


def color_distribution_score(
    lab_image,
    position,
    direction,
    half_length=7.0,
    half_width=3.0,
    dead_zone=0.5,
    sigma_across=None,
    sigma_along=None,
    min_samples_per_side=5,
    max_samples_per_side=64,
    tau=0.15,
    alpha=0.5,
    eps=1e-12,
    score_mode="final",
    return_debug=False,
    support=None,
):
    if score_mode not in ("final", "raw"):
        raise ValueError("score_mode must be either 'final' or 'raw'")

    if tau <= 0:
        raise ValueError("tau must be positive")

    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    if min_samples_per_side <= 0:
        raise ValueError("min_samples_per_side must be positive")

    if max_samples_per_side is not None and max_samples_per_side <= 0:
        raise ValueError("max_samples_per_side must be positive or None")

    lab_image = np.asarray(lab_image, dtype=np.float64)

    if lab_image.ndim != 3 or lab_image.shape[2] != 3:
        raise ValueError("lab_image must have shape (H, W, 3).")

    position = np.asarray(position, dtype=np.float64)

    if position.shape != (2,):
        raise ValueError("position must have shape (2,)")

    if support is None:
        support = make_support(
            direction,
            half_length=half_length,
            half_width=half_width,
            dead_zone=dead_zone,
            sigma_across=sigma_across,
            sigma_along=sigma_along,
        )

    if len(support["offsets"]) == 0:
        return _zero_score(return_debug, score_mode)

    limited_indices = _limited_support_indices(support, max_samples_per_side)

    if limited_indices is not None:
        sample_positions = position[None, :] + support["offsets"][limited_indices]
        colors, inside = _bilinear_sample_many(lab_image, sample_positions)

        if len(colors) == len(limited_indices):
            return color_distribution_score_from_samples(
                colors,
                support["sides"][limited_indices],
                support["weights"][limited_indices],
                min_samples_per_side=min_samples_per_side,
                max_samples_per_side=None,
                tau=tau,
                alpha=alpha,
                eps=eps,
                score_mode=score_mode,
                return_debug=return_debug,
            )

    sample_positions = position[None, :] + support["offsets"]
    colors, inside = _bilinear_sample_many(lab_image, sample_positions)

    if len(colors) == 0:
        return _zero_score(return_debug, score_mode)

    sides = support["sides"][inside]
    weights = support["weights"][inside]

    return color_distribution_score_from_samples(
        colors,
        sides,
        weights,
        min_samples_per_side=min_samples_per_side,
        max_samples_per_side=max_samples_per_side,
        tau=tau,
        alpha=alpha,
        eps=eps,
        score_mode=score_mode,
        return_debug=return_debug,
    )


def color_distribution_scores_from_lab(
    lab_image,
    directions,
    half_length=7.0,
    half_width=3.0,
    dead_zone=0.5,
    sigma_across=None,
    sigma_along=None,
    min_samples_per_side=5,
    max_samples_per_side=64,
    tau=0.15,
    alpha=0.5,
    eps=1e-12,
    score_mode="final",
):
    lab_image = np.asarray(lab_image, dtype=np.float64)

    if lab_image.ndim != 3 or lab_image.shape[2] != 3:
        raise ValueError("lab_image must have shape (H, W, 3).")

    directions = normalize_directions(directions)
    height, width = lab_image.shape[:2]

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

    scores = np.zeros(
        (height, width, len(directions)),
        dtype=np.float64
    )

    for y in range(height):
        for x in range(width):
            position = np.array([x, y], dtype=np.float64)

            for k, direction in enumerate(directions):
                scores[y, x, k] = color_distribution_score(
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
                    score_mode=score_mode,
                    support=supports[k],
                )

    return scores


def color_distribution_scores(
    image,
    directions,
    half_length=7.0,
    half_width=3.0,
    dead_zone=0.5,
    sigma_across=None,
    sigma_along=None,
    min_samples_per_side=5,
    max_samples_per_side=64,
    tau=0.15,
    alpha=0.5,
    eps=1e-12,
    score_mode="final",
):
    lab_image = prepare_lab_image(image)

    return color_distribution_scores_from_lab(
        lab_image,
        directions,
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
        score_mode=score_mode,
    )
