import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from benchmark_checkpoint import atomic_write_csv, atomic_write_json


DEFAULT_OUTPUT_ROOT = Path("dataset/synthetic_100")
DEFAULT_IMAGE_SIZE = 96
DEFAULT_RANDOM_SEED = 20260713
CASES_PER_FAMILY = 10
MANIFEST_COLUMNS = (
    "id",
    "split",
    "image",
    "gt",
    "width",
    "height",
    "n_segments",
    "source",
    "tags",
)
FAMILIES = (
    "single_luma",
    "single_chroma",
    "single_noisy",
    "parallel_luma",
    "parallel_chroma",
    "rectangle_luma",
    "rectangle_chroma",
    "triangle",
    "crossing",
    "rotated_grid",
)


def _sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _sha256_json(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))

    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")

    return vector / norm


def _line_segment_in_image(point, tangent, width, height):
    point = np.asarray(point, dtype=np.float64)
    tangent = _normalize(tangent)
    t_min = -np.inf
    t_max = np.inf

    for axis, upper in ((0, width - 1.0), (1, height - 1.0)):
        direction = float(tangent[axis])
        coordinate = float(point[axis])

        if abs(direction) <= 1e-12:
            if coordinate < 0.0 or coordinate > upper:
                raise ValueError("line does not intersect the image")
            continue

        low_t = (0.0 - coordinate) / direction
        high_t = (upper - coordinate) / direction

        if low_t > high_t:
            low_t, high_t = high_t, low_t

        t_min = max(t_min, low_t)
        t_max = min(t_max, high_t)

    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        raise ValueError("line does not form a finite image segment")

    start = point + t_min * tangent
    end = point + t_max * tangent
    return [*start.tolist(), *end.tolist()]


def _luminance(color):
    return float(np.asarray(color) @ np.array([0.299, 0.587, 0.114]))


def _luma_palette(rng):
    dark = rng.uniform(0.04, 0.24, size=3)
    light = rng.uniform(0.72, 0.96, size=3)
    return np.stack([dark, light])


def _chroma_palette(rng):
    for _ in range(1000):
        first = rng.uniform(0.05, 0.95, size=3)
        target = _luminance(first)
        red, blue = rng.uniform(0.05, 0.95, size=2)
        green = (target - 0.299 * red - 0.114 * blue) / 0.587
        second = np.array([red, green, blue], dtype=np.float64)

        if 0.05 <= green <= 0.95 and np.linalg.norm(first - second) >= 0.55:
            return np.stack([first, second])

    raise RuntimeError("failed to construct an equal-luminance color pair")


def _mixed_palette(rng, count):
    colors = rng.uniform(0.08, 0.92, size=(count, 3))
    luminances = colors @ np.array([0.299, 0.587, 0.114])

    if float(np.ptp(luminances)) < 0.35:
        colors[0] = np.array([0.08, 0.10, 0.14])
        colors[-1] = np.array([0.92, 0.88, 0.76])

    return colors


def _render_labels(labels, palette, blur_sigma, noise_std, rng):
    image = np.asarray(palette, dtype=np.float64)[labels]

    if blur_sigma > 0.0:
        image = gaussian_filter(
            image,
            sigma=(blur_sigma, blur_sigma, 0.0),
            mode="nearest",
        )

    if noise_std > 0.0:
        image += rng.normal(0.0, noise_std, size=image.shape)

    return np.clip(image, 0.0, 1.0)


def _line_geometry(rng, width, height):
    angle = rng.uniform(0.0, np.pi)
    normal = np.array([np.cos(angle), np.sin(angle)])
    tangent = np.array([-normal[1], normal[0]])
    center = np.array([(width - 1.0) / 2.0, (height - 1.0) / 2.0])
    point = center + rng.uniform(-0.14, 0.14) * min(width, height) * normal
    return point, normal, tangent


def _single_case(rng, xx, yy, width, height, chroma=False, noisy=False):
    point, normal, tangent = _line_geometry(rng, width, height)
    signed = (xx - point[0]) * normal[0] + (yy - point[1]) * normal[1]
    labels = (signed >= 0.0).astype(np.int64)
    palette = _chroma_palette(rng) if chroma else _luma_palette(rng)
    image = _render_labels(
        labels,
        palette,
        blur_sigma=rng.uniform(0.45, 0.9),
        noise_std=rng.uniform(0.015, 0.035) if noisy else 0.0,
        rng=rng,
    )
    return image, [_line_segment_in_image(point, tangent, width, height)]


def _parallel_case(rng, xx, yy, width, height, chroma=False):
    point, normal, tangent = _line_geometry(rng, width, height)
    half_width = rng.uniform(7.0, 14.0)
    signed = (xx - point[0]) * normal[0] + (yy - point[1]) * normal[1]
    labels = (np.abs(signed) <= half_width).astype(np.int64)
    palette = _chroma_palette(rng) if chroma else _luma_palette(rng)
    image = _render_labels(labels, palette, rng.uniform(0.5, 0.85), 0.0, rng)
    segments = [
        _line_segment_in_image(point - half_width * normal, tangent, width, height),
        _line_segment_in_image(point + half_width * normal, tangent, width, height),
    ]
    return image, segments


def _rotated_rectangle_case(rng, xx, yy, width, height, chroma=False):
    center = np.array(
        [
            (width - 1.0) / 2.0 + rng.uniform(-4.0, 4.0),
            (height - 1.0) / 2.0 + rng.uniform(-4.0, 4.0),
        ]
    )
    angle = rng.uniform(0.0, np.pi)
    axis_u = np.array([np.cos(angle), np.sin(angle)])
    axis_v = np.array([-axis_u[1], axis_u[0]])
    half_u = rng.uniform(22.0, 30.0)
    half_v = rng.uniform(13.0, 20.0)
    dx = xx - center[0]
    dy = yy - center[1]
    u = dx * axis_u[0] + dy * axis_u[1]
    v = dx * axis_v[0] + dy * axis_v[1]
    labels = ((np.abs(u) <= half_u) & (np.abs(v) <= half_v)).astype(np.int64)
    palette = _chroma_palette(rng) if chroma else _luma_palette(rng)
    image = _render_labels(labels, palette, rng.uniform(0.5, 0.85), 0.0, rng)
    corners = [
        center - half_u * axis_u - half_v * axis_v,
        center + half_u * axis_u - half_v * axis_v,
        center + half_u * axis_u + half_v * axis_v,
        center - half_u * axis_u + half_v * axis_v,
    ]
    segments = [
        [*corners[index].tolist(), *corners[(index + 1) % 4].tolist()]
        for index in range(4)
    ]
    return image, segments


def _polygon_mask(xx, yy, vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    twice_area = np.sum(
        vertices[:, 0] * np.roll(vertices[:, 1], -1)
        - vertices[:, 1] * np.roll(vertices[:, 0], -1)
    )
    orientation = 1.0 if twice_area >= 0.0 else -1.0
    inside = np.ones(xx.shape, dtype=bool)

    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        cross = (end[0] - start[0]) * (yy - start[1]) - (
            end[1] - start[1]
        ) * (xx - start[0])
        inside &= orientation * cross >= 0.0

    return inside


def _triangle_case(rng, xx, yy, width, height):
    center = np.array(
        [
            (width - 1.0) / 2.0 + rng.uniform(-3.0, 3.0),
            (height - 1.0) / 2.0 + rng.uniform(-3.0, 3.0),
        ]
    )
    rotation = rng.uniform(0.0, 2.0 * np.pi)
    angles = rotation + np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    radii = rng.uniform(27.0, 34.0, size=3)
    vertices = center + np.column_stack([np.cos(angles), np.sin(angles)]) * radii[:, None]
    labels = _polygon_mask(xx, yy, vertices).astype(np.int64)
    image = _render_labels(
        labels,
        _mixed_palette(rng, 2),
        rng.uniform(0.5, 0.9),
        rng.uniform(0.0, 0.012),
        rng,
    )
    segments = [
        [*vertices[index].tolist(), *vertices[(index + 1) % 3].tolist()]
        for index in range(3)
    ]
    return image, segments


def _crossing_case(rng, xx, yy, width, height):
    first_point, first_normal, first_tangent = _line_geometry(rng, width, height)
    separation = rng.uniform(np.radians(38.0), np.radians(78.0))
    angle = np.arctan2(first_normal[1], first_normal[0]) + separation
    second_normal = np.array([np.cos(angle), np.sin(angle)])
    second_tangent = np.array([-second_normal[1], second_normal[0]])
    center = np.array([(width - 1.0) / 2.0, (height - 1.0) / 2.0])
    second_point = center + rng.uniform(-5.0, 5.0) * second_normal
    first_signed = (
        (xx - first_point[0]) * first_normal[0]
        + (yy - first_point[1]) * first_normal[1]
    )
    second_signed = (
        (xx - second_point[0]) * second_normal[0]
        + (yy - second_point[1]) * second_normal[1]
    )
    labels = (first_signed >= 0.0).astype(np.int64)
    labels += 2 * (second_signed >= 0.0).astype(np.int64)
    image = _render_labels(labels, _mixed_palette(rng, 4), rng.uniform(0.5, 0.8), 0.0, rng)
    segments = [
        _line_segment_in_image(first_point, first_tangent, width, height),
        _line_segment_in_image(second_point, second_tangent, width, height),
    ]
    return image, segments


def _rotated_grid_case(rng, xx, yy, width, height):
    center = np.array([(width - 1.0) / 2.0, (height - 1.0) / 2.0])
    angle = rng.uniform(0.0, np.pi)
    axis_u = np.array([np.cos(angle), np.sin(angle)])
    axis_v = np.array([-axis_u[1], axis_u[0]])
    dx = xx - center[0]
    dy = yy - center[1]
    u = dx * axis_u[0] + dy * axis_u[1]
    v = dx * axis_v[0] + dy * axis_v[1]
    u_thresholds = np.sort(rng.uniform(-25.0, 25.0, size=2))
    v_thresholds = np.sort(rng.uniform(-25.0, 25.0, size=2))
    labels = (
        np.digitize(u, u_thresholds) + np.digitize(v, v_thresholds)
    ) % 2
    image = _render_labels(labels, _luma_palette(rng), rng.uniform(0.45, 0.75), 0.0, rng)
    segments = []

    for threshold in u_thresholds:
        point = center + threshold * axis_u
        segments.append(_line_segment_in_image(point, axis_v, width, height))

    for threshold in v_thresholds:
        point = center + threshold * axis_v
        segments.append(_line_segment_in_image(point, axis_u, width, height))

    return image, segments


def _make_case(family, rng, width, height):
    yy, xx = np.mgrid[0:height, 0:width]

    if family == "single_luma":
        return _single_case(rng, xx, yy, width, height)
    if family == "single_chroma":
        return _single_case(rng, xx, yy, width, height, chroma=True)
    if family == "single_noisy":
        return _single_case(rng, xx, yy, width, height, noisy=True)
    if family == "parallel_luma":
        return _parallel_case(rng, xx, yy, width, height)
    if family == "parallel_chroma":
        return _parallel_case(rng, xx, yy, width, height, chroma=True)
    if family == "rectangle_luma":
        return _rotated_rectangle_case(rng, xx, yy, width, height)
    if family == "rectangle_chroma":
        return _rotated_rectangle_case(rng, xx, yy, width, height, chroma=True)
    if family == "triangle":
        return _triangle_case(rng, xx, yy, width, height)
    if family == "crossing":
        return _crossing_case(rng, xx, yy, width, height)
    if family == "rotated_grid":
        return _rotated_grid_case(rng, xx, yy, width, height)

    raise ValueError(f"unknown synthetic family: {family}")


def _round_segments(segments):
    return [
        [round(float(value), 6) for value in segment]
        for segment in segments
    ]


def _validate_case(image, segments, width, height, family):
    if image.shape != (height, width, 3):
        raise ValueError(f"invalid image shape for {family}: {image.shape}")

    values = np.asarray(segments, dtype=np.float64)

    if values.ndim != 2 or values.shape[1] != 4 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid ground truth for {family}")

    starts = values[:, :2]
    ends = values[:, 2:]
    lengths = np.linalg.norm(ends - starts, axis=1)

    if np.any(lengths < 20.0):
        raise ValueError(f"ground-truth segment is too short for {family}")

    if (
        np.any(values[:, [0, 2]] < -1e-6)
        or np.any(values[:, [0, 2]] > width - 1.0 + 1e-6)
        or np.any(values[:, [1, 3]] < -1e-6)
        or np.any(values[:, [1, 3]] > height - 1.0 + 1e-6)
    ):
        raise ValueError(f"ground-truth segment leaves the image for {family}")


def _save_preview(output_root, entries, tile_size=96):
    columns = 10
    rows = (len(entries) + columns - 1) // columns
    preview = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")

    for index, entry in enumerate(entries):
        with Image.open(output_root / entry["image"]) as image:
            tile = image.convert("RGB").resize((tile_size, tile_size))
        draw = ImageDraw.Draw(tile)

        with (output_root / entry["gt"]).open() as file:
            segments = json.load(file)["segments"]

        scale_x = tile_size / entry["width"]
        scale_y = tile_size / entry["height"]

        for x1, y1, x2, y2 in segments:
            draw.line(
                (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
                fill=(255, 35, 35),
                width=1,
            )

        x = (index % columns) * tile_size
        y = (index // columns) * tile_size
        preview.paste(tile, (x, y))

    preview.save(output_root / "preview.png", optimize=True)


def prepare_synthetic_benchmark(
    output_root=DEFAULT_OUTPUT_ROOT,
    image_size=DEFAULT_IMAGE_SIZE,
    random_seed=DEFAULT_RANDOM_SEED,
    overwrite=False,
):
    output_root = Path(output_root).resolve()

    if image_size < 72:
        raise ValueError("image_size must be at least 72 pixels")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Synthetic benchmark already exists; use --overwrite: {output_root}"
            )
        shutil.rmtree(output_root)

    image_root = output_root / "images"
    gt_root = output_root / "gt"
    image_root.mkdir(parents=True)
    gt_root.mkdir(parents=True)
    rng = np.random.default_rng(random_seed)
    entries = []
    case_index = 0

    for family in FAMILIES:
        for family_index in range(CASES_PER_FAMILY):
            image, segments = _make_case(family, rng, image_size, image_size)
            segments = _round_segments(segments)
            _validate_case(image, segments, image_size, image_size, family)
            image_id = f"syn_{case_index:03d}_{family}"
            image_path = image_root / f"{image_id}.png"
            gt_path = gt_root / f"{image_id}.json"
            image_uint8 = np.rint(image * 255.0).astype(np.uint8)
            Image.fromarray(image_uint8, mode="RGB").save(image_path, optimize=True)
            atomic_write_json(gt_path, {"segments": segments})
            relative_image = Path(os.path.relpath(image_path, output_root)).as_posix()
            relative_gt = Path(os.path.relpath(gt_path, output_root)).as_posix()
            entries.append(
                {
                    "id": image_id,
                    "split": "synthetic",
                    "image": relative_image,
                    "gt": relative_gt,
                    "width": image_size,
                    "height": image_size,
                    "n_segments": len(segments),
                    "source": "deterministic synthetic geometry",
                    "tags": f"synthetic;cheap;{family};case_{family_index:02d}",
                    "family": family,
                    "image_sha256": _sha256_file(image_path),
                    "gt_sha256": _sha256_file(gt_path),
                }
            )
            case_index += 1

    manifest_rows = [
        {column: entry[column] for column in MANIFEST_COLUMNS}
        for entry in entries
    ]
    atomic_write_csv(
        output_root / "manifest.csv",
        MANIFEST_COLUMNS,
        manifest_rows,
    )
    _save_preview(output_root, entries)
    family_counts = dict(sorted(Counter(entry["family"] for entry in entries).items()))
    cohort = {
        "schema_version": 1,
        "name": "synthetic_100_lightweight_line_segments",
        "count": len(entries),
        "image_size": [image_size, image_size],
        "random_generator": "numpy.random.Generator(PCG64)",
        "random_seed": random_seed,
        "families": family_counts,
        "coordinate_format": "[x1, y1, x2, y2] in zero-based image coordinates",
        "entries": [
            {
                key: entry[key]
                for key in (
                    "id",
                    "family",
                    "image",
                    "gt",
                    "width",
                    "height",
                    "n_segments",
                    "image_sha256",
                    "gt_sha256",
                )
            }
            for entry in entries
        ],
    }
    cohort["cohort_sha256"] = _sha256_json(cohort["entries"])
    atomic_write_json(output_root / "cohort.json", cohort)
    return cohort


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create a deterministic lightweight 100-image synthetic benchmark."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cohort = prepare_synthetic_benchmark(
        output_root=args.output_root,
        image_size=args.image_size,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )
    print(
        f"Prepared {cohort['count']} synthetic images at "
        f"{Path(args.output_root).resolve()}"
    )
    print(f"Cohort fingerprint: {cohort['cohort_sha256']}")


if __name__ == "__main__":
    main()
