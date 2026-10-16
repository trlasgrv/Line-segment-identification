import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from benchmark_checkpoint import atomic_write_csv, atomic_write_json


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
DEFAULT_YORK_ROOT = Path("dataset/YorkUrban-LineSegment")
DEFAULT_WIREFRAME_ROOT = Path("dataset/Wireframe/test")
DEFAULT_OUTPUT_ROOT = Path("dataset/report_benchmark")
DEFAULT_WIREFRAME_COUNT = 102
DEFAULT_RANDOM_SEED = 42


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


def _image_size(path):
    with Image.open(path) as image:
        image.verify()

    with Image.open(path) as image:
        return image.size


def _validate_segments(segments, source):
    segments = np.asarray(segments, dtype=np.float64)

    if segments.ndim != 2 or segments.shape[1] != 4:
        raise ValueError(
            f"Expected an N x 4 segment array in {source}, found {segments.shape}"
        )

    if len(segments) == 0 or not np.all(np.isfinite(segments)):
        raise ValueError(f"Invalid or empty line annotations in {source}")

    return segments


def _load_york_entries(york_root):
    york_root = Path(york_root).resolve()
    annotation_root = york_root / "LineSegmentAnnotation"
    index_path = annotation_root / "Image_ID_List.mat"
    index = loadmat(
        index_path,
        squeeze_me=True,
        struct_as_record=False,
    )["Image_ID_List"]
    ids = [str(item.name) for item in np.atleast_1d(index)]

    if len(ids) != 102 or len(set(ids)) != 102:
        raise ValueError(
            f"York index must contain 102 unique IDs, found {len(ids)}"
        )

    indexed_ids = set(ids)
    folder_ids = {
        path.name
        for path in york_root.iterdir()
        if path.is_dir() and path.name.startswith("P")
    }
    ground_truth_ids = {
        path.name.removesuffix("_GND.mat")
        for path in annotation_root.glob("*_GND.mat")
    }

    if folder_ids != indexed_ids:
        raise ValueError("York image folders do not match Image_ID_List.mat")

    if ground_truth_ids != indexed_ids:
        raise ValueError("York ground-truth files do not match Image_ID_List.mat")

    entries = []

    for image_id in sorted(ids):
        image_path = york_root / image_id / f"{image_id}.jpg"
        source_gt_path = annotation_root / f"{image_id}_GND.mat"

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing York image: {image_path}")

        width, height = _image_size(image_path)
        segments = _validate_segments(
            loadmat(source_gt_path)["line_gnd"],
            source_gt_path,
        )
        entries.append(
            {
                "id": image_id,
                "split": "york",
                "image_path": image_path,
                "source_gt_path": source_gt_path,
                "source_annotation": segments.tolist(),
                "width": width,
                "height": height,
            }
        )

    return entries


def _wireframe_segments(record, metadata_path):
    lines = record.get("lines")

    if not isinstance(lines, list) or not lines:
        raise ValueError(
            f"Invalid Wireframe lines for {record.get('image_file')} in {metadata_path}"
        )

    segments = []

    for line in lines:
        if (
            not isinstance(line, list)
            or len(line) != 2
            or any(not isinstance(point, list) or len(point) != 2 for point in line)
        ):
            raise ValueError(
                f"Malformed Wireframe segment for {record.get('image_file')}"
            )

        segments.append([*line[0], *line[1]])

    return _validate_segments(segments, metadata_path)


def _load_wireframe_population(wireframe_root):
    wireframe_root = Path(wireframe_root).resolve()
    image_root = wireframe_root / "images"
    metadata_path = wireframe_root / "metadata.jsonl"
    entries = {}

    with metadata_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            record = json.loads(line)
            image_file = record["image_file"]
            image_id = Path(image_file).stem

            if image_id in entries:
                raise ValueError(f"Duplicate Wireframe ID: {image_id}")

            image_path = image_root / image_file

            if not image_path.is_file():
                raise FileNotFoundError(f"Missing Wireframe image: {image_path}")

            width, height = _image_size(image_path)

            if [width, height] != record.get("image_size"):
                raise ValueError(
                    f"Wireframe dimensions disagree for {image_file} "
                    f"on metadata line {line_number}"
                )

            entries[image_id] = {
                "id": image_id,
                "split": "wireframe",
                "image_path": image_path,
                "source_gt_path": metadata_path,
                "source_annotation": _wireframe_segments(
                    record,
                    metadata_path,
                ).tolist(),
                "source_record_sha256": _sha256_json(record),
                "width": width,
                "height": height,
            }

    image_ids = {path.stem for path in image_root.glob("*.png")}

    if len(entries) != 462 or len(image_ids) != 462:
        raise ValueError(
            "Wireframe test split must contain 462 metadata records and images"
        )

    if image_ids != set(entries):
        raise ValueError("Wireframe image files do not match metadata.jsonl")

    return entries


def _relative_path(path, root):
    return Path(os.path.relpath(Path(path).resolve(), Path(root).resolve())).as_posix()


def _write_entry(entry, output_root):
    ground_truth_path = (
        output_root
        / "gt"
        / entry["split"]
        / f"{entry['id']}.json"
    )
    atomic_write_json(
        ground_truth_path,
        {"segments": entry["source_annotation"]},
    )
    return {
        "id": entry["id"],
        "split": entry["split"],
        "image": _relative_path(entry["image_path"], output_root),
        "gt": _relative_path(ground_truth_path, output_root),
        "width": entry["width"],
        "height": entry["height"],
        "n_segments": len(entry["source_annotation"]),
        "source": (
            "YorkUrban-LineSegment"
            if entry["split"] == "york"
            else "Wireframe official test split"
        ),
        "tags": "real;official_gt;report_cohort",
        "image_sha256": _sha256_file(entry["image_path"]),
        "source_gt_sha256": (
            entry.get("source_record_sha256")
            or _sha256_file(entry["source_gt_path"])
        ),
        "generated_gt_sha256": _sha256_file(ground_truth_path),
    }


def prepare_report_benchmark(
    york_root=DEFAULT_YORK_ROOT,
    wireframe_root=DEFAULT_WIREFRAME_ROOT,
    output_root=DEFAULT_OUTPUT_ROOT,
    wireframe_count=DEFAULT_WIREFRAME_COUNT,
    random_seed=DEFAULT_RANDOM_SEED,
    overwrite=False,
):
    output_root = Path(output_root).resolve()

    if wireframe_count <= 0:
        raise ValueError("wireframe_count must be positive")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output cohort already exists; use --overwrite: {output_root}"
            )

        shutil.rmtree(output_root)

    york_entries = _load_york_entries(york_root)
    wireframe_population = _load_wireframe_population(wireframe_root)

    if wireframe_count > len(wireframe_population):
        raise ValueError("wireframe_count exceeds the available population")

    generator = random.Random(random_seed)
    selected_wireframe_ids = sorted(
        generator.sample(sorted(wireframe_population), wireframe_count)
    )
    selected_entries = york_entries + [
        wireframe_population[image_id]
        for image_id in selected_wireframe_ids
    ]
    frozen_entries = [
        _write_entry(entry, output_root)
        for entry in selected_entries
    ]
    manifest_rows = [
        {column: entry[column] for column in MANIFEST_COLUMNS}
        for entry in frozen_entries
    ]
    atomic_write_csv(
        output_root / "manifest.csv",
        MANIFEST_COLUMNS,
        manifest_rows,
    )
    cohort = {
        "schema_version": 1,
        "name": "report_benchmark_102_york_102_wireframe",
        "counts": {
            "total": len(frozen_entries),
            "york": len(york_entries),
            "wireframe_population": len(wireframe_population),
            "wireframe_selected": len(selected_wireframe_ids),
        },
        "wireframe_sampling": {
            "method": "uniform random sample without replacement",
            "population_order": "lexicographically sorted image IDs",
            "random_generator": "Python random.Random",
            "seed": random_seed,
            "selected_ids": selected_wireframe_ids,
        },
        "york_selection": {
            "method": "all IDs in Image_ID_List.mat",
            "selected_ids": [entry["id"] for entry in york_entries],
        },
        "entries": [
            {
                "id": entry["id"],
                "split": entry["split"],
                "image": entry["image"],
                "ground_truth": entry["gt"],
                "width": entry["width"],
                "height": entry["height"],
                "n_segments": entry["n_segments"],
                "image_sha256": entry["image_sha256"],
                "source_gt_sha256": entry["source_gt_sha256"],
                "generated_gt_sha256": entry["generated_gt_sha256"],
            }
            for entry in frozen_entries
        ],
    }
    cohort["cohort_sha256"] = _sha256_json(cohort["entries"])
    atomic_write_json(output_root / "cohort.json", cohort)
    return cohort


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Validate the full YorkUrban and Wireframe test datasets and create "
            "a reproducible 102+102 report benchmark manifest."
        )
    )
    parser.add_argument("--york-root", type=Path, default=DEFAULT_YORK_ROOT)
    parser.add_argument(
        "--wireframe-root",
        type=Path,
        default=DEFAULT_WIREFRAME_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--wireframe-count",
        type=int,
        default=DEFAULT_WIREFRAME_COUNT,
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cohort = prepare_report_benchmark(
        york_root=args.york_root,
        wireframe_root=args.wireframe_root,
        output_root=args.output_root,
        wireframe_count=args.wireframe_count,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )
    counts = cohort["counts"]
    print(
        f"Prepared {counts['total']} images: {counts['york']} York and "
        f"{counts['wireframe_selected']} of {counts['wireframe_population']} "
        "Wireframe."
    )
    print(f"Cohort fingerprint: {cohort['cohort_sha256']}")
    print(f"Manifest: {Path(args.output_root).resolve() / 'manifest.csv'}")


if __name__ == "__main__":
    main()
