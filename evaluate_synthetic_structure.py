import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grid_candidates import structure_tensor_grid_candidates
from structure_refinement import structure_tensor_refined_candidates


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    point: tuple[float, float]
    normal: tuple[float, float]
    softness: float
    noise_std: float = 0.0


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError("zero vector cannot be normalized")

    return vector / norm


def make_synthetic_edge(case, image_shape, rng):
    height, width = image_shape
    point = np.asarray(case.point, dtype=np.float64)
    normal = normalize(case.normal)

    yy, xx = np.mgrid[0:height, 0:width]
    signed_distance = (xx - point[0]) * normal[0] + (yy - point[1]) * normal[1]

    softness = max(case.softness, 1e-6)
    alpha = 0.5 + 0.5 * np.tanh(signed_distance / softness)

    dark = np.array([0.08, 0.10, 0.14], dtype=np.float64)
    light = np.array([0.93, 0.86, 0.72], dtype=np.float64)
    image = dark + alpha[:, :, None] * (light - dark)

    if case.noise_std > 0:
        image = image + rng.normal(0.0, case.noise_std, size=image.shape)

    return np.clip(image, 0.0, 1.0)


def sample_ground_truth_line(point, normal, image_shape, num_samples=300):
    height, width = image_shape
    point = np.asarray(point, dtype=np.float64)
    normal = normalize(normal)
    tangent = np.array([-normal[1], normal[0]], dtype=np.float64)

    diagonal = np.hypot(width, height)
    ts = np.linspace(-diagonal, diagonal, num_samples * 8)
    samples = point[None, :] + ts[:, None] * tangent[None, :]

    inside = (
        (0.0 <= samples[:, 0])
        & (samples[:, 0] <= width - 1)
        & (0.0 <= samples[:, 1])
        & (samples[:, 1] <= height - 1)
    )
    samples = samples[inside]

    if samples.size == 0:
        raise ValueError(f"ground-truth line for point {point} misses the image")

    if len(samples) > num_samples:
        indices = np.linspace(0, len(samples) - 1, num_samples).astype(int)
        samples = samples[indices]

    return samples


def candidates_to_arrays(candidates):
    if len(candidates) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    positions = np.vstack([candidate[0] for candidate in candidates]).astype(np.float64)
    directions = np.vstack([normalize(candidate[1]) for candidate in candidates])

    return positions, directions


def orientation_errors(directions, true_normal):
    true_normal = normalize(true_normal)

    if len(directions) == 0:
        return np.empty(0, dtype=np.float64)

    dots = np.abs(directions @ true_normal)
    dots = np.clip(dots, -1.0, 1.0)

    return np.arccos(dots)


def evaluate_candidates(
    candidates,
    true_point,
    true_normal,
    gt_samples,
    distance_tolerance,
    angle_tolerance,
    coverage_tolerance,
):
    positions, directions = candidates_to_arrays(candidates)
    true_point = np.asarray(true_point, dtype=np.float64)
    true_normal = normalize(true_normal)

    if len(positions) == 0:
        return {
            "candidates": 0,
            "true_positives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_distance_px": np.nan,
            "median_distance_px": np.nan,
            "mean_angle_deg": np.nan,
            "median_angle_deg": np.nan,
        }

    perpendicular_distances = np.abs((positions - true_point) @ true_normal)
    angle_errors = orientation_errors(directions, true_normal)
    true_positive_mask = (
        (perpendicular_distances <= distance_tolerance)
        & (angle_errors <= angle_tolerance)
    )

    precision = float(np.mean(true_positive_mask))

    angle_ok_positions = positions[angle_errors <= angle_tolerance]
    if len(angle_ok_positions) == 0:
        recall = 0.0
    else:
        squared_distances = np.sum(
            (gt_samples[:, None, :] - angle_ok_positions[None, :, :]) ** 2,
            axis=2,
        )
        covered = np.min(squared_distances, axis=1) <= coverage_tolerance**2
        recall = float(np.mean(covered))

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    tp_distances = perpendicular_distances[true_positive_mask]
    tp_angles = np.degrees(angle_errors[true_positive_mask])

    return {
        "candidates": int(len(positions)),
        "true_positives": int(np.sum(true_positive_mask)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_distance_px": float(np.mean(tp_distances)) if len(tp_distances) else np.nan,
        "median_distance_px": float(np.median(tp_distances)) if len(tp_distances) else np.nan,
        "mean_angle_deg": float(np.mean(tp_angles)) if len(tp_angles) else np.nan,
        "median_angle_deg": float(np.median(tp_angles)) if len(tp_angles) else np.nan,
    }


def run_detector(method, image, args):
    start = time.perf_counter()

    if method == "structure_tensor_grid":
        candidates = structure_tensor_grid_candidates(
            image,
            num_directions=args.num_directions,
            sigma=args.sigma,
            score_threshold=args.score_threshold,
            strength_threshold=args.strength_threshold,
        )
    elif method == "structure_tensor_refined":
        candidates = structure_tensor_refined_candidates(
            image,
            num_directions=args.num_directions,
            sigma=args.sigma,
            initial_score_threshold=args.score_threshold,
            final_score_threshold=args.score_threshold,
            strength_threshold=args.strength_threshold,
            max_iterations=args.max_iterations,
            position_step=args.position_step,
            angle_step=np.radians(args.angle_step_deg),
            min_position_step=args.min_position_step,
            min_angle_step=np.radians(args.min_angle_step_deg),
            min_improvement=args.min_improvement,
            use_nms=not args.disable_nms,
            nms_position_distance=args.nms_position_distance,
            nms_angle_distance=np.radians(args.nms_angle_distance_deg),
        )
    else:
        raise ValueError(f"unknown method: {method}")

    runtime = time.perf_counter() - start
    return candidates, runtime


def default_cases(width, height):
    center = (0.5 * (width - 1), 0.5 * (height - 1))

    return [
        SyntheticCase(
            name="horizontal_subpixel",
            point=(center[0], 35.35),
            normal=(0.0, 1.0),
            softness=0.9,
        ),
        SyntheticCase(
            name="vertical_subpixel",
            point=(48.65, center[1]),
            normal=(1.0, 0.0),
            softness=0.9,
        ),
        SyntheticCase(
            name="diagonal_23deg",
            point=center,
            normal=(np.cos(np.radians(23.0)), np.sin(np.radians(23.0))),
            softness=1.0,
        ),
        SyntheticCase(
            name="blurred_noisy_37deg",
            point=(center[0] - 3.2, center[1] + 4.4),
            normal=(np.cos(np.radians(37.0)), np.sin(np.radians(37.0))),
            softness=2.2,
            noise_std=0.035,
        ),
    ]


def metric_text(metrics, runtime):
    return (
        f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
        f"F1={metrics['f1']:.3f}\n"
        f"dist={metrics['mean_distance_px']:.3f}px, "
        f"angle={metrics['mean_angle_deg']:.2f}deg, "
        f"n={metrics['candidates']}, t={runtime:.2f}s"
    )


def plot_overlay(ax, image, gt_samples, candidates, true_point, true_normal, args, title):
    positions, directions = candidates_to_arrays(candidates)
    true_point = np.asarray(true_point, dtype=np.float64)
    true_normal = normalize(true_normal)

    ax.imshow(image)
    ax.plot(gt_samples[:, 0], gt_samples[:, 1], color="#34c759", linewidth=2.0)

    if len(positions):
        distances = np.abs((positions - true_point) @ true_normal)
        angles = orientation_errors(directions, true_normal)
        tp_mask = (
            (distances <= args.distance_tolerance)
            & (angles <= np.radians(args.angle_tolerance_deg))
        )

        if np.any(~tp_mask):
            ax.scatter(
                positions[~tp_mask, 0],
                positions[~tp_mask, 1],
                s=9,
                c="#ff3b30",
                alpha=0.45,
                linewidths=0,
                label="false positives",
            )

        if np.any(tp_mask):
            ax.scatter(
                positions[tp_mask, 0],
                positions[tp_mask, 1],
                s=9,
                c="#18a0fb",
                alpha=0.60,
                linewidths=0,
                label="matched candidates",
            )

        step = max(1, len(positions) // 45)
        shown = np.arange(0, len(positions), step)
        ax.quiver(
            positions[shown, 0],
            positions[shown, 1],
            directions[shown, 0],
            directions[shown, 1],
            color="white",
            angles="xy",
            scale_units="xy",
            scale=0.22,
            width=0.004,
            alpha=0.75,
        )

    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, image.shape[1] - 1)
    ax.set_ylim(image.shape[0] - 1, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def save_case_figure(output_dir, case, image, gt_samples, results, args):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    axes[0].imshow(image)
    axes[0].plot(gt_samples[:, 0], gt_samples[:, 1], color="#34c759", linewidth=2.0)
    axes[0].set_title(f"{case.name}\nground truth", fontsize=10)
    axes[0].set_xlim(0, image.shape[1] - 1)
    axes[0].set_ylim(image.shape[0] - 1, 0)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for ax, method in zip(axes[1:], ("structure_tensor_grid", "structure_tensor_refined")):
        plot_overlay(
            ax,
            image,
            gt_samples,
            results[method]["candidates"],
            case.point,
            case.normal,
            args,
            f"{method}\n{metric_text(results[method]['metrics'], results[method]['runtime'])}",
        )

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)

    fig.savefig(output_dir / f"{case.name}.png", dpi=180)
    plt.close(fig)


def save_summary_chart(output_dir, rows):
    import matplotlib.pyplot as plt

    cases = sorted({row["case"] for row in rows})
    metrics = ("precision", "recall", "f1", "mean_distance_px", "mean_angle_deg")
    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "mean_distance_px": "Mean distance px",
        "mean_angle_deg": "Mean angle deg",
    }

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 13), constrained_layout=True)
    x = np.arange(len(cases))
    width = 0.36

    for ax, metric in zip(axes, metrics):
        grid_values = [
            next(row[metric] for row in rows if row["case"] == case and row["method"] == "structure_tensor_grid")
            for case in cases
        ]
        refined_values = [
            next(row[metric] for row in rows if row["case"] == case and row["method"] == "structure_tensor_refined")
            for case in cases
        ]

        ax.bar(x - width / 2, grid_values, width, label="structure_tensor_grid", color="#18a0fb")
        ax.bar(x + width / 2, refined_values, width, label="structure_tensor_refined", color="#ff7a45")
        ax.set_ylabel(labels[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(cases, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="best")
    fig.savefig(output_dir / "summary_metrics.png", dpi=180)
    plt.close(fig)


def write_csv(output_dir, rows):
    fieldnames = [
        "case",
        "method",
        "candidates",
        "true_positives",
        "precision",
        "recall",
        "f1",
        "mean_distance_px",
        "median_distance_px",
        "mean_angle_deg",
        "median_angle_deg",
        "runtime",
    ]

    with (output_dir / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    header = (
        "case",
        "method",
        "cand",
        "tp",
        "prec",
        "rec",
        "f1",
        "dist",
        "angle",
        "time",
    )
    print(
        f"{header[0]:<22} {header[1]:<6} {header[2]:>6} {header[3]:>6} "
        f"{header[4]:>7} {header[5]:>7} {header[6]:>7} {header[7]:>8} "
        f"{header[8]:>8} {header[9]:>7}"
    )
    print("-" * 96)

    for row in rows:
        print(
            f"{row['case']:<22} {row['method']:<6} "
            f"{row['candidates']:>6} {row['true_positives']:>6} "
            f"{row['precision']:>7.3f} {row['recall']:>7.3f} "
            f"{row['f1']:>7.3f} {row['mean_distance_px']:>8.3f} "
            f"{row['mean_angle_deg']:>8.3f} {row['runtime']:>7.2f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthetic ground-truth comparison for structure-tensor detectors."
    )
    parser.add_argument("--output-dir", default="synthetic_results")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--num-directions", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--strength-threshold", type=float, default=0.8)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--position-step", type=float, default=0.5)
    parser.add_argument("--angle-step-deg", type=float, default=1.0)
    parser.add_argument("--min-position-step", type=float, default=0.05)
    parser.add_argument("--min-angle-step-deg", type=float, default=0.25)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument("--disable-nms", action="store_true")
    parser.add_argument("--nms-position-distance", type=float, default=1.25)
    parser.add_argument("--nms-angle-distance-deg", type=float, default=5.0)

    parser.add_argument("--distance-tolerance", type=float, default=1.0)
    parser.add_argument("--coverage-tolerance", type=float, default=1.5)
    parser.add_argument("--angle-tolerance-deg", type=float, default=5.0)

    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache_dir = output_dir / ".matplotlib"
    matplotlib_cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir.resolve()))

    rng = np.random.default_rng(args.seed)
    image_shape = (args.height, args.width)
    angle_tolerance = np.radians(args.angle_tolerance_deg)

    rows = []
    cases = default_cases(args.width, args.height)

    for case in cases:
        image = make_synthetic_edge(case, image_shape, rng)
        gt_samples = sample_ground_truth_line(case.point, case.normal, image_shape)
        case_results = {}

        for method in ("structure_tensor_grid", "structure_tensor_refined"):
            candidates, runtime = run_detector(method, image, args)
            metrics = evaluate_candidates(
                candidates,
                case.point,
                case.normal,
                gt_samples,
                distance_tolerance=args.distance_tolerance,
                angle_tolerance=angle_tolerance,
                coverage_tolerance=args.coverage_tolerance,
            )

            row = {
                "case": case.name,
                "method": method,
                "runtime": runtime,
                **metrics,
            }
            rows.append(row)
            case_results[method] = {
                "candidates": candidates,
                "metrics": metrics,
                "runtime": runtime,
            }

        save_case_figure(output_dir, case, image, gt_samples, case_results, args)

    write_csv(output_dir, rows)
    save_summary_chart(output_dir, rows)
    print_table(rows)
    print(f"\nSaved visual results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
