# Color Line Segment Detection

This is an ongoing computer vision project implementing a pipeline for detecting line segments in color images. It follows the project specification and evaluation protocol described in the references below. The implementation is still being developed; larger-scale performance testing on an HPC system is planned.

## Pipeline

The project implements four candidate detectors:

| Model | Candidate scoring | Search strategy |
| --- | --- | --- |
| 2a | Grayscale structure tensor | Image grid |
| 2b | Color distributions compared with Earth Mover's Distance (EMD) | Image grid |
| 2c | Grayscale structure tensor | Iterative refinement |
| 2d | Color distributions compared with EMD | Iterative refinement |

Candidates are compared by line similarity, partitioned into clusters using the GAEC and KLj heuristics, then converted into line segments. The implementation is primarily Python. MATLAB is used by default for the official benchmark evaluation; it is not required to run the detector itself.

## Setup

Run commands from the repository root. Python dependencies are listed in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To generate the deterministic synthetic benchmark (100 small images with generated ground-truth segments):

```bash
python source_code/prepare_synthetic_benchmark.py --overwrite
```

The output is written to `dataset/synthetic_100/`. The YorkUrban and Wireframe datasets are separate inputs; obtain them from their respective dataset sources and place them in the expected layout (`dataset/YorkUrban-LineSegment/` and `dataset/Wireframe/test/`). Then prepare a real-image benchmark pack with:

```bash
python source_code/prepare_report_benchmark.py
```

This writes the manifest and ground truth under `dataset/report_benchmark/`; image paths in the manifest refer to the source dataset files.

## MATLAB and the official evaluator

The official evaluator for reference [3] is distributed with the authors' `line_segment_review` project. MATLAB must be installed and available on the machine that runs the evaluation. The evaluator files are not part of this repository by default. Obtain the official project from [roylin1229/line_segment_review](https://github.com/roylin1229/line_segment_review), and ensure this directory exists:

```text
external/line_segment_review/eva_survey_p/
```

That directory must include `eva_detection_w_gt.m`, `process_results.p`, and `eva_unit.p`. The MATLAB executable must be on `PATH`, or its path must be supplied with `--matlab-bin` or the `MATLAB_BIN` environment variable. On a cluster, load the MATLAB module (if used there) and set `MATLAB_BIN` to the executable available on compute nodes. MATLAB licensing and module setup are cluster-specific.

The Python runner calls MATLAB in batch mode once per image/model job. For an initial HPC run, start with a small workload and one worker; increase concurrency only after checking the cluster's MATLAB license and resource limits.

Example: run the official MATLAB evaluator on two synthetic images and all four models:

```bash
python source_code/evaluate_task4_paper3.py \
  --test-pack dataset/synthetic_100 \
  --limit 2 \
  --workers 1 \
  --evaluator matlab \
  --paper3-evaluator-dir external/line_segment_review/eva_survey_p \
  --output-csv results/synthetic_matlab.csv \
  --overwrite
```

If MATLAB is not available, use `--evaluator python` for a quick development check. This uses a local reconstruction of the paper's evaluation logic, not the official MATLAB implementation, so use MATLAB for results intended to follow the official protocol.

The runner writes a CSV, a metadata sidecar, and a directory containing per-image/model segment JSON files. To continue a compatible interrupted run, use `--resume` in place of `--overwrite`. Keep outputs on a filesystem suitable for checkpoints, and retain the CSV and metadata together.

## Running one image through the detector

The detector CLI accepts an input image and writes detected segments as JSON Lines:

```bash
python source_code/line_pipeline.py \
  --image path/to/image.jpg \
  --algorithm color_distribution_refined \
  --output-jsonl results/segments.jsonl
```

Available algorithm names are `structure_tensor_grid`, `color_distribution_grid`, `structure_tensor_refined`, and `color_distribution_refined`.

## References

1. J. Irmai and B. Andres, “A State-of-the-Art Cutting Plane Algorithm for Clique Partitioning,” 2025. [doi:10.1007/978-3-031-85181-0_2](https://doi.org/10.1007/978-3-031-85181-0_2).
2. E. Levinkov, A. Kirillov, and B. Andres, “A Comparative Study of Local Search Algorithms for Correlation Clustering,” 2017. [doi:10.1007/978-3-319-66709-6_9](https://doi.org/10.1007/978-3-319-66709-6_9).
3. X. Lin, Y. Zhou, Y. Liu, and C. Zhu, “A Comprehensive Review of Image Line Segment Detection and Description: Taxonomies, Comparisons, and Challenges,” *IEEE TPAMI*, 2024. [doi:10.1109/TPAMI.2024.3400881](https://doi.org/10.1109/TPAMI.2024.3400881).
