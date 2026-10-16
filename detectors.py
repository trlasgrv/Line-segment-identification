import numpy as np

from color_grid_candidates import color_distribution_grid_candidates
from color_refinement import color_distribution_refined_candidates
from grid_candidates import structure_tensor_grid_candidates
from structure_refinement import structure_tensor_refined_candidates


DETECTOR_ALIASES = {
    "structure_tensor_grid": structure_tensor_grid_candidates,
    "color_distribution_grid": color_distribution_grid_candidates,
    "structure_tensor_refined": structure_tensor_refined_candidates,
    "color_distribution_refined": color_distribution_refined_candidates,
}


def available_detectors():
    return tuple(DETECTOR_ALIASES.keys())


def get_detector(name):
    try:
        return DETECTOR_ALIASES[name]
    except KeyError as error:
        raise ValueError(
            "unknown detector "
            + repr(name)
            + "; expected one of "
            + ", ".join(available_detectors())
        ) from error


def run_candidate_detector(name, image, detector_kwargs=None):
    detector_kwargs = dict(detector_kwargs or {})
    detector = get_detector(name)

    controlled_output_options = {
        key
        for key in ("return_scores", "return_arrays")
        if key in detector_kwargs
    }

    if controlled_output_options:
        options = ", ".join(sorted(controlled_output_options))
        raise ValueError(
            f"Detector output option(s) are controlled by the pipeline: {options}"
        )

    detector_kwargs["return_scores"] = True

    if detector is color_distribution_grid_candidates:
        detector_kwargs["return_arrays"] = False

    candidates, scores = detector(image, **detector_kwargs)
    return candidates, np.asarray(scores, dtype=np.float64)


def cap_candidates(candidates, scores=None, max_candidates=None):
    count = len(candidates)

    if max_candidates is None or count <= max_candidates:
        if scores is None:
            return list(candidates), None

        return list(candidates), np.asarray(scores, dtype=np.float64)

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive or None")

    max_candidates = int(max_candidates)

    if scores is None:
        keep = np.linspace(0, count - 1, max_candidates)
        keep = np.unique(np.round(keep).astype(np.int64))
    else:
        scores = np.asarray(scores, dtype=np.float64)

        if scores.shape != (count,):
            raise ValueError("scores must contain one value per candidate")

        keep = np.argsort(-scores, kind="mergesort")[:max_candidates]

    capped_candidates = [
        candidates[int(index)]
        for index in keep
    ]

    if scores is None:
        return capped_candidates, None

    return capped_candidates, scores[keep].copy()
