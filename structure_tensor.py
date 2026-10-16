import numpy as np
from scipy import ndimage


def rgb_to_gray(image):
    image = np.asarray(image, dtype=np.float64)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have shape (H, W, 3).")

    if image.max() > 1.0:
        image = image / 255.0

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    gray = 0.299 * R + 0.587 * G + 0.114 * B

    return gray


def compute_gradient(gray):
    gray = np.asarray(gray, dtype=np.float64)

    if gray.ndim != 2:
        raise ValueError("image is not a grayscale image")

    Ix = ndimage.sobel(gray, axis=1, mode="reflect")
    Iy = ndimage.sobel(gray, axis=0, mode="reflect")

    return Ix, Iy


def structure_tensor(Ix, Iy, sigma=1.0):
    Ix = np.asarray(Ix, dtype=np.float64)
    Iy = np.asarray(Iy, dtype=np.float64)

    if Ix.shape != Iy.shape:
        raise ValueError("Ix and Iy are different sizes")

    Txx = Ix * Ix
    Txy = Ix * Iy
    Tyy = Iy * Iy

    if sigma > 0:
        Txx = ndimage.gaussian_filter(Txx, sigma=sigma, mode="reflect")
        Txy = ndimage.gaussian_filter(Txy, sigma=sigma, mode="reflect")
        Tyy = ndimage.gaussian_filter(Tyy, sigma=sigma, mode="reflect")

    return Txx, Txy, Tyy


def prepare_structure_tensor_image(image, sigma=1.0):
    # precompute the data needed by Algorithm 1a.

    gray = rgb_to_gray(image)
    Ix, Iy = compute_gradient(gray)
    Txx, Txy, Tyy = structure_tensor(Ix, Iy, sigma=sigma)

    return Txx, Txy, Tyy


def normalize_directions(directions):
    directions = np.asarray(directions, dtype=np.float64)

    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError("directions must have shape (K, 2)")

    norms = np.linalg.norm(directions, axis=1)

    if np.any(norms == 0):
        raise ValueError("direction vector cannot be zero")

    directions = directions / norms[:, None]

    return directions


def largest_eigenvalue_image(Txx, Txy, Tyy):
    
    trace_half = 0.5 * (Txx + Tyy)
    diff_half = 0.5 * (Txx - Tyy)

    lambda_max = trace_half + np.sqrt(
        diff_half * diff_half + Txy * Txy
    )

    return lambda_max


def structure_tensor_scores_from_tensor(Txx, Txy, Tyy, directions, eps=1e-12):

    if Txx.shape != Txy.shape or Txx.shape != Tyy.shape:
        raise ValueError("Txx, Txy, and Tyy must have the same shape")

    directions = normalize_directions(directions)

    nx = directions[:, 0]
    ny = directions[:, 1]

    # Per-direction quadratic coefficients for n^T T n:
    # score = nx^2 * Txx + 2*nx*ny * Txy + ny^2 * Tyy
    q = np.stack([
        nx ** 2,
        2.0 * nx * ny,
        ny ** 2
    ], axis=1)  # shape (K, 3)

    # Stack tensor components into shape (H, W, 3)
    T_flat = np.stack([Txx, Txy, Tyy], axis=2)

    # Raw score: einsum over the 3 tensor components
    # scores_raw[h, w, k] = sum_c T_flat[h, w, c] * q[k, c]
    scores_raw = np.einsum("hwc,kc->hwk", T_flat, q)

    scores_raw = np.maximum(scores_raw, 0.0)

    # Largest eigenvalue of the structure tensor
    strength = largest_eigenvalue_image(Txx, Txy, Tyy)

    # Normalized score
    scores = np.zeros_like(scores_raw)

    valid_pixels = strength > eps

    scores[valid_pixels, :] = (
        scores_raw[valid_pixels, :] / strength[valid_pixels, None]
    )

    scores = np.clip(scores, 0.0, 1.0)

    return scores, strength


def structure_tensor_scores(image, directions, sigma=1.0, eps=1e-12):

    Txx, Txy, Tyy = prepare_structure_tensor_image(image, sigma=sigma)

    scores, strength = structure_tensor_scores_from_tensor(
        Txx,
        Txy,
        Tyy,
        directions,
        eps=eps
    )

    return scores, strength
