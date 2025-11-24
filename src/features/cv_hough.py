from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import cv2 as cv

    _HAS_CV2 = True
except Exception:  # pragma: no cover - optional dependency
    cv = None
    _HAS_CV2 = False

LOG = logging.getLogger(__name__)


def _read_gray(path: Path) -> np.ndarray | None:
    if not _HAS_CV2:
        return None
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if img is None:
        LOG.warning("Failed to load image for CV features: %s", path)
    return img


def _edge_features(img: np.ndarray, params: Dict) -> tuple[Dict[str, float], np.ndarray]:
    feats = {}
    low = int(params.get("canny_low", 40))
    high = int(params.get("canny_high", 120))
    edges = cv.Canny(img, low, high)
    feats["cv_edge_density"] = float(np.count_nonzero(edges)) / max(1.0, edges.size)
    hist, _ = np.histogram(edges, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist)) if hist.size else 0.0
    feats["cv_edge_entropy"] = float(entropy)
    return feats, edges


def _hough_features(edges: np.ndarray, params: Dict) -> Dict[str, float]:
    feats = {
        "cv_has_lines": 0.0,
        "cv_num_lines": 0.0,
        "cv_theta_mean": 0.0,
        "cv_theta_std": 0.0,
        "cv_accumulator_max": 0.0,
        "cv_frac_horizontal": 0.0,
        "cv_frac_upward": 0.0,
        "cv_frac_downward": 0.0,
        "cv_line_angle_diff_mean": 0.0,
    }
    rho = float(params.get("hough_rho", 1.0))
    theta = np.deg2rad(float(params.get("hough_theta_deg", 1.0)))
    threshold = int(params.get("hough_threshold", 80))
    lines = cv.HoughLines(edges, rho, theta, threshold)
    feats["cv_has_lines"] = 1.0 if lines is not None else 0.0
    if lines is None or len(lines) == 0:
        return feats

    thetas = []
    accum = []
    for rtheta in lines:
        rho_val, theta_val = rtheta[0]
        accum.append(abs(rho_val))
        deg = float(np.rad2deg(theta_val))
        if deg >= 90:
            deg -= 180
        thetas.append(deg)

    thetas = np.array(thetas, dtype=float)
    accum = np.array(accum, dtype=float)
    feats["cv_num_lines"] = float(len(thetas))
    feats["cv_theta_mean"] = float(np.mean(thetas)) if len(thetas) else 0.0
    feats["cv_theta_std"] = float(np.std(thetas)) if len(thetas) else 0.0
    feats["cv_accumulator_max"] = float(np.max(accum)) if len(accum) else 0.0

    if len(thetas):
        horiz = np.sum(np.abs(thetas) <= 10)
        upward = np.sum(thetas > 10)
        downward = np.sum(thetas < -10)
        total = max(len(thetas), 1)
        feats["cv_frac_horizontal"] = float(horiz / total)
        feats["cv_frac_upward"] = float(upward / total)
        feats["cv_frac_downward"] = float(downward / total)
        if len(thetas) > 1:
            diffs = []
            for i in range(len(thetas)):
                for j in range(i + 1, len(thetas)):
                    diffs.append(abs(thetas[i] - thetas[j]))
            feats["cv_line_angle_diff_mean"] = float(np.mean(diffs)) if diffs else 0.0

    return feats


def _hog_features(img: np.ndarray, params: Dict) -> Dict[str, float]:
    feats = {"cv_hog_mean": 0.0, "cv_hog_std": 0.0, "cv_hog_max": 0.0}
    if not _HAS_CV2:
        return feats
    win = int(params.get("hog_win", 64))
    try:
        h = cv.HOGDescriptor(_winSize=(win, win))
        resized = cv.resize(img, (win, win))
        desc = h.compute(resized)
        if desc is not None and len(desc):
            vals = desc.flatten()
            feats["cv_hog_mean"] = float(np.mean(vals))
            feats["cv_hog_std"] = float(np.std(vals))
            feats["cv_hog_max"] = float(np.max(vals))
    except Exception:
        pass
    return feats


def _gradient_hist_features(img: np.ndarray, params: Dict) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    bins = int(params.get("grad_bins", 9))
    try:
        gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)
        mag, ang = cv.cartToPolar(gx, gy, angleInDegrees=True)
        hist, _ = np.histogram(ang, bins=bins, range=(0, 180), weights=mag)
        total = np.sum(hist)
        if total > 0:
            hist = hist / total
        for i, v in enumerate(hist):
            feats[f"cv_grad_bin_{i}"] = float(v)
    except Exception:
        for i in range(bins):
            feats[f"cv_grad_bin_{i}"] = 0.0
    return feats


def _contour_features(edges: np.ndarray) -> Dict[str, float]:
    feats = {"cv_contour_count": 0.0, "cv_contour_area_mean": 0.0, "cv_contour_area_max": 0.0}
    try:
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        areas = [cv.contourArea(c) for c in contours] if contours else []
        feats["cv_contour_count"] = float(len(areas))
        feats["cv_contour_area_mean"] = float(np.mean(areas)) if areas else 0.0
        feats["cv_contour_area_max"] = float(np.max(areas)) if areas else 0.0
    except Exception:
        pass
    return feats


def _template_score(img: np.ndarray, params: Dict) -> Dict[str, float]:
    feats = {"cv_hs_template_corr": 0.0}
    try:
        w = int(params.get("template_width", 64))
        resized = cv.resize(img, (w, w))
        profile = resized.mean(axis=0).astype(float)
        profile = (profile - profile.mean()) / max(profile.std(), 1e-6)
        tpl = np.array([0.6, 0.2, 1.0, 0.2, 0.6], dtype=float)
        tpl = np.interp(np.linspace(0, 4, w), np.arange(5), tpl)
        tpl = (tpl - tpl.mean()) / max(tpl.std(), 1e-6)
        corr = np.correlate(profile, tpl, mode="valid")
        feats["cv_hs_template_corr"] = float(np.max(corr)) if corr.size else 0.0
    except Exception:
        pass
    return feats


def extract_cv_features(image_path: str | Path, config: Dict | None) -> Dict[str, float]:
    """
    Run lightweight CV feature extraction on a standardized chart image.
    Combines edges/entropy, Hough lines, HOG summaries, gradient histograms,
    contour stats, and a simple H&S template correlation.
    """
    feats: Dict[str, float] = {
        "cv_has_lines": 0.0,
        "cv_num_lines": 0.0,
        "cv_theta_mean": 0.0,
        "cv_theta_std": 0.0,
        "cv_accumulator_max": 0.0,
        "cv_frac_horizontal": 0.0,
        "cv_frac_upward": 0.0,
        "cv_frac_downward": 0.0,
        "cv_edge_density": 0.0,
        "cv_edge_entropy": 0.0,
        "cv_line_angle_diff_mean": 0.0,
        "cv_hog_mean": 0.0,
        "cv_hog_std": 0.0,
        "cv_hog_max": 0.0,
        "cv_contour_count": 0.0,
        "cv_contour_area_mean": 0.0,
        "cv_contour_area_max": 0.0,
        "cv_hs_template_corr": 0.0,
    }
    if not _HAS_CV2:
        LOG.debug("cv2 not available, skipping CV features for %s", image_path)
        return feats

    params = config or {}
    img = _read_gray(Path(image_path))
    if img is None:
        return feats

    edge_feats, edges = _edge_features(img, params)
    feats.update(edge_feats)
    feats.update(_hough_features(edges, params))
    feats.update(_hog_features(img, params))
    feats.update(_gradient_hist_features(img, params))
    feats.update(_contour_features(edges))
    feats.update(_template_score(img, params))
    return feats


# Backward-compatible alias
extract_hough_features = extract_cv_features

__all__ = ["extract_cv_features", "extract_hough_features"]
