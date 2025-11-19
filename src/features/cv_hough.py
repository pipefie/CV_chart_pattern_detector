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


def extract_hough_features(image_path: str | Path, config: Dict | None) -> Dict[str, float]:
    """
    Run Canny + Hough on a standardized chart image and return aggregate stats.
    """
    feats = {
        "cv_has_lines": 0.0,
        "cv_num_lines": 0.0,
        "cv_theta_mean": 0.0,
        "cv_theta_std": 0.0,
        "cv_accumulator_max": 0.0,
        "cv_frac_horizontal": 0.0,
        "cv_frac_upward": 0.0,
        "cv_frac_downward": 0.0,
    }
    if not _HAS_CV2:
        LOG.debug("cv2 not available, skipping Hough features for %s", image_path)
        return feats

    params = config or {}
    path = Path(image_path)
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if img is None:
        LOG.warning("Failed to load image for hough features: %s", path)
        return feats

    low = int(params.get("canny_low", 40))
    high = int(params.get("canny_high", 120))
    edges = cv.Canny(img, low, high)

    rho = float(params.get("hough_rho", 1.0))
    theta = np.deg2rad(float(params.get("hough_theta_deg", 1.0)))
    threshold = int(params.get("hough_threshold", 80))
    lines = cv.HoughLines(edges, rho, theta, threshold)

    feats["cv_has_lines"] = 1.0 if lines is not None else 0.0
    if lines is None:
        return feats

    thetas = []
    accumulator_max = 0.0
    for rtheta in lines:
        rho_val, theta_val = rtheta[0]
        accumulator_max = max(accumulator_max, abs(rho_val))
        deg = float(np.rad2deg(theta_val))
        if deg >= 90:
            deg -= 180
        thetas.append(deg)

    thetas = np.array(thetas, dtype=float)
    feats["cv_num_lines"] = float(len(thetas))
    feats["cv_theta_mean"] = float(np.mean(thetas)) if len(thetas) else 0.0
    feats["cv_theta_std"] = float(np.std(thetas)) if len(thetas) else 0.0
    feats["cv_accumulator_max"] = float(accumulator_max)

    if len(thetas):
        horiz = np.sum(np.abs(thetas) <= 10)
        upward = np.sum(thetas > 10)
        downward = np.sum(thetas < -10)
        total = max(len(thetas), 1)
        feats["cv_frac_horizontal"] = float(horiz / total)
        feats["cv_frac_upward"] = float(upward / total)
        feats["cv_frac_downward"] = float(downward / total)

    return feats


__all__ = ["extract_hough_features"]
