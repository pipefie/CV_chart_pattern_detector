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


def extract_orb_descriptors(image_path: str | Path, config: Dict | None) -> np.ndarray:
    """
    Extract ORB descriptors from a grayscale image.

    Parameters
    ----------
    image_path : str | Path
        Path to a standardized chart image.
    config : Dict | None
        ORB settings (e.g., max_keypoints).

    Returns
    -------
    np.ndarray
        Array of shape (n_keypoints, descriptor_dim) or empty array if no keypoints/ORB unavailable.
    """
    if not _HAS_CV2:
        LOG.debug("cv2 not available; skipping ORB for %s", image_path)
        return np.empty((0, 32), dtype=np.uint8)
    params = config or {}
    max_kp = int(params.get("max_keypoints", 500))
    orb = cv.ORB_create(nfeatures=max_kp)
    img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if img is None:
        LOG.debug("Failed to read image for ORB: %s", image_path)
        return np.empty((0, 32), dtype=np.uint8)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    if descriptors is None or len(descriptors) == 0:
        return np.empty((0, 32), dtype=np.uint8)
    return descriptors


def bovw_histogram(descriptors: np.ndarray, vocab_model, n_clusters: int, norm: str = "l2") -> np.ndarray:
    """
    Build a Bag-of-Visual-Words histogram from ORB descriptors using a fitted KMeans/MiniBatchKMeans model.

    Parameters
    ----------
    descriptors : np.ndarray
        ORB descriptor array of shape (n_keypoints, descriptor_dim).
    vocab_model :
        Fitted sklearn KMeans/MiniBatchKMeans model with a predict method.
    n_clusters : int
        Number of clusters (vocabulary size).
    norm : str
        Normalization mode: "l1", "l2", or "none".

    Returns
    -------
    np.ndarray
        Histogram of shape (n_clusters,) dtype=float.
    """
    hist = np.zeros(int(n_clusters), dtype=float)
    if descriptors is None or len(descriptors) == 0:
        return hist
    try:
        labels = vocab_model.predict(descriptors)
    except Exception:
        return hist
    hist = np.bincount(labels, minlength=int(n_clusters)).astype(float)
    if norm == "l1":
        s = hist.sum()
        if s > 0:
            hist /= s
    elif norm == "l2":
        s = np.linalg.norm(hist)
        if s > 0:
            hist /= s
    return hist


__all__ = ["extract_orb_descriptors", "bovw_histogram"]
