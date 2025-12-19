# scripts/lf_rules.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np

try:
    import cv2 as cv
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

ABSTAIN = -1

def lf_grid_presence(img) -> int:
    """Detects presence of grid-like structure; helpful sanity check for charts."""
    if not HAS_CV2: return ABSTAIN
    edges = cv.Canny(img, 50, 150)
    lines = cv.HoughLines(edges, 1, np.pi/180, 120)
    if lines is None: return 0
    # if many lines near 0° or 90°, likely a chart grid
    degs = []
    for rtheta in lines[:60]:
        rho, theta = rtheta[0]
        deg = np.degrees(theta)
        if deg >= 90: deg -= 180
        degs.append(abs(deg))
    verticals = sum(d > 80 for d in degs)
    horizontals = sum(d < 10 for d in degs)
    return 1 if (verticals + horizontals) >= 4 else 0

def lf_triangle_like(img) -> int:
    """Generic triangle proxy: many converging oblique lines, not too horizontal/vertical."""
    if not HAS_CV2: return ABSTAIN
    edges = cv.Canny(img, 60, 160)
    lines = cv.HoughLines(edges, 1, np.pi/180, 90)
    if lines is None: return 0
    degs = []
    for rtheta in lines[:60]:
        rho, theta = rtheta[0]
        deg = np.degrees(theta)
        if deg >= 90: deg -= 180
        degs.append(deg)
    # prefer a mix of positive and negative slopes (converging feel)
    pos = sum(d > 10 for d in degs)
    neg = sum(d < -10 for d in degs)
    return 1 if (pos >= 2 and neg >= 2) else 0

def lf_text_axis_density(img) -> int:
    """If OCR-ish density near borders is super high, picture may be cropped UI instead of chart region."""
    if not HAS_CV2: return ABSTAIN
    h, w = img.shape[:2]
    k = max(5, w//80)
    left = img[:, :k]; right = img[:, -k:]
    top = img[:k, :]; bottom = img[-k:, :]
    edges = cv.Canny(img, 50, 150)
    score = (edges[:, :k].mean() + edges[:, -k:].mean() + edges[:k, :].mean() + edges[-k:, :].mean())/4.0
    return 0 if score > 100 else 1  # abstain from low-quality UI crops

def read_image_gray(path: Path):
    if not HAS_CV2:
        raise RuntimeError("cv2 not available")
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    return img

def apply_all(img_path: Path, claimed_class: str) -> Dict[str, int]:
    """Return dict of LF name -> vote in {1,0,-1} (present/absent/abstain)."""
    if not HAS_CV2:
        return {"lf_grid_presence": ABSTAIN, "lf_triangle_like": ABSTAIN, "lf_text_axis_density": ABSTAIN}
    img = read_image_gray(img_path)
    votes = {
        "lf_grid_presence": lf_grid_presence(img),
        "lf_text_axis_density": lf_text_axis_density(img),
    }
    if "triangle" in claimed_class.lower():
        votes["lf_triangle_like"] = lf_triangle_like(img)
    return votes

def combine_votes(votes: Dict[str, int]) -> Dict[str, Any]:
    """
    Weighted majority vote (toy). You can later learn weights from your gold set.
    - abstain=-1 is ignored
    """
    weights = {name: 1.0 for name in votes.keys()}  # equal weights for now
    pos, neg, tot = 0.0, 0.0, 0.0
    active = {}
    for name, v in votes.items():
        if v == -1:
            continue
        w = weights[name]
        active[name] = v
        tot += w
        if v == 1: pos += w
        else:      neg += w
    prob = (pos / tot) if tot > 0 else 0.0
    decision = 1 if prob >= 0.6 else 0 if prob <= 0.4 else ""  # undecided in the gray zone
    return {"prob": float(prob), "decision": decision, "meta": {"active_votes": active, "total_weight": tot}}
