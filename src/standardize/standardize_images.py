# src/standardize/standardize_images.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2 as cv
import numpy as np
from PIL import Image

def _order_quad(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as [TL, TR, BR, BL]: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    We need a consistent order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    This helps ensure that the corners are always in the same order, which is important for the perspective transformation.
    Trick:

        Sum of coordinates x+y: smallest → top-left, largest → bottom-right.

        Difference x−y: smallest → top-right, largest → bottom-left.

    This heuristic works for convex quadrilaterals robustly.
    """
    # pts: (4,2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _largest_rect_contour(gray: np.ndarray) -> np.ndarray | None:
    """Find the largest rectangular-ish contour likely to be the plot area."""
    # Edge detection 
    edges = cv.Canny(gray, 50, 150)
    # connect edges
    # Builds a 5×5 rectangular structuring element (kernel) for morphological ops.
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    # Apply morphological closing to fill small gaps in the edges. Morphological closing = dilation followed by erosion
    # Dilation grows white regions (fills small holes), erosion then trims the growth but keeps gaps closed.
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)
    # Find contours (RETR_EXTERNAL retrieves only the outermost contours (we just need the biggest frame))
    cnts, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    # try to approximate first few large contours
    for c in cnts[:5]:
        # Calculate the perimeter length of the contour
        peri = cv.arcLength(c, True)
        # Approximate the contour with a polygon.
        # 0.02 * peri: Epsilon factor: 2% of the contour's perimeter length.
        # True: Close the polygon (connect the last and first points).
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        # rectangular-ish: 4 points, convex, area threshold
        if len(approx) == 4 and cv.isContourConvex(approx): # Checks if the polygon is convex. Chart frames are convex rectangles, so this filters odd shapes
            area = cv.contourArea(approx)
            if area > 0.1 * (gray.shape[0] * gray.shape[1]):  # Select contours whose area is at least a fraction (e.g., 10%) of the whole image
                return approx.reshape(-1, 2).astype(np.float32)
    return None

def _fallback_content_box(gray: np.ndarray) -> np.ndarray:
    """Fallback: compute a central content box by trimming low-entropy/flat margins.
    If the frame/axes aren’t clear, we use a texture heuristic to crop active content
    """
    # Blur the image to reduce noise and smooth out texture variations, while preserving overall image structure
    blur = cv.GaussianBlur(gray, (0,0), 1.5)
    # Compute the Laplacian of the blurred image. The Laplacian is a second-order derivative operator that highlights edges and changes in intensity.
    lap = cv.Laplacian(blur, cv.CV_32F)
    # Compute the magnitude of the Laplacian. The magnitude is the absolute value of the Laplacian.
    mag = np.abs(lap)
    mag = (mag / (mag.max() + 1e-6)) # Normalize the magnitude to the range [0,1]
    # threshold on activity to get content mask: We set a threshold (0.04) to identify areas of high texture activity (i.e., where the image is changing rapidly).
    # margins/titles/wide empty areas have low Laplacian magnitude; the plot area with candles/wicks/grid has higher local variation
    thr = 0.04
    mask = (mag > thr).astype(np.uint8) * 255
    # close small holes
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    # bounding box
    ys, xs = np.where(mask > 0)
    if len(xs) < 50 or len(ys) < 50:
        # extreme fallback: 90% central box
        h, w = gray.shape
        padx, pady = int(0.05*w), int(0.05*h)
        return np.array([[padx, pady],[w-padx, pady],[w-padx, h-pady],[padx, h-pady]], dtype=np.float32)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([[x0, y0],[x1, y0],[x1, y1],[x0, y1]], dtype=np.float32)

def standardize_one(img_path: Path, out_dir: Path, W: int = 960, H: int = 640, inset: float = 0.02) -> dict:
    """
    Returns a dict with transform info and writes the standardized image.
    inset: fraction of width/height to crop inside the warped rectangle to remove spines.
    """
    # Load with OpenCV (BGR) and keep original dims
    img_bgr = cv.imread(str(img_path), cv.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot read {img_path}")
    h0, w0 = img_bgr.shape[:2]

    # Downscale for robustness if huge
    scale = 1.0
    max_side = max(h0, w0)
    if max_side > 2000:
        scale = 2000.0 / max_side
        img_small = cv.resize(img_bgr, (int(w0*scale), int(h0*scale)), interpolation=cv.INTER_AREA)
    else:
        img_small = img_bgr.copy()

    # Convert to grayscale
    gray = cv.cvtColor(img_small, cv.COLOR_BGR2GRAY)

    # Find the largest rectangular-ish contour likely to be the plot area
    quad = _largest_rect_contour(gray)
    if quad is None:
        quad = _fallback_content_box(gray)

    # map quad back to original coords if we downscaled
    if scale != 1.0:
        quad = quad / float(scale)

    # order corners and compute homography
    src_quad = _order_quad(quad)  # TL, TR, BR, BL

    # target rectangle (optionally inset to remove borders)
    # We define the destination rectangle as (W×H), with a small inset (e.g., 2%) on each side to cut out axes spines/borders.
    Wt, Ht = W, H
    inset_x = inset * Wt
    inset_y = inset * Ht
    dst_quad = np.array([
        [0+inset_x, 0+inset_y],
        [Wt-inset_x, 0+inset_y],
        [Wt-inset_x, Ht-inset_y],
        [0+inset_x, Ht-inset_y]
    ], dtype=np.float32)

    # Compute the perspective transformation (3×3 projective transform) matrix H that maps the source quadrilateral to the destination quadrilateral.
    Hmat = cv.getPerspectiveTransform(src_quad, dst_quad)
    # Apply the perspective transformation to the original image to get the warped image.
    warped = cv.warpPerspective(img_bgr, Hmat, (Wt, Ht), flags=cv.INTER_LINEAR)

    # optional light denoise / contrast normalize
    # (keep gentle to not distort candles)
    warped_lab = cv.cvtColor(warped, cv.COLOR_BGR2LAB)
    L, A, B = cv.split(warped_lab)
    L = cv.equalizeHist(L)
    warped = cv.cvtColor(cv.merge([L,A,B]), cv.COLOR_LAB2BGR)

    # write PNG
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / (img_path.stem + ".png")
    cv.imwrite(str(out_png), warped)

    # transform record
    transform = {
        "source_path": str(img_path),
        "orig_w": int(w0), "orig_h": int(h0),
        "std_w": int(Wt),  "std_h": int(Ht),
        "src_quad": src_quad.tolist(),   # TL,TR,BR,BL
        "dst_quad": dst_quad.tolist(),
        "H": Hmat.tolist(),
        "inset_frac": float(inset)
    }
    (out_dir / (img_path.stem + ".transform.json")).write_text(
        json.dumps(transform, indent=2), encoding="utf-8"
    )

    return transform

def main():
    ap = argparse.ArgumentParser(description="Standardize chart images to a canonical, rectified ROI.")
    ap.add_argument("--inp", required=True, help="input folder (e.g., data/images/real/screenshots)")
    ap.add_argument("--out", required=True, help="output folder (e.g., data/images/standardized/screenshots)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--inset", type=float, default=0.02)
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    exts = {".png",".jpg",".jpeg",".bmp",".webp"}
    imgs = [p for p in inp.rglob("*") if p.suffix.lower() in exts]
    print(f"Found {len(imgs)} images.")

    for p in imgs:
        try:
            standardize_one(p, out, W=args.width, H=args.height, inset=args.inset)
            print(f"✅ {p.name}")
        except Exception as e:
            print(f"⚠️ {p.name}: {e}")

if __name__ == "__main__":
    main()
