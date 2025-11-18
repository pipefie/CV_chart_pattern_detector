# scripts/audit_kaggle.py
from __future__ import annotations
import argparse, csv, json, random, shutil, sys, textwrap
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont

# Optional: labeling functions plugin (safe import)
try:
    import lf_rules  # your LFs live here (optional)
    HAS_LF = True
except Exception:
    HAS_LF = False

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def find_classes(root: Path) -> Dict[str, List[Path]]:
    """Expect a directory dataset like root/<class_name>/*.png"""
    class_to_imgs: Dict[str, List[Path]] = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir(): continue
        imgs = [p for p in sub.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        if imgs:
            class_to_imgs[sub.name] = imgs
    return class_to_imgs

def sample_per_class(class_to_imgs: Dict[str, List[Path]], k: int, seed: int) -> Dict[str, List[Path]]:
    rng = random.Random(seed)
    picks: Dict[str, List[Path]] = {}
    for cls, imgs in class_to_imgs.items():
        if len(imgs) <= k:
            picks[cls] = imgs
        else:
            picks[cls] = rng.sample(imgs, k)
    return picks

def make_contact_sheet(images: List[Path], out_path: Path, cols: int = 6, thumb: int = 256) -> None:
    if not images:
        return
    rows = (len(images) + cols - 1) // cols
    W = cols * thumb
    H = rows * thumb
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    for i, p in enumerate(images):
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb, thumb))
        except Exception:
            im = Image.new("RGB", (thumb, thumb), (220, 220, 220))
        r, c = divmod(i, cols)
        x, y = c * thumb, r * thumb
        canvas.paste(im, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

def write_html_gallery(class_samples: Dict[str, List[Path]], out_html: Path, rel_root: Path) -> None:
    """Simple static HTML pointing to thumbnails + originals."""
    lines = []
    lines.append("<html><head><meta charset='utf-8'><style>")
    lines.append("body{font-family:system-ui,Arial,sans-serif;max-width:1200px;margin:24px auto;}")
    lines.append(".row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}")
    lines.append(".card{display:inline-block;border:1px solid #ddd;padding:8px;border-radius:8px;background:#fff}")
    lines.append("img{max-width:240px;height:auto;display:block}")
    lines.append("</style></head><body>")
    lines.append("<h1>Kaggle Dataset Audit</h1>")
    for cls, imgs in class_samples.items():
        lines.append(f"<h2>{cls} ({len(imgs)})</h2><div class='row'>")
        for p in imgs:
            rel = p.relative_to(rel_root).as_posix()
            lines.append(f"<div class='card'><a href='{rel}' target='_blank'><img src='{rel}'/></a><div style='font-size:12px;color:#666'>{rel}</div></div>")
        lines.append("</div>")
    lines.append("</body></html>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(lines), encoding="utf-8")

def run_lfs_on_image(img_path: Path, class_name: str) -> Dict:
    """
    Calls your labeling functions (if lf_rules.py is present).
    Returns a dict with raw LF outputs and a combined score.
    """
    if not HAS_LF:
        return {"has_lf": False}
    try:
        outs = lf_rules.apply_all(img_path, class_name)
        combo = lf_rules.combine_votes(outs)   # prob + decision + meta
        return {"has_lf": True, "lf_raw": outs, "lf_combo": combo}
    except Exception as e:
        return {"has_lf": True, "lf_error": str(e)}

def main():
    ap = argparse.ArgumentParser(description="Audit a Kaggle-style dataset of chart patterns.")
    ap.add_argument("--dataset_root", required=True, help="Folder with subfolders per class (e.g., head_and_shoulders/, double_top/, ...)")
    ap.add_argument("--out_dir", default="reports/audit/kaggle")
    ap.add_argument("--per_class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy_samples", action="store_true", help="Copy sampled images into the audit folder for easy sharing")
    ap.add_argument("--make_contact_sheets", action="store_true")
    ap.add_argument("--html_gallery", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Discover classes
    class_to_imgs = find_classes(dataset_root)
    if not class_to_imgs:
        print("No classes with images found. Check dataset structure.")
        sys.exit(1)

    print("Found classes:")
    for k, v in class_to_imgs.items():
        print(f" - {k}: {len(v)} images")

    # 2) Sample per class
    class_samples = sample_per_class(class_to_imgs, args.per_class, args.seed)
    total = sum(len(v) for v in class_samples.values())
    print(f"Sampling {total} images across {len(class_samples)} classes.")

    # 3) Optionally copy sampled images into out_dir/samples/<class>/
    if args.copy_samples:
        for cls, imgs in class_samples.items():
            dst = out_dir / "samples" / cls
            dst.mkdir(parents=True, exist_ok=True)
            for p in imgs:
                # keep relative structure minimal: copy with original filename; disambiguate collisions
                stem = p.stem; ext = p.suffix.lower()
                new_name = stem + ext
                i = 1
                while (dst / new_name).exists():
                    new_name = f"{stem}_{i}{ext}"; i += 1
                shutil.copy2(p, dst / new_name)

    # 4) Contact sheets & HTML
    if args.make_contact_sheets:
        for cls, imgs in class_samples.items():
            sheet = out_dir / "contact_sheets" / f"{cls}.jpg"
            make_contact_sheet(imgs, sheet, cols=6, thumb=240)
    if args.html_gallery:
        write_html_gallery(class_samples, out_dir / "gallery.html", rel_root=dataset_root)

    # 5) Create an audit CSV to manually rate a subset quickly
    #    Columns: image_path, class_name, lf_decision, lf_prob, notes, keep(0/1), correct_class(0/1)
    csv_path = out_dir / "audit_sheet.csv"
    fields = [
        "image_path","class_name",
        "lf_has","lf_decision","lf_prob","lf_meta","lf_error",
        "keep","correct_class","notes"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cls, imgs in class_samples.items():
            for p in imgs:
                lf = run_lfs_on_image(p, cls)
                row = {
                    "image_path": str(p),
                    "class_name": cls,
                    "lf_has": int(lf.get("has_lf", False)),
                    "lf_decision": "",
                    "lf_prob": "",
                    "lf_meta": "",
                    "lf_error": lf.get("lf_error",""),
                    "keep": "",
                    "correct_class": "",
                    "notes": "",
                }
                if lf.get("has_lf") and "lf_combo" in lf:
                    combo = lf["lf_combo"]
                    row["lf_decision"] = combo.get("decision","")
                    row["lf_prob"]     = combo.get("prob","")
                    row["lf_meta"]     = json.dumps(combo.get("meta",{}))
                w.writerow(row)

    # 6) Print next steps
    msg = textwrap.dedent(f"""
    ✅ Audit sheet ready: {csv_path}
       - Open it in a spreadsheet tool.
       - For each row, fill:
         * keep: 1 if the image is good quality / usable; 0 otherwise
         * correct_class: 1 if the folder label looks correct; 0 otherwise
         * notes: write why (e.g., misclass, low res, cropped, no pattern)
       - (Optional) Compare your ratings vs. lf_decision/lf_prob (if LFs enabled).

    Optional: If you ran --copy_samples, the sampled images are in {out_dir/'samples'}.
              Contact sheets (if any): {out_dir/'contact_sheets'}
              HTML gallery (if any):   {out_dir/'gallery.html'}

    When done, you can summarize precision/coverage per class with a tiny pandas script,
    or I can give you a helper to parse 'audit_sheet.csv' and print stats.
    """).strip()
    print(msg)

if __name__ == "__main__":
    main()
