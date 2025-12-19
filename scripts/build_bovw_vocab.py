#!/usr/bin/env python
"""
Build a Bag-of-Visual-Words vocabulary from ORB descriptors on standardized images.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from typing import Dict, List

import numpy as np
import yaml
from joblib import dump
from sklearn.cluster import MiniBatchKMeans

from src.features.cv_local import extract_orb_descriptors
from src.features.dataset import load_yaml as load_pipeline_yaml, samples_from_manifest


def sample_images(manifest_path: Path | None, images_root: Path, limit: int, seed: int) -> List[Path]:
    rng = random.Random(seed)
    imgs: List[Path] = []
    if manifest_path and manifest_path.exists():
        for sample in samples_from_manifest(manifest_path):
            imgs.append(sample.image_path)
    else:
        for split in ("train", "val", "test"):
            imgs.extend(sorted((images_root / split).glob("*.png")))
    rng.shuffle(imgs)
    return imgs[:limit] if limit > 0 else imgs


def main():
    ap = argparse.ArgumentParser(description="Build ORB BoVW vocabulary.")
    ap.add_argument("--pipeline_cfg", default="configs/pipeline.yaml")
    ap.add_argument("--manifest", default=None, help="Optional render manifest JSON to enumerate images.")
    ap.add_argument("--images_root", default="data/images/standardized")
    ap.add_argument("--out_dir", default="reports/cv_vocab")
    args = ap.parse_args()

    cfg = load_pipeline_yaml(Path(args.pipeline_cfg))
    bovw_cfg: Dict = (cfg.get("cv_features") or {}).get("bovw", {}) or {}
    if not bovw_cfg:
        raise ValueError("cv_features.bovw section missing in pipeline config.")
    n_clusters = int(bovw_cfg.get("n_clusters", 64))
    max_kp = int(bovw_cfg.get("max_keypoints", 500))
    sample_imgs = int(bovw_cfg.get("sample_images_for_vocab", 2000))
    max_desc = int(bovw_cfg.get("max_descriptors_total", 200000))
    seed = int(bovw_cfg.get("random_state", 42))

    manifest_path = Path(args.manifest) if args.manifest else None
    images_root = Path(args.images_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = sample_images(manifest_path, images_root, sample_imgs, seed)
    if not candidates:
        raise RuntimeError("No images found to build vocabulary.")

    rng = np.random.default_rng(seed)
    desc_list = []
    used_images = 0
    for img_path in candidates:
        desc = extract_orb_descriptors(img_path, {"max_keypoints": max_kp})
        if desc is None or len(desc) == 0:
            continue
        desc_list.append(desc)
        used_images += 1
        if sum(len(d) for d in desc_list) >= max_desc:
            break

    if not desc_list:
        raise RuntimeError("No ORB descriptors extracted; check images/standardization.")

    all_desc = np.vstack(desc_list)
    if len(all_desc) > max_desc:
        all_desc = all_desc[:max_desc]

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=2048)
    kmeans.fit(all_desc)

    vocab_name = f"bovw_k{n_clusters}_seed{seed}.joblib"
    vocab_path = out_dir / vocab_name
    dump(kmeans, vocab_path)

    manifest = {
        "n_clusters": n_clusters,
        "max_keypoints": max_kp,
        "sample_images_requested": sample_imgs,
        "sample_images_used": used_images,
        "descriptors_used": int(len(all_desc)),
        "random_state": seed,
        "vocab_path": str(vocab_path),
        "source_manifest": str(manifest_path) if manifest_path else None,
        "images_root": str(images_root),
    }
    (out_dir / f"{vocab_name}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"✅ Saved BoVW vocab to {vocab_path}")
    print(f"Images used: {used_images} | Descriptors: {len(all_desc)} | n_clusters: {n_clusters}")


if __name__ == "__main__":
    main()
