from __future__ import annotations
import json, hashlib, subprocess, sys, importlib
from datetime import datetime, timezone
from pathlib import Path
import yaml
from collections import Counter

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def package_version(name: str) -> str:
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return "not-installed"

def get_env_versions() -> dict:
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "pandas": package_version("pandas"),
        "matplotlib": package_version("matplotlib"),
        "mplfinance": package_version("mplfinance"),
        "Pillow": package_version("PIL"),
        "pyyaml": package_version("yaml"),
        "ccxt": package_version("ccxt"),
    }

def count_pngs(out_root: Path) -> dict:
    counts = Counter()
    for split in ["train","val","test"]:
        p = out_root / split
        counts[f"{split}_count"] = len(list(p.glob("*.png"))) if p.exists() else 0
    return counts

def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def write_render_manifest(
    out_root: Path,
    backtest_yaml: Path,
    render_yaml: Path,
    patterns_yaml: Path,
    run_id: str,
    window_bars: int,
    stride_bars: int,
    seed: int,
    save_dir: Path,
):
    back = read_yaml(backtest_yaml)
    env = get_env_versions()
    hashes = {
        "backtest.yaml": sha256_file(backtest_yaml),
        "render.yaml": sha256_file(render_yaml),
        "patterns.yaml": sha256_file(patterns_yaml) if patterns_yaml.exists() else "absent",
    }
    counts = count_pngs(out_root)
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": get_git_commit(),
        "env": env,
        "configs": {
            "backtest_yaml_path": str(backtest_yaml),
            "render_yaml_path": str(render_yaml),
            "patterns_yaml_path": str(patterns_yaml),
            "hashes": hashes,
        },
        "render_params": {
            "window_bars": window_bars,
            "stride_bars": stride_bars,
            "seed": seed,
        },
        "universes": back.get("universes", {}),
        "outputs": {
            **counts,
            "out_root": str(out_root),
        },
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
