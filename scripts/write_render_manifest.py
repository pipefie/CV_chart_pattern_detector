from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime
from src.utils.run_manifest import write_render_manifest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="data/images/rendered")
    ap.add_argument("--backtest", default="configs/backtest.yaml")
    ap.add_argument("--render", default="configs/render.yaml")
    ap.add_argument("--patterns", default="configs/patterns.yaml")
    ap.add_argument("--window_bars", type=int, required=True)
    ap.add_argument("--stride_bars", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--runs_dir", default="reports/runs")
    args = ap.parse_args()

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M") + f"_seed{args.seed}"
    save_dir = Path(args.runs_dir) / run_id
    manifest = write_render_manifest(
        out_root=Path(args.out_root),
        backtest_yaml=Path(args.backtest),
        render_yaml=Path(args.render),
        patterns_yaml=Path(args.patterns),
        run_id=run_id,
        window_bars=args.window_bars,
        stride_bars=args.stride_bars,
        seed=args.seed,
        save_dir=save_dir,
    )
    print(f"✅ Render manifest written: {save_dir/'render_manifest.json'}")

if __name__ == "__main__":
    main()
