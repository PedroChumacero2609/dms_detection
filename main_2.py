#!/usr/bin/env python3
"""Run the independent DMS collision and video pipeline (not part of main_1)."""
import argparse
import json
import runpy
import sys
import time
from pathlib import Path

from dms.auto_gps import process_collision_videos

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
STAGES = (
    ("DMS collision detection", "dms.zones"),
    ("Collision point-cloud extraction", "dms.split"),
    ("Collision report and image export", "dms.export_info"),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the independent DMS pipeline")
    parser.add_argument("--rosbag", type=Path, help="MCAP or ROS 2 bag path (otherwise discover in others/)")
    args = parser.parse_args(argv)
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if type(config.get("GPU")) is not bool:
        parser.error('config.json must contain "GPU": true or false')

    started = time.perf_counter()
    print(f"[INFO] [DMS] Running 4 stages | GPU={config['GPU']}", flush=True)
    for index, (description, module) in enumerate(STAGES, 1):
        print(f"[INFO] [DMS] {index}/4: {description}", flush=True)
        runpy.run_module(module, run_name="__main__")
    print("[INFO] [DMS] 4/4: GPS collision video extraction", flush=True)
    count = process_collision_videos(config_path=str(CONFIG_PATH),
                                     rosbag_path=str(args.rosbag.resolve()) if args.rosbag else None)
    if count == 0:
        raise RuntimeError("No compatible collision videos were generated")
    elapsed = time.perf_counter() - started
    print(f"[INFO] [DMS] Completed: {count} videos | elapsed {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
