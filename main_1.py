"""Run the first-stage pipeline: Extractor -> Cable MT -> Connector -> Points -> Classifier -> Rebuild."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.json"

STEPS = [
    ("Extractor", "extractor.base_extractor"),
    ("Cable MT (interactive)", "cable_mt.cable_mt"),
    ("Connector", "connector.base_connector"),
    ("Points: wire equations", "points.wires"),
    ("Points: support points", "points.points"),
    ("Points: refined equations", "points.wires_2"),
    ("Classifier (interactive)", "classifier.base_classifier"),
    ("Rebuild (Open3D viewer)", "rebuild.base_rebuild"),
]


def main():
    parser = argparse.ArgumentParser(description="Run the first-stage MV line pipeline")
    parser.add_argument("-i", "--input", help="Input PLY filename in data_dir")
    parser.add_argument("-o", "--output", help="Output report directory")
    args = parser.parse_args()

    # Overrides only affect filenames and paths, never algorithm settings.
    if args.input or args.output:
        with CONFIG_FILE.open(encoding="utf-8") as f:
            config = json.load(f)
        if args.input:
            config["input_ply_raw"] = args.input
        if args.output:
            config["output_dir"] = args.output
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            f.write("\n")

    started = time.perf_counter()
    for index, (label, module) in enumerate(STEPS, 1):
        print(f"[INFO] [PIPELINE] {index}/{len(STEPS)}: {label}", flush=True)
        try:
            subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] Step {index} ({label}) failed (exit code {exc.returncode}).", file=sys.stderr)
            return exc.returncode or 1

    elapsed = time.perf_counter() - started
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"[PIPELINE] First stage completed: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
