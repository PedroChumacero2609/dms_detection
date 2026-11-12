
import os
from pathlib import Path

from pyproj import datadir

# =========================================================
# FIX: PROJ DATABASE INITIALIZATION
# =========================================================

conda_prefix = os.environ.get("CONDA_PREFIX")

if conda_prefix:
    proj_dir = Path(conda_prefix) / "share" / "proj"

    if (proj_dir / "proj.db").is_file():
        datadir.set_data_dir(str(proj_dir))

# =========================================================
# ROUTE IMPORTS
# =========================================================

from .extractors.route_extractor import extract_route
from .extractors.video_extractor import extract_video


def extract_all() -> tuple:
    print(
        '[INFO] [ROUTE] Starting route and video export...',
        flush=True
    )

    png = extract_route()
    mp4 = extract_video()

    print(
        '[INFO] [ROUTE] Route and video export completed.',
        flush=True
    )

    return png, mp4


if __name__ == "__main__":
    extract_all()