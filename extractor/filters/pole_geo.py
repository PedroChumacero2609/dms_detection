from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from pyproj import Transformer
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree


def find_georef_yaml(
    ply_path: Path,
    extra_dirs: list[Path] | None = None,
    requested: Path | None = None,
) -> Path:
    if requested is not None:
        if not requested.is_file():
            raise FileNotFoundError(f"YAML file not found: {requested}")
        return requested

    dirs: list[Path] = [ply_path.parent]
    if extra_dirs:
        dirs.extend(Path(d) for d in extra_dirs)

    valid: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.glob("*.yaml")):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                document = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                geo = document.get("geo_coord", {}) if isinstance(document, dict) else {}
                if all(key in geo for key in ("lat", "lon", "altitude")):
                    valid.append(candidate)
            except (OSError, UnicodeError, yaml.YAMLError):
                continue

    if len(valid) == 1:
        return valid[0]
    if not valid:
        raise ValueError(
            "No YAML with geo_coord.lat, geo_coord.lon, and "
            "geo_coord.altitude was found beside the PLY or in others_dir."
        )
    names = ", ".join(str(p) for p in valid)
    raise ValueError(f"Multiple georeferencing YAML files found ({names}).")


def read_enu_origin(yaml_path: Path) -> tuple[float, float, float]:
    with yaml_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    try:
        geo = document["geo_coord"]
        return float(geo["lat"]), float(geo["lon"]), float(geo["altitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{yaml_path} does not contain valid geo_coord values for lat, lon, and altitude."
        ) from exc


def build_enu_to_wgs84(lat: float, lon: float, altitude: float) -> Transformer:
    pipeline = (
        "+proj=pipeline "
        f"+step +inv +proj=topocentric +lat_0={lat:.17g} "
        f"+lon_0={lon:.17g} +h_0={altitude:.17g} +ellps=WGS84 "
        "+step +inv +proj=cart +ellps=WGS84"
    )
    return Transformer.from_pipeline(pipeline)


def read_trajectory(trajectory_path: Path) -> np.ndarray:
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")
    data = np.loadtxt(trajectory_path, comments="#", ndmin=2)
    if data.shape[1] < 4:
        raise ValueError(
            f"La trayectoria {trajectory_path} must contain at least timestamp, x, y, z."
        )
    return data[:, 1:4]


def find_full_gps_trajectory(
    ply_path: Path,
    others_dir: Path,
    tum_path: Path | None = None,
    requested: Path | None = None,
) -> tuple[Path, np.ndarray]:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(Path(requested).resolve())
    if tum_path is not None and Path(tum_path).is_file():
        candidates.append(Path(tum_path).resolve())

    search_dirs = [others_dir, ply_path.parent, ply_path.parent.parent]
    for directory in search_dirs:
        if Path(directory).is_dir():
            candidates.extend(sorted(Path(directory).glob("gps_tum_*.txt")))
            candidates.extend(sorted(Path(directory).glob("*trajectory*tum*")))

    seen: set[Path] = set()
    rejected: list[str] = []
    for candidate in candidates:
        candidate = Path(candidate).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            xyz = read_trajectory(candidate)
        except (OSError, ValueError):
            continue
        horizontal_extent = float(np.linalg.norm(np.ptp(xyz[:, :2], axis=0)))
        if len(xyz) >= 50 and horizontal_extent >= 20.0:
            return candidate, xyz
        rejected.append(
            f"{candidate.name} ({len(xyz)} poses, {horizontal_extent:.3f} m)"
        )

    detail = ", ".join(rejected) if rejected else "no candidates"
    raise ValueError(
        "No complete GPS ENU trajectory was found to estimate heading. "
        f"Rejected candidates: {detail}."
    )


def estimate_map_to_enu_yaw(
    pole_centers: np.ndarray, gps_xy: np.ndarray
) -> tuple[float, np.ndarray]:
    tree = cKDTree(gps_xy)

    def evaluate(angle: float) -> float:
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array(((c, -s), (s, c)))
        rotated = pole_centers @ rot.T
        dist, _ = tree.query(rotated)
        return float(np.mean(np.square(dist)))

    grid = np.linspace(-np.pi, np.pi, 1441)
    scores = np.asarray([evaluate(a) for a in grid])
    initial = float(grid[np.argmin(scores)])
    step = float(grid[1] - grid[0])
    result = minimize_scalar(
        evaluate,
        bounds=(initial - 2 * step, initial + 2 * step),
        method="bounded",
        options={"xatol": 1e-12},
    )
    yaw = float(result.x)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array(((c, -s), (s, c)))
    rotated = pole_centers @ rot.T
    dist, _ = tree.query(rotated)
    return yaw, dist