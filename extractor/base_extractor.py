"""
Base Extractor
==============
Georeferences labeled MV (Medium Voltage) poles from a local PLY file, filters those
within the route (KML), and extracts geometry (individual PLY + JSON).

Workflow:
1.  Load config.json
2.  Read labeled PLY (MV Poles class (ID 6) from labels.json)
3.  Locate YAML with geo_coord (next to PLY or in others_dir)
4.  Cluster poles via interface.py (current method: DBSCAN)
5.  Estimate map -> ENU rotation using the full GPS trajectory
6.  ENU -> WGS84 (lat/lon)
7.  Write provisional GPS.txt to others_dir
8.  Filter poles within the route (KML + provisional GPS.txt)
9.  Renumber and write final GPS.txt to data_dir
10. Extract geometry: pole_XXX.ply + poles_info.json

Horizontal unit vector is obtained from the outlier points (those outside the
cylinder of radius `horizontal_radius`) projected onto the XY plane, then
orthogonalized against the vertical unit vector.
"""

from __future__ import annotations

import faulthandler
import gc
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from pyproj import datadir

conda_prefix = os.environ.get("CONDA_PREFIX")

if conda_prefix:
    proj_dir = Path(conda_prefix) / "share" / "proj"

    if (proj_dir / "proj.db").is_file():
        datadir.set_data_dir(str(proj_dir))
        
from . import interface, scalable
from .filters import kml_filter, pole_geo


# =========================================================
# CONFIG
# =========================================================
CONFIG_FILE = "config.json"
LABELS_FILE = "labels.json"
# Non-DBSCAN extractor parameters remain in the existing source file.
PLY_CHUNK_VERTICES = 200_000
MAX_GEOMETRY_POINTS = 40_000
MAX_RETURN_POINTS = 250_000


LABEL_CANDIDATES = (
    "scalar_Label",
    "label",
    "Label",
    "classification",
    "class",
)


def load_config(path=CONFIG_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_pole_label(config_path: Path) -> int:
    """Read the MV Poles semantic class from the shared labels.json catalog.

    Entries use numeric IDs and descriptive titles, with optional groups.
    Reserved ranges are metadata and must not overlap defined classes.
    """
    labels_path = Path(config_path).resolve().parent / LABELS_FILE
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as f:
        catalog = json.load(f)

    if not isinstance(catalog, dict) or not isinstance(catalog.get("labels"), list):
        raise ValueError("labels.json must contain a 'labels' array")

    seen_ids = set()
    target_id = None
    for index, entry in enumerate(catalog["labels"]):
        if not isinstance(entry, dict):
            raise ValueError(f"labels.json: labels[{index}] must be an object")
        label_id = entry.get("id")
        title = entry.get("title")
        if type(label_id) is not int or label_id < 0:
            raise ValueError(f"labels.json: labels[{index}].id must be a non-negative integer")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"labels.json: labels[{index}].title must be non-empty")
        if label_id in seen_ids:
            raise ValueError(f"labels.json: duplicate label id {label_id}")
        seen_ids.add(label_id)
        if title.strip().casefold() == "mv poles":
            if target_id is not None:
                raise ValueError("labels.json: multiple entries named 'MV Poles'")
            target_id = label_id

    for index, reserved in enumerate(catalog.get("available_ranges", [])):
        if not isinstance(reserved, dict):
            raise ValueError(f"labels.json: available_ranges[{index}] must be an object")
        first, last = reserved.get("start"), reserved.get("end")
        if type(first) is not int or type(last) is not int or first < 0 or last < first:
            raise ValueError(f"labels.json: invalid available range at index {index}")
        if any(first <= value <= last for value in seen_ids):
            raise ValueError(f"labels.json: available range {first}-{last} overlaps defined labels")

    if target_id is None:
        raise ValueError("labels.json: required class 'MV Poles' not found")
    return target_id


# =========================================================
# PLY READING
# =========================================================
def read_labeled_points(ply_path: Path, target_label: int, *,
                        scratch_dir: Path | None = None,
                        chunk_size: int = 200_000,
                        in_memory_limit: int = 500_000) -> np.ndarray:
    """Select only labelled vertices; avoid allocating giant masks/copies.

    For binary PLY, plyfile normally memory-maps a fixed-width vertex element.
    ASCII PLY may still require a large in-memory parse by plyfile; converting
    extremely large ASCII sources to binary PLY is recommended.
    """
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise ValueError(f"{ply_path} does not contain a 'vertex' element.")

    vertex = ply["vertex"].data
    fields = vertex.dtype.names or ()
    missing_xyz = [f for f in ("x", "y", "z") if f not in fields]
    if missing_xyz:
        raise ValueError(f"PLY is missing coordinate fields: {missing_xyz}")

    label_field = next((n for n in LABEL_CANDIDATES if n in fields), None)
    if label_field is None:
        raise ValueError(
            "PLY has no label field. Accepted names: "
            + ", ".join(LABEL_CANDIDATES)
        )
    n_vertices = len(vertex)

    total = 0
    for start in range(0, n_vertices, chunk_size):
        total += int(np.count_nonzero(vertex[label_field][start:start + chunk_size]
                                      == target_label))
    if not total:
        raise ValueError(f"PLY contains no points with label {target_label}.")

    if total <= in_memory_limit or scratch_dir is None:
        selected = np.empty((total, 3), dtype=np.float64)
    else:
        temp_path = Path(scratch_dir) / "selected_points.npy"
        selected = np.lib.format.open_memmap(str(temp_path), mode="w+",
                                             dtype=np.float64, shape=(total, 3))

    offset = 0
    for start in range(0, n_vertices, chunk_size):
        block = vertex[start:start + chunk_size]
        mask = (block[label_field] == target_label)
        n = int(np.count_nonzero(mask))
        if n:
            # Separate columns avoid stacking and then converting the full set.
            selected[offset:offset + n, 0] = block["x"][mask]
            selected[offset:offset + n, 1] = block["y"][mask]
            selected[offset:offset + n, 2] = block["z"][mask]
            offset += n
    if isinstance(selected, np.memmap):
        selected.flush()
    del ply, vertex
    return selected


def write_ply_xyz_stream(path: Path, points: np.ndarray,
                         chunk_size: int = 200_000) -> None:
    """Write binary PLY without building an extra whole-cloud Open3D object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property double x\nproperty double y\nproperty double z\n"
              "end_header\n")
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        for start in range(0, len(points), chunk_size):
            data = np.asarray(points[start:start + chunk_size], dtype="<f8",
                              order="C")
            data.tofile(f)




# =========================================================
# FULL PLY WITH POLE-TOP IDS
# =========================================================
def export_pole_top_ply(source: Path, destination: Path, tops: list[tuple[int, np.ndarray]],
                        pole_label: int, chunk_size: int = 200_000) -> None:
    """Copy all source vertices/properties and mark exactly one existing point per pole.

    The candidate is the highest-Z point of the extracted pole cluster. Its
    original vertex is located with a small coordinate tolerance, restricted
    to points carrying the configured pole label. No vertices are added or
    reordered. A temporary output is atomically moved on successful export.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('The output PLY must differ from the original PLY.')
    ply = PlyData.read(str(source), mmap=True)
    vertex = ply['vertex'].data
    names = vertex.dtype.names or ()
    if 'poles_id' in names:
        raise ValueError('The input PLY already contains poles_id; use an unmodified source.')
    label_field = next((n for n in LABEL_CANDIDATES if n in names), None)
    if label_field is None:
        raise ValueError('The source PLY has no classification field.')
    if not tops:
        raise ValueError('No pole tops to export.')
    top_xyz = np.asarray([xyz for _, xyz in tops], dtype=np.float64)
    ids = [int(pid) for pid, _ in tops]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate Pole_ID detected.')
    tree = cKDTree(top_xyz)
    best_dist = np.full(len(tops), np.inf)
    best_index = np.full(len(tops), -1, dtype=np.int64)
    # Match only original points with the pole class; candidates are points
    # selected from the original cloud, so differences should be negligible.
    tolerance = 1e-4
    for start in range(0, len(vertex), chunk_size):
        block = vertex[start:start + chunk_size]
        local = np.flatnonzero(block[label_field] == pole_label)
        if not len(local):
            continue
        xyz = np.column_stack((block['x'][local], block['y'][local], block['z'][local]))
        dist, near = tree.query(xyz, distance_upper_bound=tolerance)
        for j in np.flatnonzero(np.isfinite(dist)):
            k = int(near[j])
            idx = start + int(local[j])
            if dist[j] < best_dist[k] or (dist[j] == best_dist[k] and idx < best_index[k]):
                best_dist[k], best_index[k] = float(dist[j]), idx
    missing = [ids[k] for k in np.flatnonzero(best_index < 0)]
    if missing:
        raise ValueError(f'Could not find original top vertices for Pole_ID: {missing}')
    if len(set(best_index.tolist())) != len(best_index):
        raise ValueError('Two poles matched the same original vertex; refusing ambiguous IDs.')

    new_dtype = np.dtype(list(vertex.dtype.descr) + [('poles_id', '<i4')])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix='_pole_ply_', dir=destination.parent))
    try:
        temp_array = temp_dir / 'vertices.dat'
        out = np.memmap(temp_array, dtype=new_dtype, mode='w+', shape=(len(vertex),))
        marks = {int(index): int(ids[k]) for k, index in enumerate(best_index)}
        for start in range(0, len(vertex), chunk_size):
            end = min(start + chunk_size, len(vertex))
            block = vertex[start:end]
            for name in names:
                out[name][start:end] = block[name]
            out['poles_id'][start:end] = -1
            for idx, pid in marks.items():
                if start <= idx < end:
                    out['poles_id'][idx] = pid
        out.flush()
        # Preserve non-vertex elements and PLY metadata as far as plyfile supports.
        elements = [PlyElement.describe(out, 'vertex', comments=ply['vertex'].comments)]
        elements.extend(element for element in ply.elements if element.name != 'vertex')
        result = PlyData(elements, text=False, byte_order='<',
                         comments=ply.comments, obj_info=ply.obj_info)
        pending = temp_dir / 'result.ply'
        result.write(str(pending))
        del out
        os.replace(pending, destination)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    print(f'[INFO] [EXTRACTOR] PLY with poles_id: {destination} '
          f'({len(vertex):,} points, {len(ids)} pole tops)', flush=True)


# =========================================================
# GEOMETRY — BASE PRIMITIVES
# =========================================================
def _pca_axis(points: np.ndarray):
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    if direction[2] < 0:
        direction = -direction
    return direction, centroid


def _perp_distances(points, direction, anchor):
    rel = points - anchor
    t = rel @ direction
    perp = np.linalg.norm(rel - np.outer(t, direction), axis=1)
    return perp, t


def _ransac_vertical_line(
    points: np.ndarray,
    n_iter: int = 400,
    radius: float = 0.20,
    min_verticality: float = 0.80,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n = len(points)
    if n < 2:
        return None

    z_axis = np.array([0.0, 0.0, 1.0])
    best = None

    for _ in range(n_iter):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        d = points[j] - points[i]
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            continue
        d = d / norm
        if abs(d @ z_axis) < min_verticality:
            continue

        anchor = 0.5 * (points[i] + points[j])
        perp, _ = _perp_distances(points, d, anchor)
        count = int(np.sum(perp < radius))

        if best is None or count > best[0]:
            best = (count, d, anchor)

    if best is None:
        return None
    count, direction, anchor = best
    if direction[2] < 0:
        direction = -direction
    return direction, anchor, count


def _refine_axis_parametric(
    points: np.ndarray,
    direction: np.ndarray,
    anchor: np.ndarray,
    max_iter: int = 50,
    k_sigma: float = 2.5,
    tol: float = 1e-10,
):
    prev_dir = direction.copy()
    prev_anchor = anchor.copy()
    mask = np.ones(len(points), dtype=bool)

    for _ in range(max_iter):
        rel = points - anchor
        t = rel @ direction
        perp_vec = rel - np.outer(t, direction)
        perp = np.linalg.norm(perp_vec, axis=1)

        med = float(np.median(perp))
        mad = float(np.median(np.abs(perp - med)))
        sigma = 1.4826 * mad if mad > 1e-9 else float(np.std(perp))
        thresh = med + k_sigma * sigma

        new_mask = perp <= thresh
        if new_mask.sum() < 3:
            break

        direction, anchor = _pca_axis(points[new_mask])

        moved = (
            np.linalg.norm(direction - prev_dir)
            + np.linalg.norm(anchor - prev_anchor)
        )
        mask = new_mask
        if moved < tol:
            break
        prev_dir = direction.copy()
        prev_anchor = anchor.copy()

    return direction, anchor, mask


# =========================================================
# GEOMETRY — ROBUST EXTRACTION HELPERS
# =========================================================
def _project_to_perpendicular(p0, d, ref):
    d = np.asarray(d, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    return p0 + ((ref - p0) @ d) * d


def _axis_from_normals(points, initial_axis, radius_hint=None, max_nn=30):
    n = len(points)
    if n < 20:
        return None

    if radius_hint is None:
        tree = cKDTree(points)
        d, _ = tree.query(points, k=2)
        radius_hint = float(np.median(d[:, 1]) * 5.0)
    radius_hint = max(radius_hint, 1e-3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_hint, max_nn=max_nn
        )
    )
    normals = np.asarray(pcd.normals)
    if len(normals) == 0:
        return None

    if initial_axis is not None:
        ia = np.asarray(initial_axis, dtype=float)
        ia = ia / (np.linalg.norm(ia) + 1e-12)
        cos_ang = np.abs(normals @ ia)
        mask = cos_ang < 0.5
        if mask.sum() < 10:
            mask = np.ones(len(normals), dtype=bool)
        normals = normals[mask]

    Cn = normals.T @ normals / len(normals)
    eigvals, eigvecs = np.linalg.eigh(Cn)
    direction = eigvecs[:, int(np.argmin(eigvals))]
    if direction[2] < 0:
        direction = -direction
    return direction


def _fit_cylinder(points, p0_init, d_init, r_init, robust_scale=0.05):
    if len(points) < 10:
        return p0_init, d_init, r_init

    d_init = np.asarray(d_init, dtype=float)
    d_init = d_init / (np.linalg.norm(d_init) + 1e-12)
    p0_init = _project_to_perpendicular(
        np.asarray(p0_init, dtype=float), d_init, points.mean(axis=0)
    )

    params0 = np.concatenate([p0_init, d_init, [float(r_init)]])

    def residuals(p):
        p0 = p[0:3]
        d = p[3:6]
        r = p[6]
        dn = np.linalg.norm(d)
        if dn < 1e-9:
            return np.full(len(points), 1e3)
        d = d / dn
        rel = points - p0
        t = rel @ d
        perp = np.linalg.norm(rel - np.outer(t, d), axis=1)
        return perp - r

    try:
        res = least_squares(
            residuals, params0,
            method="trf", loss="soft_l1", f_scale=robust_scale,
            xtol=1e-10, ftol=1e-10, max_nfev=200,
        )
    except Exception:
        return p0_init, d_init, r_init

    p0 = res.x[0:3]
    d = res.x[3:6]
    d = d / (np.linalg.norm(d) + 1e-12)
    r = float(res.x[6])
    if d[2] < 0:
        d = -d
    return p0, d, r


def _classify_points(points, p0, d, r_pole, extra=0.05):
    rel = points - p0
    t = rel @ d
    perp = np.linalg.norm(rel - np.outer(t, d), axis=1)
    is_pole = perp < (r_pole + extra)
    return perp, is_pole


def _orthogonalize_horizontal(vertical_dir, h_target):
    d = np.asarray(vertical_dir, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    h = np.asarray(h_target, dtype=float)
    h = h - (h @ d) * d
    nrm = np.linalg.norm(h)
    if nrm > 1e-12:
        h = h / nrm
    return h


def _horizontal_from_outliers_xy(
    points: np.ndarray,
    base: np.ndarray,
    unit_vector: np.ndarray,
    height: float,
    radius_threshold: float = 0.30,
    min_z_frac: float = 0.5,
):
    """Compute the horizontal unit vector from the outlier points (upper half).

    1. Classify each point by its perpendicular distance to the axis
       `base + t * unit_vector`; points with distance > radius_threshold
       are outliers (crossarms, insulators, vegetation, noise).
    2. Keep only outliers whose height above the base is greater than
       `min_z_frac * height` (by default the upper half). The lower half
       is discarded because it usually contains no crossarm/attachment.
    3. Project the kept outliers onto the XY plane.
    4. Fit the dominant 2D direction via PCA (that's the horizontal line).
    5. Apply perpendicularity against `unit_vector` (h · u = 0) to obtain
       the true 3D horizontal unit vector.

    Returns
    -------
    h_unit : (3,) float ndarray
    confidence : float in [0, 1]  (explained variance ratio of the 2D PCA)
    outliers_used : int
    """
    pts = np.asarray(points, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64).reshape(3)
    u = np.asarray(unit_vector, dtype=np.float64).reshape(3)
    u = u / (np.linalg.norm(u) + 1e-12)

    # 1. Classify by perpendicular distance to the axis
    rel = pts - base
    t = rel @ u
    perp_vec = rel - np.outer(t, u)
    perp_dist = np.linalg.norm(perp_vec, axis=1)
    is_outlier = perp_dist > radius_threshold

    # 2. Keep only those in the upper part (z > height/2 above base)
    z_local = pts[:, 2] - base[2]
    z_min = min_z_frac * max(height, 1e-6)
    is_upper = z_local > z_min

    keep_mask = is_outlier & is_upper
    outliers = pts[keep_mask]

    # Fallback if too few outliers: any horizontal perpendicular to u
    if len(outliers) < 5:
        x_ref = np.array([1.0, 0.0, 0.0])
        if abs(x_ref @ u) > 0.9:
            x_ref = np.array([0.0, 1.0, 0.0])
        h = _orthogonalize_horizontal(u, x_ref)
        if h[0] < 0 or (abs(h[0]) < 1e-12 and h[1] < 0):
            h = -h
        return h, 0.0, int(len(outliers))

    # 3. Project to XY
    xy = outliers[:, :2]
    c = xy.mean(axis=0)
    centered = xy - c

    # 4. 2D PCA
    cov = centered.T @ centered / max(len(centered), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_max = int(np.argmax(eigvals))
    d2 = eigvecs[:, idx_max]

    h_xy = np.array([d2[0], d2[1], 0.0])
    nrm = np.linalg.norm(h_xy)
    if nrm < 1e-12:
        x_ref = np.array([1.0, 0.0, 0.0])
        if abs(x_ref @ u) > 0.9:
            x_ref = np.array([0.0, 1.0, 0.0])
        h = _orthogonalize_horizontal(u, x_ref)
        if h[0] < 0 or (abs(h[0]) < 1e-12 and h[1] < 0):
            h = -h
        return h, 0.0, int(len(outliers))
    h_xy /= nrm

    # 5. Perpendicularity: h · u = 0
    h = _orthogonalize_horizontal(u, h_xy)

    # Sign convention: prefer +X, then +Y
    if h[0] < 0 or (abs(h[0]) < 1e-12 and h[1] < 0):
        h = -h

    total_var = float(eigvals.sum())
    conf = float(eigvals[idx_max] / total_var) if total_var > 1e-12 else 0.0

    return h, conf, int(len(outliers))


# =========================================================
# GEOMETRY — FULL EXTRACT
# =========================================================
def extract_geometry(
    points: np.ndarray,
    ransac_iters: int = 400,
    ransac_radius: float = 0.20,
    max_outer_iters: int = 6,
    tol_outer: float = 1e-6,
    classify_extra: float = 0.04,
    horizontal_radius: float = 0.30,
):
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 3:
        raise ValueError("Not enough points to estimate a direction.")

    # 1. RANSAC init
    ransac = _ransac_vertical_line(
        points, n_iter=ransac_iters, radius=ransac_radius
    )
    if ransac is None:
        direction, anchor = _pca_axis(points)
    else:
        direction, anchor, _ = ransac

    # 2. Parametric refinement
    direction, anchor, shaft_mask = _refine_axis_parametric(
        points, direction, anchor
    )
    if shaft_mask.sum() < 3:
        shaft_mask = np.ones(n, dtype=bool)
    direction, anchor = _pca_axis(points[shaft_mask])

    # 3. Rough pole radius
    rel = points - anchor
    t_proj = rel @ direction
    perp = np.linalg.norm(rel - np.outer(t_proj, direction), axis=1)
    low = perp[perp < np.percentile(perp, 30)]
    r_pole = float(np.median(low)) if len(low) else 0.10
    r_pole = max(r_pole, 0.02)

    # 4. Axis from normals covariance
    d_normals = _axis_from_normals(
        points, direction, radius_hint=max(3.0 * r_pole, 0.05)
    )
    if d_normals is not None and (d_normals @ direction) > 0.7:
        direction = d_normals
        anchor = _project_to_perpendicular(
            anchor, direction, points.mean(axis=0)
        )

    # 5. Outer refinement loop (cylinder fitting only)
    p0 = anchor.copy()
    d_axis = direction.copy()
    is_pole = np.ones(n, dtype=bool)

    for _ in range(max_outer_iters):
        _, is_pole = _classify_points(
            points, p0, d_axis, r_pole, extra=classify_extra
        )
        n_pole = int(is_pole.sum())

        if n_pole >= 30:
            pole_pts_fit = points[is_pole]
            p0_new, d_axis_new, r_pole_new = _fit_cylinder(
                pole_pts_fit, p0, d_axis, r_pole
            )
            if 0.02 < r_pole_new < 1.0:
                p0, d_axis, r_pole = p0_new, d_axis_new, r_pole_new

        moved = (
            np.linalg.norm(d_axis - direction)
            + np.linalg.norm(p0 - anchor)
        )
        direction = d_axis.copy()
        anchor = p0.copy()
        if moved < tol_outer:
            break

    # 6. Final geometry: height and base point
    if is_pole.sum() >= 2:
        _, t_pole = _perp_distances(points[is_pole], d_axis, p0)
        t_min, t_max = float(t_pole.min()), float(t_pole.max())
    else:
        _, t_all_ = _perp_distances(points, d_axis, p0)
        t_min, t_max = float(t_all_.min()), float(t_all_.max())
    height = t_max - t_min
    base_point = p0 + d_axis * t_min

    # 7. Horizontal direction from the outliers (upper half only)
    h_final, h_conf, n_outliers = _horizontal_from_outliers_xy(
        points,
        base_point,
        d_axis,
        height,
        radius_threshold=horizontal_radius,
        min_z_frac=0.5,   # z > height/2
    )

    n_att = int(n - is_pole.sum())

    return {
        "center": [float(base_point[0]), float(base_point[1]), 0.0],
        "base_z": float(base_point[2]),
        "height": float(height),
        "direction": d_axis.tolist(),
        "direction_xy": h_final.tolist(),
        "pole_radius": float(r_pole),
        "num_pole_points": int(is_pole.sum()),
        "num_attachment_points": int(n_att),
        "num_outliers_used": int(n_outliers),
        "horizontal_confidence": float(h_conf),
    }


# =========================================================
# JSON HELPERS
# =========================================================
_RAW_START = "__RAW_FLOAT_9F8E7D__"
_RAW_END = "__END_RAW_FLOAT_9F8E7D__"
_PRECISION = 20


class _RawNum:
    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


class _RawEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, _RawNum):
            return _RAW_START + o.text + _RAW_END
        return super().default(o)


def num(value) -> _RawNum:
    return _RawNum(f"{float(value):.{_PRECISION}f}")


def _dump_json(obj, indent: int = 4) -> str:
    s = json.dumps(obj, cls=_RawEncoder, indent=indent)
    pattern = re.compile('"' + _RAW_START + r'(-?\d+\.\d+)' + _RAW_END + '"')
    return pattern.sub(r"\1", s)


# =========================================================
# GPS.txt
# =========================================================
def write_gps_txt(poles: list[dict], path: Path, precision: int = 15) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(poles, start=1):
            f.write(
                f"Pole {i}: "
                f"{p['latitude']:.{precision}f}, "
                f"{p['longitude']:.{precision}f}\n"
            )


# =========================================================
# KML DISCOVERY
# =========================================================
def find_kml_path(others_dir: Path, configured_name: str | None = None) -> Path:
    """Find the project's KML even when its filename changes.

    Prefer an existing configured file; otherwise, use the only KML in
    others_dir. If several exist, refuse to guess which route to use.
    """
    others_dir = Path(others_dir)
    if configured_name:
        configured_path = others_dir / configured_name
        if configured_path.is_file():
            return configured_path.resolve()

    candidates = sorted(
        (p for p in others_dir.iterdir()
         if p.is_file() and p.suffix.lower() == ".kml"),
        key=lambda p: p.name.lower(),
    ) if others_dir.is_dir() else []

    if not candidates:
        raise FileNotFoundError(
            f"No KML file found in {others_dir}. "
            "Place exactly one .kml file in others_dir."
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(
            f"Multiple KML files found in {others_dir}: {names}. "
            "Keep only the relevant .kml file in others_dir."
        )
    return candidates[0].resolve()


# =========================================================
# MAIN
# =========================================================
def _detect_MT_poles_impl(config_path: Path) -> list[dict]:
    print('[INFO] [EXTRACTOR] Reading input point cloud...', flush=True)
    config = load_config(config_path)
    data_dir = Path(config["data_dir"])
    others_dir = Path(config["others_dir"])
    output_dir = Path(config["output_dir"])
    models_dir = others_dir / config["models_dir"]

    ply_path = (data_dir / config["input_ply_raw"]).resolve()
    tum_path = (others_dir / config["trajectory_tum"]).resolve()
    if not ply_path.is_file():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    kml_path = find_kml_path(others_dir, config.get("kml"))
    print(f"[INFO] [EXTRACTOR] Using KML: {kml_path.name}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / config["json_extractor_raw"]
    gps_provisional = others_dir / config["coords_txt"]
    gps_final = data_dir / config["coords_txt"]

    chunk_size = PLY_CHUNK_VERTICES
    max_geometry = MAX_GEOMETRY_POINTS
    target_label = load_pole_label(config_path)
    print(f"[INFO] [EXTRACTOR] Pole label: {target_label}", flush=True)

    work_dir = Path(tempfile.mkdtemp(prefix="_extractor_work_", dir=output_dir))
    try:
        yaml_path = pole_geo.find_georef_yaml(ply_path, extra_dirs=[others_dir])
        lat0, lon0, alt0 = pole_geo.read_enu_origin(yaml_path)

        points = read_labeled_points(
            ply_path, target_label, scratch_dir=work_dir, chunk_size=chunk_size
        )
        cluster_ids, ids = interface.extract_poles(
            points, chunk_size=chunk_size, scratch_dir=work_dir
        )

        counts, centers, z_min, z_max = scalable.cluster_statistics(
            points, cluster_ids, ids, chunk_size=chunk_size
        )
        gps_path, gps_xyz = pole_geo.find_full_gps_trajectory(
            ply_path, others_dir, tum_path
        )
        yaw, fit_dists = pole_geo.estimate_map_to_enu_yaw(centers, gps_xyz[:, :2])
        transformer = pole_geo.build_enu_to_wgs84(lat0, lon0, alt0)
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array(((c, -s), (s, c)))

        poles: list[dict] = []
        for j, cid in enumerate(ids):
            center_xy = centers[j]
            enu_xy = rot @ center_xy
            base_z = float(z_min[j])
            longitude, latitude, _ = transformer.transform(
                float(enu_xy[0]), float(enu_xy[1]), base_z
            )
            poles.append({
                "cluster_id": int(cid),
                "center_xy": center_xy,
                "enu_xy": enu_xy,
                "base_z": base_z,
                "height": float(z_max[j] - z_min[j]),
                "num_points": int(counts[j]),
                "latitude": float(latitude),
                "longitude": float(longitude),
            })
        if not poles:
            raise ValueError("No poles were detected.")

        write_gps_txt(poles, gps_provisional)
        in_route = kml_filter.get_poles_in_route(gps_provisional, kml_path)
        kept = [p for i, p in enumerate(poles, start=1) if in_route.get(i, False)]
        if not kept:
            raise ValueError(
                "No poles remain inside the KML route. Check the route, "
                "GPS origin, ENU rotation, and ROUTE_TOLERANCE."
            )
        tree = cKDTree(gps_xyz[:, :2])
        for p in kept:
            _, idx = tree.query(p["enu_xy"])
            p["traj_idx"] = int(idx)
        kept.sort(key=lambda p: p["traj_idx"])
        write_gps_txt(kept, gps_final)
        del tree, gps_xyz

        stored_ids = sorted(p["cluster_id"] for p in kept)
        counts_by_id = {cid: int(count) for cid, count in zip(ids, counts)}
        stored_counts = np.array([counts_by_id[cid] for cid in stored_ids],
                                 dtype=np.int64)
        stored, offsets = scalable.group_points_to_disk(
            points, cluster_ids, stored_ids, stored_counts, work_dir,
            chunk_size=chunk_size
        )
        locations = {
            cid: (int(offsets[i]), int(offsets[i + 1]))
            for i, cid in enumerate(stored_ids)
        }
        del points, cluster_ids
        gc.collect()

        print('[INFO] [EXTRACTOR] Starting point extraction...', flush=True)
        poles_json: list[dict] = []
        pole_tops: list[tuple[int, np.ndarray]] = []
        for i, p in enumerate(kept, start=1):
            a, b = locations[p["cluster_id"]]
            cloud_points = stored[a:b]
            stride = max(1, int(np.ceil(len(cloud_points) / max_geometry)))
            fit_points = cloud_points[::stride]
            geom = extract_geometry(fit_points)
            # Highest existing point in the detected pole cluster (not a new vertex).
            top_point = np.array(cloud_points[int(np.argmax(cloud_points[:, 2]))], copy=True)
            pole_tops.append((i, top_point))
            # The saved PLY includes ALL assigned points, not only fit_points.
            ply_out = models_dir / f"pole_{i:03d}.ply"
            write_ply_xyz_stream(ply_out, cloud_points, chunk_size=chunk_size)
            center_xy = p["center_xy"]
            p["diameter"] = 2.0 * float(np.median(np.linalg.norm(
                fit_points[:, :2] - center_xy, axis=1
            )))
            p["model_path"] = str(ply_out)

            poles_json.append({
                "Pole_ID": i,
                "Info": {
                    "num_points": int(len(cloud_points)),
                    "Base": [num(geom["center"][0]), num(geom["center"][1]),
                             num(geom["base_z"])],
                    "height": num(geom["height"]),
                    "unit_vector": [
                        num(geom["direction"][0]),
                        num(geom["direction"][1]),
                        num(geom["direction"][2]),
                    ],
                    "unit_vector_horizontal": [
                        num(geom["direction_xy"][0]),
                        num(geom["direction_xy"][1]),
                        num(geom["direction_xy"][2]),
                    ],
                },
            })
            print(
                f"[INFO] [EXTRACTOR] Extracted pole {i}/{len(kept)}: {ply_out.name} "
                f"({len(cloud_points):,} points)",
                flush=True,
            )

        output_ply = data_dir / config.get('input_ply_poles', 'GlobalMap_labeled_1.ply')
        export_pole_top_ply(ply_path, output_ply, pole_tops,
                            target_label, chunk_size=chunk_size)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(_dump_json(poles_json, indent=4))
        del fit_points, cloud_points
        # Backwards compatibility: keep the old p["points"] return field
        # for small runs; for large files avoid retaining huge arrays in RAM.
        return_limit = MAX_RETURN_POINTS
        if sum(p["num_points"] for p in kept) <= return_limit:
            for p in kept:
                a, b = locations[p["cluster_id"]]
                p["points"] = np.array(stored[a:b], copy=True)
        # For large runs, the PLY files are used instead of allocating copies.
        del stored
        gc.collect()
        print(f"[INFO] [EXTRACTOR] Extraction completed: {len(poles_json)} poles.", flush=True)
        return kept
    finally:
        # In Windows all mmap views must be collected before files are deleted.
        gc.collect()
        try:
            shutil.rmtree(work_dir)
        except OSError as exc:
            print(f"[WARNING] [EXTRACTOR] Unable to remove temporary directory {work_dir}: {exc}",
                  file=sys.stderr, flush=True)


def detect_MT_poles(config_path: str | Path = CONFIG_FILE) -> list[dict]:
    """Extract poles with minimal English progress; do not create log files.

    Native crash details, if any, are written to stderr only.
    """
    own_fault_handler = False
    if not faulthandler.is_enabled():
        try:
            faulthandler.enable()
            own_fault_handler = True
        except (OSError, RuntimeError):
            pass
    try:
        return _detect_MT_poles_impl(Path(config_path))
    finally:
        if own_fault_handler:
            faulthandler.disable()


def main():
    try:
        detect_MT_poles()
    except Exception as exc:
        print(f"[ERROR] [EXTRACTOR] Point extraction failed: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        if os.name == "nt" and sys.stdin is not None and sys.stdin.isatty():
            try:
                input("Press Enter to close...")
            except EOFError:
                pass
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
