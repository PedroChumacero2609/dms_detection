"""
ZONES: object-first LiDAR DMS classification for three reconstructed MV wires.

Pipeline:
  1. Assign each eligible point to the lowest-ID intersecting pole section.
  2. Segment physical objects by section and original class.
  3. Test the complete union of the three wire influence volumes.
  4. Resolve one wire and one level per eligible DMS point.
  5. Select critical points from the most critical accepted level using the configured 3D spacing.
  6. Export the complete annotated PLY and a red/green section PNG.

GPU, when available, accelerates conservative candidate filtering only.
All DMS membership and distances are evaluated on the CPU in float64.
An object-level zones_report.json is generated using a compact, section-first schema.
No Excel file is generated.

Run: python -m dms.zones (geometry constants live in this file)
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from ._labels import load_label_maps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree, ConvexHull, QhullError

# ============================================================================
# ALGORITHM SETTINGS — keep numeric tuning here, not in config.json.
# File paths and the GPU toggle are read from config.json.
# ============================================================================

# 1. CONFIGURATION FILE AND OUTPUTS
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
OUTPUT_PLY = None                # Annotated PLY path; None = config.json
OUTPUT_JSON = None               # Master JSON report path; None = config.json
OUTPUT_IMAGE = None              # Connection image path; None = config.json
EXPORT_PNG = True               # Export the connection image
GPU_BATCH_SIZE = 100000        # GPU batch size is a Python algorithm setting

VERBOSE = False                  # Verbose progress logging; warnings remain visible

# 2. PHYSICAL OBJECT SEGMENTATION (meters / points)
OBJECT_EPS = 1.8
EPS_MIN = 0.15
EPS_MAX = 0.90
EPS_FACTOR = 1.70
MIN_OBJECT_POINTS = 100

# 3. DMS CLASSIFICATION AND CRITICAL POINTS
MIN_DMS_POINTS = 80
MIN_LEVEL_POINTS = 10
CP_SEPARATION = 10.0            # Meters; preserves the supplied value

# 4. SEARCH GEOMETRY AND NUMERICAL TOLERANCES
BOX_LENGTH_MARGIN = 2.0
BOX_X_EXTRA_MIN = -0.5
BOX_X_EXTRA_MAX = 0.5
ZERO_TOL = 1e-9
FIGURE_PATH_RES = 48
SECTION_ASSIGN_BATCH_SIZE = 65536
DMS_CLASSIFY_BATCH_SIZE = 8192
OBJECT_MERGE_BATCH_SIZE = 8192

# 5. IMAGE VISUALIZATION
SCALE_X = 8.0                 # Visualization scaling (algorithm setting)
SCALE_Y = 2.0
VIS_TUBE_RADIUS = 3.0
COLOR_RED = [0.95, 0.05, 0.05]
COLOR_GREEN = [0.10, 0.85, 0.10]

# 6. SEMANTIC CLASSES — loaded from labels.json at startup
CLASS_NAMES = {}
DMS_LABELS = {}
CP_LABELS = {}
DMS_CODE = {}
RECON_LABELS = set()

# 7. CLASS-SPECIFIC INFLUENCE VOLUMES (meters)
SNAKE_LEVEL_SETTINGS = {
    "Vegetation": [0.3, 1.0, 2.0, 4.0],
    "LV Wires": [1.0, 1.8],
    "Ground": [5.5, 7.0],
}
FIGURE_LEVEL_SETTINGS = {
    "Ads": [0, 1], "Buildings": [0, 1],
    "LV Poles": [2, 3], "Rubble": [2, 3], "Scaffolding": [2, 3],
    "Ladder": [2, 3], "Bricks": [2, 3], "MV Electrical Works": [2, 3],
}
FIGURE_PARAMS_BY_CLASS = {name: (2.0, 2.5, 4.0, 2.0)
                          for name in FIGURE_LEVEL_SETTINGS}
SNAKE_LEVELS = {}
FIGURE_LEVELS = {}
FIGURE_PARAMS = {}

# 8. DERIVED CONSTANTS
DMS_CLASSES = set()
FIGURE_CLASSES = set()
SNAKE_CLASSES = set()
GLOBAL_MAX_RADIUS = max(
    *(max(levels) for levels in SNAKE_LEVEL_SETTINGS.values()),
    *(max(*params[:3], float(np.hypot(params[1], params[3] + params[2])))
      for params in FIGURE_PARAMS_BY_CLASS.values())
)
BOX_RADIUS = GLOBAL_MAX_RADIUS
_FIGURE_PATHS_CACHE = {}


def configure_labels(cfg):
    """Apply the shared semantic IDs to all lookup tables before processing."""
    maps = cfg["_label_maps"]
    ids = maps["ids"]
    CLASS_NAMES.clear(); CLASS_NAMES.update(maps["original"])
    DMS_LABELS.clear(); DMS_LABELS.update(maps["dms"])
    CP_LABELS.clear(); CP_LABELS.update(maps["cp"])
    DMS_CODE.clear(); DMS_CODE.update(maps["code"])
    RECON_LABELS.clear(); RECON_LABELS.update(maps["recon"])
    SNAKE_LEVELS.clear(); SNAKE_LEVELS.update(
        {ids[name]: levels for name, levels in SNAKE_LEVEL_SETTINGS.items()})
    FIGURE_LEVELS.clear(); FIGURE_LEVELS.update(
        {ids[name]: levels for name, levels in FIGURE_LEVEL_SETTINGS.items()})
    FIGURE_PARAMS.clear(); FIGURE_PARAMS.update(
        {ids[name]: params for name, params in FIGURE_PARAMS_BY_CLASS.items()})
    DMS_CLASSES.clear(); DMS_CLASSES.update(DMS_LABELS)
    FIGURE_CLASSES.clear(); FIGURE_CLASSES.update(FIGURE_LEVELS)
    SNAKE_CLASSES.clear(); SNAKE_CLASSES.update(SNAKE_LEVELS)
    _FIGURE_PATHS_CACHE.clear()


def debug_log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def _class_max_radius(cls):
    if cls in SNAKE_LEVELS:
        return max(SNAKE_LEVELS[cls])
    r1, r2, r3, length = FIGURE_PARAMS[cls]
    return max(r1, r2, r3, float(np.hypot(r2, length + r3)))



def load_config(path):
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    root = config_path.parent

    def resolve(folder_key, filename_key=None):
        folder = Path(cfg[folder_key]).expanduser()
        if not folder.is_absolute():
            folder = root / folder
        return str(folder / cfg[filename_key]) if filename_key else str(folder)

    cfg["_ply_in"] = resolve("data_dir", "input_ply_reconstructed")
    cfg["_wires_json"] = resolve("output_dir", "wires_final")
    cfg["_poles_json"] = resolve("output_dir", "poles_final")
    cfg["_ply_out"] = resolve("collision_dir", "input_ply_new")
    cfg["_image"] = resolve("collision_dir", "collisions")
    cfg["_report_json"] = resolve("collision_dir", "zones_report_json")
    cfg["_label_maps"] = load_label_maps(cfg, config_path)
    cfg["_remove_classes"] = cfg["_label_maps"]["excluded"]
    if type(cfg.get("GPU")) is not bool:
        raise ValueError("config.json must define GPU as a JSON boolean (true or false)")
    return cfg


def load_ply(path):
    ply = PlyData.read(path, mmap="r")
    vertex = ply["vertex"].data
    required = {"x", "y", "z", "class"}
    if not required.issubset(vertex.dtype.names):
        raise ValueError(f"PLY vertex is missing fields: {required-set(vertex.dtype.names)}")
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    classes = np.asarray(vertex["class"], dtype=np.int32).copy()
    print(f"[INFO] [DMS/ZONES] Loaded {len(xyz):,} points from {path}", flush=True)
    return xyz, classes


def save_augmented_ply(input_path, output_path, fields, drop_fields=()):
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("Input and output PLY paths must be different")
    source = PlyData.read(input_path, mmap="r")
    original = source["vertex"].data
    names = [name for name in original.dtype.names if name not in drop_fields]
    new_dtype = []
    for name in names:
        if name not in fields:
            new_dtype.append((name, original.dtype.fields[name][0]))
        else:
            new_dtype.append((name, np.asarray(fields[name]).dtype))
    for name, data in fields.items():
        if name not in names:
            new_dtype.append((name, np.asarray(data).dtype))
    output = np.empty(len(original), dtype=new_dtype)
    for name in names:
        if name not in fields:
            output[name] = original[name]
    for name, data in fields.items():
        if len(data) != len(output):
            raise ValueError(f"Unexpected size in output field {name}")
        output[name] = data
    other_elements = [element for element in source.elements if element.name != "vertex"]
    elements = [PlyElement.describe(output, "vertex")] + other_elements
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    PlyData(elements, text=False, byte_order="<",
            comments=source.comments, obj_info=source.obj_info).write(output_path)
    print(f"[INFO] [DMS/ZONES] Exported complete annotated PLY: {output_path}", flush=True)


def _wire_is_zero(wire):
    eq = wire.get("equation")
    if not isinstance(eq, dict):
        return True
    try:
        coeffs = np.array([[float(eq[axis][k]) for k in ("a", "b", "c")]
                           for axis in ("x", "y", "z")], dtype=np.float64)
    except (KeyError, ValueError, TypeError):
        return True
    return not np.all(np.isfinite(coeffs)) or np.all(np.abs(coeffs) <= ZERO_TOL)


def _section_has_any_wire(section):
    return any(not _wire_is_zero(w) for w in section.get("wires", []))


# ============================================================
# CONFIG
# ============================================================


def norm_pole_id(pid):
    if isinstance(pid, str):
        s = pid.strip()
        try:
            f = float(s)
            return str(int(f)) if f.is_integer() else s
        except ValueError:
            return s
    if isinstance(pid, float):
        return str(int(pid)) if pid.is_integer() else str(pid)
    return str(pid)


def pole_base_id(pid):
    s = str(pid).strip()
    try:
        return int(float(s)) if "." in s else int(s)
    except (ValueError, TypeError):
        return None


def field_of(p, name):
    if name in p:
        return p[name]
    if "Info" in p and name in p["Info"]:
        return p["Info"][name]
    return None


def resolve_pole(poles_json, pole_id):
    target   = norm_pole_id(pole_id)
    base_num = target.split(".")[0]

    matching = []
    for p in poles_json:
        pid = norm_pole_id(field_of(p, "Pole_ID") or "")
        if not pid:
            continue
        if "." in target:
            if pid == target:
                matching.append(p)
        else:
            if pid == target or pid.startswith(base_num + "."):
                matching.append(p)
    if not matching:
        return None

    bases, bases_avg, heights, unit_vectors = [], [], [], []
    for p in matching:
        ptype  = field_of(p, "Type") or "Monoposte"
        b_main = field_of(p, "Base")
        b1, b2 = field_of(p, "Base_1"), field_of(p, "Base_2")

        if ptype == "Biposte" and b1 is not None and b2 is not None:
            b1 = np.array(b1, dtype=float)
            b2 = np.array(b2, dtype=float)
            bases.extend([b1.tolist(), b2.tolist()])
            bases_avg.append((b1 + b2) / 2.0)
        elif b_main is not None:
            b_main = np.array(b_main, dtype=float)
            bases.append(b_main.tolist())
            bases_avg.append(b_main)
        elif b1 is not None:
            b1 = np.array(b1, dtype=float)
            bases.append(b1.tolist())
            bases_avg.append(b1)

        h = field_of(p, "height")
        if h is not None:
            heights.append(float(h))
        uv = field_of(p, "unit_vector")
        if uv is not None:
            unit_vectors.append(np.array(uv, dtype=float))

    if not bases_avg:
        return None

    base_avg   = np.mean(bases_avg, axis=0)
    height_avg = float(np.mean(heights)) if heights else None
    if unit_vectors:
        uv_avg = np.mean(unit_vectors, axis=0)
        n = float(np.linalg.norm(uv_avg))
        uv_avg = uv_avg / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    else:
        uv_avg = np.array([0.0, 0.0, 1.0])

    return {
        "id":       int(base_num),
        "bases":    bases,
        "base_avg": base_avg.tolist(),
        "height":   height_avg,
        "dir_x":    float(uv_avg[0]),
        "dir_y":    float(uv_avg[1]),
        "dir_z":    float(uv_avg[2]),
    }


def choose_base(pole, other_coord):
    bases = pole.get("bases", [])
    if not bases:
        return np.array(pole["base_avg"], dtype=float)
    if len(bases) == 1:
        return np.array(bases[0], dtype=float)
    arrs = [np.array(b, dtype=float) for b in bases]
    d = [float(np.linalg.norm(b - other_coord)) for b in arrs]
    return arrs[int(np.argmax(d))]


def pole_view(pole, base_xyz):
    p = dict(pole)
    p["center_x"] = float(base_xyz[0])
    p["center_y"] = float(base_xyz[1])
    p["base_z"]   = float(base_xyz[2])
    return p


# ============================================================
# SEARCH BOX
# ============================================================
def build_search_box(from_p, to_p, radius):
    u1 = np.array([from_p["dir_x"], from_p["dir_y"], from_p["dir_z"]], dtype=float)
    u2 = np.array([to_p["dir_x"],   to_p["dir_y"],   to_p["dir_z"]],   dtype=float)
    if np.linalg.norm(u1) > 1e-9:
        u1 /= np.linalg.norm(u1)
    if np.linalg.norm(u2) > 1e-9:
        u2 /= np.linalg.norm(u2)

    base1 = np.array([from_p["center_x"], from_p["center_y"], from_p["base_z"]], dtype=float)
    base2 = np.array([to_p["center_x"],   to_p["center_y"],   to_p["base_z"]],   dtype=float)

    length_vec = base2 - base1
    length     = float(np.linalg.norm(length_vec))
    l_dir      = np.array([1.0, 0.0, 0.0]) if length < 1e-6 else length_vec / length
    length     = max(length, 1e-6)

    h_dir = 0.5 * (u1 + u2)
    if np.linalg.norm(h_dir) < 1e-6:
        h_dir = np.array([0.0, 0.0, 1.0])
    h_dir /= np.linalg.norm(h_dir)
    h_dir = h_dir - np.dot(h_dir, l_dir) * l_dir
    if np.linalg.norm(h_dir) < 1e-6:
        h_dir = np.array([0.0, 0.0, 1.0])
        h_dir = h_dir - np.dot(h_dir, l_dir) * l_dir
        if np.linalg.norm(h_dir) < 1e-6:
            h_dir = np.array([0.0, 1.0, 0.0])
    h_dir /= np.linalg.norm(h_dir)

    w_dir = np.cross(h_dir, l_dir)
    if np.linalg.norm(w_dir) < 1e-6:
        w_dir = np.array([1.0, 0.0, 0.0])
    w_dir /= np.linalg.norm(w_dir)

    h_max      = max(float(from_p["height"] or 0.0), float(to_p["height"] or 0.0))
    box_length = length + BOX_LENGTH_MARGIN
    box_width  = radius * 2.0
    box_height = h_max + radius
    origin     = base1
    center     = origin + (length / 2.0) * l_dir + (box_height / 2.0) * h_dir

    return {
        "origin": origin, "center": center,
        "length": box_length, "width": box_width, "height": box_height,
        "l_dir": l_dir, "w_dir": w_dir, "h_dir": h_dir,
        "length_axis": length, "h_max": h_max,
    }


def world_to_local(points, box):
    R = np.column_stack((box["l_dir"], box["w_dir"], box["h_dir"]))
    return (points - box["origin"]) @ R


def inside_box_local(local_pts, box):
    half_w = box["width"] / 2.0
    return (
        (local_pts[:, 0] >= BOX_X_EXTRA_MIN) &
        (local_pts[:, 0] <= box["length_axis"] + BOX_X_EXTRA_MAX) &
        (local_pts[:, 1] >= -half_w) &
        (local_pts[:, 1] <= half_w) &
        (local_pts[:, 2] >= 0.0) &
        (local_pts[:, 2] <= box["height"])
    )


# ============================================================
# WIRE GEOMETRY
# ============================================================


def _as_xyz(v):
    if isinstance(v, dict):
        return np.array([v["x"], v["y"], v["z"]], dtype=np.float64)
    return np.array(v, dtype=np.float64)


def wire_coefficients(eq):
    return (np.array([float(eq[axis][k]) for axis in ("x", "y", "z")], dtype=np.float64)
            for k in ("a", "b", "c"))


def wire_point(t, eq):
    A, B, C = wire_coefficients(eq)
    return A * t * t + B * t + C


def point_wire_distance(point, a, b):
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-20:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def closest_parameter(point, eq):
    """Exact squared-distance minimum on a quadratic curve, 0 <= t <= 1."""
    A, B, C = wire_coefficients(eq)
    D = C - np.asarray(point, dtype=np.float64)
    coefficients = np.array([
        2.0 * np.dot(A, A),
        3.0 * np.dot(A, B),
        np.dot(B, B) + 2.0 * np.dot(A, D),
        np.dot(B, D),
    ], dtype=np.float64)
    scale = np.max(np.abs(coefficients))
    candidates = [0.0, 1.0]
    if scale > 0.0:
        poly = coefficients / scale
        nz = np.flatnonzero(np.abs(poly) > 1e-14)
        if nz.size and nz[0] < len(poly) - 1:
            roots = np.roots(poly[nz[0]:])
            for root in roots:
                if abs(root.imag) < 1e-9 and 0.0 <= root.real <= 1.0:
                    candidates.append(float(root.real))
    best_t = min(candidates, key=lambda t: np.dot(A*t*t+B*t+D, A*t*t+B*t+D))
    residual = A*best_t*best_t + B*best_t + D
    return float(np.linalg.norm(residual)), float(best_t)


def point_wire_distance_curve(point, eq, _unused_min=None, _unused_max=None):
    return closest_parameter(point, eq)


def wire_local_frame(wire):
    """Preserve the original parabola frame; handle straight wires too."""
    if "_frame" in wire:
        return wire["_frame"]
    A, B, _ = wire_coefficients(wire["equation"])
    n = np.cross(A, B)
    if np.linalg.norm(n) >= 1e-12 and np.linalg.norm(A) >= 1e-12:
        n_hat = n / np.linalg.norm(n)
        e2_hat = A / np.linalg.norm(A)
        m_hat = np.cross(n_hat, e2_hat)
        m_hat /= np.linalg.norm(m_hat)
    else:
        t = B + A
        if np.linalg.norm(t) < 1e-12:
            wire["_frame"] = None
            return None
        t = t / np.linalg.norm(t)
        ref = np.array([0., 0., 1.])
        if abs(np.dot(ref, t)) > .98:
            ref = np.array([0., 1., 0.])
        n_hat = np.cross(ref, t)
        n_hat /= np.linalg.norm(n_hat)
        e2_hat = np.cross(t, n_hat)
        e2_hat /= np.linalg.norm(e2_hat)
        m_hat = t
    wire["_frame"] = (n_hat, e2_hat, m_hat)
    return wire["_frame"]

def project_point_to_wire(point, wire, tol_t=1e-6):
    frame = wire_local_frame(wire)
    if frame is None:
        return []
    n_hat, e2_hat, m_hat = frame
    eq = wire["equation"]
    A, B, C = wire_coefficients(eq)
    C2 = float(np.dot(A, m_hat))
    C1 = float(np.dot(B, m_hat))
    C0 = float(np.dot(C - point, m_hat))
    if abs(C2) < 1e-15:
        roots = [-C0 / C1] if abs(C1) >= 1e-15 else []
    else:
        roots = np.roots([C2, C1, C0])
    results = []
    for root in roots:
        if isinstance(root, complex) or np.iscomplexobj(root):
            if abs(root.imag) > 1e-9:
                continue
            t = float(root.real)
        else:
            t = float(root)
        if t < -tol_t or t > 1.0 + tol_t:
            continue
        t = float(np.clip(t, 0.0, 1.0))
        d = np.asarray(point, dtype=np.float64) - wire_point(t, eq)
        results.append((t, float(d @ n_hat), float(d @ e2_hat),
                        float(np.linalg.norm(d))))
    return results


# ============================================================
# SECTION DATA AND X-INDEX
# ============================================================


def build_section_data(wires_data):
    sections = []
    skipped = 0
    for section in wires_data:
        wire_limits = []
        x_min_sec = np.inf
        x_max_sec = -np.inf
        for wire in section.get("wires", []):
            if _wire_is_zero(wire):
                continue
            eq = wire["equation"]
            if wire.get("point_min") is None or wire.get("point_max") is None:
                continue
            A, B, C = wire_coefficients(eq)
            ts = [0.0, 1.0]
            if abs(A[0]) > 1e-14:
                critical_t = -B[0] / (2.0 * A[0])
                if 0.0 < critical_t < 1.0:
                    ts.append(critical_t)
            xs = [float(wire_point(t, eq)[0]) for t in ts]
            x_lo, x_hi = min(xs), max(xs)
            # Calculate exact coordinate-wise bounds of the quadratic curve.
            # Extrema can occur at t=-B/(2*A), not only at its endpoints.
            extrema_lo = np.minimum(C, A + B + C)
            extrema_hi = np.maximum(C, A + B + C)
            for axis in range(3):
                if abs(A[axis]) > 1e-14:
                    t_crit = -B[axis] / (2.0 * A[axis])
                    if 0.0 < t_crit < 1.0:
                        val = (A[axis] * t_crit + B[axis]) * t_crit + C[axis]
                        extrema_lo[axis] = min(extrema_lo[axis], val)
                        extrema_hi[axis] = max(extrema_hi[axis], val)
            wire_limits.append({
                "wire_id": normalize_wire_id(wire["wire_id"]), "equation": eq,
                "a": C.copy(), "b": (A + B + C).copy(),
                "x_min": x_lo, "x_max": x_hi,
                "bbox_lo": extrema_lo, "bbox_hi": extrema_hi,
                # Exact quadratic-to-chord deviation maximum: ||A|| / 4.
                "chord_max_deviation": float(np.linalg.norm(A) * 0.25),
            })
            x_min_sec = min(x_min_sec, x_lo)
            x_max_sec = max(x_max_sec, x_hi)
        if not wire_limits:
            skipped += 1
            continue
        sections.append({"section_id": section["section_id"], "wires": wire_limits,
                         "x_min": float(x_min_sec), "x_max": float(x_max_sec)})
    if skipped:
        print(f"[WARNING] [DMS/ZONES] Skipped {skipped} sections with no valid wires")
    return sections


def _circle_polygon(r, res, theta0=0.0, direction=1.0):
    n = max(res * 4, 24)
    theta = np.linspace(theta0, theta0 + direction * 2.0 * np.pi, n, endpoint=False)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def _poly_signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_orientation(poly, target_sign):
    a = _poly_signed_area(poly)
    if (a > 0 and target_sign < 0) or (a < 0 and target_sign > 0):
        return poly[::-1].copy()
    return poly.copy()


def _full_shape_1(R2, R3, res):
    theta_top = np.linspace(np.pi, 0.0, res)
    arc_top = np.column_stack((R2 * np.cos(theta_top), R2 * np.sin(theta_top)))
    y3 = -np.sqrt(max(R3 * R3 - R2 * R2, 0.0))
    t3 = np.arctan2(y3,  R2)
    t4 = np.arctan2(y3, -R2)
    theta_bot = np.linspace(t3, t4, res)
    arc_bot = np.column_stack((R3 * np.cos(theta_bot), R3 * np.sin(theta_bot)))
    return np.vstack([arc_top, arc_bot])


def _build_region(level, params, res=FIGURE_PATH_RES):
    R1, R2, R3, L = params
    if level == 0:
        return ("solid", _circle_polygon(R1, res))
    if level == 1:
        outer = _full_shape_1(R2, R3, res)
        sgn = np.sign(_poly_signed_area(outer))
        inner = _circle_polygon(R1, res, theta0=np.pi, direction=sgn)
        inner = _ensure_orientation(inner, sgn)
        return ("shell", outer, inner)
    if level == 2:
        y3 = -np.sqrt(max(R3 * R3 - R2 * R2, 0.0))
        t3 = np.arctan2(y3,  R2)
        t4 = np.arctan2(y3, -R2)
        theta = np.linspace(t3, t4, res)
        arc = np.column_stack((R3 * np.cos(theta), R3 * np.sin(theta)))
        poly = np.vstack([[R2, -R3], arc, [-R2, -R3]])
        return ("solid", poly)
    if level == 3:
        return ("solid", np.array([[R2, -R3], [R2, -L-R3],
                                   [-R2, -L-R3], [-R2, -R3]]))
    raise ValueError(f"invalid figure level {level}")


def get_figure_paths(cls):
    if cls in _FIGURE_PATHS_CACHE:
        return _FIGURE_PATHS_CACHE[cls]
    params = FIGURE_PARAMS[cls]
    built = {}
    for level in FIGURE_LEVELS[cls]:
        region = _build_region(level, params)
        if region[0] == "solid":
            built[level] = ("solid", [MplPath(region[1], closed=True)])
        else:
            built[level] = ("shell", [MplPath(region[1], closed=True),
                                       MplPath(region[2], closed=True)])
    _FIGURE_PATHS_CACHE[cls] = built
    return built


def compute_snake_level(dist, radii):
    for i, r in enumerate(radii):
        if dist <= r:
            return i
    return None


def compute_figure_level(u, v, cls):
    """Level membership including outer boundaries (1e-7 m tolerance)."""
    paths = get_figure_paths(cls)
    for level in sorted(paths):
        kind, plist = paths[level]
        if kind == "solid":
            if plist[0].contains_point((u, v), radius=2e-7):
                return level
        else:
            outer, inner = plist
            if (outer.contains_point((u, v), radius=2e-7)
                    and not inner.contains_point((u, v), radius=-2e-7)):
                return level
    return None


# ============================================================
# POINT-VS-SECTION ANALYSIS — union of all wire influence volumes
# ============================================================


class GPUBackend:
    """GPU broad-phase with vectorized, conservative CPU fallback.

    Every accepted candidate still goes through the original exact CPU
    geometry. Both filters return supersets of true collisions. Failures
    in CuPy (including NVRTC headers and GPU memory) switch to CPU and
    DO NOT disable broad-phase filtering.
    """
    def __init__(self, requested=False, batch_size=GPU_BATCH_SIZE):
        self.requested = bool(requested)
        self.enabled = False
        self.batch_size = max(1000, int(batch_size))
        self.cp = None
        self.nvml = None
        self.handle = None
        self.device_name = 'CPU'
        if self.requested:
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError('No CUDA GPU detected')
                cp.cuda.Device(0).use()
                # IMPORTANT: getDeviceCount does not compile a kernel! A real
                # elementwise calculation forces NVRTC to check CUDA headers.
                test = cp.arange(128, dtype=cp.float64)
                result = float(cp.sum(test * test).get())
                if not np.isclose(result, sum(float(i*i) for i in range(128))):
                    raise RuntimeError('CuPy arithmetic self-test returned an unexpected result')
                self.cp = cp
                self.enabled = True
                name = cp.cuda.runtime.getDeviceProperties(0)['name']
                self.device_name = name.decode(errors='replace') if isinstance(name, bytes) else str(name)
                print(f'[INFO] [DMS/ZONES] Compute backend: GPU broad-phase on {self.device_name}', flush=True)
            except Exception as exc:
                print(f'[WARNING] [DMS/ZONES] GPU initialization failed: {exc}', flush=True)
                print('[WARNING] [DMS/ZONES] Using vectorized CPU filtering. '
                      'For missing CUDA headers try: python -m pip install --upgrade "cupy-cuda12x[ctk]"', flush=True)
        else:
            print('[INFO] [DMS/ZONES] Compute backend: CPU (GPU=false)', flush=True)
        if self.enabled:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml = pynvml
            except Exception as exc:
                print(f'[WARNING] [DMS/ZONES] GPU monitor unavailable ({exc}); '
                      'install nvidia-ml-py', flush=True)

    def status(self):
        if not self.enabled:
            return 'Backend: CPU | GPU: N/A'
        if self.nvml is None:
            return 'Backend: GPU broad-phase | GPU utilization: N/A'
        try:
            util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            mem = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            return (f'Backend: GPU broad-phase | GPU: {util}% | '
                    f'VRAM: {mem.used / 2**30:.2f}/{mem.total / 2**30:.2f} GiB')
        except Exception:
            return 'Backend: GPU broad-phase | GPU utilization: N/A'

    @staticmethod
    def _filter_batch(xyz, classes, wires, xp, transfer_to_host):
        """Superset broad-phase: per-class bounds, exact curve bounding box,
        and chord distance + analytic maximum curve-to-chord deviation.
        A numerical margin prevents borderline points being dropped.
        """
        size = len(classes)
        radii = xp.zeros(size, dtype=xp.float64)
        for cls in DMS_CLASSES:
            radii = xp.where(classes == cls, _class_max_radius(cls) + 1e-5, radii)
        possible = radii > 0
        if not bool(transfer_to_host(xp.any(possible))):
            return np.empty(0, dtype=np.int64)

        keep = xp.zeros(size, dtype=xp.bool_)
        for wire in wires:
            lo = xp.asarray(wire['bbox_lo'], dtype=xp.float64)
            hi = xp.asarray(wire['bbox_hi'], dtype=xp.float64)
            in_box = possible & xp.all((xyz >= (lo - radii[:, None])) &
                                            (xyz <= (hi + radii[:, None])), axis=1)
            # Avoid generating many temporaries for points outside the box;
            # still compute batched coordinates to keep a simple GPU path.
            a = xp.asarray(wire['a'], dtype=xp.float64)
            b = xp.asarray(wire['b'], dtype=xp.float64)
            ab = b - a
            denom = float(transfer_to_host(xp.sum(ab * ab)))
            delta = xyz - a
            if denom <= 1e-20:
                d2 = xp.sum(delta * delta, axis=1)
            else:
                t = xp.clip(xp.sum(delta * ab, axis=1) / denom, 0.0, 1.0)
                residual = delta - t[:, None] * ab
                d2 = xp.sum(residual * residual, axis=1)
            bound = radii + wire['chord_max_deviation'] + 1e-5
            keep |= in_box & (d2 <= bound * bound)
        selected = xp.flatnonzero(keep)
        return np.asarray(transfer_to_host(selected), dtype=np.int64)

    def select_indices(self, xyz, labels, candidate_indices, wires, section_id=None):
        """Return GLOBAL point indices after GPU (or CPU fallback) prefilter.
        Never copies the entire section's point coordinates in one go.
        """
        if len(candidate_indices) == 0:
            return np.empty(0, dtype=np.int64)
        selected = []
        total = len(candidate_indices)
        n_batches = (total + self.batch_size - 1) // self.batch_size
        report_step = max(1, n_batches // 10)
        accepted_so_far = 0
        for batch_no, start in enumerate(range(0, total, self.batch_size), 1):
            stop = min(start + self.batch_size, len(candidate_indices))
            idx = candidate_indices[start:stop]
            block_xyz = np.asarray(xyz[idx], dtype=np.float64)
            block_cls = np.asarray(labels[idx], dtype=np.int32)
            local = None
            if self.enabled:
                try:
                    cp = self.cp
                    with cp.cuda.Device(0):
                        gpu_xyz = cp.asarray(block_xyz)
                        gpu_cls = cp.asarray(block_cls)
                        local = self._filter_batch(gpu_xyz, gpu_cls, wires, cp, cp.asnumpy)
                        # Release temporary references early on 4 GB GPUs.
                        del gpu_xyz, gpu_cls
                except Exception as exc:
                    print(f'[WARNING] [DMS/ZONES] GPU batch failed: {exc}', flush=True)
                    print('[WARNING] [DMS/ZONES] Continuing with CPU vectorized filtering '
                          '(not an unfiltered scan)', flush=True)
                    self.enabled = False
                    try:
                        self.cp.get_default_memory_pool().free_all_blocks()
                    except Exception:
                        pass
            if local is None:
                local = self._filter_batch(block_xyz, block_cls, wires, np, np.asarray)
            accepted_so_far += len(local)
            selected.append(idx[local])
        return np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)


# Vectorized figure-level narrow phase. Projection roots are calculated for an
# entire batch; matplotlib performs polygon containment in its compiled C code.
# Only positive matches require the expensive exact quadratic-curve distance.
def _figure_matches_vectorized(points, cls, wire):
    frame = wire_local_frame(wire)
    if frame is None or len(points) == 0:
        return []
    n_hat, e2_hat, m_hat = frame
    A, B, C = wire_coefficients(wire["equation"])
    c2 = float(A @ m_hat)
    c1 = float(B @ m_hat)
    c0 = (C - points) @ m_hat
    roots = []
    if abs(c2) < 1e-15:
        if abs(c1) >= 1e-15:
            roots = [-c0 / c1]
    else:
        disc = c1*c1 - 4*c2*c0
        valid = disc >= 0
        sq = np.sqrt(np.maximum(disc, 0))
        roots = [np.where(valid, (-c1+sq)/(2*c2), np.nan),
                 np.where(valid, (-c1-sq)/(2*c2), np.nan)]
    best = np.full(len(points), 99, dtype=np.int16)
    paths = get_figure_paths(cls)
    for root in roots:
        valid = np.isfinite(root) & (root >= -1e-6) & (root <= 1+1e-6)
        if not valid.any():
            continue
        indices = np.flatnonzero(valid)
        t = np.clip(root[indices], 0, 1)
        residual = points[indices] - (t[:, None]**2*A + t[:, None]*B + C)
        uv = np.column_stack((residual @ n_hat, residual @ e2_hat))
        for level in sorted(paths):
            kind, plist = paths[level]
            if kind == "solid":
                inside = plist[0].contains_points(uv, radius=2e-7)
            else:
                inside = (plist[0].contains_points(uv, radius=2e-7) &
                          ~plist[1].contains_points(uv, radius=-2e-7))
            best[indices[inside]] = np.minimum(best[indices[inside]], level)
    return [(int(i), int(level)) for i, level in enumerate(best) if level != 99]


def _analyze_batch_vectorized(points, classes, section):
    """Batch geometry, preserving original exact distance for each positive match."""
    results = [[] for _ in range(len(points))]
    for cls in np.unique(classes):
        cls = int(cls)
        if cls not in DMS_CLASSES:
            continue
        loc = np.flatnonzero(classes == cls)
        pts = points[loc]
        radius = _class_max_radius(cls) + 1e-6
        for wire in section['wires']:
            lo, hi = wire['bbox_lo'], wire['bbox_hi']
            mask = np.all((pts >= lo-radius) & (pts <= hi+radius), axis=1)
            mask &= (pts[:, 0] >= wire['x_min']-radius) & (pts[:, 0] <= wire['x_max']+radius)
            sub = np.flatnonzero(mask)
            if not len(sub):
                continue
            psub = pts[sub]
            a, b = wire['a'], wire['b']
            ab = b-a
            denom = float(ab @ ab)
            delta = psub-a
            if denom <= 1e-20:
                d2 = np.einsum('ij,ij->i', delta, delta)
            else:
                t = np.clip((delta @ ab)/denom, 0, 1)
                residual = delta-t[:, None]*ab
                d2 = np.einsum('ij,ij->i', residual, residual)
            sub = sub[d2 <= (radius+wire['chord_max_deviation'])**2]
            if not len(sub):
                continue
            if cls in FIGURE_CLASSES:
                candidates = _figure_matches_vectorized(pts[sub], cls, wire)
                for j, level in candidates:
                    idx = int(loc[sub[j]])
                    dist, _ = closest_parameter(points[idx], wire['equation'])
                    results[idx].append((level, int(wire['wire_id']), dist))
            else:
                for j in sub:
                    idx = int(loc[j])
                    dist, _ = closest_parameter(points[idx], wire['equation'])
                    level = compute_snake_level(dist, SNAKE_LEVELS[cls])
                    if level is not None:
                        results[idx].append((level, int(wire['wire_id']), dist))
    return results

# ============================================================
# DETECTION — per-section progress, unique physical points
# ============================================================


def _rss_gib():
    """Current resident memory on Linux, without optional dependencies."""
    try:
        with open('/proc/self/status', encoding='ascii') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / (1024.0 ** 2)
    except OSError:
        pass
    return float('nan')


def run_connected_components(points, eps):
    """Exact radius-connected components using indexed spatial cells.

    All original float64 coordinates are retained; no voxel downsampling
    or all-neighbors materialization is performed. Cross-cell edges are
    checked using actual Euclidean distances. Minimum cluster size is one.
    """
    from itertools import product

    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n == 0:
        return []
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('DBSCAN eps must be positive and finite')

    total_start = time.perf_counter()
    core_indices = np.arange(n, dtype=np.int64)
    # With no minimum object size, every point is a core point.
    # Consequently, dense-radius neighbor counts are unnecessary.

    # A cell has diagonal < eps: every point pair in one cell
    # belongs to the same connected component. The cell index accelerates
    # neighbor searches; every cross-cell test uses original coordinates.
    cell_side = eps / (np.sqrt(3.0) * (1.0 + 1e-10))
    core_xyz = points  # View: no extra full-cloud coordinate copy.
    origin = np.min(core_xyz, axis=0)
    cell_coord = np.floor((core_xyz - origin) / cell_side).astype(np.int64)
    unique_cells, cell_inverse = np.unique(
        cell_coord, axis=0, return_inverse=True)
    n_cells = len(unique_cells)
    core_order = np.argsort(cell_inverse, kind='stable')
    boundaries = np.r_[0, np.flatnonzero(np.diff(cell_inverse[core_order])) + 1,
                       len(core_indices)]
    # Each group contains indices into the original `points` array.
    cell_points = [core_indices[core_order[boundaries[i]:boundaries[i+1]]]
                   for i in range(n_cells)]
    cell_lookup = {tuple(int(v) for v in coords): i
                   for i, coords in enumerate(unique_cells)}
    del core_xyz, cell_coord, core_order

    parent = np.arange(n_cells, dtype=np.int32 if n_cells < 2**31 else np.int64)
    rank = np.zeros(n_cells, dtype=np.uint8)

    def root(i):
        i = int(i)
        while parent[i] != i:
            parent[i] = parent[int(parent[i])]
            i = int(parent[i])
        return i

    def union_cells(i, j):
        a, b = root(i), root(j)
        if a == b:
            return False
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1
        return True

    eps_sq = float(eps * eps)
    # With cell_side ~ eps/sqrt(3), an edge cannot span >2 cells per axis.
    # Also skip offsets whose two cell AABBs are necessarily > eps apart.
    offsets = []
    for offset in product(range(-2, 3), repeat=3):
        if offset <= (0, 0, 0):
            continue
        sep = np.maximum(np.abs(offset) - 1, 0) * cell_side
        if float(sep @ sep) <= eps_sq:
            offsets.append(offset)

    # Built on demand; a point belongs to exactly one spatial cell.
    cell_trees = {}

    def have_exact_core_edge(a, b):
        """True iff any ORIGINAL core-point pair across two cells is <= eps."""
        ids_a, ids_b = cell_points[a], cell_points[b]
        # Fast exact test on one pair, often sufficient in dense clouds.
        delta = points[int(ids_a[0])] - points[int(ids_b[0])]
        if float(delta @ delta) <= eps_sq:
            return True
        # Small pair-product: exact vectorized checks avoid tiny KD-tree builds.
        if len(ids_a) * len(ids_b) <= 1024:
            pa, pb = points[ids_a], points[ids_b]
            diffs = pa[:, None, :] - pb[None, :, :]
            d_sq = np.einsum('ijk,ijk->ij', diffs, diffs)
            return bool(np.any(d_sq <= eps_sq))
        if len(ids_a) > len(ids_b):
            ids_a, ids_b = ids_b, ids_a
            # The target tree is the original a-cell.
            tree_cell = a
        else:
            tree_cell = b
        local_tree = cell_trees.get(tree_cell)
        if local_tree is None:
            local_tree = cKDTree(points[ids_b], leafsize=16)
            cell_trees[tree_cell] = local_tree
        distances, nearest = local_tree.query(
            points[ids_a], k=1,
            distance_upper_bound=np.nextafter(eps, np.inf), workers=1)
        valid = np.flatnonzero(np.isfinite(distances))
        if not len(valid):
            return False
        # Validate returned pairs by squared ORIGINAL-coordinate distances.
        deltas = points[ids_a[valid]] - points[ids_b[nearest[valid]]]
        return bool(np.any(np.einsum('ij,ij->i', deltas, deltas) <= eps_sq))

    link_start = time.perf_counter()
    tested_pairs = 0
    linked_pairs = 0
    # Cells need not form a fully connected graph just because they touch.
    for a, coord in enumerate(unique_cells):
        x, y, z = (int(v) for v in coord)
        for dx, dy, dz in offsets:
            b = cell_lookup.get((x+dx, y+dy, z+dz))
            if b is None or root(a) == root(b):
                continue
            tested_pairs += 1
            if have_exact_core_edge(a, b):
                linked_pairs += int(union_cells(a, b))
        if n_cells >= 2000 and (a+1) % 2000 == 0:
            debug_log(f'[INFO] [CLUSTER] core-cell union: {a+1:,}/{n_cells:,} '
                  f'({time.perf_counter()-link_start:.1f}s)', flush=True)

    # Label all points using their component's root cell.
    root_cells = np.fromiter((root(i) for i in range(n_cells)),
                             dtype=np.int64, count=n_cells)
    db_labels = np.full(n, -1, dtype=np.int64)
    db_labels[core_indices] = root_cells[cell_inverse]
    debug_log(f'[INFO] [CLUSTER] {n_cells:,} spatial cells; '
          f'{tested_pairs:,} exact cell-pair checks; '
          f'core union {time.perf_counter()-link_start:.2f}s', flush=True)

    valid = np.flatnonzero(db_labels >= 0)
    if len(valid) == 0:
        return []
    order = np.argsort(db_labels[valid], kind='stable')
    sorted_indices = valid[order]
    sorted_labels = db_labels[sorted_indices]
    ends = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1, len(valid)]
    clusters = [sorted_indices[ends[i]:ends[i+1]]
                for i in range(len(ends)-1)
                if ends[i+1]-ends[i] >= 1]
    debug_log(f'[INFO] [CLUSTER] completed: {len(clusters):,} objects '
          f'({time.perf_counter()-total_start:.2f}s)', flush=True)
    return clusters


def section_sort_key(sid):
    try:
        return tuple(int(p) for p in str(sid).split("_"))
    except ValueError:
        return (float("inf"),)


def export_connections_png(poles_dict, wires_data, critical_sections, image_path,
                           scale_x=None, scale_y=None):
    """Export a 2D plan of sections: red = confirmed collision, green = clear.

    Only plotting coordinates are scaled; source pole and wire data stay unchanged.
    """
    scale_x = 8.0 if scale_x is None else scale_x
    scale_y = 2.0 if scale_y is None else scale_y
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("scale_x and scale_y must be greater than zero")

    # Include only sections that have at least one valid reconstructed wire.
    segments = []
    used_ids = set()
    for section in wires_data:
        if not _section_has_any_wire(section):
            continue
        sid = str(section.get("section_id", ""))
        parts = sid.split("_")
        if len(parts) < 2:
            continue
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if a not in poles_dict or b not in poles_dict:
            print(f"[WARNING] [DMS/ZONES] Missing pole for section {sid}")
            continue
        segments.append((sid, a, b))
        used_ids.update((a, b))

    fig, ax = plt.subplots(figsize=(16, 20))
    if used_ids:
        ids = sorted(used_ids)
        original = np.array([poles_dict[i]["base_avg"][:2] for i in ids], dtype=float)
        center = original.mean(axis=0)
        scaled = center + (original - center) * np.array([scale_x, scale_y])
        coords = dict(zip(ids, scaled))

        for sid, a, b in segments:
            x1, y1 = coords[a]
            x2, y2 = coords[b]
            ax.plot([x1, x2], [y1, y2],
                    color="red" if sid in critical_sections else "green",
                    linewidth=2.2, solid_capstyle="round", zorder=1)

        ax.scatter(scaled[:, 0], scaled[:, 1], s=180, facecolor="skyblue",
                   edgecolor="black", linewidth=0.8, zorder=3)
        for pid in ids:
            x, y = coords[pid]
            ax.text(x, y, str(pid), fontsize=8, ha="center", va="center",
                    fontweight="bold", zorder=4)
    else:
        ax.text(0.5, 0.5, "No valid connections to display",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(x=0.15, y=0.08)
    ax.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax.set_title("DMS — Connections", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("X coordinate (visualization scale)")
    ax.set_ylabel("Y coordinate (visualization scale)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color="red", lw=2.2, label="Collision"),
                       Line2D([0], [0], color="green", lw=2.2, label="No collision")],
              loc="best")
    image_path = os.fspath(image_path)
    os.makedirs(os.path.dirname(os.path.abspath(image_path)), exist_ok=True)
    fig.savefig(image_path, dpi=350, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[INFO] [DMS/ZONES] Connections PNG exported to: {image_path}")
    print(f"[INFO] [DMS/ZONES] Sections: "
          f"{sum(sid in critical_sections for sid, _, _ in segments)} red, "
          f"{sum(sid not in critical_sections for sid, _, _ in segments)} green")


# ============================================================
# MAIN
# ============================================================



# ============================================================
# SECTION AND OBJECT SEGMENTATION
# ============================================================
def normalize_wire_id(value):
    """Accept numeric, L1/L2/L3, and wire_1/wire_2/wire_3 identifiers."""
    if isinstance(value, (int, np.integer)):
        result = int(value)
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    else:
        raw = str(value).strip().lower()
        match = re.fullmatch(r"(?:wire[_ -]?|l)?([123])", raw)
        if match is None:
            raise ValueError(f"Unsupported wire identifier: {value!r}")
        result = int(match.group(1))
    if result not in (1, 2, 3):
        raise ValueError(f"Wire identifier must be 1, 2, or 3: {value!r}")
    return result


def build_section_boxes(sections, poles_dict, radius):
    """Number valid sections in ascending physical section-ID order."""
    available = []
    for section in sorted(sections, key=lambda item: section_sort_key(item["section_id"])):
        sid = str(section["section_id"])
        parts = sid.split("_")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            print(f"[WARNING] [DMS/ZONES] Invalid pole section identifier {sid!r}; skipped", flush=True)
            continue
        start_id, end_id = map(int, parts)
        if start_id not in poles_dict or end_id not in poles_dict:
            print(f"[WARNING] [DMS/ZONES] Missing pole geometry for {sid}; skipped", flush=True)
            continue
        pole_a, pole_b = poles_dict[start_id], poles_dict[end_id]
        base_a = choose_base(pole_a, np.asarray(pole_b["base_avg"], dtype=np.float64))
        base_b = choose_base(pole_b, base_a)
        base_a = choose_base(pole_a, base_b)
        box = build_search_box(pole_view(pole_a, base_a),
                               pole_view(pole_b, base_b), radius)
        corners = np.array([
            box["origin"] + t*box["l_dir"] + w*box["w_dir"] + h*box["h_dir"]
            for t in (BOX_X_EXTRA_MIN, box["length_axis"] + BOX_X_EXTRA_MAX)
            for w in (-box["width"]/2.0, box["width"]/2.0)
            for h in (0.0, box["height"])
        ], dtype=np.float64)
        available.append({
            "section": section, "box": box,
            "x_min": float(np.min(corners[:, 0])),
            "x_max": float(np.max(corners[:, 0])),
            "section_number": len(available) + 1,
        })
    return available


def assign_sections(xyz, original_classes, boxes, excluded_classes,
                    batch_size=SECTION_ASSIGN_BATCH_SIZE):
    """Claim section membership once, giving the lowest section ID priority."""
    membership = np.zeros(len(xyz), dtype=np.int32)
    eligible = np.isfinite(xyz).all(axis=1)
    eligible &= np.isin(original_classes, list(DMS_CLASSES))
    eligible &= ~np.isin(original_classes, list(excluded_classes))

    eligible_indices = np.flatnonzero(eligible)
    sort_order = np.argsort(xyz[eligible_indices, 0], kind="stable")
    sorted_indices = eligible_indices[sort_order]
    sorted_x = xyz[sorted_indices, 0]
    by_section = {}
    for item in boxes:
        section_number = item["section_number"]
        lo = np.searchsorted(sorted_x, item["x_min"], side="left")
        hi = np.searchsorted(sorted_x, item["x_max"], side="right")
        parts = []
        for start in range(lo, hi, batch_size):
            chunk = sorted_indices[start:min(hi, start + batch_size)]
            chunk = chunk[membership[chunk] == 0]
            if len(chunk) == 0:
                continue
            inside = inside_box_local(world_to_local(xyz[chunk], item["box"]),
                                      item["box"])
            found = chunk[inside]
            if len(found):
                membership[found] = section_number
                parts.append(found)
        selected = (np.concatenate(parts) if parts else
                    np.empty(0, dtype=np.int64))
        by_section[section_number] = selected
        debug_log(f"[INFO] [DMS/ZONES] Section {section_number} "
              f"({item['section']['section_id']}): {len(selected):,} assigned points",
              flush=True)
    print(f"[INFO] [DMS/ZONES] Assigned {np.count_nonzero(membership):,} eligible "
          "points to unique sections", flush=True)
    return membership, by_section


def adaptive_object_eps(points, min_eps, max_eps, factor):
    """Bound the radius using local spacing, avoiding an unbounded global eps."""
    n = len(points)
    if n <= 1:
        return min_eps
    tree = cKDTree(points, leafsize=32)
    sample_count = min(n, 5000)
    sample_ids = np.linspace(0, n - 1, sample_count, dtype=np.int64)
    k = min(n, 5)
    distances, _ = tree.query(points[sample_ids], k=k,
                              workers=min(4, os.cpu_count() or 1))
    kth = distances if k == 1 else distances[:, -1]
    finite = kth[np.isfinite(kth)]
    if not len(finite):
        return min_eps
    target = float(np.percentile(finite, 75)) * factor
    return float(np.clip(target, min_eps, max_eps))


def merge_small_components(points, clusters, min_points):
    """Attach undersized components to the nearest qualifying component.

    Never discard points; only merge within the same section and original class.
    If no qualifying component exists, seed the largest one and absorb the rest.
    """
    if len(clusters) <= 1 or min_points <= 1:
        return clusters
    sizes = np.array([len(c) for c in clusters], dtype=np.int64)
    large = np.flatnonzero(sizes >= min_points)
    small = np.flatnonzero(sizes < min_points)
    if not len(small):
        return clusters
    if not len(large):
        large = np.array([int(np.argmax(sizes))])
        small = small[small != large[0]]
    # Spatial nearest-point assignment, rather than centroid-only assignment.
    # The KD-tree maps each small component to its closest established object.
    base_ids = np.concatenate([clusters[int(i)] for i in large])
    base_owner = np.concatenate([
        np.full(len(clusters[int(i)]), j, dtype=np.int32)
        for j, i in enumerate(large)])
    tree = cKDTree(points[base_ids], leafsize=32)
    additions = [[] for _ in large]
    for i in small:
        component = clusters[int(i)]
        best_dist = np.inf
        best_owner = 0
        for start in range(0, len(component), OBJECT_MERGE_BATCH_SIZE):
            dist, nearest = tree.query(points[component[start:start + OBJECT_MERGE_BATCH_SIZE]], k=1)
            candidate = int(np.argmin(dist))
            if float(dist[candidate]) < best_dist:
                best_dist = float(dist[candidate])
                best_owner = int(base_owner[int(nearest[candidate])])
        additions[best_owner].append(component)
    merged = []
    for j, i in enumerate(large):
        parts = [clusters[int(i)]] + additions[j]
        merged.append(np.concatenate(parts) if len(parts) > 1 else parts[0])
    if sum(map(len, merged)) != len(points):
        raise RuntimeError('Object merging lost or duplicated points')
    return merged


def segment_objects(xyz, original_classes, boxes, points_by_section,
                    object_eps, eps_min, eps_max, eps_factor, min_object_points=100):
    """Create section-local object IDs ordered from the start pole onward."""
    object_ids = np.zeros(len(xyz), dtype=np.int32)
    objects = []
    for entry in boxes:
        section_number = entry["section_number"]
        section_indices = points_by_section[section_number]
        if len(section_indices) == 0:
            continue
        box = entry["box"]
        raw_objects = []
        for cls in sorted(DMS_CLASSES):
            idx = section_indices[original_classes[section_indices] == cls]
            if len(idx) == 0:
                continue
            points = xyz[idx]
            eps = (float(object_eps) if object_eps is not None else
                   adaptive_object_eps(points, eps_min, eps_max, eps_factor))
            debug_log(f"[INFO] [DMS/ZONES] Clustering section={section_number}, "
                  f"class={cls}, points={len(idx):,}, eps={eps:.4f} m, "
                  f"RSS={_rss_gib():.2f} GiB", flush=True)
            clusters = run_connected_components(points, eps=eps)
            before = len(clusters)
            clusters = merge_small_components(points, clusters, min_object_points)
            debug_log(f"[INFO] [DMS/ZONES] Section {section_number}, class {cls}: "
                  f"{before:,} components -> {len(clusters):,} objects "
                  f"(minimum target={min_object_points}; no points discarded)", flush=True)
            for component in clusters:
                full_indices = idx[component]
                centroid = xyz[full_indices].mean(axis=0)
                position = float((centroid - box["origin"]) @ box["l_dir"])
                raw_objects.append((position, int(cls), int(full_indices.min()),
                                    full_indices))
        # Object numbers are unique across all classes within each section.
        raw_objects.sort(key=lambda item: (item[0], item[1], item[2]))
        for object_number, (_, cls, _, indices) in enumerate(raw_objects, 1):
            object_ids[indices] = object_number
            objects.append({"section_id": section_number,
                            "object_id": object_number,
                            "original_class": cls,
                            "indices": indices})
        debug_log(f"[INFO] [DMS/ZONES] Section {section_number}: "
              f"{len(raw_objects):,} physical objects", flush=True)
    return object_ids, objects


# ============================================================
# COMPLETE 3D UNION / UNIQUE WIRE AND LEVEL ASSIGNMENT
# ============================================================
def choose_wire_candidate(matches, tolerance=1e-9):
    """Order by lowest level, shortest perpendicular distance, highest wire ID."""
    lowest = min(item[0] for item in matches)
    same_level = [item for item in matches if item[0] == lowest]
    shortest = min(item[2] for item in same_level)
    tied = [item for item in same_level
            if abs(float(item[2]) - float(shortest)) <= tolerance]
    return max(tied, key=lambda item: item[1])


def classify_union(xyz, original_classes, boxes, points_by_section,
                   gpu_backend, batch_size=DMS_CLASSIFY_BATCH_SIZE):
    """Evaluate swept 3D volumes and assign each point a single wire/level."""
    n = len(xyz)
    wire_id = np.zeros(n, dtype=np.int8)
    level_id = np.full(n, -1, dtype=np.int8)
    distance_m = np.full(n, -1.0, dtype=np.float64)
    for entry in boxes:
        section_number = entry["section_number"]
        section_indices = points_by_section[section_number]
        if not len(section_indices):
            continue
        candidates = gpu_backend.select_indices(
            xyz, original_classes, section_indices,
            entry["section"]["wires"], section_id=section_number)
        accepted = 0
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start:start+batch_size]
            matches_batch = _analyze_batch_vectorized(
                xyz[chunk], original_classes[chunk], entry["section"])
            for index, matches in zip(chunk, matches_batch):
                if not matches:
                    continue
                lev, wid, dist = choose_wire_candidate(matches)
                wire_id[index] = wid
                level_id[index] = lev
                distance_m[index] = dist
                accepted += 1
        print(f"[INFO] [DMS/ZONES] Section {section_number}: "
              f"{accepted:,} DMS union points, "
              f"{gpu_backend.status()}", flush=True)
    return wire_id, level_id, distance_m


def select_spaced_critical_points(xyz, candidates, wire_id, distance_m,
                                  min_separation_m=CP_SEPARATION):
    """Greedy TOP ranking with configurable, exact 3D suppression.

    A candidate's priority is its DMS perpendicular distance to its assigned
    conductor, then higher wire number, then smaller global PLY index.
    TOP 1 is selected, all remaining candidates closer than the configured
    separation are removed, and TOP 2 is chosen from the survivors.

    cKDTree accelerates neighborhood searches; Euclidean distances are checked
    again in float64 to honor the strict cutoff (points exactly at the
    configured separation remain eligible).
    The method never discards/edits the actual DMS points in the PLY.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    if not len(candidates):
        return np.empty(0, dtype=np.int64)
    if not (np.isfinite(min_separation_m) and min_separation_m > 0):
        raise ValueError('min_separation_m must be a positive finite distance')
    pts = np.asarray(xyz[candidates], dtype=np.float64)
    if not np.isfinite(pts).all():
        raise ValueError('CP candidate coordinates must be finite')

    # Ranking uses the original DMS wire/level assignment, unchanged.
    ordering = np.lexsort((candidates, -wire_id[candidates], distance_m[candidates]))
    active = np.ones(len(candidates), dtype=np.bool_)
    tree = cKDTree(pts, leafsize=32)
    # Query slightly outside the radius, then enforce strict < r with float64.
    radius_sq = float(min_separation_m) ** 2
    query_radius = np.nextafter(float(min_separation_m), np.inf)
    accepted = []
    for position in ordering:
        if not active[position]:
            continue
        accepted.append(int(candidates[position]))
        neighbors = np.asarray(tree.query_ball_point(pts[position], r=query_radius,
                                                     workers=1), dtype=np.int64)
        if len(neighbors):
            delta = pts[neighbors] - pts[position]
            d2 = np.einsum('ij,ij->i', delta, delta)
            active[neighbors[d2 < radius_sq]] = False
        # Always remove TOP 1 itself from future rounds.
        active[position] = False
    return np.asarray(accepted, dtype=np.int64)


def finalize_dms_and_cp(xyz, original_classes, objects, wire_id, level_id,
                        distance_m, min_dms_points=1, min_level_points=1,
                        cp_separation_m=CP_SEPARATION):
    """Keep DMS levels intact; select CPs ONLY from each object's most critical level.

    The most critical accepted level is the smallest level_id (0 > 1 > 2 > 3
    in severity). Rank only that level's candidates across all wires, then
    greedily suppress candidates within < cp_separation_m in 3D.
    """
    updated_classes = original_classes.copy()
    is_dms = np.zeros(len(original_classes), dtype=np.uint8)
    is_cp = np.zeros(len(original_classes), dtype=np.uint8)
    cp_rank = np.zeros(len(original_classes), dtype=np.int32)
    cp_count = 0
    for obj in objects:
        original_class = obj['original_class']
        indices = obj['indices']
        mask = (wire_id[indices] > 0) & (level_id[indices] >= 0)
        candidates = indices[mask]
        if not len(candidates):
            continue
        # Exactly as in the baseline: DMS minimum per (object, wire, level).
        pairs = np.column_stack((wire_id[candidates], level_id[candidates]))
        _, inverse, counts = np.unique(
            pairs, axis=0, return_inverse=True, return_counts=True)
        valid = candidates[counts[inverse] >= min_level_points]
        if len(valid) < min_dms_points:
            continue
        is_dms[valid] = 1
        updated_classes[valid] = DMS_LABELS[original_class]

        critical_level = int(np.min(level_id[valid]))
        level_candidates = valid[level_id[valid] == critical_level]
        chosen = select_spaced_critical_points(
            xyz, level_candidates, wire_id, distance_m,
            min_separation_m=cp_separation_m)
        rank = len(chosen)
        if rank:
            is_cp[chosen] = 1
            updated_classes[chosen] = CP_LABELS[original_class]
            cp_rank[chosen] = np.arange(1, rank + 1, dtype=np.int32)
        cp_count += rank
        if rank:
            debug_log(f"[INFO] [DMS/ZONES] Section {obj['section_id']}, "
                  f"object {obj['object_id']}, class {original_class}: "
                  f"{len(valid):,} DMS points, critical level {critical_level} -> {rank:,} CPs "
                  f"(3D cutoff {cp_separation_m:.3f} m)", flush=True)

    rejected = is_dms == 0
    wire_id[rejected] = 0
    level_id[rejected] = -1
    distance_m[rejected] = -1.0
    assert np.all((is_dms == 0) | ((wire_id >= 1) & (wire_id <= 3) & (level_id >= 0)))
    assert np.all((is_dms == 1) | ((wire_id == 0) & (level_id == -1)))
    assert np.all((is_cp == 0) | (is_dms == 1))
    assert cp_count == int(is_cp.sum())
    print(f'[INFO] [DMS/ZONES] Accepted DMS points={int(is_dms.sum()):,}; '
          f'official critical points={cp_count:,}', flush=True)
    return updated_classes, is_dms, is_cp, cp_rank


def renumber_collision_objects(objects, is_dms, point_count):
    """Prioritize accepted collision IDs 1..N independently inside each section.

    All segmented physical objects in collision-bearing sections retain a
    distinct positive object_id. Non-collision objects receive IDs after
    accepted objects, preserving their gray context in export_info figures.
    Sections with no confirmed collision export object_id=0.
    """
    object_ids = np.zeros(point_count, dtype=np.int32)
    by_section = defaultdict(list)
    for obj in objects:
        by_section[int(obj["section_id"])].append(obj)

    for section_objects in by_section.values():
        ordered = sorted(section_objects, key=lambda obj: int(obj["object_id"]))
        accepted = [obj for obj in ordered if np.any(is_dms[obj["indices"]])]
        if not accepted:
            continue
        rejected = [obj for obj in ordered if not np.any(is_dms[obj["indices"]])]
        for local_id, obj in enumerate(accepted + rejected, 1):
            obj["object_id"] = local_id
            object_ids[obj["indices"]] = local_id
    return object_ids


# ============================================================
# COMPACT SECTION / OBJECT / WIRE / LEVEL JSON REPORT
# ============================================================
def dms_volume(points):
    """3D convex-hull volume of the object's accepted DMS points (m^3)."""
    if len(points) < 4:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return 0.0


def export_collision_json(path, boxes, objects, xyz, original_classes,
                          is_dms, is_cp, wire_id, level_id, distance_m, cp_rank):
    """Write the requested section-first schema without duplicate CP metadata.

    Every level contains num_points, min_distance_m and critical_points.
    Critical points contain only XYZ coordinates; levels without CPs use null.
    CP identity and rank are kept in the annotated PLY, not duplicated in JSON.
    """
    objects_by_section = defaultdict(list)
    total_critical_points = 0
    for obj in objects:
        indices = obj['indices']
        valid = indices[is_dms[indices] == 1]
        if not len(valid):
            continue
        cls = int(obj['original_class'])
        points = xyz[valid]
        volume = dms_volume(points)
        wires_affected = {}
        object_critical_points = 0

        for wid in (1, 2, 3):
            wire_indices = valid[wire_id[valid] == wid]
            if not len(wire_indices):
                continue
            levels = {}
            for lev in sorted(int(value) for value in np.unique(level_id[wire_indices])):
                level_indices = wire_indices[level_id[wire_indices] == lev]
                critical_indices = level_indices[is_cp[level_indices] == 1]
                critical_indices = critical_indices[
                    np.argsort(cp_rank[critical_indices], kind='stable')]
                critical_points = ([{'xyz': xyz[int(idx)].tolist()}
                                    for idx in critical_indices]
                                   if len(critical_indices) else None)
                distances = distance_m[level_indices]
                valid_distances = distances[np.isfinite(distances) & (distances >= 0)]
                levels[f'level_{lev}'] = {
                    'num_points': int(len(level_indices)),
                    'min_distance_m': (float(np.min(valid_distances))
                                       if len(valid_distances) else None),
                    'critical_points': critical_points,
                }
                object_critical_points += len(critical_indices)
            wires_affected[f'L{wid}'] = {'levels': levels}

        if object_critical_points != int(is_cp[valid].sum()):
            raise ValueError('PLY and report contain different critical-point counts')
        if object_critical_points == 0:
            raise ValueError('An accepted DMS object has no critical points')
        total_critical_points += object_critical_points
        objects_by_section[int(obj['section_id'])].append({
            'class_name': CLASS_NAMES[cls],
            'object_id': int(obj['object_id']),
            'original_class_id': cls,
            '# DMS': DMS_CODE[cls],
            'num_points': int(len(valid)),
            'volume_m3': volume,
            'density_pts_m3': (float(len(valid) / volume) if volume > 0 else None),
            'centroid': points.mean(axis=0).tolist(),
            'affected_wires': len(wires_affected),
            'wires_affected': wires_affected,
        })

    report = {'sections': []}
    physical_to_report_section = {}

    for entry in boxes:
        original_sid = int(entry['section_number'])
        collisions = sorted(
            objects_by_section[original_sid],
            key=lambda obj: obj['object_id']
        )

        if not collisions:
            continue

        from_pole, to_pole = map(
            int,
            str(entry['section']['section_id']).split('_')
        )

        new_sid = len(report['sections']) + 1
        physical_to_report_section[original_sid] = new_sid

        report['sections'].append({
            'section_id': new_sid,
            'from_pole': from_pole,
            'to_pole': to_pole,
            'collisions': collisions,
        })

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[INFO] [DMS/ZONES] JSON exported: {path} "
          f"({sum(len(s['collisions']) for s in report['sections']):,} objects; "
          f"{total_critical_points:,} critical points)", flush=True)
    return physical_to_report_section


# ============================================================
# EXECUTION
# ============================================================
def main():
    """Run the complete pipeline using top-of-file settings only."""
    if MIN_DMS_POINTS < 1 or MIN_LEVEL_POINTS < 1:
        raise ValueError("Minimum DMS and level point counts must be at least 1")
    if not (0 < EPS_MIN <= EPS_MAX) or EPS_FACTOR <= 0:
        raise ValueError("Require 0 < EPS_MIN <= EPS_MAX and EPS_FACTOR > 0")
    if MIN_OBJECT_POINTS < 1 or OBJECT_EPS <= 0 or BOX_RADIUS <= 0:
        raise ValueError("Object and box parameters must be positive")
    if not np.isfinite(CP_SEPARATION) or CP_SEPARATION <= 0:
        raise ValueError("CP_SEPARATION must be positive and finite")
    if int(GPU_BATCH_SIZE) < 1:
        raise ValueError("GPU_BATCH_SIZE must be positive")

    cfg = load_config(CONFIG_PATH)
    configure_labels(cfg)
    output_ply = OUTPUT_PLY or cfg["_ply_out"]
    output_png = OUTPUT_IMAGE or cfg["_image"]
    output_json = OUTPUT_JSON or cfg["_report_json"]
    if os.path.abspath(output_ply) == os.path.abspath(cfg["_ply_in"]):
        raise ValueError("Input PLY cannot be overwritten")
    print("[INFO] [DMS/ZONES] Starting object-first collision classification", flush=True)
    print(f"[INFO] [DMS/ZONES] Min DMS={MIN_DMS_POINTS}; "
          f"min level={MIN_LEVEL_POINTS}; min object size={MIN_OBJECT_POINTS}; "
          f"object eps={OBJECT_EPS:.2f} m", flush=True)
    xyz, original_classes = load_ply(cfg["_ply_in"])
    with open(cfg["_wires_json"], "r", encoding="utf-8") as handle:
        wires_data = json.load(handle)
    with open(cfg["_poles_json"], "r", encoding="utf-8") as handle:
        poles_data = json.load(handle)
    if isinstance(poles_data, dict):
        poles_data = list(poles_data.values())
    poles_dict = {}
    for pole in poles_data:
        pole_id = pole_base_id(field_of(pole, "Pole_ID") or "")
        if pole_id is not None and pole_id not in poles_dict:
            resolved = resolve_pole(poles_data, str(pole_id))
            if resolved is not None:
                poles_dict[pole_id] = resolved

    sections = build_section_data(wires_data)
    box_radius = max(GLOBAL_MAX_RADIUS, BOX_RADIUS)
    boxes = build_section_boxes(sections, poles_dict, box_radius)
    if not boxes:
        raise ValueError("No valid sections with both poles and wires were found")
    print(f"[INFO] [DMS/ZONES] Valid sections={len(boxes):,}; "
          f"physical grouping radius={box_radius:.2f} m", flush=True)
    section_id, by_section = assign_sections(
        xyz, original_classes, boxes, cfg["_remove_classes"])
    object_id, objects = segment_objects(
        xyz, original_classes, boxes, by_section, OBJECT_EPS,
        EPS_MIN, EPS_MAX, EPS_FACTOR, MIN_OBJECT_POINTS)

    backend = GPUBackend(cfg["GPU"], GPU_BATCH_SIZE)
    wire_id, level_id, distance_m = classify_union(
        xyz, original_classes, boxes, by_section, backend)
    updated_classes, is_dms, is_cp, cp_rank = finalize_dms_and_cp(
        xyz, original_classes, objects, wire_id, level_id, distance_m,
        min_dms_points=MIN_DMS_POINTS,
        min_level_points=MIN_LEVEL_POINTS,
        cp_separation_m=CP_SEPARATION)

    # Renumber only accepted collisions; keep report and PLY identifiers aligned.
    object_id = renumber_collision_objects(objects, is_dms, len(xyz))
    physical_to_report = export_collision_json(
        output_json, boxes, objects, xyz, original_classes,
        is_dms, is_cp, wire_id, level_id, distance_m, cp_rank)
    section_lut = np.zeros(len(boxes) + 1, dtype=np.int32)
    for old_id, new_id in physical_to_report.items():
        section_lut[old_id] = new_id
    report_section_id = section_lut[section_id]

    # Preserve the full source cloud and all original properties except distance_m.
    save_augmented_ply(cfg["_ply_in"], output_ply, {
        "class": updated_classes.astype(np.int32, copy=False),
        "original_class": original_classes.astype(np.int32, copy=False),
        "section_id": report_section_id,
        "object_id": object_id,
        "is_dms": is_dms,
        "wire_id": wire_id,
        "level_id": level_id,
        "is_cp": is_cp,
        "cp_rank": cp_rank,
    }, drop_fields={"distance_m"})

    critical_ids = {entry["section"]["section_id"]
                    for entry in boxes
                    if np.any(is_dms[by_section[entry["section_number"]]])}
    if EXPORT_PNG:
        export_connections_png(
            poles_dict, wires_data, critical_ids, output_png,
            scale_x=SCALE_X, scale_y=SCALE_Y)
    else:
        print("[INFO] [DMS/ZONES] PNG export skipped (EXPORT_PNG=False)", flush=True)
    print(f"[INFO] [DMS/ZONES] Completed: {len(objects):,} physical objects, "
          f"{len(critical_ids):,} affected sections. PLY and JSON generated together.", flush=True)


if __name__ == "__main__":
    main()