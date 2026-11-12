import os
import json
import copy
import numpy as np
from plyfile import PlyData

from ._paths import load_config, configured_path, output_path, label_ids

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MIN_POINTS = 10
SEP_DEFAULT = 1.0


# Box dimensions used by the oriented search box
BOX_WIDTH = 2.0
BOX_WIDTH_MAX = 5.0
BOX_WIDTH_STEP = 1.0

# Number of decimals used for the "radius" value
RADIUS_DECIMALS = 3

# Maximum allowed radius
MAX_RADIUS = 0.08
SYNTHETIC_RADIUS = 0.04  # Generic radius when no valid curve can be fitted

# -------------------------------------------------
# CURVE FIT
# -------------------------------------------------
CURVE_MIN_PTS = 100       # minimum points to accept a quadratic fit
SEGMENT_MAX_DISTANCE = 20.0  # base-to-base 3D distance; strictly less uses straight segments

# -------------------------------------------------
# CONFIDENCE TABLE
# -------------------------------------------------
CONF_TABLE = {
    3: 100.0,
    2: 75.0,
    1: 100.0,
    0: 25.0
}

# -------------------------------------------------
# OUTPUT DECIMALS
# -------------------------------------------------
DECIMALS = 20

def fmt_num(value, decimals):
    return f"{float(value):.{decimals}f}"

# -------------------------------------------------
# LOAD CONFIG
# -------------------------------------------------

# -------------------------------------------------
# LOAD WIRE LABELS
# -------------------------------------------------
def load_wire_labels(cfg):
    return label_ids(cfg, ["MV Wires 1", "MV Wires 2", "MV Wires 3"])

# -------------------------------------------------
# NORMALIZE NEW POLE FORMAT
# -------------------------------------------------
def normalize_pole(p):
    info = p["Info"]
    base = info["Base"]
    uv   = info["unit_vector"]
    return {
        "id":       int(p["Pole_ID"]),
        "center_x": float(base[0]),
        "center_y": float(base[1]),
        "base_z":   float(base[2]),
        "height":   float(info["height"]),
        "dir_x":    float(uv[0]),
        "dir_y":    float(uv[1]),
        "dir_z":    float(uv[2]),
    }

def _coord(p):
    return np.array([p["center_x"], p["center_y"], p["base_z"]], dtype=float)


def resolve_endpoint(id_value, poles_dict, other_coord=None):

    if isinstance(id_value, str) and "_" in id_value:
        a_id_str, b_id_str = id_value.split("_")
        a_id, b_id = int(a_id_str), int(b_id_str)
        pa, pb = poles_dict[a_id], poles_dict[b_id]

        A = _coord(pa)
        B = _coord(pb)
        seg = B - A
        seg_len2 = float(np.dot(seg, seg))

        if other_coord is not None and seg_len2 > 1e-9:
            t = float(np.dot(other_coord - A, seg) / seg_len2)
            t = float(np.clip(t, 0.0, 1.0))
        else:
            t = 0.5

        tie_point = A + t * seg

        tie = {
            "id":        id_value,
            "center_x": float(tie_point[0]),
            "center_y": float(tie_point[1]),
            "base_z":   float(tie_point[2]),
            "height":   (1 - t) * pa.get("height", 0.0) + t * pb.get("height", 0.0),
            "dir_x":    (1 - t) * pa.get("dir_x", 0.0) + t * pb.get("dir_x", 0.0),
            "dir_y":    (1 - t) * pa.get("dir_y", 0.0) + t * pb.get("dir_y", 0.0),
            "dir_z":    (1 - t) * pa.get("dir_z", 1.0) + t * pb.get("dir_z", 1.0),
        }
        return tie, True

    pole = poles_dict[int(id_value)]
    return pole, False

# -------------------------------------------------
# Resolve both ends of a connection simultaneously
# -------------------------------------------------
def resolve_connection_endpoints(conn, poles_dict):
    from_is_seg = isinstance(conn["from_id"], str) and "_" in conn["from_id"]
    to_is_seg   = isinstance(conn["to_id"], str) and "_" in conn["to_id"]

    if from_is_seg and not to_is_seg:
        to_p, _ = resolve_endpoint(conn["to_id"], poles_dict)
        from_p, _ = resolve_endpoint(conn["from_id"], poles_dict,
                                      other_coord=_coord(to_p))
    elif to_is_seg and not from_is_seg:
        from_p, _ = resolve_endpoint(conn["from_id"], poles_dict)
        to_p, _ = resolve_endpoint(conn["to_id"], poles_dict,
                                    other_coord=_coord(from_p))
    else:
        from_p, _ = resolve_endpoint(conn["from_id"], poles_dict)
        to_p, _ = resolve_endpoint(conn["to_id"], poles_dict)

    return from_p, to_p

# -------------------------------------------------
# GEOMETRY — points inside oriented box
# -------------------------------------------------
def points_in_box(points, center, length, width, height,
                  l_dir, w_dir, h_dir):
    R   = np.column_stack((l_dir, w_dir, h_dir))
    rel = points - center
    loc = rel @ R
    mask = (
        (np.abs(loc[:, 0]) <= length / 2) &
        (np.abs(loc[:, 1]) <= width  / 2) &
        (np.abs(loc[:, 2]) <= height / 2)
    )
    return np.where(mask)[0]

# -------------------------------------------------
# PARAMETRIC 3D CURVE: P(t) = A*t**2 + B*t + C, 0 <= t <= 1
# Coordinates are always global XYZ.
def _coeff_matrix(equation):
    return np.array([[float(equation[axis][k]) for k in ("a", "b", "c")]
                     for axis in ("x", "y", "z")], dtype=float)


def curve_point(t, equation):
    coeff = _coeff_matrix(equation)
    return coeff @ np.array([t * t, t, 1.0], dtype=float)




def _nearest_curve_parameters(points, equation, initial_t=None, iterations=12):
    points = np.asarray(points, dtype=float)
    coeff = _coeff_matrix(equation)
    a, b, c = coeff[:, 0], coeff[:, 1], coeff[:, 2]
    if initial_t is None:
        t = np.clip((points-c) @ b / max(np.dot(b, b), 1e-12), 0.0, 1.0)
    else:
        t = np.clip(np.asarray(initial_t, dtype=float).copy(), 0.0, 1.0)
    for _ in range(iterations):
        curve = t[:, None]**2*a + t[:, None]*b + c
        tangent = 2*t[:, None]*a + b
        residual = curve - points
        first = np.einsum('ij,ij->i', residual, tangent)
        second = (np.einsum('ij,ij->i', tangent, tangent)
                  + 2 * residual @ a)
        step = np.divide(first, second, out=np.zeros_like(first),
                         where=np.abs(second)>1e-12)
        t = np.clip(t - np.clip(step, -0.25, 0.25), 0.0, 1.0)
    candidate_t = np.column_stack((t, np.zeros_like(t), np.ones_like(t)))
    candidate = (candidate_t[..., None]**2*a + candidate_t[..., None]*b + c)
    squared = np.sum((candidate-points[:, None, :])**2, axis=2)
    closest = np.argmin(squared, axis=1)
    return candidate_t[np.arange(len(points)), closest], np.sqrt(squared[np.arange(len(points)), closest])


def _fit_robust_quadratic(t, values, iterations=10):
    A = np.column_stack((t*t, t, np.ones_like(t)))
    weights = np.ones(len(t))
    coeff = np.linalg.lstsq(A, values, rcond=None)[0]
    for _ in range(iterations):
        errors = A @ coeff - values
        deviation = np.linalg.norm(errors, axis=1)
        median = np.median(deviation)
        scale = 1.4826*np.median(np.abs(deviation-median))
        delta = max(1.345*scale, 0.015)
        weights = np.minimum(1.0, delta / np.maximum(deviation, 1e-12))
        weighted_A = A * np.sqrt(weights[:, None])
        weighted_values = values * np.sqrt(weights[:, None])
        coeff = np.linalg.lstsq(weighted_A, weighted_values, rcond=None)[0]
    return coeff


def build_curve(points, center, length, l_dir, w_dir, h_dir):
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < CURVE_MIN_PTS or length <= 1e-8:
        return None
    local = (points - center) @ np.column_stack((l_dir, w_dir, h_dir))
    t = local[:, 0] / length + 0.5
    if np.ptp(t) < 1e-5:
        return None
    transverse_coeff = _fit_robust_quadratic(t, local[:, 1:3])
    longitudinal_b = length * l_dir
    longitudinal_c = center - 0.5*length*l_dir
    global_a = w_dir*transverse_coeff[0, 0] + h_dir*transverse_coeff[0, 1]
    global_b = longitudinal_b + w_dir*transverse_coeff[1, 0] + h_dir*transverse_coeff[1, 1]
    global_c = longitudinal_c + w_dir*transverse_coeff[2, 0] + h_dir*transverse_coeff[2, 1]
    equation = {axis: {"a": float(global_a[i]), "b": float(global_b[i]),
                       "c": float(global_c[i])}
                for i, axis in enumerate(("x", "y", "z"))}
    fitted_t, distances = _nearest_curve_parameters(points, equation, t)
    residual_2d = np.column_stack((t*t, t, np.ones(n))) @ transverse_coeff - local[:, 1:3]
    robust_residual = np.linalg.norm(residual_2d, axis=1)
    median = np.median(robust_residual)
    scale = 1.4826 * np.median(np.abs(robust_residual-median))
    threshold = max(3.0*scale, 0.03)
    mask = robust_residual <= threshold
    if np.count_nonzero(mask) >= CURVE_MIN_PTS and np.count_nonzero(mask) < n:
        transverse_coeff = np.linalg.lstsq(
            np.column_stack((t[mask]**2, t[mask], np.ones(np.count_nonzero(mask)))),
            local[mask, 1:3], rcond=None)[0]
        global_a = w_dir*transverse_coeff[0, 0] + h_dir*transverse_coeff[0, 1]
        global_b = longitudinal_b + w_dir*transverse_coeff[1, 0] + h_dir*transverse_coeff[1, 1]
        global_c = longitudinal_c + w_dir*transverse_coeff[2, 0] + h_dir*transverse_coeff[2, 1]
        equation = {axis: {"a": float(global_a[i]), "b": float(global_b[i]),
                           "c": float(global_c[i])}
                    for i, axis in enumerate(("x", "y", "z"))}
        fitted_t, distances = _nearest_curve_parameters(points, equation, t)
    else:
        mask = np.ones(n, dtype=bool)
    return {
        "equation": equation,
        "mask": mask,
        "rmse": float(np.sqrt(np.mean(distances[mask]**2))),
        "inliers": int(mask.sum()),
        "outliers": int((~mask).sum()),
        "inlier_ratio": float(mask.mean()),
        "proj_monotonic": bool(np.all(np.diff(fitted_t[np.argsort(t)]) >= -1e-4)),
    }


def build_straight_segment(points, center, length, l_dir, w_dir, h_dir):
    """Fit a straight 3D segment, without quadratic sag or robust rejection."""
    if len(points) < MIN_POINTS or length <= 1e-8:
        return None
    local = (points - center) @ np.column_stack((l_dir, w_dir, h_dir))
    t = local[:, 0] / length + 0.5
    if np.ptp(t) < 1e-5:
        return None
    design = np.column_stack((t, np.ones_like(t)))
    transverse = np.linalg.lstsq(design, local[:, 1:3], rcond=None)[0]
    b = length * l_dir + transverse[0, 0] * w_dir + transverse[0, 1] * h_dir
    c = center - 0.5 * length * l_dir + transverse[1, 0] * w_dir + transverse[1, 1] * h_dir
    equation = {
        axis: {"a": 0.0, "b": float(b[i]), "c": float(c[i])}
        for i, axis in enumerate(("x", "y", "z"))
    }
    return {"equation": equation, "mask": np.ones(len(points), dtype=bool)}


def get_extreme_points(points):
    if points is None or len(points) == 0:
        return None
    return {
        "point_min": {k: float(v) for k, v in zip(("x", "y", "z"), points[0])},
        "point_max": {k: float(v) for k, v in zip(("x", "y", "z"), points[-1])},
    }






def extend_extreme_points(equation, center, length, width, height,
                          l_dir, w_dir, h_dir, extremes):
    if equation is None:
        return extremes
    p0, p1 = curve_point(0.0, equation), curve_point(1.0, equation)
    return {
        "point_min": dict(zip(("x", "y", "z"), map(float, p0))),
        "point_max": dict(zip(("x", "y", "z"), map(float, p1))),
    }


# OFFSET EQUATION  (vertical shift in z)
# -------------------------------------------------
def offset_equation(eq, delta_z=0.0):
    new_eq = copy.deepcopy(eq)
    new_eq["z"]["c"] += delta_z
    return new_eq

# -------------------------------------------------
# Shift an extreme point (same logic, in z)
# -------------------------------------------------
def shift_point(point, delta_z=0.0):
    if point is None:
        return None
    new_point = dict(point)
    new_point["z"] = new_point["z"] + delta_z
    return new_point

# -------------------------------------------------
# BUILD ORIENTED SEARCH BOX
# -------------------------------------------------
def build_search_box(from_p, to_p):
    u1 = np.array([from_p["dir_x"], from_p["dir_y"], from_p["dir_z"]], dtype=float)
    u2 = np.array([to_p["dir_x"],   to_p["dir_y"],   to_p["dir_z"]],   dtype=float)
    n1 = np.linalg.norm(u1)
    n2 = np.linalg.norm(u2)
    if n1 > 1e-9:
        u1 = u1 / n1
    if n2 > 1e-9:
        u2 = u2 / n2

    base1 = np.array([from_p["center_x"], from_p["center_y"], from_p["base_z"]], dtype=float)
    base2 = np.array([to_p["center_x"],   to_p["center_y"],   to_p["base_z"]],   dtype=float)

    h_dir = 0.5 * (u1 + u2)
    if np.linalg.norm(h_dir) < 1e-6:
        h_dir = np.array([0.0, 0.0, 1.0])
    h_dir = h_dir / np.linalg.norm(h_dir)

    length_vec = base2 - base1
    length = float(np.linalg.norm(length_vec))
    if length < 1e-6:
        l_dir = np.array([1.0, 0.0, 0.0])
        length = 1e-6
    else:
        l_dir = length_vec / length

    w_dir = np.cross(l_dir, h_dir)
    if np.linalg.norm(w_dir) < 1e-6:
        w_dir = np.array([1.0, 0.0, 0.0])
    w_dir = w_dir / np.linalg.norm(w_dir)

    h_dir = np.cross(w_dir, l_dir)
    if np.linalg.norm(h_dir) < 1e-6:
        h_dir = np.array([0.0, 0.0, 1.0])
    h_dir = h_dir / np.linalg.norm(h_dir)

    mid_xy = 0.5 * (base1 + base2)

    if from_p["height"] >= to_p["height"]:
        base_max = base1
        u_max    = u1
        h_max    = float(from_p["height"])
    else:
        base_max = base2
        u_max    = u2
        h_max    = float(to_p["height"])

    J = base_max + (h_max / 2.0) * u_max
    center = np.array([mid_xy[0], mid_xy[1], J[2]], dtype=float)

    box_length = length - 1.0
    box_width  = BOX_WIDTH
    box_height = h_max + 8.0

    return center, box_length, box_width, box_height, l_dir, w_dir, h_dir

# -------------------------------------------------
# GENERATE SYNTHETIC WIRES
# -------------------------------------------------
def generate_synthetic_wires(from_p, to_p, straight=False):
    wires = []
    base1 = _coord(from_p)
    base2 = _coord(to_p)
    dir1 = np.array([from_p["dir_x"], from_p["dir_y"], from_p["dir_z"]])
    dir2 = np.array([to_p["dir_x"], to_p["dir_y"], to_p["dir_z"]])
    dir1 /= max(np.linalg.norm(dir1), 1e-12)
    dir2 /= max(np.linalg.norm(dir2), 1e-12)
    for i in range(3):
        top1 = base1 + dir1 * (from_p["height"] - i * SEP_DEFAULT)
        top2 = base2 + dir2 * (to_p["height"] - i * SEP_DEFAULT)
        sag = 0.0 if straight else np.linalg.norm(top2[:2] - top1[:2])**2 / 3200.0
        a = np.array([0.0, 0.0, 4.0*sag])
        b = top2-top1-a
        c = top1
        equation = {axis: {"a": float(a[j]), "b": float(b[j]), "c": float(c[j])}
                    for j, axis in enumerate(("x", "y", "z"))}
        wires.append({
            "wire_id": i+1, "n_points": 0, "equation": equation,
            "point_min": dict(zip(("x", "y", "z"), map(float, top1))),
            "point_max": dict(zip(("x", "y", "z"), map(float, top2))),
            "source": "synthetic",
        })
    return wires



# FILL MISSING WIRES (case of 2 detected wires)
# -------------------------------------------------
def fill_missing_wires(wires_data):
    detected = {
        w["wire_id"]: w
        for w in wires_data
        if w["source"] == "detected"
    }
    by_id = {w["wire_id"]: w for w in wires_data}

    def _set_estimated(wire_id, source_id, delta):
        src = detected[source_id]
        wire = by_id[wire_id]
        wire["equation"]   = offset_equation(src["equation"], delta)
        wire["point_min"]  = shift_point(src["point_min"], delta)
        wire["point_max"]  = shift_point(src["point_max"], delta)
        wire["source"]     = "estimated"

    if set(detected) == {1, 2}:
        _set_estimated(3, 2, -SEP_DEFAULT)

    elif set(detected) == {1, 3}:
        _set_estimated(2, 1, -SEP_DEFAULT)

    elif set(detected) == {2, 3}:
        _set_estimated(1, 2, +SEP_DEFAULT)

    return wires_data

# -------------------------------------------------
# Special case of a single detected cable ("twisted pairs")
# -------------------------------------------------
def handle_single_wire(wires_data):
    detected_id = next(
        w["wire_id"] for w in wires_data if w["source"] == "detected"
    )
    zero_eq = {
        "x": {"a": 0.0, "b": 0.0, "c": 0.0},
        "z": {"a": 0.0, "b": 0.0, "c": 0.0},
        "y": {"a": 0.0, "b": 0.0, "c": 0.0},
    }
    for w in wires_data:
        if w["wire_id"] != detected_id:
            w["equation"]  = copy.deepcopy(zero_eq)
            w["point_min"] = None
            w["point_max"] = None
            w["source"]    = None

    return wires_data

# -------------------------------------------------
# Final format (fixed decimals) before exporting to JSON
# -------------------------------------------------
def format_equation(eq):
    if eq is None:
        return None
    return {
        axis: {c: fmt_num(v, DECIMALS) for c, v in coeffs.items()}
        for axis, coeffs in eq.items()
    }

def format_point(pt):
    if pt is None:
        return None
    return {k: fmt_num(v, DECIMALS) for k, v in pt.items()}

def format_wire(wire):
    out = {
        "wire_id":   wire["wire_id"],
        "n_points":  wire.get("n_points", 0),
        "equation":  format_equation(wire["equation"]),
        "point_min": format_point(wire.get("point_min")),
        "point_max": format_point(wire.get("point_max")),
        "source":    wire["source"],
    }
    return out

def format_span(span):
    return {
        "section_id": span["section_id"],
        "n_points":   span["n_points"],
        "density":    fmt_num(span["density"], DECIMALS),
        "confidence": span["confidence"],
        "status":     span["status"],
        "case":       span["case"],
        "radius":     fmt_num(span["radius"], RADIUS_DECIMALS),
        "wires":      [format_wire(w) for w in span["wires"]],
    }

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    cfg = load_config()

    ply = PlyData.read(configured_path(cfg, "data_dir", "input_ply"))
    WIRE_LABELS = load_wire_labels(cfg)

    v = ply["vertex"].data
    points = np.vstack([v["x"], v["y"], v["z"]]).T

    label_field = next(
        (f for f in ["class", "classification", "label"] if f in v.dtype.names),
        None,
    )
    labels = np.array(v[label_field])

    with open(configured_path(cfg, "output_dir", "json_connector")) as f:
        connections = json.load(f)

    with open(configured_path(cfg, "output_dir", "json_extractor")) as f:
        poles_json = json.load(f)

    poles_dict = {
        int(p["Pole_ID"]): normalize_pole(p)
        for p in poles_json
    }

    all_spans = []

    for conn in connections:
        section_id = f"{conn['from_id']}_{conn['to_id']}"

        from_p, to_p = resolve_connection_endpoints(
            conn, poles_dict
        )

        # Measure the original 3D distance between pole bases, not box length.
        base_distance = float(np.linalg.norm(_coord(to_p) - _coord(from_p)))
        use_segments = base_distance < SEGMENT_MAX_DISTANCE

        center, length, width, height, l_dir, w_dir, h_dir = build_search_box(
            from_p, to_p
        )

        # Expand the search until all three cables are detected or the maximum width is reached.
        # Each attempt recalculates the points and the curves.
        search_width = width
        while True:
            wires_data   = []
            total_points = 0
            # Perpendicular residuals of ALL points for the "radius"
            all_distances = []

            for wire_id, wire_label in enumerate(WIRE_LABELS):
                wire_points = points[labels == wire_label]

                idx       = points_in_box(wire_points, center, length, search_width, height,
                                          l_dir, w_dir, h_dir)
                extracted = wire_points[idx]
                n_points  = len(extracted)
                total_points += n_points

                if n_points < MIN_POINTS:
                    wires_data.append({
                        "wire_id": wire_id + 1,
                        "n_points": int(n_points),
                        "equation": None,
                        "point_min": None,
                        "point_max": None,
                        "source":   "missing",
                    })
                    continue

                # ---- Curve fit using all points ----
                curve_result = (
                    build_straight_segment(extracted, center, length, l_dir, w_dir, h_dir)
                    if use_segments else
                    build_curve(extracted, center, length, l_dir, w_dir, h_dir)
                )
                if curve_result is None:
                    wires_data.append({
                        "wire_id":  wire_id + 1,
                        "n_points": int(n_points),
                        "equation": None,
                        "point_min": None,
                        "point_max": None,
                        "source":   "missing",
                    })
                    continue

                equation      = curve_result["equation"]
                inlier_mask   = curve_result["mask"]
                extracted_in  = extracted[inlier_mask]    # == extracted

                # Initial extremes from all points
                extremes = get_extreme_points(extracted_in)

                # Extend them along the curve until they hit the walls
                extremes = extend_extreme_points(
                    equation, center, length, search_width, height,
                    l_dir, w_dir, h_dir, extremes
                )
                point_min = extremes["point_min"]
                point_max = extremes["point_max"]

                _, distances = _nearest_curve_parameters(extracted_in, equation)
                all_distances.extend(distances.tolist())

                wires_data.append({
                    "wire_id":  wire_id + 1,
                    "n_points": int(n_points),
                    "equation": equation,
                    "point_min": point_min,
                    "point_max": point_max,
                    "source":   "detected"
                })
            n_detected = sum(w["source"] == "detected" for w in wires_data)
            if n_detected == len(WIRE_LABELS) or search_width >= BOX_WIDTH_MAX:
                break
            search_width = min(search_width + BOX_WIDTH_STEP, BOX_WIDTH_MAX)


        if all_distances:
            radius = float(np.mean(all_distances))

            # Limit the radius to the maximum allowed value
            if radius > MAX_RADIUS:
                radius = MAX_RADIUS
        else:
            radius = 0.0

        n_detected = sum(1 for w in wires_data if w["source"] == "detected")

        if use_segments:
            # All three outputs must be straight segments. For labels without
            # enough points, use a straight geometric estimate, retaining raw counts.
            synthetic = {w["wire_id"]: w for w in generate_synthetic_wires(from_p, to_p, straight=True)}
            for wire in wires_data:
                if wire["source"] != "detected":
                    estimated = synthetic[wire["wire_id"]]
                    wire["equation"] = estimated["equation"]
                    wire["point_min"] = estimated["point_min"]
                    wire["point_max"] = estimated["point_max"]
                    wire["source"] = "synthetic"
            case = "segments"
            status = "OK" if n_detected == 3 else "WARNING"
            confidence = CONF_TABLE[n_detected]
            if n_detected == 0:
                radius = SYNTHETIC_RADIUS

        elif n_detected == 0:
            # No cable passed the point-count and curve-fit requirements.
            raw_counts = {w["wire_id"]: w["n_points"] for w in wires_data}
            wires_data = generate_synthetic_wires(from_p, to_p)
            for wire in wires_data:
                wire["n_points"] = raw_counts[wire["wire_id"]]
            radius = SYNTHETIC_RADIUS
            case   = "no_data"
            status = "REPROCESS"
            confidence = CONF_TABLE[0]

        elif n_detected == 1:
            wires_data = handle_single_wire(wires_data)
            case   = "complete"
            status = "OK"
            confidence = 100.0

        else:
            wires_data = fill_missing_wires(wires_data)
            case   = {3: "complete", 2: "partial"}.get(n_detected, "no_data")
            status = {3: "OK", 2: "WARNING"}.get(n_detected, "REPROCESS")
            confidence = CONF_TABLE[n_detected]

        density = total_points / max(length, 1e-6)

        all_spans.append({
            "section_id": section_id,
            "n_points":   int(sum(w["n_points"] for w in wires_data)),
            "density":    float(density),
            "confidence": float(confidence),
            "status":     status,
            "case":       case,
            "radius":     float(radius),
            "wires":      wires_data,
        })

    out_path = output_path(cfg, "json_wires")
    formatted_spans = [format_span(span) for span in all_spans]
    with open(out_path, "w") as f:
        json.dump(formatted_spans, f, indent=4)

    print('[INFO] [POINTS] Wires JSON exported successfully.')


if __name__ == "__main__":
    main()