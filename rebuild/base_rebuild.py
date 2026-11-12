"""
=========================================================
Base rebuild MT

Reconstruction: CPU
Visualization:  Open3D

The point cloud is reduced ONLY for visualization.
The reconstruction itself remains unchanged.

Press S to save screenshot.
=========================================================
"""

import os
import gc
import json
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
pole_radius = 0.14
crossarm_radius = 0.1

transformer_width = 1.0
transformer_length = 0.8
transformer_height = 0.8

CONNECTING_ELEMENT = 0.2
CONNECTING_CUBE_SIDE = CONNECTING_ELEMENT
CONNECTING_CUBE_POINTS = 500

pole_color = [0.55, 0.27, 0.07]
transformer_color = [0.4, 0.4, 0.4]
wire_color = [0.0, 0.0, 0.0]
cube_color = [0.0, 0.0, 0.0]

TOL = 1e-6

RECON_LABELS = {}  # Populated from labels.json by load_config()

POLE_POINTS = 500
CROSSARM_POINTS = 500
WIRE_POINTS = 500


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
def load_config():
    """Load only paths/filenames from config; class IDs come from labels.json."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    with (root / "config.json").open(encoding="utf-8") as f:
        config = json.load(f)

    def directory(key):
        value = Path(config[key])
        return value if value.is_absolute() else root / value

    data_dir = directory("data_dir")
    output_dir = directory("output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    config["PLY_PATH"] = str(data_dir / config["input_ply"])
    config["POLES_JSON"] = str(output_dir / config["poles_final"])
    config["WIRES_JSON"] = str(output_dir / config["wires_final"])
    config["OUTPUT_IMAGE"] = str(output_dir / config["reconstruction"])
    config["RECON_FULL_PLY_PATH"] = str(data_dir / config["input_ply_reconstructed"])

    labels_path = Path(config["labels_json"])
    if not labels_path.is_absolute():
        labels_path = root / labels_path
    with labels_path.open(encoding="utf-8") as f:
        labels_data = json.load(f)
    labels = {entry["title"]: int(entry["id"]) for entry in labels_data["labels"]}
    global RECON_LABELS
    RECON_LABELS = {name: labels[name] for name in (
        "Reconstruction MV Poles", "Reconstruction MV Crossarms",
        "Reconstruction MV Wires", "Reconstruction MV Connecting")}
    config["MV_WIRE_LABELS"] = [labels[name] for name in (
        "MV Wires", "MV Wires 1", "MV Wires 2", "MV Wires 3")]
    return config


COLOR_MAP = {
    0:  [0.95, 0.95, 0.95],
    1:  [0.50, 0.50, 0.50],
    2:  [0.80, 0.80, 0.80],
    3:  [0.78, 0.78, 0.78],
    4:  [0.75, 0.75, 0.75],
    5:  [0.70, 0.70, 0.70],
    8:  [0.66, 0.66, 0.66],
    9:  [0.64, 0.64, 0.64],
    10: [0.88, 0.88, 0.88],
    11: [0.62, 0.62, 0.62],
    12: [0.72, 0.72, 0.72],
    13: [0.69, 0.69, 0.69],
    14: [0.60, 0.60, 0.60],
    15: [0.58, 0.58, 0.58],
}


# ---------------------------------------------------------
# Visualization point cloud
# ---------------------------------------------------------
def load_and_color_pointcloud(config, ply_path=None, max_points=500_000):
    if ply_path is None:
        ply_path = config["PLY_PATH"]

    print('[INFO] [REBUILD] Loading point cloud for visualization...')
    plydata = PlyData.read(ply_path)
    v = plydata["vertex"].data
    total_points = len(v)
    print(f"[INFO] [REBUILD] Original points: {total_points:,}")

    if max_points is None or total_points <= max_points:
        indices = np.arange(total_points)
    else:
        step = int(np.ceil(total_points / max_points))
        indices = np.arange(0, total_points, step)

    print(f"[INFO] [REBUILD] Visualization sample points: {len(indices):,}")

    points = np.column_stack((v["x"][indices], v["y"][indices], v["z"][indices]))
    labels = np.asarray(v["class"][indices])

    remove_classes = [0, *config["MV_WIRE_LABELS"]]
    mask = ~np.isin(labels, remove_classes)
    points = points[mask]
    labels = labels[mask]
    print(f"[INFO] [REBUILD] Points after class filtering: {len(points):,} points")

    colors = np.zeros((len(labels), 3), dtype=np.float32)
    for cls, rgb in COLOR_MAP.items():
        colors[labels == cls] = rgb

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    del plydata, v
    gc.collect()
    return pcd


# ---------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------
def rotation_from_z_to_vector(v):
    z = np.array([0.0, 0.0, 1.0])
    v = v / np.linalg.norm(v)
    axis = np.cross(z, v)
    angle = np.arccos(np.clip(np.dot(z, v), -1.0, 1.0))
    if np.linalg.norm(axis) < TOL:
        return np.eye(3)
    axis /= np.linalg.norm(axis)
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)


def cylinder_between(p1, p2, radius, color):
    v = p2 - p1
    length = np.linalg.norm(v)
    if length < TOL:
        return None

    direction = v / length
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius, length)
    cyl.compute_vertex_normals()
    cyl.paint_uniform_color(color)

    R = rotation_from_z_to_vector(direction)
    cyl.rotate(R, center=(0.0, 0.0, 0.0))
    cyl.translate((p1 + p2) / 2.0)
    return cyl


def parallelepiped_between(p1, p2, side, color, up_ref=None):
    v = p2 - p1
    length = np.linalg.norm(v)
    if length < TOL:
        return None

    z_axis = v / length

    if up_ref is None:
        up_ref = np.array([0.0, 0.0, 1.0])
    else:
        up_ref = np.asarray(up_ref, dtype=float)
        if np.linalg.norm(up_ref) < TOL:
            up_ref = np.array([0.0, 0.0, 1.0])

    y_axis = up_ref - np.dot(up_ref, z_axis) * z_axis
    yn = np.linalg.norm(y_axis)
    if yn < TOL:
        alt = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(alt, z_axis)) > 1.0 - TOL:
            alt = np.array([1.0, 0.0, 0.0])
        y_axis = alt - np.dot(alt, z_axis) * z_axis
        yn = np.linalg.norm(y_axis)
    y_axis = y_axis / yn

    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    R = np.column_stack((x_axis, y_axis, z_axis))

    box = o3d.geometry.TriangleMesh.create_box(side, side, length)
    box.compute_vertex_normals()
    box.paint_uniform_color(color)
    box.translate(-box.get_center())
    box.rotate(R, center=(0.0, 0.0, 0.0))
    box.translate((p1 + p2) / 2.0)
    return box


def is_same_point(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b)) < TOL


def farthest_index(base, points):
    return int(np.argmax([np.linalg.norm(p - base) for p in points]))


def orthonormal_frame(x_dir, z_ref):
    xn = np.linalg.norm(x_dir)
    x = x_dir / xn if xn > TOL else np.array([1.0, 0.0, 0.0])

    z = z_ref - np.dot(z_ref, x) * x
    zn = np.linalg.norm(z)
    if zn < TOL:
        alt = np.array([0.0, 0.0, 1.0])
        z = alt - np.dot(alt, x) * x
        zn = np.linalg.norm(z)
    z = z / zn

    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    return x, y, z


# ---------------------------------------------------------
# Connecting elements (label 34)
# ---------------------------------------------------------
def build_connecting_cubes(H, C, up_ref):
    up = np.asarray(up_ref, dtype=float)
    if np.linalg.norm(up) < TOL:
        up = np.array([0.0, 0.0, 1.0])

    ref_dir = None
    for Hi, Ci in zip(H, C):
        v = np.asarray(Ci, dtype=float) - np.asarray(Hi, dtype=float)
        if np.linalg.norm(v) > TOL:
            ref_dir = v
            break
    if ref_dir is None:
        ref_dir = up

    cubes = []
    for Hi, Ci in zip(H, C):
        Hi = np.asarray(Hi, dtype=float)
        Ci = np.asarray(Ci, dtype=float)

        v = Ci - Hi
        if np.linalg.norm(v) < TOL:
            v = ref_dir
        vn = np.linalg.norm(v)
        if vn < TOL:
            continue
        z_axis = v / vn

        y_axis = up - np.dot(up, z_axis) * z_axis
        yn = np.linalg.norm(y_axis)
        if yn < TOL:
            alt = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(alt, z_axis)) > 1.0 - TOL:
                alt = np.array([1.0, 0.0, 0.0])
            y_axis = alt - np.dot(alt, z_axis) * z_axis
            yn = np.linalg.norm(y_axis)
        y_axis = y_axis / yn

        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        R = np.column_stack((x_axis, y_axis, z_axis))

        side = CONNECTING_CUBE_SIDE
        cube = o3d.geometry.TriangleMesh.create_box(side, side, side)
        cube.compute_vertex_normals()
        cube.paint_uniform_color(cube_color)
        cube.translate(-cube.get_center())
        cube.rotate(R, center=(0.0, 0.0, 0.0))
        center = np.array([Ci[0], Ci[1], Ci[2] + CONNECTING_ELEMENT])
        cube.translate(center)
        cubes.append(cube)

    return cubes


def build_connecting_elements(H, C_by_H, radius=crossarm_radius):
    elements = []
    for Hi, c_list in zip(H, C_by_H):
        Hi = np.asarray(Hi, dtype=float)
        for Ci in c_list:
            Ci = np.asarray(Ci, dtype=float)
            if np.linalg.norm(Ci - Hi) < TOL:
                continue
            cyl = cylinder_between(Hi, Ci, radius, cube_color)
            if cyl is not None:
                elements.append(cyl)
    return elements


# ---------------------------------------------------------
# Pole builders
# ---------------------------------------------------------
def create_monopole(base, H, C, unit_vector=None,
                    as_connecting=False, C_by_H=None):
    pole_geoms = []
    crossarm_geoms = []
    connecting_geoms = []

    top_idx = farthest_index(base, H)
    H_top = H[top_idx]

    pole_dir = H_top - base
    norm_dir = np.linalg.norm(pole_dir)
    pole_top = H_top + (pole_dir / norm_dir) * 0.1 if norm_dir > TOL else H_top

    pole = cylinder_between(base, pole_top, pole_radius, pole_color)
    if pole is not None:
        pole_geoms.append(pole)

    if unit_vector is None or np.linalg.norm(unit_vector) < TOL:
        up_ref = pole_dir
    else:
        up_ref = np.asarray(unit_vector, dtype=float)

    crossarm_axes = []
    if as_connecting:
        if C_by_H is not None:
            connecting_geoms.extend(build_connecting_elements(H, C_by_H))
        else:
            connecting_geoms.extend(
                build_connecting_elements(H, [[c] for c in C])
            )
    else:
        for Hi, Ci in zip(H, C):
            if is_same_point(Hi, Ci):
                continue
            ca = parallelepiped_between(
                Hi, Ci, crossarm_radius * 2.0, pole_color, up_ref=up_ref
            )
            if ca is not None:
                crossarm_geoms.append(ca)
                crossarm_axes.append((Hi, Ci))

        connecting_geoms.extend(build_connecting_cubes(H, C, up_ref))

    points_info = {
        "Type": "Monoposte",
        "base": base.tolist(),
        "C": [c.tolist() for c in C],
        "PoleAxis": (base, pole_top),
        "Crossarms": crossarm_axes,
    }
    return pole_geoms, crossarm_geoms, connecting_geoms, points_info


def create_bipole(base_A, base_B, H_A, C_A, H_B, C_B, height,
                  unit_vector=None, as_connecting=True,
                  C_A_by_H=None, C_B_by_H=None):
    pole_A, cross_A, conn_A, info_A = create_monopole(
        base_A, H_A, C_A, unit_vector=unit_vector,
        as_connecting=as_connecting, C_by_H=C_A_by_H
    )
    pole_B, cross_B, conn_B, info_B = create_monopole(
        base_B, H_B, C_B, unit_vector=unit_vector,
        as_connecting=as_connecting, C_by_H=C_B_by_H
    )

    pole_geoms = pole_A + pole_B
    crossarm_geoms = cross_A + cross_B
    connecting_cubes = conn_A + conn_B
    geoms = pole_geoms + crossarm_geoms

    H_A_top = H_A[farthest_index(base_A, H_A)]
    pole_dir = (H_A_top - base_A) / np.linalg.norm(H_A_top - base_A)

    O1 = base_A + pole_dir * (height / 2.0)
    O2 = base_B + pole_dir * (height / 2.0)
    Op1 = base_A + pole_dir * (0.75 * height)
    Op2 = base_B + pole_dir * (0.75 * height)

    M1 = (O1 + O2) / 2.0
    Mp1 = (Op1 + Op2) / 2.0

    x_axis, y_axis, z_axis = orthonormal_frame(O2 - O1, pole_dir)
    R = np.column_stack((x_axis, y_axis, z_axis))

    box_length = np.linalg.norm(O1 - O2)
    box_side = 2.0 * pole_radius

    for center in (M1, Mp1):
        box = o3d.geometry.TriangleMesh.create_box(box_length, box_side, box_side)
        box.compute_vertex_normals()
        box.paint_uniform_color(pole_color)
        box.translate(-box.get_center())
        box.rotate(R, center=(0.0, 0.0, 0.0))
        box.translate(center)
        geoms.append(box)
        pole_geoms.append(box)

    d_m = Mp1 - M1
    up_dir = d_m / np.linalg.norm(d_m) if np.linalg.norm(d_m) > TOL else pole_dir
    T = M1 + up_dir * (pole_radius + transformer_height)

    trans = o3d.geometry.TriangleMesh.create_box(
        transformer_length, transformer_width, transformer_height
    )
    trans.compute_vertex_normals()
    trans.paint_uniform_color(transformer_color)
    trans.translate(-trans.get_center())
    trans.rotate(R, center=(0.0, 0.0, 0.0))
    trans.translate(T)
    geoms.append(trans)

    points_info = {
        "Type": "Biposte",
        "base_A": base_A.tolist(),
        "base_B": base_B.tolist(),
        "C_A": [c.tolist() for c in C_A],
        "C_B": [c.tolist() for c in C_B],
        "PoleAxis_A": info_A["PoleAxis"],
        "PoleAxis_B": info_B["PoleAxis"],
        "Crossarms_A": info_A["Crossarms"],
        "Crossarms_B": info_B["Crossarms"],
    }
    return geoms, pole_geoms, crossarm_geoms, [trans], connecting_cubes, points_info


# ---------------------------------------------------------
# JSON loaders
# ---------------------------------------------------------
def load_poles_json(json_path):
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r") as f:
        return json.load(f)


def load_wires_json(json_path):
    if not json_path or not os.path.exists(json_path):
        return []
    with open(json_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------
# Pole reconstruction
# ---------------------------------------------------------
def _read_C_points(pts, case, z_shift):
    C = []
    for i in (1, 2, 3):
        key_suffixed = f"Point_C{i}1"
        key_plain = f"Point_C{i}"
        key = key_suffixed if key_suffixed in pts else key_plain
        C.append(np.asarray(pts[key], dtype=float) - z_shift)
    return C


def _read_C_points_case_I(pts, z_shift, pole_suffix=""):
    C_by_H = []
    for i in (1, 2, 3):
        base_key = f"Point_C{i}{pole_suffix}"
        keys = sorted(
            k for k in pts
            if k == base_key or (k.startswith(base_key) and k[len(base_key):].isdigit())
        )
        C_by_H.append([np.asarray(pts[k], dtype=float) - z_shift for k in keys])
    return C_by_H


def reconstruct_poles(config):
    data = load_poles_json(config["POLES_JSON"])

    geoms = []
    poles = {}
    crossarms = {}
    connecting_geoms = []
    poles_points = {}

    for item in data:
        pole_id = item["Pole_ID"]
        pole_type = item["Type"]
        info = item["Info"]
        pts = item["Points"]
        
        if not pts or float(info.get("height", 0.0)) <= TOL:
            continue

        height = float(info["height"])
        unit_v = np.asarray(
            info.get("unit_vector", [0.0, 0.0, 1.0]), dtype=float
        )

        case = item.get("Case", "")
        dz = 0.0 if case == "I" else CONNECTING_ELEMENT + crossarm_radius
        z_shift = np.array([0.0, 0.0, dz])

        if pole_type == "Monoposte":
            base = np.asarray(info["Base"], dtype=float)
            H = [np.asarray(pts[f"Point_H{i}"], dtype=float) - z_shift for i in (1, 2, 3)]

            if case == "I":
                C_by_H = _read_C_points_case_I(pts, z_shift, pole_suffix="")
                C = [lst[0] if lst else H[i] for i, lst in enumerate(C_by_H)]
                pole_parts, crossarm_parts, conn_parts, points_info = create_monopole(
                    base, H, C, unit_vector=unit_v,
                    as_connecting=True, C_by_H=C_by_H
                )
            else:
                C = _read_C_points(pts, case, z_shift)
                pole_parts, crossarm_parts, conn_parts, points_info = create_monopole(
                    base, H, C, unit_vector=unit_v, as_connecting=False
                )
            geoms.extend(pole_parts)
            geoms.extend(crossarm_parts)

        elif pole_type == "Biposte":
            base_A = np.asarray(info["Base_1"], dtype=float)
            base_B = np.asarray(info["Base_2"], dtype=float)

            H_A = [np.asarray(pts[f"Point_H{i}1"], dtype=float) - z_shift for i in (1, 2, 3)]
            C_A = [np.asarray(pts[f"Point_C{i}1"], dtype=float) - z_shift for i in (1, 2, 3)]
            H_B = [np.asarray(pts[f"Point_H{i}2"], dtype=float) - z_shift for i in (1, 2, 3)]
            C_B = [np.asarray(pts[f"Point_C{i}2"], dtype=float) - z_shift for i in (1, 2, 3)]

            as_connecting = (case == "I")

            if case == "I":
                C_A_by_H = _read_C_points_case_I(pts, z_shift, pole_suffix="1")
                C_B_by_H = _read_C_points_case_I(pts, z_shift, pole_suffix="2")
            else:
                C_A_by_H = None
                C_B_by_H = None

            geom_parts, pole_parts, crossarm_parts, trans_parts, conn_parts, points_info = (
                create_bipole(base_A, base_B, H_A, C_A, H_B, C_B, height,
                              unit_vector=unit_v, as_connecting=as_connecting,
                              C_A_by_H=C_A_by_H, C_B_by_H=C_B_by_H)
            )
            geoms.extend(geom_parts)
            pole_parts = pole_parts + trans_parts

        else:
            continue

        pole_mesh = o3d.geometry.TriangleMesh()
        for g in pole_parts:
            pole_mesh += g
        pole_mesh.compute_vertex_normals()
        poles[pole_id] = pole_mesh

        crossarm_mesh = o3d.geometry.TriangleMesh()
        for g in crossarm_parts:
            crossarm_mesh += g
        if crossarm_parts:
            crossarm_mesh.compute_vertex_normals()
        crossarms[pole_id] = crossarm_mesh

        connecting_geoms.extend(conn_parts)
        poles_points[pole_id] = points_info

    print(f"[INFO] [REBUILD] Connecting geometries generated: {len(connecting_geoms)}")
    return geoms, poles, crossarms, connecting_geoms, poles_points


# ---------------------------------------------------------
# Wire reconstruction
# ---------------------------------------------------------
def generate_wire_mesh(p1, p2, eq_x, eq_y, eq_z, wire_radius=0.02,
                       color=(0.0, 0.0, 0.0), samples=80):
    coefficients = np.asarray([eq_x, eq_y, eq_z], dtype=float)
    ts = np.linspace(0.0, 1.0, max(2, int(samples)))
    curve = (coefficients[:, 0][None, :] * ts[:, None] ** 2
             + coefficients[:, 1][None, :] * ts[:, None]
             + coefficients[:, 2][None, :])

    curve[0] = np.asarray(p1, dtype=float)
    curve[-1] = np.asarray(p2, dtype=float)

    meshes = []
    for i in range(len(curve) - 1):
        cyl = cylinder_between(curve[i], curve[i + 1], wire_radius, color)
        if cyl is not None:
            meshes.append(cyl)

    return meshes, curve


def _as_xyz(v):
    if isinstance(v, dict):
        keys = {k.lower(): k for k in v.keys()}
        return np.array(
            [float(v[keys["x"]]), float(v[keys["y"]]), float(v[keys["z"]])],
            dtype=float,
        )
    return np.asarray(v, dtype=float)


def _as_eq(eq):
    if isinstance(eq, dict):
        return (float(eq["a"]), float(eq["b"]), float(eq["c"]))
    return (float(eq[0]), float(eq[1]), float(eq[2]))


def _eq_is_zero(eq_x, eq_y, eq_z):
    return all(abs(v) < TOL for axis in (eq_x, eq_y, eq_z) for v in axis)


def _wire_is_zero(wire):
    eq = wire.get("equation")
    if not isinstance(eq, dict):
        return True
    try:
        eq_x = _as_eq(eq["x"])
        eq_y = _as_eq(eq["y"])
        eq_z = _as_eq(eq["z"])
    except (KeyError, TypeError, ValueError):
        return True
    return _eq_is_zero(eq_x, eq_y, eq_z)


def reconstruct_wires(config, poles_points, color=(0.0, 0.0, 0.0), samples=80):
    wires_data = load_wires_json(config["WIRES_JSON"])

    geoms = []
    curves_data = []
    processed_bipoles = set()

    for section in wires_data:
        sec_id = section["section_id"]
        parts = sec_id.split("_")

        if len(parts) != 2:
            continue
        p1_id, p2_id = int(parts[0]), int(parts[1])
        section_radius = float(section.get("radius", 0.02))

        for w in section["wires"]:
            if _wire_is_zero(w):
                continue

            if w.get("point_min") is None or w.get("point_max") is None:
                continue

            wid = w["wire_id"]
            eq_x = _as_eq(w["equation"]["x"])
            eq_y = _as_eq(w["equation"]["y"])
            eq_z = _as_eq(w["equation"]["z"])

            C1 = _as_xyz(w["point_min"])
            C2 = _as_xyz(w["point_max"])

            meshes, curve = generate_wire_mesh(
                C1, C2, eq_x, eq_y, eq_z,
                wire_radius=section_radius,
                color=color,
                samples=samples,
            )

            geoms.extend(meshes)
            curves_data.append({
                "section_id": sec_id,
                "wire_id": wid,
                "curve": curve,
                "meshes": meshes,
            })

        for pole_id in (p1_id, p2_id):
            if pole_id in processed_bipoles:
                continue
            pole = poles_points.get(pole_id)
            if not pole or pole.get("Type") != "Biposte":
                continue
            processed_bipoles.add(pole_id)

            base_A = np.asarray(pole["base_A"], dtype=float)
            base_B = np.asarray(pole["base_B"], dtype=float)
            offset = base_B - base_A

            for w in section["wires"]:
                if _wire_is_zero(w):
                    continue

                if w.get("point_min") is None or w.get("point_max") is None:
                    continue

                wid = w["wire_id"]

                if pole_id == p1_id:
                    attach = _as_xyz(w["point_min"])
                else:
                    attach = _as_xyz(w["point_max"])

                if np.linalg.norm(attach - base_A) <= np.linalg.norm(attach - base_B):
                    other = attach + offset
                else:
                    other = attach - offset

                bar = cylinder_between(attach, other, section_radius, color)
                if bar is None:
                    continue

                geoms.append(bar)
                curves_data.append({
                    "section_id": f"{pole_id}_bipole",
                    "wire_id": wid,
                    "curve": np.array([attach, other]),
                    "meshes": [bar],
                })

    return geoms, curves_data


# ---------------------------------------------------------
# Point sampling / export
# ---------------------------------------------------------
def iter_geometries(geometries):
    if geometries is None:
        return []
    if isinstance(geometries, dict):
        return geometries.values()
    return geometries


def add_geometry(geom, label, n_points, all_points, all_labels):
    if isinstance(geom, o3d.geometry.TriangleMesh):
        if len(geom.triangles) == 0:
            return
        sampled = geom.sample_points_uniformly(number_of_points=n_points)
        pts = np.asarray(sampled.points)
    elif isinstance(geom, o3d.geometry.PointCloud):
        pts = np.asarray(geom.points)
    else:
        return

    if len(pts) == 0:
        return

    all_points.append(pts)
    all_labels.append(np.full(len(pts), label, dtype=np.uint16))


def build_reconstruction_points(poles=None, crossarms=None, wires=None,
                                connecting=None,
                                pole_points=POLE_POINTS,
                                crossarm_points=CROSSARM_POINTS,
                                wire_points=WIRE_POINTS,
                                connecting_points=CONNECTING_CUBE_POINTS):
    poles = poles or []
    crossarms = crossarms or []
    wires = wires or []
    connecting = connecting or []

    all_points = []
    all_labels = []

    pole_label = RECON_LABELS["Reconstruction MV Poles"]
    crossarm_label = RECON_LABELS["Reconstruction MV Crossarms"]
    wire_label = RECON_LABELS["Reconstruction MV Wires"]
    connecting_label = RECON_LABELS["Reconstruction MV Connecting"]

    for geom in iter_geometries(poles):
        add_geometry(geom, pole_label, pole_points, all_points, all_labels)

    for geom in iter_geometries(crossarms):
        add_geometry(geom, crossarm_label, crossarm_points, all_points, all_labels)

    for geom in iter_geometries(wires):
        add_geometry(geom, wire_label, wire_points, all_points, all_labels)

    for geom in iter_geometries(connecting):
        add_geometry(geom, connecting_label, connecting_points, all_points, all_labels)

    if not all_points:
        return {
            "points": np.empty((0, 3)),
            "labels": np.empty((0,), dtype=np.uint16),
        }

    return {
        "points": np.vstack(all_points),
        "labels": np.hstack(all_labels),
    }


def write_ply(path, points, labels):
    n = len(points)
    vertex = np.empty(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("class", "i4")])
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["class"] = labels
    PlyData([PlyElement.describe(vertex, "vertex")]).write(path)


def export_full_cloud(config, poles=None, crossarms=None, wires=None, connecting=None):
    output_path = config["RECON_FULL_PLY_PATH"]

    recon = build_reconstruction_points(
        poles=poles, crossarms=crossarms, wires=wires, connecting=connecting
    )
    recon_pts = recon["points"].astype(np.float32, copy=False)
    recon_labels = recon["labels"].astype(np.uint16, copy=False)

    print('[INFO] [REBUILD] Loading full point cloud for final merge...')
    plydata = PlyData.read(config["PLY_PATH"])
    v = plydata["vertex"].data
    n_full = len(v["x"])
    n_recon = len(recon_pts)
    n_total = n_full + n_recon
    print(f"[INFO] [REBUILD] Point-cloud merge summary: {n_full:,}  +  recon: {n_recon:,}  =  {n_total:,}")

    # Preserve every original vertex property, including poles_id.
    vertex = np.empty(n_total, dtype=v.dtype)
    vertex[:n_full] = v
    for name in vertex.dtype.names:
        if name in ("x", "y", "z", "class"):
            continue
        vertex[name][n_full:] = -1 if name == "poles_id" else 0
    vertex["x"][n_full:] = recon_pts[:, 0]
    vertex["y"][n_full:] = recon_pts[:, 1]
    vertex["z"][n_full:] = recon_pts[:, 2]
    vertex["class"][n_full:] = recon_labels

    del plydata, v
    gc.collect()

    PlyData([PlyElement.describe(vertex, "vertex")]).write(output_path)
    print(f"[INFO] [REBUILD] Merged point cloud exported to: {output_path} ({n_total:,} points)")


# ---------------------------------------------------------
# Viewer
# ---------------------------------------------------------
def save_current_view(vis, output_path):
    vis.capture_screen_image(output_path)
    print('[INFO] [REBUILD] Visualization saved successfully.')
    return False


def main():
    config = load_config()

    pcd = load_and_color_pointcloud(config, ply_path=None, max_points=5_000_000)
    print(f"[INFO] [REBUILD] Visualization point count: {len(pcd.points):,} points")

    print('[INFO] [REBUILD] Starting pole reconstruction...')
    pole_geoms, poles_dict, crossarms_dict, connecting_geoms, poles_points = reconstruct_poles(config)
    print(f"[INFO] [REBUILD] Pole geometries reconstructed: {len(pole_geoms)}")
    print(f"[INFO] [REBUILD] Connecting geometries reconstructed: {len(connecting_geoms)}")

    print('[INFO] [REBUILD] Starting wire reconstruction...')
    wire_geoms, wire_curves = reconstruct_wires(config, poles_points)
    print(f"[INFO] [REBUILD] Wire geometries reconstructed: {len(wire_geoms)}")

    print('[INFO] [REBUILD] Opening Open3D viewer...')

    def merge_meshes(meshes):
        merged = o3d.geometry.TriangleMesh()
        for mesh in meshes:
            if isinstance(mesh, o3d.geometry.TriangleMesh):
                merged += mesh
        return merged


    def make_wire_preview(curves_data):
        points = []
        lines = []

        for item in curves_data:
            curve = np.asarray(item["curve"], dtype=np.float64)
            if len(curve) < 2:
                continue

            offset = len(points)
            points.extend(curve.tolist())

            lines.extend(
                [[offset + i, offset + i + 1]
                for i in range(len(curve) - 1)]
            )

        preview = o3d.geometry.LineSet()
        preview.points = o3d.utility.Vector3dVector(
            np.asarray(points, dtype=np.float64).reshape(-1, 3)
        )
        preview.lines = o3d.utility.Vector2iVector(
            np.asarray(lines, dtype=np.int32).reshape(-1, 2)
        )
        preview.paint_uniform_color([0.0, 0.0, 0.0])
        return preview


    print("[INFO] Preparing optimized viewer...")

    pole_preview = merge_meshes(pole_geoms)
    connecting_preview = merge_meshes(connecting_geoms)
    wire_preview = make_wire_preview(wire_curves)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        "MV Line Reconstruction | Press S to save",
        width=1200,
        height=800
    )

    vis.add_geometry(pcd)

    for geom in [pole_preview, connecting_preview, wire_preview]:
        if isinstance(geom, o3d.geometry.TriangleMesh):
            if len(geom.vertices) == 0:
                continue
        elif isinstance(geom, o3d.geometry.LineSet):
            if len(geom.lines) == 0:
                continue

        vis.add_geometry(geom, reset_bounding_box=False)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.line_width = 2.0
    opt.mesh_show_back_face = True

    vis.reset_view_point(True)

    vis.register_key_callback(
        ord("S"),
        lambda v: save_current_view(v, config["OUTPUT_IMAGE"])
    )

    print("[INFO] Viewer ready.")
    vis.run()
    vis.destroy_window()


    del pcd
    gc.collect()

    export_full_cloud(
        config,
        poles=poles_dict,
        crossarms=crossarms_dict,
        wires=wire_geoms,
        connecting=connecting_geoms,
    )


if __name__ == "__main__":
    main()