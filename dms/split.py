"""
=========================================================
Split:
Extracts local point cloud regions around detected MT
collisions. Uses pole positions and collision reports
to generate bounding boxes and export collision PLY files.
=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================
import os
import json
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement

# =========================================================
# SETTINGS (edit here; no command-line arguments)
# =========================================================
BOX_LENGTH_MARGIN = 6.0
BOX_WIDTH_EXTRA = 6.0
BOX_HEIGHT_MARGIN = 20.0
BOX_BASE_OFFSET = 5.0
DEFAULT_BOX_RADIUS = 4.0  # Used when the collision report has no tube_radius.

# =========================================================
# LOAD CONFIGURATION
# =========================================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

COLLISION_DIR = os.path.join(BASE_DIR, config["collision_dir"])
OUTPUT_DIR    = os.path.join(BASE_DIR, config["output_dir"])
OTHERS_DIR    = os.path.join(BASE_DIR, config.get("others_dir", "others"))

INPUT_PLY        = os.path.join(COLLISION_DIR, config["input_ply_new"])
COLLISION_REPORT = os.path.join(
    COLLISION_DIR, config["zones_report_json"]
)
POLES_JSON       = os.path.join(OUTPUT_DIR,     config["poles_final"])

# =========================================================
# LOAD DATA
# =========================================================
def load_ply_with_properties(ply_path):
    """Load XYZ for geometry while retaining every original vertex property.

    The structured vertex array is the source of truth. No column is removed,
    converted, renamed, or regenerated when saving the extracted points.
    """
    plydata = PlyData.read(ply_path, mmap="r")
    vertices = plydata["vertex"].data
    fields = vertices.dtype.names or ()
    missing = {"x", "y", "z"} - set(fields)
    if missing:
        raise ValueError(f"Input PLY is missing required XYZ properties: {sorted(missing)}")

    # Floating-point working coordinates are for spatial selection ONLY.
    # Original XYZ values and all other fields are exported unchanged.
    pts = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    pts = pts.astype(np.float64, copy=False)
    print(f"[INFO] [DMS/SPLIT] Input PLY: {len(vertices):,} points")
    return plydata, vertices, pts


source_ply, source_vertices, pts = load_ply_with_properties(INPUT_PLY)

# =========================================================
# POLE ID NORMALIZATION
# =========================================================
def norm_pole_id(pid):
    if isinstance(pid, str):
        s = pid.strip()
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
            return s
        except ValueError:
            return s
    if isinstance(pid, float):
        if pid.is_integer():
            return str(int(pid))
        return str(pid)
    return str(pid)


# =========================================================
# LOAD POLES FROM poles_final.json
# =========================================================
with open(POLES_JSON, "r", encoding="utf-8") as f:
    poles_data = json.load(f)

poles_dict  = {}
poles_bases = {}

for entry in poles_data:
    pid   = norm_pole_id(entry["Pole_ID"])
    info  = entry["Info"]
    uvec  = info["unit_vector"]
    ptype = entry.get("Type", "Monoposte")

    if ptype == "Biposte" and "Base_1" in info and "Base_2" in info:
        b1 = np.array(info["Base_1"], dtype=float)
        b2 = np.array(info["Base_2"], dtype=float)
        center = (b1 + b2) / 2.0
        poles_bases[pid] = (b1, b2)
    else:
        b = np.array(info["Base"], dtype=float)
        center = b
        poles_bases[pid] = (b, None)

    poles_dict[pid] = {
        "Pole_ID":  pid,
        "Center_X": float(center[0]),
        "Center_Y": float(center[1]),
        "Base_Z":   float(center[2]),
        "Dir_X":    float(uvec[0]),
        "Dir_Y":    float(uvec[1]),
        "Dir_Z":    float(uvec[2]),
        "Height_m": float(info["height"]),
        "Type":     ptype,
    }

poles_df = pd.DataFrame(list(poles_dict.values()))


def find_pole(poles_df, pole_id):
    key = norm_pole_id(pole_id)
    rows = poles_df.loc[poles_df["Pole_ID"] == key]
    if not rows.empty:
        return rows.iloc[0]
    alt = f"{key}.1"
    rows = poles_df.loc[poles_df["Pole_ID"] == alt]
    if not rows.empty:
        return rows.iloc[0]
    return None


def get_bases(row):
    return poles_bases[row["Pole_ID"]]


def choose_base(base_1, base_2, reference):
    if base_2 is None:
        return base_1
    d1 = np.linalg.norm(base_1 - reference)
    d2 = np.linalg.norm(base_2 - reference)
    return base_1 if d1 >= d2 else base_2


with open(COLLISION_REPORT, "r", encoding="utf-8") as f:
    collisions_report = json.load(f)

RADIUS = float(collisions_report.get("tube_radius", DEFAULT_BOX_RADIUS))

# =========================================================
# BOX GEOMETRY
# =========================================================
def build_search_box(B1, B2, u1, u2, h_max, radius):
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    if np.linalg.norm(u1) > 1e-9:
        u1 = u1 / np.linalg.norm(u1)
    if np.linalg.norm(u2) > 1e-9:
        u2 = u2 / np.linalg.norm(u2)

    length_vec = B2 - B1
    length = float(np.linalg.norm(length_vec))
    if length < 1e-6:
        l_dir  = np.array([1.0, 0.0, 0.0])
        length = 1e-6
    else:
        l_dir = length_vec / length

    h_dir = 0.5 * (u1 + u2)
    if np.linalg.norm(h_dir) < 1e-6:
        h_dir = np.array([0.0, 0.0, 1.0])
    h_dir = h_dir / np.linalg.norm(h_dir)
    h_dir = h_dir - np.dot(h_dir, l_dir) * l_dir
    if np.linalg.norm(h_dir) < 1e-6:
        h_dir = np.array([0.0, 0.0, 1.0])
        h_dir = h_dir - np.dot(h_dir, l_dir) * l_dir
        if np.linalg.norm(h_dir) < 1e-6:
            h_dir = np.array([0.0, 1.0, 0.0])
    h_dir = h_dir / np.linalg.norm(h_dir)

    w_dir = np.cross(h_dir, l_dir)
    w_dir = w_dir / np.linalg.norm(w_dir)

    box_length = length + BOX_LENGTH_MARGIN
    box_width  = radius * 2.0 + BOX_WIDTH_EXTRA
    box_height = h_max + BOX_HEIGHT_MARGIN

    origin = B1 - (BOX_LENGTH_MARGIN / 2.0) * l_dir - BOX_BASE_OFFSET * h_dir

    return {
        "origin":      origin,
        "length":      box_length,
        "width":       box_width,
        "height":      box_height,
        "l_dir":       l_dir,
        "w_dir":       w_dir,
        "h_dir":       h_dir,
        "length_axis": length,
    }


def world_to_local(points, box):
    R   = np.column_stack((box["l_dir"], box["w_dir"], box["h_dir"]))
    rel = points - box["origin"]
    return rel @ R


def inside_box_local(local_pts, box):
    lx, ly, lz = local_pts[:, 0], local_pts[:, 1], local_pts[:, 2]
    half_w = box["width"] / 2.0
    mask = (
        (lx >= 0.0) & (lx <= box["length"]) &
        (ly >= -half_w) & (ly <= half_w) &
        (lz >= 0.0) & (lz <= box["height"])
    )
    return mask


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def save_ply(filename, selected_vertices, original_ply):
    """
    Save selected points preserving all original vertex properties,
    field names, order, and data types.
    """

    source_element = original_ply["vertex"]

    vertex_element = PlyElement(
        "vertex",
        source_element.properties,
        len(selected_vertices),
        comments=source_element.comments,
    )

    vertex_element.data = selected_vertices

    exported = PlyData(
        [vertex_element],
        text=original_ply.text,
        byte_order=original_ply.byte_order,
        comments=original_ply.comments,
        obj_info=original_ply.obj_info,
    )

    exported.write(filename)


# =========================================================
# RESOLVE SECOND ENDPOINT (real pole or tap splice point)
# =========================================================
def resolve_second_endpoint(section, poles_df, u1):
    """Resolve the second pole from the physical section in zones_report.json."""
    to_id = section.get("to_pole")
    if to_id is None:
        return None, None
    row = find_pole(poles_df, to_id)
    if row is None:
        return None, None
    B2 = np.array([row["Center_X"], row["Center_Y"], row["Base_Z"]], dtype=float)
    u2 = np.array([row["Dir_X"], row["Dir_Y"], row["Dir_Z"]], dtype=float)
    return B2, u2


# =========================================================
# MAIN PROCESS
# =========================================================

def has_confirmed_collision(section):
    """A confirmed DMS collision has at least one critical point (is_cp=1)."""
    for collision in section.get("collisions", []):
        for wire in collision.get("wires_affected", {}).values():
            for level in wire.get("levels", {}).values():
                if level.get("critical_points"):
                    return True
    return False


cid = 1

for section in collisions_report["sections"]:
    if not has_confirmed_collision(section):
        continue

    from_id = section["from_pole"]

    B1_row = find_pole(poles_df, from_id)
    if B1_row is None:
        print(f"[WARNING] [DMS/SPLIT] Skipping section {section.get('section_id')}: "
              f"pole {from_id} not found in {POLES_JSON}")
        continue

    u1 = np.array([B1_row["Dir_X"], B1_row["Dir_Y"], B1_row["Dir_Z"]],
                  dtype=float)

    B2_rough, u2 = resolve_second_endpoint(section, poles_df, u1)
    if B2_rough is None:
        print(f"[WARNING] [DMS/SPLIT] Skipping section {section.get('section_id')}: "
              f"could not resolve second endpoint "
              f"(to_pole={section.get('to_pole')}, "
              f"to_segment={section.get('to_segment')})")
        continue

    from_b1, from_b2 = get_bases(B1_row)
    B1 = choose_base(from_b1, from_b2, B2_rough)

    to_row = None
    if section.get("to_pole") is not None:
        to_row = find_pole(poles_df, section["to_pole"])
        if to_row is not None and to_row["Type"] == "Biposte":
            to_b1, to_b2 = get_bases(to_row)
            B2 = choose_base(to_b1, to_b2, B1)
        else:
            B2 = B2_rough
    else:
        B2 = B2_rough

    h1 = float(B1_row["Height_m"]) if B1_row is not None else 0.0
    h2 = float(to_row["Height_m"]) if to_row is not None else 0.0
    h_max = max(h1, h2)

    box = build_search_box(B1, B2, u1, u2, h_max, RADIUS)

    local_pts = world_to_local(pts, box)
    mask      = inside_box_local(local_pts, box)

    # Boolean indexing copies complete structured records, preserving all
    # original point attributes (class, RGB, DMS/CP flags, wire, level, etc.).
    extracted_vertices = source_vertices[mask]

    collision_subdir = os.path.join(COLLISION_DIR, f"collision_{cid:02d}")
    os.makedirs(collision_subdir, exist_ok=True)

    out_path = os.path.join(
        collision_subdir,
        f"collision_extract_{cid:02d}.ply"
    )

    save_ply(out_path, extracted_vertices, source_ply)

    print(f"[INFO] [DMS/SPLIT] Section {section['section_id']}: {len(extracted_vertices):,} points -> {out_path}")

    cid += 1
print(f"[INFO] [DMS/SPLIT] Exported {cid - 1} collision PLY files")
