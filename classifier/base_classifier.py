import os
import json
import numpy as np

from pathlib import Path

# Los nombres de entrada y salida se obtienen exclusivamente de config.json.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.json"

with CONFIG_FILE.open(encoding="utf-8") as f:
    config = json.load(f)

output_dir = Path(config["output_dir"])
if not output_dir.is_absolute():
    output_dir = PROJECT_ROOT / output_dir

json_extractor_path = output_dir / config["json_extractor"]
json_points_path = output_dir / config["json_points"]
json_wires_path = output_dir / config["json_wires_2"]
poles_final_path = output_dir / config["poles_final"]
wires_final_path = output_dir / config["wires_final"]

INFO_DECIMALS     = 20
BIPOLE_SPACING    = 2.2
CONNECTING_ELEMENT = 0.01
EPS               = 1e-6


with open(json_extractor_path, "r") as f:
    extractor_poles = json.load(f)

with open(json_points_path, "r") as f:
    points_list = json.load(f)

with open(json_wires_path, "r") as f:
    wires_sections = json.load(f)

subs_by_pole = {}
for item in points_list:
    subs_by_pole.setdefault(item["Pole_ID"], []).append(item)


def compute_unit_vector(base, points_data):
    farthest = None
    farthest_d = -1.0
    for key in ("Point_H1", "Point_H2", "Point_H3"):
        p = points_data.get(key)
        if p is None:
            continue
        d = float(np.linalg.norm(np.array(p, dtype=float) - base))
        if d > farthest_d:
            farthest_d = d
            farthest = np.array(p, dtype=float)

    if farthest is None:
        return [0.0, 0.0, 0.0]

    v = farthest - base
    n = float(np.linalg.norm(v))
    if n <= 0:
        return [0.0, 0.0, 0.0]
    return (v / n).tolist()


def expand_bipole_points(points_data, uvh, spacing):
    uvh = np.array(uvh, dtype=float)
    offset = (spacing / 2.0) * uvh
    new_pts = {}
    for k in (1, 2, 3):
        H = np.array(points_data[f"Point_H{k}"], dtype=float)
        C = np.array(points_data[f"Point_C{k}"], dtype=float)
        new_pts[f"Point_H{k}1"] = (H - offset).tolist()
        new_pts[f"Point_H{k}2"] = (H + offset).tolist()
        new_pts[f"Point_C{k}1"] = (C - offset).tolist()
        new_pts[f"Point_C{k}2"] = (C + offset).tolist()
    return new_pts


def refit_parametric(eq, start, end):
    """Preserve each quadratic coefficient and pass exactly through t=0 and t=1."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    eq = eq or {}
    updated = {}
    for idx, axis in enumerate(("x", "y", "z")):
        a = float(eq.get(axis, {}).get("a", 0.0))
        c = float(start[idx])
        b = float(end[idx] - start[idx] - a)
        updated[axis] = {"a": a, "b": b, "c": c}
    return updated


def evaluate_parametric(eq, t):
    return np.array([
        float(eq[axis]["a"]) * t * t
        + float(eq[axis]["b"]) * t
        + float(eq[axis]["c"])
        for axis in ("x", "y", "z")
    ], dtype=float)


def get_anchor_points(pole_entry, side, section_id=None):
    pts = pole_entry["Points"]
    is_biposte = pole_entry.get("Type") == "Biposte"
    is_case_i  = pole_entry.get("Case") == "I"

    anchors = []
    for k in range(3):
        if is_case_i:
            suffix = 1
            if section_id is not None:
                suffix = pole_entry.get("CaseI_Sections", {}).get(section_id, 1)
            key = f"Point_C{k+1}{suffix}"
        elif is_biposte:
            key = f"Point_C{k+1}2" if side == "from" else f"Point_C{k+1}1"
        else:
            key = f"Point_C{k+1}"
        anchors.append(np.array(pts[key], dtype=float))
    return anchors


def parse_orig_id(pole_id):
    s = str(pole_id)
    return int(s.split(".")[0]) if "." in s else int(s)


def build_pole_usage_order(wires_sections):
    usage = {}
    for section in wires_sections:
        sid = section.get("section_id", "")
        if "_" not in sid:
            continue
        a_str, b_str = sid.split("_")
        try:
            a_id, b_id = int(a_str), int(b_str)
        except ValueError:
            continue
        usage.setdefault(a_id, []).append(sid)
        usage.setdefault(b_id, []).append(sid)
    return usage


def pick_pole_for_section(poles_final, orig_id, section_id, usage_order):
    cands = [p for p in poles_final
             if parse_orig_id(p["Pole_ID"]) == orig_id]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]

    cands_sorted = sorted(cands, key=lambda p: str(p["Pole_ID"]))
    sections_for_pole = usage_order.get(orig_id, [])
    try:
        idx = sections_for_pole.index(section_id)
    except ValueError:
        idx = 0
    return cands_sorted[idx % len(cands_sorted)]


def ask_biposte_ids(extractor_poles):
    ids = [p["Pole_ID"] for p in extractor_poles]
    print('[INFO] [CLASSIFIER] Available pole identifiers:')
    print(", ".join(str(i) for i in ids))
    print('[INFO] [CLASSIFIER] Select biposte pole identifiers (comma-separated; press Enter for none):')
    raw = input("> ").strip()
    if not raw:
        return set()

    biposte = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            biposte.add(int(part))
        except ValueError:
            print(f"[WARNING] [CLASSIFIER] Ignoring invalid pole identifier: {part}")
    return biposte


def compute_case_i_c_points(poles_final, wires_sections, usage_order):
    appearances = {}
    for section in wires_sections:
        sid = section.get("section_id", "")
        if "_" not in sid:
            continue
        a_str, b_str = sid.split("_")
        try:
            a_id, b_id = int(a_str), int(b_str)
        except ValueError:
            continue

        pa = pick_pole_for_section(poles_final, a_id, sid, usage_order)
        pb = pick_pole_for_section(poles_final, b_id, sid, usage_order)
        if pa is None or pb is None:
            continue

        appearances.setdefault(pa["Pole_ID"], []).append({
            "section_id": sid, "side": "from",
            "section": section, "pa": pa, "pb": pb
        })
        appearances.setdefault(pb["Pole_ID"], []).append({
            "section_id": sid, "side": "to",
            "section": section, "pa": pa, "pb": pb
        })

    for pole in poles_final:
        if pole.get("Case") != "I":
            continue
        apps = appearances.get(pole["Pole_ID"], [])
        if not apps:
            continue
        pole["CaseI_Sections"] = {}
        for idx, app in enumerate(apps):
            pole["CaseI_Sections"][app["section_id"]] = idx + 1

    for pole in poles_final:
        if pole.get("Case") != "I":
            continue
        apps = appearances.get(pole["Pole_ID"], [])
        if not apps:
            continue

        pts = pole["Points"]

        orig_C = {}
        for k in range(3):
            key = f"Point_C{k+1}"
            if key in pts:
                orig_C[k] = np.array(pts[key], dtype=float)
            else:
                orig_C[k] = np.array(pts.get(f"Point_H{k+1}", [0, 0, 0]),
                                     dtype=float)

        for k in range(1, 4):
            pts.pop(f"Point_C{k}", None)

        for app in apps:
            suffix = pole["CaseI_Sections"][app["section_id"]]
            sid = app["section_id"]
            section = app["section"]

            if app["side"] == "from":
                own, other = app["pa"], app["pb"]
            else:
                own, other = app["pb"], app["pa"]

            for wire in section.get("wires", []):
                wid = wire.get("wire_id", 1)
                k = wid - 1
                if k < 0 or k >= 3:
                    continue

                own_h_key = f"Point_H{k+1}"
                if own_h_key not in pts:
                    continue

                if other.get("Type") == "Biposte":
                    other_suffix_h = 1 if app["side"] == "from" else 2
                    other_h_key = f"Point_H{k+1}{other_suffix_h}"
                else:
                    other_h_key = f"Point_H{k+1}"

                if other_h_key not in other["Points"]:
                    continue

                H_own   = np.array(pts[own_h_key], dtype=float)
                H_other = np.array(other["Points"][other_h_key], dtype=float)

                own_info = own["Info"]
                other_info = other["Info"]
                own_base = np.array(own_info.get("Base", own_info.get("Base_1")), dtype=float)
                other_base = np.array(other_info.get("Base", other_info.get("Base_1")), dtype=float)
                if "Base_2" in own_info:
                    own_base = (own_base + np.array(own_info["Base_2"], dtype=float)) / 2.0
                if "Base_2" in other_info:
                    other_base = (other_base + np.array(other_info["Base_2"], dtype=float)) / 2.0
                if other.get("Case") == "I":
                    other_suffix = other.get("CaseI_Sections", {}).get(sid, 1)
                    other_key = f"Point_C{k+1}{other_suffix}"
                elif other.get("Type") == "Biposte":
                    other_key = (f"Point_C{k+1}1" if app["side"] == "from"
                                else f"Point_C{k+1}2")
                else:
                    other_key = f"Point_C{k+1}"

                C_other    = np.array(other["Points"].get(other_key, H_other),
                                      dtype=float)
                C_own_orig = orig_C[k]

                if app["side"] == "from":
                    first, last = C_own_orig, C_other
                    t_c = CONNECTING_ELEMENT
                else:
                    first, last = C_other, C_own_orig
                    t_c = 1.0 - CONNECTING_ELEMENT

                eq_local = refit_parametric(wire.get("equation"), first, last)
                C_new = evaluate_parametric(eq_local, t_c)
                pts[f"Point_C{k+1}{suffix}"] = C_new.tolist()


def main():
    biposte_ids = ask_biposte_ids(extractor_poles)

    poles_final = []

    for pole in extractor_poles:
        pole_id = pole["Pole_ID"]
        info    = pole["Info"]

        base       = [round(float(v), INFO_DECIMALS) for v in info["Base"]]
        num_points = info.get("num_points", 0)
        uvh_orig   = info.get("unit_vector_horizontal", [0.0, 0.0, 0.0])

        configuration = pole.get("configuration")
        if configuration is None:
            configuration = info.get("configuration", "")

        subs = subs_by_pole.get(pole_id, [])
        if not subs:
            subs = [{"Sub_ID": 0, "Case": "", "Points": {},
                     "t-values": {}, "Crosshead_Lengths": {}}]

        type_str = "Biposte" if pole_id in biposte_ids else "Monoposte"

        uvh_out = ([round(float(v), INFO_DECIMALS) for v in uvh_orig]
                   if type_str == "Biposte" else [0.0, 0.0, 0.0])

        multiple = len(subs) > 1

        for sub in subs:
            sub_id      = sub.get("Sub_ID", 0)
            case        = sub.get("Case", "")
            points_data = sub.get("Points", {}) or {}
            t_values    = sub.get("t-values", {}) or {}
            crosshead   = sub.get("Crosshead_Lengths", {}) or {}

            height = round(max(t_values.get("t1", 0.0),
                               t_values.get("t2", 0.0),
                               t_values.get("t3", 0.0)), INFO_DECIMALS)

            uv = [round(float(v), INFO_DECIMALS)
                  for v in compute_unit_vector(np.array(base, dtype=float),
                                               points_data)]

            if type_str == "Biposte" and points_data:
                points_data = expand_bipole_points(points_data,
                                                   uvh_out,
                                                   BIPOLE_SPACING)

            new_pole_id = f"{pole_id}.{sub_id + 1}" if multiple else pole_id

            if type_str == "Biposte":
                base_arr = np.array(base, dtype=float)
                uvh_arr  = np.array(uvh_out, dtype=float)
                offset   = (BIPOLE_SPACING / 2.0) * uvh_arr

                base_1 = [round(float(v), INFO_DECIMALS)
                          for v in (base_arr - offset)]
                base_2 = [round(float(v), INFO_DECIMALS)
                          for v in (base_arr + offset)]

                info_out = {
                    "num_points": num_points,
                    "Base_1": base_1,
                    "Base_2": base_2,
                    "height": height,
                    "unit_vector": uv,
                    "unit_vector_horizontal": uvh_out,
                }
            else:
                info_out = {
                    "num_points": num_points,
                    "Base": base,
                    "height": height,
                    "unit_vector": uv,
                    "unit_vector_horizontal": uvh_out,
                }

            poles_final.append({
                "Pole_ID": new_pole_id,
                "Case": case,
                "Type": type_str,
                "Info": info_out,
                "configuration": configuration,
                "Points": points_data,
                "t-values": t_values,
                "Crosshead_Lengths": crosshead,
            })

    usage_order = build_pole_usage_order(wires_sections)
    compute_case_i_c_points(poles_final, wires_sections, usage_order)

    print('[INFO] [CLASSIFIER] Detected pole classifications:')
    for p in poles_final:
        print(f"[INFO] [CLASSIFIER] Pole {p['Pole_ID']}: {p['Type']}")

    for section in wires_sections:
        sid = section.get("section_id", "")
        if "_" not in sid:
            continue

        a_str, b_str = sid.split("_")
        try:
            a_id, b_id = int(a_str), int(b_str)
        except ValueError:
            continue

        pa = pick_pole_for_section(poles_final, a_id, sid, usage_order)
        pb = pick_pole_for_section(poles_final, b_id, sid, usage_order)
        if pa is None or pb is None:
            print(f"[WARNING] [CLASSIFIER] Section {sid}: pole not found")
            continue

        try:
            P_from = get_anchor_points(pa, "from", sid)
            P_to   = get_anchor_points(pb, "to",   sid)
        except KeyError as e:
            print(f"[WARNING] [CLASSIFIER] Section {sid}: missing key {e}")
            continue

        for i, wire in enumerate(section.get("wires", [])):
            wid = wire.get("wire_id", i + 1)
            k = wid - 1
            if k < 0 or k >= 3:
                continue

            start = P_from[k]
            end = P_to[k]
            wire["equation"] = refit_parametric(wire.get("equation"), start, end)
            wire["point_min"] = {
                axis: float(start[j]) for j, axis in enumerate(("x", "y", "z"))
            }
            wire["point_max"] = {
                axis: float(end[j]) for j, axis in enumerate(("x", "y", "z"))
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(wires_final_path, "w", encoding="utf-8") as f:
        json.dump(wires_sections, f, indent=4)
    print(f"[INFO] [CLASSIFIER] Wires JSON exported to: {wires_final_path}")

    for p in poles_final:
        p.pop("CaseI_Sections", None)

    with open(poles_final_path, "w", encoding="utf-8") as f:
        json.dump(poles_final, f, indent=4)
    print(f"[INFO] [CLASSIFIER] Poles JSON exported to: {poles_final_path}")


if __name__ == "__main__":
    main()