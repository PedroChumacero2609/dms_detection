import json
import os
import json.encoder
import numpy as np
from scipy.optimize import minimize_scalar

from ._paths import load_config, configured_path, output_path, label_ids
EPS = 1e-6
OUT_DECIMALS = 20
DEFAULT_SAG_PARAMETER = 800.0


class D20Encoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        markers = {} if self.check_circular else None
        enc = (json.encoder.encode_basestring_ascii
               if self.ensure_ascii else json.encoder.encode_basestring)

        def floatstr(x):
            if not np.isfinite(x):
                raise ValueError("Non-finite float in JSON")
            return repr(float(x))

        _indent = None if self.indent is None else (
            self.indent if isinstance(self.indent, str) else ' ' * self.indent)
        it = json.encoder._make_iterencode(
            markers, self.default, enc, _indent, floatstr,
            self.key_separator, self.item_separator, self.sort_keys,
            self.skipkeys, _one_shot,
        )
        return it(o, 0)




def parse_point(pt):
    if pt is None:
        return None
    return {axis: float(pt[axis]) for axis in ("x", "y", "z")}


def parse_equation(eq):
    if eq is None:
        return None
    return {
        axis: {key: float(eq[axis][key]) for key in ("a", "b", "c")}
        for axis in ("x", "y", "z")
    }


def cable_pt(t, eq):
    t = float(t)
    return np.array([
        eq[axis]["a"] * t * t + eq[axis]["b"] * t + eq[axis]["c"]
        for axis in ("x", "y", "z")
    ], dtype=float)


def distance_point_to_curve(point, eq):
    if eq is None:
        return float("inf")
    P = np.asarray(point, dtype=float)
    result = minimize_scalar(
        lambda t: float(np.sum((cable_pt(t, eq) - P) ** 2)),
        bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-12}
    )
    distances = [float(np.linalg.norm(cable_pt(t, eq) - P))
                 for t in (0.0, 1.0, float(result.x))]
    return min(distances)


def score_sub(pole_entry, wire_eqs):
    total = 0.0
    pts = pole_entry["Points"]
    for k in range(3):
        C = np.array(pts[f"Point_C{k+1}"], dtype=float)
        eq = wire_eqs[k] if k < len(wire_eqs) else None
        d = distance_point_to_curve(C, eq)
        if np.isfinite(d):
            total += d
    return total


def find_sub_for_section(pole_id, section_case, wire_eqs, points_data, verbose=False):
    candidates = [p for p in points_data if p["Pole_ID"] == pole_id]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    case_matched = [c for c in candidates if c.get("Case") == section_case]
    if len(case_matched) == 1:
        if verbose:
            print(f"[INFO] [POINTS] Pole {pole_id}: Case={section_case} -> "
                  f"Sub {case_matched[0]['Sub_ID']}")
        return case_matched[0]

    pool = case_matched if case_matched else candidates
    scored = [(score_sub(c, wire_eqs), c) for c in pool]
    scored.sort(key=lambda item: (item[0], item[1]["Sub_ID"]))
    if verbose:
        print(f"[INFO] [POINTS] Pole {pole_id}: " + ", ".join(
            f"Sub {c['Sub_ID']}(Case={c.get('Case')}) -> {s:.4f}"
            for s, c in scored))
    return scored[0][1]


def resolve_anchor_points(section_id, span, points_data, taps_dict, verbose=False):
    parts = section_id.split("_")
    section_case = span.get("case")
    wire_eqs = [parse_equation(w.get("equation")) for w in span["wires"]]

    if len(parts) == 2:
        from_id, to_id = map(int, parts)
        pole_from = find_sub_for_section(from_id, section_case, wire_eqs,
                                         points_data, verbose)
        pole_to = find_sub_for_section(to_id, section_case, wire_eqs,
                                       points_data, verbose)
        if pole_from is None or pole_to is None:
            raise KeyError(f"Pole {from_id} or {to_id} not found in points_data")
        P_from = [np.array(pole_from["Points"][f"Point_C{i+1}"], dtype=float)
                  for i in range(3)]
        P_to = [np.array(pole_to["Points"][f"Point_C{i+1}"], dtype=float)
                for i in range(3)]
        return P_from, P_to

    tap_pole_id = int(parts[0])
    pole_from = find_sub_for_section(tap_pole_id, section_case, wire_eqs,
                                     points_data, verbose)
    if pole_from is None:
        raise KeyError(f"Tap pole {tap_pole_id} not found in points_data")

    P_from = [np.array(pole_from["Points"][f"Point_C{i+1}"], dtype=float)
              for i in range(3)]
    tap_entry = taps_dict.get(section_id)
    if tap_entry is None:
        raise KeyError(f"No se pudo calcular la intersección de la derivación '{section_id}'")
    tap_info = tap_entry["Info"]
    P_to = [np.array(tap_info[f"Point_W{i+1}"], dtype=float) for i in range(3)]
    return P_from, P_to


def refit_equation(eq_old, anchor_from, anchor_to):
    A = np.asarray(anchor_from, dtype=float)
    B = np.asarray(anchor_to, dtype=float)

    if eq_old is not None:
        old_start = cable_pt(0.0, eq_old)
        old_end = cable_pt(1.0, eq_old)
        direct = np.linalg.norm(old_start - A) + np.linalg.norm(old_end - B)
        reverse = np.linalg.norm(old_start - B) + np.linalg.norm(old_end - A)
        if reverse < direct:
            A, B = B, A
        a = np.array([eq_old[axis]["a"] for axis in ("x", "y", "z")], dtype=float)
    else:
        horizontal_span = float(np.linalg.norm((B - A)[:2]))
        a = np.array([0.0, 0.0,
                      horizontal_span ** 2 / (4.0 * DEFAULT_SAG_PARAMETER)], dtype=float)

    b = B - A - a
    c = A
    eq_new = {
        axis: {"a": float(a[i]), "b": float(b[i]), "c": float(c[i])}
        for i, axis in enumerate(("x", "y", "z"))
    }
    return eq_new


def main():
    cfg = load_config()
    wires_path = configured_path(cfg, "output_dir", "json_wires")
    points_path = configured_path(cfg, "output_dir", "json_points")

    with open(wires_path) as f:
        wires_data = json.load(f)
    with open(points_path) as f:
        points_data = json.load(f)

    # Intersecciones de derivaciones calculadas en memoria: no se genera JSON intermedio.
    from .points import compute_tap_intersections
    per_pole = {}
    for entry in points_data:
        pid = int(entry["Pole_ID"])
        coords = entry["Points"]
        per_pole.setdefault(pid, []).append({
            "case": entry["Case"],
            "C": {i: np.asarray(coords[f"Point_C{i}"], dtype=float) for i in (1, 2, 3)},
        })
    taps_data = compute_tap_intersections(wires_data, per_pole)
    taps_dict = {item["id"]: item for item in taps_data}

    new_wires = []
    for span in wires_data:
        section_id = span["section_id"]
        try:
            P_from, P_to = resolve_anchor_points(
                section_id, span, points_data, taps_dict, verbose=True)
        except (KeyError, IndexError, ValueError) as e:
            print(f"[WARNING] [POINTS] Skipping invalid span '{section_id}': {e}")
            new_wires.append(span)
            continue

        wires_new = []
        for i, wire in enumerate(span["wires"]):
            if i >= len(P_from):
                wires_new.append(wire)
                continue
            eq_old = parse_equation(wire.get("equation"))
            new_eq = refit_equation(eq_old, P_from[i], P_to[i])
            new_wire = dict(wire)
            new_wire["equation"] = new_eq
            new_wire["point_min"] = dict(zip(("x", "y", "z"),
                                             map(float, cable_pt(0.0, new_eq))))
            new_wire["point_max"] = dict(zip(("x", "y", "z"),
                                             map(float, cable_pt(1.0, new_eq))))
            wires_new.append(new_wire)

        new_span = dict(span)
        new_span["wires"] = wires_new
        new_wires.append(new_span)

    out_path = output_path(cfg, "json_wires_2")
    with open(out_path, "w") as f:
        json.dump(new_wires, f, indent=4, cls=D20Encoder, allow_nan=False)
    print(f"[INFO] [POINTS] Adjusted wires JSON exported to: {out_path}")


if __name__ == "__main__":
    main()
