import json
import os
import numpy as np

from ._paths import load_config, configured_path, output_path, label_ids

EPS = 1e-6

ANGLE_T_MAX = 110.0
ANGLE_P_MAX = 165.0
ANGLE_O_MAX = 10.0
C_RATIO_TOL = 0.10
ANGLE_ALIGN_TOL = 15.0
DIR_MATCH_TOL = 15.0
F_OFFSET_TOL = 0.30
OUT_DECIMALS = 20
CROSSHEAD_LEN_TOL = 0.15
L_DIST_TOL = 0.9

MAX_T_VALUE = 14.0
MAX_CROSSHEAD_LEN = 2.0
MIN_CROSSHEAD_LEN = 0.6
SUBCASE_H_DIFF_TOL = 2.0
T_GAP_TOL = 2.0




def norm(v):
    n = np.linalg.norm(v)
    return v if n < EPS else v / n


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def normalize_pole(p):
    info = p["Info"]
    base = info["Base"]
    uv   = info["unit_vector"]

    uvh = info.get("unit_vector_horizontal")
    if uvh is None:
        uvh = uv
    uvh = np.array([float(uvh[0]), float(uvh[1]), float(uvh[2])], dtype=float)

    if np.linalg.norm(uvh) < EPS:
        horiz = np.array([1.0, 0.0, 0.0])
    else:
        horiz = uvh / np.linalg.norm(uvh)

    config = info.get("configuration", p.get("configuration", ""))
    if config is None:
        config = ""

    return {
        "id":       int(p["Pole_ID"]),
        "center_x": float(base[0]),
        "center_y": float(base[1]),
        "base_z":   float(base[2]),
        "height":   float(info["height"]),
        "dir_x":    float(uv[0]),
        "dir_y":    float(uv[1]),
        "dir_z":    float(uv[2]),
        "dir_h_x":  float(horiz[0]),
        "dir_h_y":  float(horiz[1]),
        "dir_h_z":  float(horiz[2]),
        "config":   str(config).strip(),
    }


def pole_coord(pole):
    return np.array([pole["center_x"], pole["center_y"], pole["base_z"]], dtype=float)


def pole_dir(pole):
    return norm(np.array([pole["dir_x"], pole["dir_y"], pole["dir_z"]], dtype=float))


def pole_dir_horizontal(pole):
    return norm(np.array([pole["dir_h_x"], pole["dir_h_y"], pole["dir_h_z"]], dtype=float))


def cable_pt(t, eq):
    return np.array([
        eq[axis]["a"] * t * t + eq[axis]["b"] * t + eq[axis]["c"]
        for axis in ("x", "y", "z")
    ], dtype=float)


def parse_equation(eq):
    if eq is None:
        return None
    return {
        axis: {c: float(eq[axis][c]) for c in ("a", "b", "c")}
        for axis in ("x", "y", "z")
    }


def parse_point(pt):
    if pt is None:
        return None
    return np.array([float(pt["x"]), float(pt["y"]), float(pt["z"])])


def eq_is_zero(eq):
    if eq is None:
        return True
    for ax in ("x", "y", "z"):
        for c in ("a", "b", "c"):
            if abs(eq[ax][c]) > EPS:
                return False
    return True


def direction_perp_to_pole(d_raw, u):
    d = d_raw - np.dot(d_raw, u) * u
    return norm(d)


def effective_wire(span, k):
    w = span["wires"][k]
    eq = parse_equation(w.get("equation"))
    pmin = parse_point(w.get("point_min"))
    pmax = parse_point(w.get("point_max"))

    if eq is not None and not eq_is_zero(eq):
        return eq, pmin, pmax

    for w2 in span["wires"]:
        eq2 = parse_equation(w2.get("equation"))
        if eq2 is not None and not eq_is_zero(eq2):
            return eq2, parse_point(w2.get("point_min")), parse_point(w2.get("point_max"))

    return None, None, None


def count_nonzero_equations(span):
    if span is None:
        return None
    count = 0
    for k in range(3):
        w = span["wires"][k]
        eq = parse_equation(w.get("equation"))
        if eq is not None and not eq_is_zero(eq):
            count += 1
    return count


def unify_crosshead_lengths(lengths, tol=CROSSHEAD_LEN_TOL):
    big, mid, small = sorted(lengths, reverse=True)
    if big < EPS:
        return 0.0
    if (big - small) / big <= tol:
        return (big + mid + small) / 3.0

    pairs = [(big, mid), (mid, small), (big, small)]
    a, b = min(pairs, key=lambda p: abs(p[0] - p[1]))
    return (a + b) / 2.0


def _curve_coeff_vectors(eq):
    a = np.array([eq[k]["a"] for k in ("x", "y", "z")], dtype=float)
    b = np.array([eq[k]["b"] for k in ("x", "y", "z")], dtype=float)
    c = np.array([eq[k]["c"] for k in ("x", "y", "z")], dtype=float)
    return a, b, c


def _stationary_parameters(a, b, c):
    # Roots of (a*t² + b*t + c) dot (2*a*t + b) = 0.
    coeffs = np.array([
        float(np.dot(b, c)),
        float(np.dot(b, b) + 2.0 * np.dot(a, c)),
        float(3.0 * np.dot(a, b)),
        float(2.0 * np.dot(a, a)),
    ])
    scale = float(np.max(np.abs(coeffs)))
    if scale < 1e-20:
        return []
    coeffs /= scale
    while len(coeffs) > 1 and abs(coeffs[-1]) < 1e-14:
        coeffs = coeffs[:-1]
    if len(coeffs) <= 1:
        return []
    roots = np.roots(coeffs[::-1])
    return [float(np.clip(r.real, 0.0, 1.0)) for r in roots
            if abs(r.imag) < 1e-7 and -1e-9 <= r.real <= 1.0 + 1e-9]


def find_perp_roots(eq, B, u):
    u = norm(u)
    projection = np.eye(3) - np.outer(u, u)
    a, b, c = _curve_coeff_vectors(eq)
    return _stationary_parameters(
        projection @ a, projection @ b, projection @ (c - B)
    )


def nearest_curve_point(eq, Q):
    a, b, c = _curve_coeff_vectors(eq)
    candidates = [0.0, 1.0] + _stationary_parameters(a, b, c - Q)
    best_t = min(candidates, key=lambda t: np.sum((cable_pt(t, eq) - Q)**2))
    return cable_pt(best_t, eq)


def project_curve_to_pole(eq, B, u, pmin=None, pmax=None):
    if eq is None:
        return None

    u = norm(u)
    candidates = [0.0, 1.0] + find_perp_roots(eq, B, u)

    def dist_sq(t):
        d = cable_pt(t, eq) - B
        perp = d - float(np.dot(d, u)) * u
        return float(np.dot(perp, perp))

    best_t = min(candidates, key=dist_sq)
    return cable_pt(best_t, eq)


def build_pole_incidences(wires_data, poles_dict):
    incidences = {}

    def ensure(pid):
        if pid not in incidences:
            incidences[pid] = {1: [], 2: [], 3: []}
        return incidences[pid]

    for span in wires_data:
        parts = span["section_id"].split("_")
        if len(parts) == 2:
            pole_ids = [int(parts[0]), int(parts[1])]
        else:
            pole_ids = [int(parts[0])]

        eff = [effective_wire(span, k) for k in range(3)]

        for pid in pole_ids:
            if pid not in poles_dict:
                continue
            pole = poles_dict[pid]
            B = pole_coord(pole)
            u = pole_dir(pole)
            lst = ensure(pid)

            for idx, wire_id in enumerate((1, 2, 3)):
                eq, pmin, pmax = eff[idx]
                if eq is None and pmin is None and pmax is None:
                    continue

                pt = project_curve_to_pole(eq, B, u, pmin, pmax)

                if pt is None:
                    if pmin is not None and pmax is not None:
                        dmin = np.linalg.norm(pmin - B)
                        dmax = np.linalg.norm(pmax - B)
                        pt = pmin if dmin <= dmax else pmax
                    elif pmin is not None:
                        pt = pmin
                    elif pmax is not None:
                        pt = pmax
                    else:
                        continue

                lst[wire_id].append((pt, eq, pmin, pmax))

    return incidences


def build_pole_neighbors(wires_data):
    neighbors = {}
    for span in wires_data:
        parts = span["section_id"].split("_")
        if len(parts) != 2:
            continue
        a, b = int(parts[0]), int(parts[1])
        neighbors.setdefault(a, [])
        neighbors.setdefault(b, [])
        if b not in neighbors[a]:
            neighbors[a].append(b)
        if a not in neighbors[b]:
            neighbors[b].append(a)
    return neighbors



def inherit_short_pole_heights(wires_data, poles_dict, min_height=4.0):
    """Replace heights < min_height using an adjacent pole with a valid height.

    For a section A_B, A is considered the predecessor of B. Original
    (unmodified) heights are used as references, avoiding cascading errors.
    Height and both direction vectors are inherited from the same neighbor.
    The original pole's base coordinates remain unchanged.
    """
    predecessors = {}
    successors = {}
    for span in wires_data:
        parts = span["section_id"].split("_")
        if len(parts) != 2:
            continue
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        predecessors.setdefault(b, []).append(a)
        successors.setdefault(a, []).append(b)

    original_heights = {pid: float(pole["height"]) for pid, pole in poles_dict.items()}
    original_directions = {
        pid: {key: float(pole[key]) for key in
              ("dir_x", "dir_y", "dir_z", "dir_h_x", "dir_h_y", "dir_h_z")}
        for pid, pole in poles_dict.items()
    }
    for pid, pole in poles_dict.items():
        old_height = original_heights[pid]
        if old_height >= min_height:
            continue

        # Priority: previous section first, then next section.
        candidates = predecessors.get(pid, []) + successors.get(pid, [])
        selected = next(
            (other for other in candidates
             if other != pid and original_heights.get(other, -float("inf")) >= min_height),
            None,
        )
        if selected is None:
            continue

        pole["height"] = original_heights[selected]
        # Copy both directions from the exact same reference pole.
        # Source snapshots prevent chained corrections from influencing the result.
        pole.update(original_directions[selected])

def compute_P_by_section(wires_data, poles_dict):
    P_by_section = {}

    for span in wires_data:
        parts = span["section_id"].split("_")
        if len(parts) == 2:
            pole_ids = [int(parts[0]), int(parts[1])]
        else:
            pole_ids = [int(parts[0])]

        eff = [effective_wire(span, k) for k in range(3)]

        for pid in pole_ids:
            if pid not in poles_dict:
                continue
            pole = poles_dict[pid]
            B = pole_coord(pole)
            u = pole_dir(pole)

            P = {}
            for k in (1, 2, 3):
                eq, pmin, pmax = eff[k - 1]

                if eq is None and pmin is None and pmax is None:
                    P[k] = B + u
                    continue

                pt = project_curve_to_pole(eq, B, u, pmin, pmax)

                if pt is None:
                    if pmin is not None and pmax is not None:
                        dmin = np.linalg.norm(pmin - B)
                        dmax = np.linalg.norm(pmax - B)
                        pt = pmin if dmin <= dmax else pmax
                    elif pmin is not None:
                        pt = pmin
                    elif pmax is not None:
                        pt = pmax
                    else:
                        pt = B + u

                P[k] = pt

            P_by_section.setdefault(pid, []).append((span["section_id"], P))

    return P_by_section



def _is_fully_synthetic_span(span):
    """Only replace the geometry of spans made entirely from synthetic wires."""
    return bool(span.get("wires")) and all(
        wire.get("source") == "synthetic" for wire in span["wires"]
    )


def inherit_synthetic_connection_heights(wires_data, poles_dict, P_by_section):
    """Copy cable attachment heights from adjacent real spans at shared poles.

    A real span is the priority. When a synthetic span has a real neighbor
    at only one end, those three attachment heights are also transferred to
    its other end (relative to that pole's own base and axial direction).
    The same values may propagate through chains of synthetic spans.
    Original synthetic transverse positions are preserved.
    """
    section_map = {s["section_id"]: s for s in wires_data}
    by_pole = {
        pid: {section_id: P for section_id, P in entries}
        for pid, entries in P_by_section.items()
    }

    # Reference attachments are taken exclusively from non-synthetic spans,
    # preventing a synthetic span from becoming its own source of evidence.
    inherited = {}
    for pid, entries in by_pole.items():
        pole = poles_dict.get(pid)
        if pole is None:
            continue
        B, u = pole_coord(pole), pole_dir(pole)
        for section_id, P in entries.items():
            if not _is_fully_synthetic_span(section_map[section_id]):
                heights = {
                    k: float(np.dot(P[k] - B, u)) for k in (1, 2, 3)
                }
                inherited.setdefault(pid, heights)
                break

    synthetic_sections = []
    for span in wires_data:
        if not _is_fully_synthetic_span(span):
            continue
        parts = span["section_id"].split("_")
        if len(parts) != 2:
            continue
        a, b = int(parts[0]), int(parts[1])
        if a in by_pole and b in by_pole:
            synthetic_sections.append((span["section_id"], a, b))

    # Propagate the known cable heights across consecutive synthetic spans.
    # Never overwrite attachment heights already measured from a real span.
    for _ in range(len(synthetic_sections)):
        changed = False
        for _, a, b in synthetic_sections:
            if a in inherited and b not in inherited:
                inherited[b] = inherited[a].copy()
                changed = True
            elif b in inherited and a not in inherited:
                inherited[a] = inherited[b].copy()
                changed = True
        if not changed:
            break

    corrected = 0
    for section_id, a, b in synthetic_sections:
        for pid in (a, b):
            if pid not in inherited:
                continue  # No real predecessor/successor: keep original synthetic geometry.
            pole = poles_dict[pid]
            B, u = pole_coord(pole), pole_dir(pole)
            P = by_pole[pid][section_id]
            for k in (1, 2, 3):
                previous = P[k]
                old_height = float(np.dot(previous - B, u))
                lateral_offset = previous - (B + old_height * u)
                P[k] = B + inherited[pid][k] * u + lateral_offset
            corrected += 1

    return P_by_section


def merge_P(entries):
    return {k: np.mean([e[1][k] for e in entries], axis=0) for k in (1, 2, 3)}


def _angle_at(A, B, C):
    v1 = B - A
    v2 = C - A
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < EPS or n2 < EPS:
        return 0.0
    cosang = clamp(float(np.dot(v1, v2) / (n1 * n2)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def triangle_angles(P1, P2, P3):
    return {
        1: _angle_at(P1, P2, P3),
        2: _angle_at(P2, P1, P3),
        3: _angle_at(P3, P1, P2),
    }


def line_angle_deg(dir_a, dir_b):
    cosang = clamp(abs(float(np.dot(norm(dir_a), norm(dir_b)))), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def span_chord_direction(pmin, pmax):
    if pmin is None or pmax is None:
        return None
    chord = pmax - pmin
    if np.linalg.norm(chord) < EPS:
        return None
    return norm(chord)


def crosshead_direction(P, B, u_json):
    u = u_json
    d = {k: float(np.linalg.norm(P[k] - B)) for k in (1, 2, 3)}
    top = max(d, key=lambda k: d[k])

    t1 = float(np.dot(P[top] - B, u))
    Htop = B + t1 * u
    offset = P[top] - Htop

    if np.linalg.norm(offset) > EPS:
        return norm(offset)
    return direction_perp_to_pole(np.array([1.0, 0.0, 0.0]), u)


def classify_fil(P, B, u_json, inc):
    v_cross = crosshead_direction(P, B, u_json)
    deviating = 0

    for k in (1, 2, 3):
        v = P[k] - B
        dist = np.linalg.norm(v)
        if dist < EPS:
            continue

        parallel = np.dot(v, u_json) * u_json
        perp = v - parallel
        offset = np.linalg.norm(perp)

        cosang = clamp(float(np.dot(norm(v), u_json)), -1.0, 1.0)
        angle = np.degrees(np.arccos(cosang))

        if offset <= F_OFFSET_TOL and angle <= DIR_MATCH_TOL:
            continue

        forgiven = False
        for entry in inc.get(k, []):
            _, _, pmin, pmax = entry
            chord = span_chord_direction(pmin, pmax)
            if chord is None:
                continue
            if line_angle_deg(chord, v_cross) <= ANGLE_ALIGN_TOL:
                forgiven = True
                break

        if not forgiven:
            deviating += 1

    if deviating == 0:
        return "I"
    if deviating == 3:
        return "F"
    return "P"


def _is_T(P):
    angles = triangle_angles(P[1], P[2], P[3])
    A = max(angles.values())
    return A <= ANGLE_T_MAX


def refine_P_to_M(P, B, u_json):
    angles = triangle_angles(P[1], P[2], P[3])
    major_k = max(angles, key=angles.get)
    t_vals = {k: float(np.dot(P[k] - B, u_json)) for k in (1, 2, 3)}
    max_t_k = max(t_vals, key=t_vals.get)
    if major_k == max_t_k:
        return "P"
    return "M"


def classify_case(P, B, u_json, inc, config="", n_eq=None):
    angles = triangle_angles(P[1], P[2], P[3])
    A = max(angles.values())

    is_ese = str(config).strip().upper() == "E S E"

    if is_ese:
        if n_eq == 1 and A <= ANGLE_O_MAX:
            return "O-L"
        if A > ANGLE_P_MAX:
            return "L"
        return "T"

    if n_eq == 1 and A <= ANGLE_O_MAX:
        return "O"

    if A < ANGLE_T_MAX:
        return "C"

    if A <= ANGLE_P_MAX:
        return refine_P_to_M(P, B, u_json)

    case = classify_fil(P, B, u_json, inc)
    if case == "P":
        return refine_P_to_M(P, B, u_json)
    return case


def _crosshead_dir_T(B, u, P1, P2_self, neighbor_P2s):
    chord = None

    if neighbor_P2s and len(neighbor_P2s) >= 2:
        best_pair = None
        best_d = -1.0
        n = len(neighbor_P2s)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(neighbor_P2s[i] - neighbor_P2s[j]))
                if d > best_d:
                    best_d = d
                    best_pair = (neighbor_P2s[i], neighbor_P2s[j])
        if best_pair is not None and best_d >= EPS:
            chord = best_pair[1] - best_pair[0]

    elif neighbor_P2s and len(neighbor_P2s) == 1 and P2_self is not None:
        chord = P2_self - neighbor_P2s[0]

    if chord is None or np.linalg.norm(chord) < EPS:
        return None

    chord = norm(chord)

    q = np.cross(u, chord)
    nq = float(np.linalg.norm(q))
    if nq < EPS:
        return None
    q = q / nq

    proj_P1 = B + np.dot(P1 - B, u) * u
    q_vec = P1 - proj_P1
    if np.linalg.norm(q_vec) > EPS and np.dot(q, q_vec) < 0:
        q = -q

    return q


def construct_T(P, B, u_json, neighbor_P2s=None):
    P1, P2, P3 = P[1], P[2], P[3]
    u = u_json

    t1 = float(np.linalg.norm(P1 - B))
    t2 = float(np.linalg.norm(P2 - B))
    t3 = float(np.linalg.norm(P3 - B))

    ts = [t1, t2, t3]
    max_idx = int(np.argmax(ts))
    others = [i for i in range(3) if i != max_idx]
    t_avg_others = (ts[others[0]] + ts[others[1]]) / 2.0

    t_vals = list(ts)
    for i in others:
        t_vals[i] = t_avg_others

    t1, t2, t3 = t_vals

    H1 = B + t1 * u
    H2 = B + t2 * u
    H3 = B + t3 * u

    w = float(np.linalg.norm(P1 - P3))

    q = _crosshead_dir_T(B, u, P1, P2, neighbor_P2s)

    if q is None:
        proj_P1 = B + np.dot(P1 - B, u) * u
        q_vec = P1 - proj_P1
        if np.linalg.norm(q_vec) > EPS:
            q = norm(q_vec)
        else:
            q = direction_perp_to_pole(np.array([1.0, 0.0, 0.0]), u)

    Hs = [H1, H2, H3]
    Ps = [P1, P2, P3]

    C = {}
    for i in range(3):
        if i == max_idx:
            C[i + 1] = Hs[i].copy()
        else:
            perp = Ps[i] - (B + np.dot(Ps[i] - B, u) * u)
            s = float(np.sign(np.dot(perp, q)))
            if s == 0.0:
                s = 1.0 if i == 0 else -1.0
            C[i + 1] = Hs[i] + s * (w / 2.0) * q

    H = {1: H1, 2: H2, 3: H3}
    t = {1: t1, 2: t2, 3: t3}
    cl = {i: float(np.linalg.norm(H[i] - C[i])) for i in H}
    return H, C, t, cl


def construct_P(P, B, u_json, height=None):
    d = {k: float(np.linalg.norm(P[k] - B)) for k in (1, 2, 3)}
    order = sorted(d, key=lambda k: -d[k])
    top, mid, bot = order
    Ptop, Pmid, Pbot = P[top], P[mid], P[bot]

    u = u_json
    t1 = d[top]
    Htop = B + t1 * u
    Ctop = Htop.copy()

    M = (Pmid + Pbot) / 2.0
    M_proj = B + np.dot(M - B, u) * u
    v = norm(M - M_proj)

    Pmid_proj = B + np.dot(Pmid - B, u) * u
    d1 = float(np.linalg.norm(Pmid - Pmid_proj))
    t2 = float(np.linalg.norm(Pmid_proj - B))

    Pbot_proj = B + np.dot(Pbot - B, u) * u
    d2 = float(np.linalg.norm(Pbot - Pbot_proj))
    t3 = float(np.linalg.norm(Pbot_proj - B))

    if max(d1, d2) > EPS and min(d1, d2) > EPS:
        ratio = max(d1, d2) / min(d1, d2)
        if ratio > 2.0 + CROSSHEAD_LEN_TOL:
            return construct_I(P, B, u_json, height), "I"
        avg = (d1 + d2) / 2.0
        d1 = d2 = avg

    Hmid = B + t2 * u
    Hbot = B + t3 * u
    Cmid = Hmid + d1 * v
    Cbot = Hbot + d2 * v

    H = {top: Htop, mid: Hmid, bot: Hbot}
    C = {top: Ctop, mid: Cmid, bot: Cbot}
    t = {top: t1, mid: t2, bot: t3}
    cl = {top: 0.0, mid: d1, bot: d2}
    return (H, C, t, cl), "P"


def construct_I(P, B, u_json, height=None):
    u = u_json
    H, C, t, cl = {}, {}, {}, {}

    t_raw = {k: float(np.linalg.norm(P[k] - B)) for k in (1, 2, 3)}

    order = sorted((1, 2, 3), key=lambda k: -t_raw[k])
    top_k, mid_k, bot_k = order

    t_top = t_raw[top_k]
    t_mid = t_raw[mid_k]
    t_bot = t_raw[bot_k]

    g1 = t_top - t_mid
    g2 = t_mid - t_bot

    if g1 <= 0.2 or g2 <= 0.2:
        if height is not None:
            t_top = float(height)
        t_mid = (t_top + t_bot) / 2.0

    t_new = {top_k: t_top, mid_k: t_mid, bot_k: t_bot}

    for k in (1, 2, 3):
        tk = t_new[k]
        Hk = B + tk * u
        H[k] = Hk
        C[k] = Hk.copy()
        t[k] = tk
        cl[k] = 0.0

    return H, C, t, cl


def construct_O(P, B, u_json, height=None):
    u = u_json
    if height is not None:
        h = float(height)
    else:
        h = float(np.linalg.norm(P[1] - B))

    H1 = B + h * u
    H = {1: H1.copy(), 2: H1.copy(), 3: H1.copy()}
    C = {1: H1.copy(), 2: H1.copy(), 3: H1.copy()}
    t = {1: h, 2: h, 3: h}
    cl = {1: 0.0, 2: 0.0, 3: 0.0}
    return H, C, t, cl


def _f_common_setup(P, B, u_json):
    u = np.asarray(u_json, dtype=float)
    un = np.linalg.norm(u)
    if un < EPS:
        return None
    u_hat = u / un

    d = {k: float(np.linalg.norm(P[k] - B)) for k in (1, 2, 3)}
    order = sorted(d, key=lambda k: -d[k])
    top, mid, bot = order
    Ptop, Pmid, Pbot = P[top], P[mid], P[bot]

    t_vals = {
        top: float(np.dot(Ptop - B, u_hat)),
        mid: float(np.dot(Pmid - B, u_hat)),
        bot: float(np.dot(Pbot - B, u_hat)),
    }

    k_max, k_mid, k_min = sorted(t_vals, key=lambda k: -t_vals[k])
    g1 = t_vals[k_max] - t_vals[k_mid]
    g2 = t_vals[k_mid] - t_vals[k_min]
    if g1 < 0.5 or g2 < 0.5:
        t_vals[k_mid] = t_vals[k_max] - 0.8
        t_vals[k_min] = t_vals[k_mid] - 0.8

    t_top = t_vals[top]
    t_mid = t_vals[mid]
    t_bot = t_vals[bot]

    Htop = B + t_top * u_hat
    Hmid = B + t_mid * u_hat
    Hbot = B + t_bot * u_hat

    if np.linalg.norm(Ptop - Htop) > EPS:
        v = norm(Ptop - Htop)
    else:
        v = direction_perp_to_pole(np.array([1.0, 0.0, 0.0]), u_hat)

    def _is_perp(vec, uh, tol=1e-6):
        n = np.linalg.norm(vec)
        if n < EPS:
            return True
        return abs(float(np.dot(vec / n, uh))) < tol

    def _orthogonalize_keep_xy(vec, uh):
        ux, uy, uz = uh
        vx, vy, vz = vec
        if abs(uz) > 1e-6:
            vz_new = -(vx * ux + vy * uy) / uz
            out = np.array([vx, vy, vz_new], dtype=float)
        else:
            out = vec - np.dot(vec, uh) * uh
        n = np.linalg.norm(out)
        if n < EPS:
            out = direction_perp_to_pole(np.array([1.0, 0.0, 0.0]), uh)
            n = np.linalg.norm(out)
        return out / n

    if not _is_perp(v, u_hat):
        v = _orthogonalize_keep_xy(v, u_hat)

    return {
        "u_hat": u_hat,
        "top": top, "mid": mid, "bot": bot,
        "Ptop": Ptop, "Pmid": Pmid, "Pbot": Pbot,
        "t_top": t_top, "t_mid": t_mid, "t_bot": t_bot,
        "Htop": Htop, "Hmid": Hmid, "Hbot": Hbot,
        "v": v,
    }


def construct_F(P, B, u_json, height=None):
    setup = _f_common_setup(P, B, u_json)
    if setup is None:
        return construct_I(P, B, u_json, height), "I"

    top, mid, bot = setup["top"], setup["mid"], setup["bot"]
    Ptop, Pmid, Pbot = setup["Ptop"], setup["Pmid"], setup["Pbot"]
    Htop, Hmid, Hbot = setup["Htop"], setup["Hmid"], setup["Hbot"]
    t_top, t_mid, t_bot = setup["t_top"], setup["t_mid"], setup["t_bot"]
    v = setup["v"]

    raw_top = float(np.linalg.norm(Ptop - Htop))
    raw_mid = float(np.linalg.norm(Pmid - Hmid))
    raw_bot = float(np.linalg.norm(Pbot - Hbot))

    L_common = float(np.mean([raw_top, raw_mid, raw_bot]))

    if L_common < MIN_CROSSHEAD_LEN or L_common > MAX_CROSSHEAD_LEN:
        return construct_I(P, B, u_json, height), "I"

    Ctop = Htop + L_common * v
    Cmid = Hmid + L_common * v
    Cbot = Hbot + L_common * v

    H = {top: Htop, mid: Hmid, bot: Hbot}
    C = {top: Ctop, mid: Cmid, bot: Cbot}
    t = {top: t_top, mid: t_mid, bot: t_bot}
    cl = {top: L_common, mid: L_common, bot: L_common}

    return (H, C, t, cl), "F"


def construct_M(P, B, u_json, height=None):
    setup = _f_common_setup(P, B, u_json)
    if setup is None:
        return construct_I(P, B, u_json, height), "M"

    top, mid, bot = setup["top"], setup["mid"], setup["bot"]
    Ptop, Pmid, Pbot = setup["Ptop"], setup["Pmid"], setup["Pbot"]
    Htop, Hmid, Hbot = setup["Htop"], setup["Hmid"], setup["Hbot"]
    t_top, t_mid, t_bot = setup["t_top"], setup["t_mid"], setup["t_bot"]
    v = setup["v"]

    raw_top = float(np.linalg.norm(Ptop - Htop))
    raw_mid = float(np.linalg.norm(Pmid - Hmid))
    raw_bot = float(np.linalg.norm(Pbot - Hbot))

    L_ext = (raw_top + raw_bot) / 2.0
    L_top = L_ext
    L_bot = L_ext
    L_mid = raw_mid

    if L_ext <= L_mid:
        return construct_P(P, B, u_json, height)

    Ctop = Htop + L_top * v
    Cmid = Hmid + L_mid * v
    Cbot = Hbot + L_bot * v

    H = {top: Htop, mid: Hmid, bot: Hbot}
    C = {top: Ctop, mid: Cmid, bot: Cbot}
    t = {top: t_top, mid: t_mid, bot: t_bot}
    cl = {top: L_top, mid: L_mid, bot: L_bot}

    return (H, C, t, cl), "M"


def construct_C(P, B, u_json):
    u = u_json

    t_raw = {k: float(np.dot(P[k] - B, u)) for k in (1, 2, 3)}

    order_t = sorted((1, 2, 3), key=lambda k: t_raw[k])
    bot_k, mid_k, top_k = order_t

    t_lo = t_raw[bot_k]
    t_hi = t_raw[top_k]
    gap = (t_hi - t_lo) / 2.0

    t_new = {
        bot_k: t_lo,
        mid_k: t_lo + gap,
        top_k: t_hi,
    }

    lengths = []
    for k in (1, 2, 3):
        Hk_raw = B + t_raw[k] * u
        lengths.append(float(np.linalg.norm(P[k] - Hk_raw)))
    L_avg = sum(lengths) / 3.0

    H_top_raw = B + t_raw[top_k] * u
    off_top = P[top_k] - H_top_raw
    if np.linalg.norm(off_top) > EPS:
        v = norm(off_top)
    else:
        v = direction_perp_to_pole(np.array([1.0, 0.0, 0.0]), u)

    signs = {
        top_k:  1.0,
        bot_k:  1.0,
        mid_k: -1.0,
    }

    H, C, t, cl = {}, {}, {}, {}
    for k in (1, 2, 3):
        Hk = B + t_new[k] * u
        Ck = Hk + signs[k] * L_avg * v

        H[k] = Hk
        C[k] = Ck
        t[k] = t_new[k]
        cl[k] = L_avg

    return H, C, t, cl


def construct_L(P, B, u_json, height=None):
    u = u_json

    d = {k: float(np.linalg.norm(P[k] - B)) for k in (1, 2, 3)}
    t_common = sum(d.values()) / 3.0
    H_pt = B + t_common * u

    perps = []
    ts = {}
    for k in (1, 2, 3):
        off = P[k] - H_pt
        perp = off - np.dot(off, u) * u
        n = float(np.linalg.norm(perp))
        ts[k] = n
        if n > EPS:
            perps.append(perp / n)

    if perps:
        v_raw = np.sum(perps, axis=0)
        v_raw = v_raw - np.dot(v_raw, u) * u
        nv = float(np.linalg.norm(v_raw))
        if nv > EPS:
            v = v_raw / nv
        else:
            v = direction_perp_to_pole(np.array([1.0, 0.0, 0.0]), u)
    else:
        v = direction_perp_to_pole(np.array([1.0, 0.0, 0.0]), u)

    H = {1: H_pt, 2: H_pt, 3: H_pt}
    C = {}
    t = {1: t_common, 2: t_common, 3: t_common}
    cl = {}
    for k in (1, 2, 3):
        C[k] = H_pt + ts[k] * v
        cl[k] = ts[k]

    return H, C, t, cl


def construct(case, P, B, u_json, neighbor_P2s=None, height=None):
    if case == "T":
        return construct_T(P, B, u_json, neighbor_P2s), "T"
    if case == "P":
        return construct_P(P, B, u_json, height)
    if case == "M":
        return construct_M(P, B, u_json, height)
    if case == "C":
        return construct_C(P, B, u_json), "C"
    if case == "I":
        return construct_I(P, B, u_json, height), "I"
    if case == "F":
        return construct_F(P, B, u_json, height)
    if case == "L":
        return construct_L(P, B, u_json, height), "L"
    if case == "O":
        return construct_O(P, B, u_json, height), "O"
    if case == "O-L":
        return construct_L(P, B, u_json, height), "O-L"
    raise ValueError(f"Unknown case: {case}")


def fix_t_outlier(H, C, t, u, threshold=T_GAP_TOL):
    keys = sorted([1, 2, 3], key=lambda k: -t[k])
    k_max, k_mid, k_min = keys

    t_max = t[k_max]
    t_mid = t[k_mid]
    t_min = t[k_min]

    if (t_max - t_mid) > threshold or (t_mid - t_min) > threshold:
        t_new_max = t_mid + (t_mid - t_min)
        delta_t = t_new_max - t_max

        H[k_max] = H[k_max] + delta_t * u
        C[k_max] = C[k_max] + delta_t * u
        t[k_max] = t_new_max

    return H, C, t


def limit_t_values(H, C, t, u, max_t=MAX_T_VALUE):
    excess = max(t.values()) - max_t
    if excess > 0:
        shift = excess * u
        for k in (1, 2, 3):
            H[k] = H[k] - shift
            C[k] = C[k] - shift
            t[k] = t[k] - excess
    return H, C, t


def _priority_order(cases):
    if "O" in cases:
        return "O"
    if "O-L" in cases:
        return "O-L"
    if "F" in cases:
        return "F"
    if "P" in cases:
        return "P"
    if "C" in cases:
        return "C"
    if "M" in cases:
        return "M"
    return None


def decide_outputs(classified, config=""):
    if str(config).strip().upper() == "E S E":
        cases = {e[2] for e in classified}
        if len(cases) == 1:
            c = list(cases)[0]
            return [(c, merge_P(classified), list(classified))]
        if "T" in cases:
            return [("T", merge_P(classified), list(classified))]
        if "L" in cases:
            return [("L", merge_P(classified), list(classified))]
        if "O-L" in cases:
            entries = [e for e in classified if e[2] == "O-L"]
            return [("O-L", merge_P(entries), entries)]
        return [("L", merge_P(classified), list(classified))]

    unique = {c for _, _, c in classified}

    if len(unique) == 1:
        c = list(unique)[0]
        return [(c, merge_P(classified), list(classified))]

    non_i = [e for e in classified if e[2] != "I"]
    has_i = "I" in unique

    if not non_i:
        return [("I", merge_P(classified), list(classified))]

    non_i_cases = {e[2] for e in non_i}
    winner = _priority_order(non_i_cases)
    winner_entries = [e for e in non_i if e[2] == winner]

    if has_i:
        i_entries = [e for e in classified if e[2] == "I"]
        return [
            (winner, merge_P(winner_entries), winner_entries),
            ("I", merge_P(i_entries), i_entries),
        ]

    return [(winner, merge_P(winner_entries), winner_entries)]


def compute_tap_intersections(wires_data, per_pole):
    span_by_id = {span["section_id"]: span for span in wires_data}
    results = []

    for span in wires_data:
        parts = span["section_id"].split("_")
        if len(parts) < 3:
            continue

        tap_pole_id = int(parts[0])
        main_span_id = "_".join(parts[1:])

        main_span = span_by_id.get(main_span_id)
        if main_span is None:
            continue

        if tap_pole_id not in per_pole:
            continue

        for info in per_pole[tap_pole_id]:
            C = info["C"]

            W = {}
            valid = True
            for k in (1, 2, 3):
                main_eq, _, _ = effective_wire(main_span, k - 1)
                if main_eq is None:
                    valid = False
                    W[k] = None
                    continue

                W[k] = nearest_curve_point(main_eq, C[k])

            if not valid:
                continue

            def r(p):
                return [round(float(x), OUT_DECIMALS) for x in p]

            results.append({
                "id": span["section_id"],
                "tap_pole": tap_pole_id,
                "main_span": main_span_id,
                "Case": info["case"],
                "Info": {
                    "Point_W1": r(W[1]),
                    "Point_W2": r(W[2]),
                    "Point_W3": r(W[3]),
                }
            })

    results.sort(key=lambda x: x["tap_pole"])
    return results


def main():
    cfg = load_config()

    poles_path = configured_path(cfg, "output_dir", "json_extractor")
    if not os.path.exists(poles_path):
        poles_path = configured_path(cfg, "others_dir", "json_extractor")

    wires_path = configured_path(cfg, "output_dir", "json_wires")

    with open(poles_path) as f:
        poles_json = json.load(f)
    with open(wires_path) as f:
        wires_data = json.load(f)

    poles_dict = {
        int(p["Pole_ID"]): normalize_pole(p)
        for p in poles_json
    }

    # Fix short pole heights before classifying and constructing points.
    inherit_short_pole_heights(wires_data, poles_dict)

    span_by_id = {span["section_id"]: span for span in wires_data}

    incidences = build_pole_incidences(wires_data, poles_dict)

    P_by_section = compute_P_by_section(wires_data, poles_dict)
    P_by_section = inherit_synthetic_connection_heights(
        wires_data, poles_dict, P_by_section
    )

    P_by_pole_for_neighbor = {
        pid: entries[0][1]
        for pid, entries in P_by_section.items() if entries
    }

    neighbors = build_pole_neighbors(wires_data)

    per_pole = {}

    for pid, pole in poles_dict.items():
        entries = P_by_section.get(pid, [])
        if not entries:
            continue

        B = pole_coord(pole)
        u_json = pole_dir(pole)
        height = float(pole["height"])
        inc = incidences.get(pid, {1: [], 2: [], 3: []})
        config = pole.get("config", "")
        is_ese = str(config).strip().upper() == "E S E"

        classified = []
        for span_id, P in entries:
            span = span_by_id.get(span_id)
            n_eq = count_nonzero_equations(span)
            case = classify_case(P, B, u_json, inc, config, n_eq)
            classified.append((span_id, P, case))

        neighbor_P2s = [
            P_by_pole_for_neighbor[n][2]
            for n in neighbors.get(pid, [])
            if n in P_by_pole_for_neighbor
            and P_by_pole_for_neighbor[n][2] is not None
        ]

        current = list(classified)
        infos = []
        for _ in range(20):
            outputs = decide_outputs(current, config)

            infos = []
            became_i_spans = set()

            for case, merged_P, sources in outputs:
                (H, C, t, cl), final_case = construct(
                    case, merged_P, B, u_json, neighbor_P2s, height
                )

                if final_case not in ("I", "O", "O-L") and not (is_ese and final_case in ("T", "L")):
                    non_zero_cl = [v for v in cl.values() if v > EPS]
                    if non_zero_cl and (max(non_zero_cl) > MAX_CROSSHEAD_LEN
                                        or min(non_zero_cl) < MIN_CROSSHEAD_LEN):
                        H, C, t, cl = construct_I(merged_P, B, u_json, height)
                        final_case = "I"

                if final_case not in ("O", "O-L"):
                    H, C, t = fix_t_outlier(H, C, t, u_json, T_GAP_TOL)

                H, C, t = limit_t_values(H, C, t, u_json)

                infos.append({
                    "H": H, "C": C, "t": t, "cl": cl, "case": final_case,
                    "B": B, "u": u_json,
                })

                if final_case == "I" and case != "I":
                    for s, _, _ in sources:
                        became_i_spans.add(s)

            if not became_i_spans:
                break

            current = [
                (s, P, "I" if s in became_i_spans else c)
                for s, P, c in current
            ]

        if len(infos) > 1:
            max_h_dists = []
            for info in infos:
                H = info["H"]
                max_d = max(float(np.linalg.norm(H[k] - B)) for k in (1, 2, 3))
                max_h_dists.append(max_d)

            min_d = min(max_h_dists)
            max_d = max(max_h_dists)

            if (max_d - min_d) > SUBCASE_H_DIFF_TOL:
                infos = [
                    info for info, d in zip(infos, max_h_dists)
                    if abs(d - min_d) < EPS
                ]

        per_pole[pid] = infos

    def r(p):
        return [round(float(x), OUT_DECIMALS) for x in p]

    final = []
    for pid, infos in per_pole.items():
        for sub_id, info in enumerate(infos):
            H, C, t, cl = info["H"], info["C"], info["t"], info["cl"]
            final.append({
                "Pole_ID": pid,
                "Sub_ID": sub_id,
                "Case": info["case"],
                "Points": {
                    "Point_H1": r(H[1]),
                    "Point_H2": r(H[2]),
                    "Point_H3": r(H[3]),
                    "Point_C1": r(C[1]),
                    "Point_C2": r(C[2]),
                    "Point_C3": r(C[3]),
                },
                "t-values": {
                    "t1": round(float(t[1]), OUT_DECIMALS),
                    "t2": round(float(t[2]), OUT_DECIMALS),
                    "t3": round(float(t[3]), OUT_DECIMALS),
                },
                "Crosshead_Lengths": {
                    "Crosshead_1": round(float(cl[1]), OUT_DECIMALS),
                    "Crosshead_2": round(float(cl[2]), OUT_DECIMALS),
                    "Crosshead_3": round(float(cl[3]), OUT_DECIMALS),
                }
            })

    final.sort(key=lambda x: (x["Pole_ID"], x["Sub_ID"]))

    out_path = output_path(cfg, "json_points")
    with open(out_path, "w") as f:
        json.dump(final, f, indent=4)

    print(f"[INFO] [POINTS] Points JSON exported to: {out_path}")



if __name__ == "__main__":
    main()