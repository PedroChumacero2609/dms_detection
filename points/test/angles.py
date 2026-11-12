
import csv
import json
import os
import numpy as np

from pathlib import Path

from .._paths import load_config, configured_path
EPS = 1e-6
OUT_DECIMALS = 6
CSV_FILENAME = "pole_angles_by_span.csv"

ANGLE_T_MAX = 110.0
ANGLE_P_MAX = 165.0
ANGLE_O_MAX = 10.0




def norm(v):
    n = np.linalg.norm(v)
    return v if n < EPS else v / n


# ── Normalización de postes ──────────────────────────────────────────────────

def normalize_pole(p):
    info = p["Info"]

    return {
        "id": int(p["Pole_ID"]),
        "base": np.asarray(info["Base"], dtype=float),
        "direction": norm(
            np.asarray(info["unit_vector"], dtype=float)
        ),
        "config": str(
            info.get(
                "configuration",
                p.get("configuration", "")
            ) or ""
        ).strip(),
    }


# ── Lectura de puntos y ecuaciones ──────────────────────────────────────────

def parse_point(pt):
    if pt is None:
        return None

    return np.array(
        [float(pt[ax]) for ax in ("x", "y", "z")],
        dtype=float
    )


def parse_equation(eq):
    if eq is None:
        return None

    return {
        axis: {
            c: float(eq[axis][c])
            for c in ("a", "b", "c")
        }
        for axis in ("x", "y", "z")
    }


def eq_is_zero(eq):
    return eq is None or all(
        abs(eq[axis][c]) <= EPS
        for axis in ("x", "y", "z")
        for c in ("a", "b", "c")
    )


def effective_wire(span, index):
    """
    Conserva el fallback del segundo script
    para ecuaciones nulas.
    """

    w = span["wires"][index]

    eq = parse_equation(w.get("equation"))

    if not eq_is_zero(eq):
        return (
            eq,
            parse_point(w.get("point_min")),
            parse_point(w.get("point_max"))
        )

    for other in span["wires"]:
        other_eq = parse_equation(
            other.get("equation")
        )

        if not eq_is_zero(other_eq):
            return (
                other_eq,
                parse_point(other.get("point_min")),
                parse_point(other.get("point_max"))
            )

    return None, None, None


def count_nonzero_equations(span):
    return sum(
        not eq_is_zero(
            parse_equation(
                span["wires"][k].get("equation")
            )
        )
        for k in range(3)
    )


# ── Geometría de curvas ──────────────────────────────────────────────────────

def curve_coefficients(eq):
    return tuple(
        np.array(
            [
                eq[axis][coef]
                for axis in ("x", "y", "z")
            ],
            dtype=float
        )
        for coef in ("a", "b", "c")
    )


def stationary_parameters(a, b, c):
    """
    Raíces reales en [0, 1] para minimizar
    ||a*t² + b*t + c||².
    """

    coeffs = np.array([
        np.dot(b, c),
        np.dot(b, b) + 2.0 * np.dot(a, c),
        3.0 * np.dot(a, b),
        2.0 * np.dot(a, a),
    ], dtype=float)

    scale = float(np.max(np.abs(coeffs)))

    if scale < 1e-20:
        return []

    coeffs /= scale

    while (
        len(coeffs) > 1
        and abs(coeffs[-1]) < 1e-14
    ):
        coeffs = coeffs[:-1]

    if len(coeffs) <= 1:
        return []

    roots = np.roots(coeffs[::-1])

    return [
        float(np.clip(r.real, 0.0, 1.0))
        for r in roots
        if (
            abs(r.imag) < 1e-7
            and -1e-9 <= r.real <= 1.0 + 1e-9
        )
    ]


def project_curve_to_pole(eq, base, direction):
    if eq is None:
        return None

    a, b, c = curve_coefficients(eq)

    u = norm(direction)

    projector = np.eye(3) - np.outer(u, u)

    candidates = (
        [0.0, 1.0]
        + stationary_parameters(
            projector @ a,
            projector @ b,
            projector @ (c - base)
        )
    )

    def point(t):
        return a * t * t + b * t + c

    def squared_distance(t):
        off = projector @ (point(t) - base)
        return float(np.dot(off, off))

    best_t = min(
        candidates,
        key=squared_distance
    )

    return point(best_t)


# ── Extracción de P1, P2, P3 ────────────────────────────────────────────────

def extract_span_P_points(span, poles_dict):
    """
    Extrae P1, P2 y P3 para cada poste
    del tramo utilizando proyección.
    """

    parts = span["section_id"].split("_")

    if len(parts) == 2:
        pole_ids = [
            int(parts[0]),
            int(parts[1])
        ]
    else:
        pole_ids = [int(parts[0])]

    result = {}

    for pid in pole_ids:

        pole = poles_dict.get(pid)

        if pole is None:
            print(
                f"[WARN] Poste {pid} ausente "
                "en JSON extractor; se omite"
            )
            continue

        B = pole["base"]
        u = pole["direction"]

        P = {}

        for k in (1, 2, 3):

            eq, pmin, pmax = effective_wire(
                span,
                k - 1
            )

            pt = project_curve_to_pole(
                eq,
                B,
                u
            )

            if pt is None:

                if (
                    pmin is not None
                    and pmax is not None
                ):
                    dmin = np.linalg.norm(pmin - B)
                    dmax = np.linalg.norm(pmax - B)

                    pt = (
                        pmin
                        if dmin <= dmax
                        else pmax
                    )

                elif pmin is not None:
                    pt = pmin

                elif pmax is not None:
                    pt = pmax

                else:
                    pt = B + u

            P[k] = pt

        result[pid] = P

    return result


# ── Cálculo de ángulos ───────────────────────────────────────────────────────

def angle_at(A, B, C):

    v1 = B - A
    v2 = C - A

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < EPS or n2 < EPS:
        return 0.0

    cosine = np.clip(
        np.dot(v1, v2) / (n1 * n2),
        -1.0,
        1.0
    )

    return float(
        np.degrees(np.arccos(cosine))
    )


def triangle_angles(P):

    return {
        1: angle_at(P[1], P[2], P[3]),
        2: angle_at(P[2], P[1], P[3]),
        3: angle_at(P[3], P[1], P[2]),
    }


# ── Primer filtro de clasificación ──────────────────────────────────────────

def classify_first_angle_filter(a_max, config, n_eq):
    """
    Solo las condiciones angulares iniciales
    de classify_case().
    """

    is_ese = (
        str(config).strip().upper() == "E S E"
    )

    if is_ese:

        if (
            n_eq == 1
            and a_max <= ANGLE_O_MAX
        ):
            return "O-L"

        if a_max > ANGLE_P_MAX:
            return "L"

        return "T"

    if (
        n_eq == 1
        and a_max <= ANGLE_O_MAX
    ):
        return "O"

    if a_max < ANGLE_T_MAX:
        return "C"

    if a_max <= ANGLE_P_MAX:
        return "P"

    # Clasificación provisional
    return "I"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():

    cfg = load_config()

    # Rutas de archivos
    poles_path = configured_path(cfg, "output_dir", "json_extractor")

    if not os.path.exists(poles_path):
        poles_path = configured_path(cfg, "others_dir", "json_extractor")

    wires_path = configured_path(cfg, "output_dir", "json_wires")

    # Leer JSON de postes
    with open(
        poles_path,
        encoding="utf-8"
    ) as f:
        poles_json = json.load(f)

    # Leer JSON de cables
    with open(
        wires_path,
        encoding="utf-8"
    ) as f:
        wires_data = json.load(f)

    # Diccionario de postes
    poles_dict = {
        int(p["Pole_ID"]): normalize_pole(p)
        for p in poles_json
    }

    # ── 1. Recolectar filas ─────────────────────────────────────────────

    rows = []

    for span in wires_data:

        span_id = span["section_id"]

        P_by_pole = extract_span_P_points(
            span,
            poles_dict
        )

        n_eq = count_nonzero_equations(span)

        for pid, P in P_by_pole.items():

            angles = triangle_angles(P)

            a_max, a_mid, a_min = sorted(
                angles.values(),
                reverse=True
            )

            case = classify_first_angle_filter(
                a_max,
                poles_dict[pid]["config"],
                n_eq
            )

            rows.append({
                "Span_ID": span_id,
                "Pole_ID": pid,
                "angle_max": a_max,
                "angle_mid": a_mid,
                "angle_min": a_min,
                "case": case,
            })

    # ── 2. Ordenar filas por Pole_ID ──────────────────────────────────

    # Primero Pole_ID ascendente.
    # Luego Span_ID ascendente dentro de cada poste.

    rows.sort(
        key=lambda row: (
            row["Pole_ID"],
            row["Span_ID"]
        )
    )

    # ── 3. Exportar CSV ───────────────────────────────────────────────

    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / CSV_FILENAME

    fields = [
        "Span_ID",
        "Pole_ID",
        "angle_max",
        "angle_mid",
        "angle_min",
        "case",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for row in rows:

            writer.writerow({
                k: (
                    round(v, OUT_DECIMALS)
                    if k.startswith("angle_")
                    else v
                )
                for k, v in row.items()
            })

    print(
        f"[INFO] [POINTS] CSV exported to: {csv_path}"
    )

    print(
        f"[INFO] [POINTS] Rows exported: {len(rows)}"
    )


if __name__ == "__main__":
    main()
