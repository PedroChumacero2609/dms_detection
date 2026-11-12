"""Consolidate Extractor RAW metadata with Cable MT results.

Run: python -m connector.base_connector [--config config.json]
"""
import argparse

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _stretch_coordinates(poles, scale_x=8.0, scale_y=2.0):
    """
    Apply uniform axis scaling for visualization purposes only.

    Preserve pole connectivity, relative alignment, and ordering.
    Do not modify the original coordinates in poles_info.json.

    Args:
        poles: List of pole information dictionaries.
        scale_x: Horizontal visualization scaling factor.
        scale_y: Vertical visualization scaling factor.

    Returns:
        Dictionary mapping Pole_ID to scaled 2D coordinates.
    """

    coords = np.array([
        p["Info"]["Base"][:2]
        for p in poles
    ], dtype=float)

    center = np.mean(coords, axis=0)
    cx, cy = center

    stretched = {}

    for p in poles:
        pid = p["Pole_ID"]

        x, y = np.asarray(
            p["Info"]["Base"][:2],
            dtype=float
        )

        x_plot = cx + (x - cx) * scale_x
        y_plot = cy + (y - cy) * scale_y

        stretched[pid] = np.array(
            [x_plot, y_plot],
            dtype=float
        )

    return stretched



def _plot(
    poles,
    connections,
    out_png,
    scale_x=8.0,
    scale_y=2.0
):
    """
    Generate a scaled pole connection diagram.

    Calculate distances from original pole coordinates.
    Place distance labels near their corresponding connections.
    Avoid overlapping pole markers, pole IDs, and other labels.
    """

    fig, ax = plt.subplots(figsize=(16, 20))

    by_id = {
        p["Pole_ID"]: p
        for p in poles
    }

    plot_xy = _stretch_coordinates(
        poles,
        scale_x=scale_x,
        scale_y=scale_y
    )

    connection_labels = []
    connection_segments = []

    # -------------------------------
    # 1) Draw connections
    # -------------------------------
    for i, c in enumerate(connections):

        pid_a = c["from_id"]
        pid_b = c["to_id"]

        a_real = np.asarray(
            by_id[pid_a]["Info"]["Base"],
            dtype=float
        )

        b_real = np.asarray(
            by_id[pid_b]["Info"]["Base"],
            dtype=float
        )

        # Calculate the actual distance
        distance = float(np.linalg.norm(b_real - a_real))

        c["distance"] = round(distance, 2)

        # Retrieve scaled plotting coordinates
        x1, y1 = plot_xy[pid_a]
        x2, y2 = plot_xy[pid_b]

        ax.plot(
            [x1, x2],
            [y1, y2],
            color="steelblue",
            linewidth=2.2,
            solid_capstyle="round",
            zorder=1
        )

        connection_segments.append(
            ((x1, y1), (x2, y2))
        )

        connection_labels.append({
            "index": i,
            "p1": (x1, y1),
            "p2": (x2, y2),
            "distance": distance
        })

    # -------------------------------
    # 2) Draw poles
    # -------------------------------
    xs = [
        plot_xy[p["Pole_ID"]][0]
        for p in poles
    ]

    ys = [
        plot_xy[p["Pole_ID"]][1]
        for p in poles
    ]

    marker_size = 180

    ax.scatter(
        xs,
        ys,
        s=marker_size,
        facecolor="skyblue",
        edgecolor="black",
        linewidth=0.8,
        zorder=3
    )

    pole_texts = []

    for p in poles:
        pid = p["Pole_ID"]
        x, y = plot_xy[pid]

        txt = ax.text(
            x,
            y,
            str(pid),
            fontsize=8,
            ha="center",
            va="center",
            weight="bold",
            zorder=4
        )

        pole_texts.append(txt)

    # -------------------------------
    # 3) Configure plot geometry
    # -------------------------------
    ax.set_aspect(
        "equal",
        adjustable="datalim"
    )

    ax.margins(x=0.15, y=0.05)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25,
        zorder=0
    )

    ax.set_title(
        "Connections",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # -------------------------------
    # 4) Build pole exclusion areas
    # -------------------------------
    from matplotlib.transforms import Bbox

    marker_radius = (
        np.sqrt(marker_size / np.pi)
        * fig.dpi / 72.0
    )

    pole_boxes = []

    for p, txt in zip(poles, pole_texts):

        x, y = plot_xy[p["Pole_ID"]]

        px, py = ax.transData.transform((x, y))

        marker_box = Bbox.from_extents(
            px - marker_radius,
            py - marker_radius,
            px + marker_radius,
            py + marker_radius
        )

        pole_boxes.append(
            marker_box.padded(4)
        )

        pole_boxes.append(
            txt.get_window_extent(
                renderer=renderer
            ).padded(3)
        )

    # -------------------------------
    # 5) Place distance labels
    # -------------------------------
    placed_boxes = []

    offset_options = [
        12, 16, 20, 26, 34, 44
    ]

    position_options = [
        0.50, 0.40, 0.60, 0.30, 0.70
    ]

    def overlap_area(box_a, box_b):
        """
        Calculate the intersection area of two bounding boxes.
        """

        width = max(
            0.0,
            min(box_a.x1, box_b.x1)
            - max(box_a.x0, box_b.x0)
        )

        height = max(
            0.0,
            min(box_a.y1, box_b.y1)
            - max(box_a.y0, box_b.y0)
        )

        return width * height

    for item in connection_labels:

        p1 = np.asarray(item["p1"], dtype=float)
        p2 = np.asarray(item["p2"], dtype=float)

        # Calculate direction in screen coordinates
        screen_p1 = ax.transData.transform(p1)
        screen_p2 = ax.transData.transform(p2)

        direction = screen_p2 - screen_p1

        length = np.linalg.norm(direction)

        if length > 0:
            normal = np.array([
                -direction[1],
                direction[0]
            ]) / length
        else:
            normal = np.array([1.0, 0.0])

        # Prefer opposite sides for consecutive connections
        preferred_side = (
            1 if item["index"] % 2 == 0 else -1
        )

        side_options = [
            preferred_side,
            -preferred_side
        ]

        best_candidate = None
        best_score = float("inf")

        placed = False

        for offset in offset_options:

            if placed:
                break

            for fraction in position_options:

                if placed:
                    break

                anchor = (
                    p1 + fraction * (p2 - p1)
                )

                for side in side_options:

                    offset_xy = (
                        normal * side * offset
                    )

                    annotation = ax.annotate(
                        f'{item["distance"]:.1f} m',
                        xy=anchor,
                        xytext=tuple(offset_xy),
                        textcoords="offset points",
                        fontsize=8,
                        color="black",
                        ha="center",
                        va="center",
                        annotation_clip=False,
                        bbox=dict(
                            boxstyle="round,pad=0.18",
                            fc="white",
                            ec="gray",
                            lw=0.5,
                            alpha=0.95
                        ),
                        zorder=5
                    )

                    # Obtain the complete label bounding box
                    bbox = annotation.get_window_extent(
                        renderer=renderer
                    ).padded(3)

                    obstacles = (
                        pole_boxes + placed_boxes
                    )

                    total_overlap = sum(
                        overlap_area(bbox, other)
                        for other in obstacles
                    )

                    # Prefer labels near the connection midpoint
                    distance_penalty = (
                        offset * 0.05
                        + abs(fraction - 0.5) * 10
                    )

                    score = (
                        total_overlap * 1000
                        + distance_penalty
                    )

                    if score < best_score:
                        best_score = score
                        best_candidate = (
                            fraction,
                            offset,
                            side
                        )

                    if total_overlap == 0:
                        placed_boxes.append(bbox)
                        placed = True
                        break

                    annotation.remove()

        # -------------------------------
        # 6) Apply the best available position
        # -------------------------------
        if not placed and best_candidate is not None:

            fraction, offset, side = best_candidate

            anchor = (
                p1 + fraction * (p2 - p1)
            )

            offset_xy = (
                normal * side * offset
            )

            annotation = ax.annotate(
                f'{item["distance"]:.1f} m',
                xy=anchor,
                xytext=tuple(offset_xy),
                textcoords="offset points",
                fontsize=8,
                color="black",
                ha="center",
                va="center",
                annotation_clip=False,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    fc="white",
                    ec="gray",
                    lw=0.5,
                    alpha=0.95
                ),
                zorder=5
            )

            placed_boxes.append(
                annotation.get_window_extent(
                    renderer=renderer
                ).padded(3)
            )

    # -------------------------------
    # 7) Export image
    # -------------------------------
    fig.savefig(
        out_png,
        dpi=350,
        bbox_inches="tight",
        pad_inches=0.3
    )

    plt.close(fig)


def _add_configuration(poles, html_report):
    """
    Insert the configuration field into each pole's Info dictionary,
    immediately after unit_vector_horizontal when available.
    """

    html_by_id = {}

    for entry in html_report.get("poles", []) or []:
        try:
            html_by_id[int(entry["Pole_ID"])] = entry
        except (KeyError, TypeError, ValueError):
            continue

    for p in poles:
        pid = p["Pole_ID"]
        entry = html_by_id.get(pid, {})

        config = entry.get(
            "configuration",
            "S M I"
        )

        info = p.get("Info", {})

        if "configuration" in info:
            info["configuration"] = config
            continue

        new_info = {}
        inserted = False

        for k, v in info.items():
            new_info[k] = v

            if k == "unit_vector_horizontal":
                new_info["configuration"] = config
                inserted = True

        if not inserted:
            new_info["configuration"] = config

        p["Info"] = new_info


def run_merge(input_json, output_dir, config):
    """
    Merge pole configuration data and update connection distances.

    Inputs:
        - poles_info_raw.json
        - poles_config.json
        - poles_connections.json

    Outputs:
        - poles_info.json with merged configurations (raw unchanged)
        - poles_connections.json with recalculated distances
        - connections.png with scaled visualization

    Configuration options:
        - scale_x: Horizontal visualization scaling factor.
        - scale_y: Vertical visualization scaling factor.
    """

    # -------------------------------
    # 1) Load pole information
    # -------------------------------
    poles_path = Path(input_json)

    if not poles_path.exists():
        raise FileNotFoundError(f"Raw poles file not found: {poles_path}")

    with open(poles_path, "r", encoding="utf-8") as f:
        poles = json.load(f)

    # -------------------------------
    # 2) Load the optional HTML report
    # -------------------------------
    root = Path(config.get("_project_root", "."))
    html_path = root / config["others_dir"] / config["json_html"]

    html_report = {}

    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            html_report = json.load(f)
    else:
        print(
            f"[WARNING] [CONNECTOR] "
            f"Optional HTML report not found: {html_path}"
        )

    # -------------------------------
    # 3) Load connection information
    # -------------------------------
    conn_path = Path(output_dir) / config["json_connector"]

    if not conn_path.exists():
        raise FileNotFoundError(f"Cable MT connections not found: {conn_path}")

    with open(conn_path, "r", encoding="utf-8") as f:
        connections = json.load(f)

    # -------------------------------
    # 4) Update pole configurations
    # -------------------------------
    if html_report:
        _add_configuration(poles, html_report)

    # Never overwrite raw metadata: it is the immutable Extractor output.
    final_path = Path(output_dir) / config["json_extractor"]
    if final_path.resolve() == poles_path.resolve():
        raise ValueError("json_extractor and json_extractor_raw must be different files")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", encoding="utf-8") as f:
        json.dump(poles, f, indent=4, ensure_ascii=False)
    print(f"[INFO] [CONNECTOR] Consolidated pole metadata: {final_path}")

    # -------------------------------
    # 5) Load visualization scaling factors
    # -------------------------------
    scale_x = 8.0
    scale_y = 2.0

    if scale_x <= 0 or scale_y <= 0:
        raise ValueError(
            "Visualization scaling factors must be greater than zero."
        )

    # -------------------------------
    # 6) Generate the connection image
    # -------------------------------
    img_path = Path(output_dir) / "connections.png"

    img_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    _plot(
        poles,
        connections,
        img_path,
        scale_x=scale_x,
        scale_y=scale_y
    )

    # -------------------------------
    # 7) Save recalculated connection distances
    # -------------------------------
    # Connection geometry and distances are preserved in their existing report.
    # Distances are recalculated by _plot; save updates to the same report.
    with conn_path.open("w", encoding="utf-8") as f:
        json.dump(connections, f, indent=4, ensure_ascii=False)

    print(
        f"[INFO] [CONNECTOR] "
        f"Connections image exported to: {img_path}"
    )

    print(
        f"[INFO] [CONNECTOR] "
        f"Connections file: {conn_path} "
        f"({len(connections)} spans)"
    )

    return connections


def main():
    parser = argparse.ArgumentParser(description="Merge raw poles and Cable MT metadata")
    parser.add_argument("--config", default=None, help="Path to config.json")
    args = parser.parse_args()
    config_path = Path(args.config).resolve() if args.config else Path(__file__).resolve().parent.parent / "config.json"
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    root = config_path.parent
    config["_project_root"] = str(root)
    output_dir = root / config["output_dir"]
    input_json = output_dir / config["json_extractor_raw"]
    output_dir.mkdir(parents=True, exist_ok=True)
    run_merge(input_json=input_json, output_dir=output_dir, config=config)

if __name__ == "__main__":
    main()
