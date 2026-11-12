from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import LineString, Point

ROUTE_TOLERANCE = 100.0


def get_poles_in_route(
    gps_txt_path: Path,
    kml_path: Path,
    tolerance: float = ROUTE_TOLERANCE,
) -> dict[int, bool]:
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    coords: list[tuple[float, float]] = []
    for elem in root.findall(".//kml:LineString/kml:coordinates", ns):
        text = (elem.text or "").strip()
        if not text:
            continue
        for coord in text.split():
            lon, lat, *_ = map(float, coord.split(","))
            coords.append((lon, lat))
        break

    if len(coords) < 2:
        raise ValueError(f"KML file {kml_path} does not contain a valid LineString.")

    poles: dict[int, tuple[float, float]] = {}
    with open(gps_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, coords_txt = line.split(":")
            lat, lon = map(float, coords_txt.split(","))
            pid = int(name.strip().split()[-1])
            poles[pid] = (lat, lon)

    lon0, lat0 = coords[0]
    zone = int((lon0 + 180) / 6) + 1
    south = "+south " if lat0 < 0 else ""
    proj_str = (
        f"+proj=utm +zone={zone} {south}"
        "+datum=WGS84 +units=m +no_defs"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326", proj_str, always_xy=True
    )

    route_proj = [transformer.transform(lon, lat) for lon, lat in coords]
    line = LineString(route_proj)

    in_route: dict[int, bool] = {}
    for pid, (lat, lon) in poles.items():
        x, y = transformer.transform(lon, lat)
        distance = line.distance(Point(x, y))
        in_route[pid] = distance <= tolerance
    return in_route