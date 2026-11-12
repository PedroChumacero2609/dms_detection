from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*pyproj unable to set PROJ database path.*",
    category=UserWarning,
)

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from pyproj import Transformer


@dataclass(frozen=True)
class MapSettings:
    FONDO: str = "calles"
    EXIGIR_FONDO: bool = False
    DPI: int = 260
    ANCHO_PULGADAS: float = 14
    ALTO_PULGADAS: float = 10
    TAMANO_PIN: float = 0.62
    MOSTRAR_LEYENDA: bool = True
    ZOOM_EXTRA: int = 2


class KMLReader:
    @staticmethod
    def _coordenadas(texto: str) -> list[tuple[float, float]]:
        puntos = []
        for token in texto.split():
            try:
                lon, lat = map(float, token.split(',')[:2])
                if math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -85 <= lat <= 85:
                    puntos.append((lon, lat))
            except (ValueError, TypeError):
                continue
        return puntos

    def leer(self, archivo: Path) -> list[list[tuple[float, float]]]:
        raiz = ET.parse(archivo).getroot()
        rutas = []
        for nodo in raiz.iter():
            tipo = nodo.tag.split('}')[-1]
            if tipo == 'LineString':
                for hijo in nodo.iter():
                    if hijo.tag.split('}')[-1] == 'coordinates' and hijo.text:
                        puntos = self._coordenadas(hijo.text)
                        if len(puntos) >= 2:
                            rutas.append(puntos)
            elif tipo == 'Track':
                puntos = []
                for hijo in nodo:
                    if hijo.tag.split('}')[-1] == 'coord' and hijo.text:
                        try:
                            lon, lat = map(float, hijo.text.split()[:2])
                            if math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -85 <= lat <= 85:
                                puntos.append((lon, lat))
                        except ValueError:
                            pass
                if len(puntos) >= 2:
                    rutas.append(puntos)
        if rutas:
            return rutas
        puntos = []
        for nodo in raiz.iter():
            if nodo.tag.split('}')[-1] == 'Point':
                for hijo in nodo.iter():
                    if hijo.tag.split('}')[-1] == 'coordinates' and hijo.text:
                        puntos.extend(self._coordenadas(hijo.text))
        return [puntos] if len(puntos) >= 2 else []


def distancia_ruta_km(rutas: list[list[tuple[float, float]]]) -> float:
    """Suma las distancias geodésicas de cada tramo del KML, en kilómetros."""
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    metros = 0.0
    for ruta in rutas:
        for (lon1, lat1), (lon2, lat2) in zip(ruta, ruta[1:]):
            _, _, distancia = geod.inv(lon1, lat1, lon2, lat2)
            metros += distancia
    return metros / 1000


class MapIcons:
    """Pines vectoriales con relleno contenido y punta sobre la coordenada."""

    @staticmethod
    def pin(color: str, escala: float = 0.62):
        area = DrawingArea(42 * escala, 54 * escala, 0, 0)
        vertices = [
            (21, 2),
            (16, 10), (4, 24), (4, 34),
            (4, 45), (11, 52), (21, 52),
            (31, 52), (38, 45), (38, 34),
            (38, 24), (26, 10), (21, 2),
            (21, 2),
        ]
        codigos = [
            MplPath.MOVETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        vertices = [(x * escala, y * escala) for x, y in vertices]
        silueta = MplPath(vertices, codigos)
        area.add_artist(PathPatch(silueta, facecolor=color,
                                  edgecolor='white', linewidth=1.5,
                                  joinstyle='round'))
        area.add_artist(Circle((21 * escala, 35 * escala), 7.5 * escala, facecolor='white', edgecolor='none'))
        return area

    @classmethod
    def colocar(cls, eje, inicio, fin, escala=0.62):
        for punto, color, orden in (
            (inicio, '#18a558', 10),
            (fin, '#2563eb', 11),
        ):
            eje.add_artist(AnnotationBbox(
                cls.pin(color, escala), punto,
                xycoords='data', xybox=(0, -2 * escala),
                boxcoords='offset points',
                box_alignment=(0.5, 0),
                frameon=False, zorder=orden,
                annotation_clip=False,
            ))


class MapRenderer:
    def __init__(self, cfg: MapSettings):
        self.cfg = cfg

    def guardar(self, rutas: list[list[tuple[float, float]]], destino: Path, distancia_km: float):
        import contextily as ctx
        puntos = [p for ruta in rutas for p in ruta]
        xs, ys = zip(*puntos)
        extension = max(max(xs) - min(xs), max(ys) - min(ys), 200)
        margen = extension * 0.05
        centro_x, centro_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        figura, eje = plt.subplots(figsize=(self.cfg.ANCHO_PULGADAS,
                                             self.cfg.ALTO_PULGADAS), dpi=self.cfg.DPI)
        try:
            eje.set_xlim(centro_x - extension / 2 - margen, centro_x + extension / 2 + margen)
            eje.set_ylim(centro_y - extension / 2 - margen, centro_y + extension / 2 + margen)
            fuente = ctx.providers.Esri.WorldStreetMap if self.cfg.FONDO == 'calles' else ctx.providers.Esri.WorldImagery
            print('[INFO] [ROUTE] Loading map background...', flush=True)
            try:
                zoom = None
                if self.cfg.ZOOM_EXTRA:
                    from contextily.tile import _calculate_zoom
                    izq, der = eje.get_xlim()
                    abajo, arriba = eje.get_ylim()
                    a_wgs = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)
                    oeste, sur = a_wgs.transform(izq, abajo)
                    este, norte = a_wgs.transform(der, arriba)
                    zoom = min(19, max(0, _calculate_zoom(oeste, sur, este, norte) + self.cfg.ZOOM_EXTRA))
                ctx.add_basemap(eje, source=fuente, crs='EPSG:3857',
                                zoom=zoom if zoom is not None else 'auto', attribution=False)
            except Exception as exc:
                if self.cfg.EXIGIR_FONDO:
                    raise RuntimeError(f'Could not load basemap: {exc}') from exc
                print(f'[WARNING] [ROUTE] Map background unavailable: {exc}')
            for ruta in rutas:
                rx, ry = zip(*ruta)
                eje.plot(rx, ry, color='#d62828', linewidth=2.8, alpha=.95, zorder=5)
            MapIcons.colocar(eje, rutas[0][0], rutas[-1][-1], self.cfg.TAMANO_PIN)
            if self.cfg.MOSTRAR_LEYENDA:
                leyenda = [
                    Line2D([0], [0], color='#d62828', lw=2.8, label='Route'),
                    Line2D([0], [0], marker='o', linestyle='None', markersize=8,
                           markerfacecolor='#18a558', markeredgecolor='white', label='Start'),
                    Line2D([0], [0], marker='o', linestyle='None', markersize=8,
                           markerfacecolor='#2563eb', markeredgecolor='white', label='End'),
                ]
                eje.legend(handles=leyenda, loc='upper right', fontsize=10,
                           frameon=True, framealpha=0.92, facecolor='white')
            eje.text(0.02, 0.03, f"Distance traveled: {distancia_km:.2f} km",
                     transform=eje.transAxes, fontsize=12, fontweight='bold',
                     va='bottom', ha='left', zorder=20,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                               edgecolor='#cccccc', alpha=0.95))
            eje.set_aspect('equal', adjustable='box')
            eje.set_axis_off()
            figura.subplots_adjust(left=0, right=1, top=1, bottom=0)
            destino.parent.mkdir(parents=True, exist_ok=True)
            print('[INFO] [ROUTE] Saving route map as PNG...', flush=True)
            temporal = destino.with_name(destino.stem + '.tmp.png')
            try:
                figura.savefig(temporal, bbox_inches='tight', pad_inches=.05)
                temporal.replace(destino)
            finally:
                temporal.unlink(missing_ok=True)
        finally:
            plt.close(figura)


def buscar_kml(others_dir: Path, nombre_configurado: str | None = None) -> Path:
    """Usa el KML configurado si existe; si no, detecta el único KML de la carpeta."""
    if nombre_configurado:
        configurado = others_dir / nombre_configurado
        if configurado.is_file():
            return configurado
        print(
            f"[WARNING] [ROUTE] Configured KML not found: {configurado.name}. "
            "Searching in others_dir...",
            flush=True,
        )

    if not others_dir.is_dir():
        raise FileNotFoundError(f"KML directory not found: {others_dir}")

    candidatos = sorted(
        (archivo for archivo in others_dir.iterdir()
         if archivo.is_file() and archivo.suffix.lower() == ".kml"),
        key=lambda archivo: archivo.name.lower(),
    )
    if not candidatos:
        raise FileNotFoundError(f"No .kml files found in: {others_dir}")
    if len(candidatos) > 1:
        nombres = ", ".join(archivo.name for archivo in candidatos)
        raise ValueError(
            f"Multiple KML files found in {others_dir}: {nombres}. "
            "Specify the correct filename in config.json (key 'kml') "
            "or leave only one KML file in the directory."
        )

    print(f"[INFO] [ROUTE] Automatically selected KML: {candidatos[0].name}", flush=True)
    return candidatos[0]


def extract_route() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    with (project_root / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)

    destination = project_root / config["data_dir"] / config["route_png"]
    others_dir = project_root / config["others_dir"]
    kml = buscar_kml(others_dir, config.get("kml"))

    print('[INFO] [ROUTE] Starting route extraction...', flush=True)
    print(f"[INFO] [ROUTE] Reading route KML: {kml.name}", flush=True)
    routes = KMLReader().leer(kml)
    if not routes:
        raise ValueError("KML does not contain a valid route")

    distancia_km = distancia_ruta_km(routes)
    print(f"[INFO] [ROUTE] Distance traveled: {distancia_km:.2f} km", flush=True)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    projected = []
    for route in routes:
        x, y = transformer.transform(*zip(*route))
        projected.append(list(zip(x, y)))

    print('[INFO] [ROUTE] Rendering route map...', flush=True)
    MapRenderer(MapSettings()).guardar(projected, destination, distancia_km)
    print(f"[INFO] [ROUTE] Route PNG exported to: {destination}", flush=True)
    return destination
