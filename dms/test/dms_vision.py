#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visor local de puntos en una caja de sección, coloreados según export_info.

Uso:
    python visor_puntos_seccion.py --config config.json
    python visor_puntos_seccion.py --config config.json --section 4
    python visor_puntos_seccion.py --input-ply ruta/al/PLY_anotado.ply

En el navegador: elija la sección, gire o amplíe la nube 3D (también hay
vistas TOP y PROFILE) y haga clic sobre un punto para ver TODAS sus columnas
originales, sin alterar el PLY, zones.py ni export_info.py.

Requisitos: pip install numpy plyfile plotly
La interfaz trabaja sin internet. Se inicia un servidor SOLO en 127.0.0.1.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from plyfile import PlyData

# Igual que export_info.py: marco de postes y márgenes de la caja.
BOX_X_MIN = -0.5
BOX_X_MAX = 0.5
BOX_WIDTH = 10.0
BOX_HEIGHT_MARGIN = 2.0
SCAN_BATCH = 250_000
MAX_DISPLAY_POINTS = 75_000  # Solo visualizacion; el PLY completo sigue consultable

# Misma paleta de niveles DMS de export_info.py.
LEVEL_COLORS = {0: '#e60000', 1: '#ff8c00', 2: '#ffd600', 3: '#3caf45'}
GRAY_COLOR = '#858585'
BLACK_COLOR = '#000000'
# Postes, crucetas y cables (originales y reconstruidos).
STRUCTURE_CLASSES = {4, 5, 24, 25, 26, 27}

CLASS_NAMES = {
    1: 'Ground', 2: 'Vegetation', 3: 'Buildings', 4: 'LV Poles',
    5: 'LV Wires', 10: 'Ads', 11: 'Rubble', 12: 'Scaffolding',
    13: 'Ladder', 14: 'Bricks', 15: 'MV Electrical Works',
    24: 'Reconstruction MV Poles', 25: 'Reconstruction MV Crossarms',
    26: 'Reconstruction MV Wires', 27: 'Reconstruction MV Connecting',
}


def read_json(path):
    with open(path, encoding='utf-8') as file:
        return json.load(file)


def resolve_path(base, *parts):
    return Path(os.path.join(str(base), *(str(v) for v in parts))).resolve()


def pole_field(pole, key):
    value = pole.get(key)
    if value is None and isinstance(pole.get('Info'), dict):
        value = pole['Info'].get(key)
    return value


def pole_id_number(value):
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        return None


def resolve_pole(poles, target):
    """Reproducir la selección de bases y dirección de export_info.py."""
    matching = [p for p in poles if pole_id_number(pole_field(p, 'Pole_ID')) == target]
    if not matching:
        return None
    bases, mean_bases, heights, vectors = [], [], [], []
    for pole in matching:
        b1, b2 = pole_field(pole, 'Base_1'), pole_field(pole, 'Base_2')
        main = pole_field(pole, 'Base')
        if pole_field(pole, 'Type') == 'Biposte' and b1 is not None and b2 is not None:
            a, b = np.asarray(b1, dtype=float), np.asarray(b2, dtype=float)
            bases.extend((a, b))
            mean_bases.append((a+b)/2)
        else:
            raw = main if main is not None else b1
            if raw is not None:
                point = np.asarray(raw, dtype=float)
                bases.append(point)
                mean_bases.append(point)
        height = pole_field(pole, 'height')
        if height is not None:
            heights.append(float(height))
        vec = pole_field(pole, 'unit_vector')
        if vec is not None:
            vectors.append(np.asarray(vec, dtype=float))
    if not mean_bases:
        return None
    direction = np.mean(vectors, axis=0) if vectors else np.array([0., 0., 1.])
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-9 else np.array([0., 0., 1.])
    return {
        'bases': bases,
        'base_avg': np.mean(mean_bases, axis=0),
        'height': float(np.mean(heights)) if heights else None,
        'dir': direction,
    }


def choose_base(pole, other):
    bases = pole['bases']
    if len(bases) <= 1:
        return np.asarray(bases[0] if bases else pole['base_avg'], dtype=float)
    return np.asarray(max(bases, key=lambda p: float(np.linalg.norm(p-other))), dtype=float)


def section_box(start_pole, end_pole):
    """Exactamente los ejes y límites espaciales de export_info.py."""
    start = choose_base(start_pole, end_pole['base_avg'])
    end = choose_base(end_pole, start)
    start = choose_base(start_pole, end)
    hdir = 0.5 * (start_pole['dir'] + end_pole['dir'])
    if np.linalg.norm(hdir) < 1e-6:
        hdir = np.array([0., 0., 1.])
    hdir /= np.linalg.norm(hdir)
    delta = end - start
    ldir = delta - np.dot(delta, hdir) * hdir
    if np.linalg.norm(ldir) < 1e-6:
        for ref in (np.array([1., 0., 0.]), np.array([0., 1., 0.])):
            ldir = ref - np.dot(ref, hdir) * hdir
            if np.linalg.norm(ldir) >= 1e-6:
                break
    ldir /= np.linalg.norm(ldir)
    wdir = np.cross(hdir, ldir)
    wdir /= np.linalg.norm(wdir)
    length = max(float(np.dot(delta, ldir)), 1e-6)
    end_z = float(np.dot(delta, hdir))
    height_a = float(start_pole['height'] or 0.)
    height_b = float(end_pole['height'] or 0.)
    height = max(height_a, end_z+height_b, end_z, 0.) + BOX_HEIGHT_MARGIN
    return {
        'origin': start,
        'end_base': end,
        'length_axis': length,
        'height': height,
        'width': BOX_WIDTH,
        'R': np.column_stack((ldir, wdir, hdir)),
        'z2_local': end_z,
    }


def valid_wire(wire):
    if wire.get('point_min') is None or wire.get('point_max') is None:
        return False
    try:
        eq = wire['equation']
        coeff = np.asarray([[float(eq[axis][key]) for key in ('a', 'b', 'c')]
                            for axis in ('x', 'y', 'z')], dtype=float)
        return np.all(np.isfinite(coeff)) and np.max(np.abs(coeff)) > 1e-9
    except (KeyError, ValueError, TypeError):
        return False


def section_key(value):
    try:
        return tuple(int(s) for s in str(value).split('_'))
    except (TypeError, ValueError):
        return (float('inf'),)


def load_sections(cfg, pole_map, base):
    """Todas las secciones válidas de wires_final; completa con zones_report."""
    output_dir = resolve_path(base, cfg['output_dir'])
    wires_path = output_dir / cfg['wires_final']
    sections = {}
    if wires_path.is_file():
        wires = read_json(wires_path)
        if isinstance(wires, dict):
            wires = list(wires.values())
        physical = []
        for rec in wires:
            name = str(rec.get('section_id', ''))
            match = re.fullmatch(r'(\d+)_(\d+)', name)
            if not match or not any(valid_wire(w) for w in rec.get('wires', [])):
                continue
            a, b = int(match.group(1)), int(match.group(2))
            if a not in pole_map or b not in pole_map:
                continue
            physical.append((name, a, b))
        # zones.py enumera secciones válidas en orden ascendente.
        for name, a, b in sorted(physical, key=lambda s: section_key(s[0])):
            sid = len(sections) + 1
            sections[sid] = {'section_id': sid, 'from_pole': a, 'to_pole': b,
                             'name': name, 'collisions': None}
    report_path = resolve_path(base, cfg['collision_dir']) / cfg.get(
        'zones_report_json', 'zones_report.json')
    if report_path.is_file():
        report = read_json(report_path)
        for sec in report.get('sections', []):
            sid = int(sec['section_id'])
            a, b = int(sec['from_pole']), int(sec['to_pole'])
            if a in pole_map and b in pole_map:
                sections[sid] = {'section_id': sid, 'from_pole': a, 'to_pole': b,
                                 'name': f'{a}_{b}',
                                 'collisions': len(sec.get('collisions', []))}
    if not sections:
        raise ValueError('No hay secciones válidas en wires_final o zones_report.json.')
    return {key: sections[key] for key in sorted(sections)}


def json_value(value):
    """Convertir cualquier tipo de columna PLY a JSON sin perder el nombre."""
    if isinstance(value, np.ndarray):
        return [json_value(v) for v in value]
    if isinstance(value, np.void):
        return [json_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode('utf-8', errors='replace')
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    return str(value)


def box_segments(box):
    """Las 12 aristas de la caja, con separadores null para Plotly."""
    xx = (BOX_X_MIN, box['length_axis']+BOX_X_MAX)
    yy = (-box['width']/2., box['width']/2.)
    zz = (0., box['height'])
    corners = [(x, y, z) for x in xx for y in yy for z in zz]
    edges = [(a, b) for a in range(8) for b in range(a+1, 8)
             if sum(corners[a][i] != corners[b][i] for i in range(3)) == 1]
    x, y, z = [], [], []
    for a, b in edges:
        for v in (corners[a], corners[b], (None, None, None)):
            x.append(v[0]); y.append(v[1]); z.append(v[2])
    return {'x': x, 'y': y, 'z': z}


class SectionViewerData:
    def __init__(self, cfg_path, input_ply=None):
        cfg = read_json(cfg_path)
        base = Path(cfg.get('BASE_DIR', '.')).expanduser()
        if not base.is_absolute():
            # Misma convención que zones.py y export_info.py: respecto al cwd.
            base = Path.cwd() / base
        base = base.resolve()
        pole_path = resolve_path(base, cfg['output_dir'], cfg['poles_final'])
        if not pole_path.is_file():
            raise FileNotFoundError(f'No existe poles_final: {pole_path}')
        poles = read_json(pole_path)
        if isinstance(poles, dict):
            poles = list(poles.values())
        ids = {pole_id_number(pole_field(p, 'Pole_ID')) for p in poles}
        self.poles = {pid: resolved for pid in ids if pid is not None
                      if (resolved := resolve_pole(poles, pid)) is not None}
        self.sections = load_sections(cfg, self.poles, base)
        self.boxes = {sid: section_box(self.poles[rec['from_pole']],
                                       self.poles[rec['to_pole']])
                      for sid, rec in self.sections.items()}
        path = (Path(input_ply).expanduser().resolve() if input_ply else
                resolve_path(base, cfg['collision_dir'], cfg['input_ply_new']))
        if not path.is_file():
            raise FileNotFoundError(f'No existe el PLY anotado: {path}')
        self.ply_path = path
        # mmap mantiene los datos del PLY en disco cuando es binario.
        self.vertex = PlyData.read(str(path), mmap='r')['vertex'].data
        self.fields = list(self.vertex.dtype.names)
        for key in ('x', 'y', 'z'):
            if key not in self.fields:
                raise ValueError(f'PLY vertex carece del campo {key!r}')
        self.n_total = len(self.vertex)
        self._cached_sid = None
        self._cached_data = None
        self._cache_lock = threading.Lock()

    def section_points(self, sid):
        """Todos los puntos dentro de la caja, SIN muestreo ni filtro por clase."""
        sid = int(sid)
        if sid not in self.boxes:
            raise ValueError(f'Sección desconocida: {sid}')
        with self._cache_lock:
            if sid == self._cached_sid:
                return self._cached_data
            box = self.boxes[sid]
            lo = np.array([BOX_X_MIN, -box['width']/2, 0.], dtype=float)
            hi = np.array([box['length_axis']+BOX_X_MAX, box['width']/2,
                           box['height']], dtype=float)
            corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                                for y in (lo[1], hi[1])
                                for z in (lo[2], hi[2])], dtype=float)
            world_corners = box['origin'] + corners @ box['R'].T
            world_min = world_corners.min(axis=0)
            world_max = world_corners.max(axis=0)
            picked_indices = []
            picked_local = []
            for start in range(0, self.n_total, SCAN_BATCH):
                end = min(self.n_total, start+SCAN_BATCH)
                block = self.vertex[start:end]
                x = np.asarray(block['x'], dtype=np.float64)
                y = np.asarray(block['y'], dtype=np.float64)
                z = np.asarray(block['z'], dtype=np.float64)
                candidate = np.flatnonzero(
                    (x >= world_min[0]) & (x <= world_max[0]) &
                    (y >= world_min[1]) & (y <= world_max[1]) &
                    (z >= world_min[2]) & (z <= world_max[2]) &
                    np.isfinite(x) & np.isfinite(y) & np.isfinite(z))
                if not len(candidate):
                    continue
                xyz = np.column_stack((x[candidate], y[candidate], z[candidate]))
                local = (xyz - box['origin']) @ box['R']
                inside = ((local >= lo-1e-9) & (local <= hi+1e-9)).all(axis=1)
                if inside.any():
                    picked_indices.append(candidate[inside].astype(np.int64) + start)
                    picked_local.append(local[inside].astype(np.float32))
            if picked_indices:
                indices = np.concatenate(picked_indices)
                local = np.concatenate(picked_local)
            else:
                indices = np.empty(0, dtype=np.int64)
                local = np.empty((0, 3), dtype=np.float32)
            self._cached_sid = sid
            self._cached_data = {'indices': indices, 'local': local}
            print(f'[INFO] Sección {sid}: {len(indices):,} puntos en la caja '
                  f'(de {self.n_total:,} puntos del PLY)', flush=True)
            return self._cached_data

    def overview(self):
        return {
            'sections': [{**sec, 'label': f"Sección {sid} · P{sec['from_pole']}-P{sec['to_pole']}"
                          + (f" · {sec['collisions']} DMS" if sec['collisions'] is not None else '')}
                         for sid, sec in self.sections.items()],
            'fields': self.fields,
            'ply': str(self.ply_path),
            'n_total': self.n_total,
        }

    def plot_data(self, sid):
        points = self.section_points(sid)
        all_coords = points['local']
        all_indices = points['indices']
        n = len(all_indices)
        # Muestreo reproducible para no saturar la memoria WebGL del navegador.
        chosen = (np.linspace(0, n - 1, MAX_DISPLAY_POINTS, dtype=np.int64)
                  if n > MAX_DISPLAY_POINTS else np.arange(n, dtype=np.int64))
        coords = all_coords[chosen]
        indices = all_indices[chosen]
        box = self.boxes[sid]
        # Leer SOLO las propiedades de los puntos visibles: evita buffers enormes.
        displayed = self.vertex[indices]
        colors = np.full(len(indices), GRAY_COLOR, dtype='<U7')
        classes = (np.asarray(displayed['class'], dtype=np.int32)
                   if 'class' in self.fields else np.full(len(indices), -1, dtype=np.int32))
        colors[np.isin(classes, list(STRUCTURE_CLASSES))] = BLACK_COLOR
        if 'is_dms' in self.fields and 'level_id' in self.fields:
            dms = np.asarray(displayed['is_dms']) == 1
            levels = np.asarray(displayed['level_id'])
            for level, color in LEVEL_COLORS.items():
                colors[dms & (levels == level)] = color
        # Los DMS prevalecen sobre el negro y el gris, como en export_info.
        # Usar floats de precisión de render, pero consultar las columnas exactas
        # desde el PLY cuando el usuario selecciona un punto.
        return {
            'section': self.sections[sid],
            'indices': indices.tolist(),
            'colors': colors.tolist(),
            'total_points': n,
            'x': coords[:, 0].tolist(),
            'y': coords[:, 1].tolist(),
            'z': coords[:, 2].tolist(),
            'box': box_segments(box),
            'bounds': {'xmin': BOX_X_MIN, 'xmax': box['length_axis']+BOX_X_MAX,
                       'ymin': -box['width']/2, 'ymax': box['width']/2,
                       'zmin': 0., 'zmax': box['height']},
        }

    def selected_point(self, sid, index):
        points = self.section_points(sid)
        index = int(index)
        pos = int(np.searchsorted(points['indices'], index))
        if pos >= len(points['indices']) or points['indices'][pos] != index:
            raise ValueError('El punto elegido no pertenece a la caja de la sección.')
        original = self.vertex[index]
        props = {name: json_value(original[name]) for name in self.fields}
        xyz = np.array([original['x'], original['y'], original['z']], dtype=float)
        box = self.boxes[sid]
        local = (xyz - box['origin']) @ box['R']
        cls = pole_id_number(props.get('class'))
        return {'index': index, 'row_number': index+1,
                'local': dict(zip(("x'", "y'", "z'"), [float(v) for v in local])),
                'class_name': CLASS_NAMES.get(cls, 'Unknown'),
                'fields': props}


HTML = r'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Inspector LiDAR · Puntos por sección</title>
<style>
  :root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#1d2733;background:#edf1f6}
  *{box-sizing:border-box}body{margin:0}.bar{padding:14px 20px;background:white;box-shadow:0 1px 5px #0002;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
  h1{font-size:20px;margin:0 18px 0 0}.bar label{font-weight:600;font-size:13px}select,button{font:inherit;padding:8px 11px;border-radius:7px;border:1px solid #acb6c4;background:white}button{cursor:pointer}
  .layout{display:grid;grid-template-columns:minmax(480px,1fr) 360px;gap:12px;padding:12px;height:calc(100vh - 76px);min-height:570px}
  .panel{background:white;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px #0001;display:flex;flex-direction:column;min-width:0}
  #plot{width:100%;flex:1;min-height:370px}.hint{font-size:12px;color:#627185;padding:8px 14px;border-top:1px solid #e4e8ee}
  .inspect{padding:16px;overflow:auto}.inspect h2{font-size:17px;margin:0 0 10px}.meta{font-size:13px;white-space:pre-wrap;color:#48586b;line-height:1.6;margin:0 0 12px}
  table{border-collapse:collapse;width:100%;font-size:13px;table-layout:fixed}td{padding:8px 6px;border-bottom:1px solid #e7e9ee;vertical-align:top;word-break:break-word}td:first-child{width:42%;font-weight:650;color:#34485e}
  .placeholder{color:#677689;font-size:14px;line-height:1.6}.actions{display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap}button:disabled{opacity:.45;cursor:not-allowed}
  .status{font-size:13px;color:#4c6379}.ok{color:#157245}.error{color:#ad2929}
  @media(max-width:950px){.layout{display:flex;flex-direction:column;height:auto}.panel:first-child{height:530px}.panel:last-child{min-height:260px}.inspect{max-height:600px}}
</style>
<script src="/plotly.js"></script>
</head><body>
<div class="bar">
  <h1>Inspector LiDAR</h1>
  <label for="section">Sección</label><select id="section"></select>
  <label for="mode">Vista</label>
  <select id="mode"><option value="3d">Caja 3D</option><option value="top">TOP (X'–Y')</option><option value="profile">PROFILE (X'–Z')</option></select>
  <span class="status" id="status">Cargando…</span>
  <span style="font-size:12px">🔴 DMS 0 · 🟠 DMS 1 · 🟡 DMS 2 · 🟢 DMS 3 · ⚫ Postes/cables · Gris: resto</span>
</div>
<div class="layout">
  <div class="panel"><div id="plot"></div><div class="hint">Arrastra para rotar (3D) o desplazarte (2D); usa la rueda para acercar. Haz clic en cualquier punto para inspeccionarlo. DMS: rojo/naranja/amarillo/verde; postes y cables: negro; resto: gris. Se muestra una muestra visual de hasta 75 000 puntos.</div></div>
  <div class="panel inspect">
    <h2>Propiedades del punto</h2>
    <div class="actions"><button id="copy" disabled>Copiar todas las columnas (JSON)</button></div>
    <p class="meta" id="meta"></p>
    <div id="properties" class="placeholder">Elige una sección y haz clic en un punto de la nube.</div>
  </div>
</div>
<script>
'use strict';
let current = null, selected = null, selectedRecord = null, epoch = 0, currentMode = '3d';
const sectionSelect = document.getElementById('section');
const modeSelect = document.getElementById('mode');
const plot = document.getElementById('plot');
const statusEl = document.getElementById('status');
const propsEl = document.getElementById('properties');
const metaEl = document.getElementById('meta');
const copyEl = document.getElementById('copy');

function setStatus(message, cls=''){statusEl.textContent=message;statusEl.className='status '+cls;}
function el(tag, text){const item=document.createElement(tag);if(text!==undefined)item.textContent=text;return item;}
async function api(path){const response=await fetch(path);const data=await response.json();if(!response.ok)throw new Error(data.error || response.statusText);return data;}
function clearSelection(){selected=null;selectedRecord=null;copyEl.disabled=true;metaEl.textContent='';propsEl.textContent='Haz clic sobre un punto para ver todas sus columnas originales del PLY.';}

async function loadSection(){
  const request=++epoch; const sid=Number(sectionSelect.value);
  current=null;clearSelection();setStatus('Leyendo todos los puntos de la caja…');
  try{
    const data=await api('/api/section?sid='+encodeURIComponent(sid));
    if(request!==epoch)return;
    current=data;
    setStatus(data.indices.length.toLocaleString('es')+' puntos visibles de '+data.total_points.toLocaleString('es')+' en la caja · sección '+sid,'ok');
    await renderPlot();
  }catch(err){if(request===epoch){setStatus('Error: '+err.message,'error');propsEl.textContent=err.message;}}
}
function box2d(d, vertical){
  const b=d.bounds;
  const a=b.xmin,c=b.xmax,lo=vertical==='y'?b.ymin:b.zmin,hi=vertical==='y'?b.ymax:b.zmax;
  return {x:[a,c,c,a,a],y:[lo,lo,hi,hi,lo],type:'scatter',mode:'lines',name:'Caja de sección',hoverinfo:'skip',line:{color:'#2563a0',width:2},showlegend:false};
}
function renderConfig(){return {responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToRemove:['toImage']};}
async function renderPlot(){
 if(!current)return;
 const mode=modeSelect.value;
 const b=current.bounds;
 const is3d=mode==='3d';
 let trace, rectangle, layout;
 const hover = "Punto global #%{customdata}<extra>Haz clic para ver todas las columnas</extra>";
 if(is3d){
   trace={type:'scatter3d',mode:'markers',x:current.x,y:current.y,z:current.z,
     customdata:current.indices,name:'Puntos',
     marker:{size:2,color:current.colors,opacity:0.88},hovertemplate:hover};
   rectangle={type:'scatter3d',mode:'lines',...current.box,name:'Caja de sección',
     hoverinfo:'skip',showlegend:false,line:{color:'#2266a1',width:4}};
   layout={margin:{l:0,r:0,t:38,b:0},title:{text:'Sección '+current.section.section_id+' · caja 3D',font:{size:16}},
     scene:{aspectmode:'data',xaxis:{title:"X' (m)",range:[b.xmin,b.xmax]},
       yaxis:{title:"Y' (m)",range:[b.ymin,b.ymax]},
       zaxis:{title:"Z' (m)",range:[b.zmin,b.zmax]}},showlegend:false};
 }else{
   const vertical=mode==='top'?'y':'z';
   trace={type:'scattergl',mode:'markers',x:current.x,y:current[vertical],
     customdata:current.indices,name:'Puntos',marker:{size:3,color:current.colors,opacity:.88},hovertemplate:hover};
   rectangle=box2d(current,vertical);
   layout={margin:{l:60,r:20,t:52,b:55},title:{text:'Sección '+current.section.section_id+' · '+mode.toUpperCase(),font:{size:16}},
     xaxis:{title:"X' (m)",range:[b.xmin,b.xmax]},
     yaxis:{title:(vertical==='y'?"Y'":"Z'")+" (m)",scaleanchor:'x',scaleratio:1,
       range:vertical==='y'?[b.ymin,b.ymax]:[b.zmin,b.zmax]},
     showlegend:false,dragmode:'zoom'};
 }
 const traces=[trace,rectangle];
 // La traza de seleccion siempre existe; actualizarla no duplica el buffer WebGL.
 const selectedTrace = is3d ?
   {type:'scatter3d',mode:'markers',x:selected?[selected.local["x'"]]:[],
    y:selected?[selected.local["y'"]]:[],z:selected?[selected.local["z'"]]:[],
    marker:{size:9,color:'#dc2626'},hoverinfo:'skip',showlegend:false} :
   {type:'scatter',mode:'markers',x:selected?[selected.local["x'"]]:[],
    y:selected?[selected.local[mode==='top'?"y'":"z'"]]:[],
    marker:{size:13,color:'#dc2626'},hoverinfo:'skip',showlegend:false};
 traces.push(selectedTrace);
 await Plotly.react(plot,traces,layout,renderConfig());
 currentMode=mode;
 if(typeof plot.removeAllListeners==='function')plot.removeAllListeners('plotly_click');
 plot.on('plotly_click',async event=>{
   const point=event.points && event.points[0];
   if(!point || point.curveNumber!==0)return;
   const index=point.customdata;
   if(!Number.isInteger(index))return;
   const sid=Number(sectionSelect.value);
   try{
     const result=await api('/api/point?sid='+sid+'&index='+index);
     if(!current || Number(current.section.section_id)!==sid)return;
     selected=result;selectedRecord=result;
     showProperties(result);
     // Restyle modifica solo una traza pequena: evita deleteTraces/addTraces.
     if(modeSelect.value==='3d'){
       await Plotly.restyle(plot,{x:[[result.local["x'"]]],
         y:[[result.local["y'"]]],z:[[result.local["z'"]]]},[2]);
     }else{
       await Plotly.restyle(plot,{x:[[result.local["x'"]]],
         y:[[result.local[modeSelect.value==='top'?"y'":"z'"]]]},[2]);
     }
   }catch(err){setStatus('Error consultando punto: '+err.message,'error');}
 });
}
function showProperties(result){
 metaEl.textContent='Fila del PLY: '+result.row_number+' (índice '+result.index+')\n'+
   'Clase interpretada: '+result.class_name+'\n'+
   "Posición local X': "+result.local["x'"].toFixed(4)+' m | '+
   "Y': "+result.local["y'"].toFixed(4)+' m | '+
   "Z': "+result.local["z'"].toFixed(4)+' m';
 const table=el('table');const tbody=el('tbody');
 for(const [name,value] of Object.entries(result.fields)){
   const tr=el('tr');tr.appendChild(el('td',name));
   tr.appendChild(el('td',typeof value==='object'?JSON.stringify(value):String(value)));
   tbody.appendChild(tr);
 }
 table.appendChild(tbody);propsEl.replaceChildren(table);copyEl.disabled=false;
}
copyEl.addEventListener('click',async()=>{
 if(!selectedRecord)return;
 try{await navigator.clipboard.writeText(JSON.stringify(selectedRecord,null,2));
   copyEl.textContent='¡Copiado!';setTimeout(()=>copyEl.textContent='Copiar todas las columnas (JSON)',1300);
 }catch(err){alert('No se pudo copiar: '+err.message);}
});
sectionSelect.addEventListener('change',loadSection);
modeSelect.addEventListener('change',()=>{if(current)renderPlot();});
async function init(){
 try{
   const config=await api('/api/info');
   for(const sec of config.sections){const option=el('option',sec.label);option.value=sec.section_id;sectionSelect.appendChild(option);}
   const params=new URLSearchParams(location.search);
   if(params.has('section') && [...sectionSelect.options].some(o=>o.value===params.get('section'))){sectionSelect.value=params.get('section');}
   await loadSection();
 }catch(err){setStatus('Error: '+err.message,'error');}
}
init();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    viewer = None
    plotly_js = None

    def send_bytes(self, code, payload, ctype, compress=False):
        if compress and len(payload) > 40_000:
            payload = gzip.compress(payload, compresslevel=3)
            encoding = 'gzip'
        else:
            encoding = None
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        if encoding:
            self.send_header('Content-Encoding', encoding)
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, value, code=200):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False,
                             separators=(',', ':')).encode('utf-8')
        self.send_bytes(code, payload, 'application/json; charset=utf-8', True)

    def do_GET(self):
        url = urlsplit(self.path)
        try:
            if url.path == '/':
                return self.send_bytes(200, HTML.encode('utf-8'), 'text/html; charset=utf-8')
            if url.path == '/plotly.js':
                return self.send_bytes(200, self.plotly_js, 'text/javascript; charset=utf-8', True)
            if url.path == '/api/info':
                return self.send_json(self.viewer.overview())
            query = parse_qs(url.query)
            if url.path == '/api/section':
                sid = int(query['sid'][0])
                return self.send_json(self.viewer.plot_data(sid))
            if url.path == '/api/point':
                sid = int(query['sid'][0])
                index = int(query['index'][0])
                return self.send_json(self.viewer.selected_point(sid, index))
            return self.send_json({'error': 'Ruta desconocida.'}, 404)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            self.send_json({'error': str(exc)}, 400)
        except Exception as exc:
            print(f'[ERROR] {exc}', flush=True)
            self.send_json({'error': f'{type(exc).__name__}: {exc}'}, 500)

    def log_message(self, fmt, *args):
        # Evita llenar la consola con cada consulta de punto.
        if 'GET /api/section' in (fmt % args):
            print('[HTTP] '+(fmt % args), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Visor 3D y selector de puntos por caja de sección DMS')
    ap.add_argument('--config', default='config.json', help='Config JSON del pipeline DMS')
    ap.add_argument('--input-ply', default=None, help='PLY anotado alternativo')
    ap.add_argument('--section', type=int, default=None, help='Sección inicial (también seleccionable desde la interfaz)')
    ap.add_argument('--port', type=int, default=0, help='Puerto local; 0 para automático')
    ap.add_argument('--no-browser', action='store_true', help='No abrir navegador automáticamente')
    args = ap.parse_args(argv)
    if args.port < 0 or args.port > 65535:
        ap.error('--port debe estar entre 0 y 65535')
    viewer = SectionViewerData(args.config, args.input_ply)
    if args.section is not None and args.section not in viewer.sections:
        ap.error(f'No existe sección {args.section}. Disponibles: {list(viewer.sections)}')
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise SystemExit('Falta plotly: pip install plotly') from exc
    Handler.viewer = viewer
    Handler.plotly_js = get_plotlyjs().encode('utf-8')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    port = server.server_address[1]
    url = f'http://127.0.0.1:{port}/'
    if args.section is not None:
        url += f'?section={args.section}'
    print(f'[INFO] PLY: {viewer.ply_path}', flush=True)
    print(f'[INFO] Columnas originales ({len(viewer.fields)}): {", ".join(viewer.fields)}', flush=True)
    print(f'[INFO] Secciones disponibles: {", ".join(map(str, viewer.sections))}', flush=True)
    print(f'[INFO] Abre el visor en: {url}', flush=True)
    print('[INFO] Ctrl+C cierra el servidor local.', flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print('\n[INFO] Visor cerrado.', flush=True)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
