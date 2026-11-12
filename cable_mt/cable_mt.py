import asyncio
import json
import os
import sys
import time
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import partial

import numpy as np

try:
    from plyfile import PlyData
except ImportError:
    print('[ERROR] [CABLE_MT] Required dependency missing. Install with: pip install plyfile')
    sys.exit(1)
try:
    from scipy.spatial import KDTree, cKDTree
except ImportError:
    print('[ERROR] [CABLE_MT] Required dependency missing. Install with: pip install scipy')
    sys.exit(1)
try:
    import websockets
except ImportError:
    print('[ERROR] [CABLE_MT] Required dependency missing. Install with: pip install websockets')
    sys.exit(1)


SLOT_COLORS = [
    [0, 229, 255],
    [255, 214, 0],
    [224, 64, 251],
]
MAX_CABLE_PREVIEW = 40_000
MAX_VIZ_PTS = 80_000

# Geometric span detection parameters (kept in code, not config.json).
SPAN_POLE_LATERAL_TOLERANCE_M = 2.0
SPAN_ENDPOINT_MARGIN_M = 5.0
SPAN_MIN_TRACE_COVERAGE = 0.75


def _load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config.json')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Project configuration not found: {config_path}')
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    config['_project_root'] = project_root
    return config


def _resolve_path(name):
    path = os.fspath(name)
    return path if os.path.isabs(path) else os.path.join(CONFIG['_project_root'], path)


def _configured_file(folder_key, file_key):
    return os.path.join(_resolve_path(CONFIG[folder_key]), CONFIG[file_key])


CONFIG = _load_config()


def _labels_from_catalog():
    path = _resolve_path(CONFIG['labels_json'])
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Labels catalog not found: {path}')
    with open(path, encoding='utf-8') as handle:
        catalog = json.load(handle)
    entries = catalog['labels']
    by_title = {item['title'].strip().casefold(): int(item['id']) for item in entries}
    required = ['MV Poles', 'MV Wires', 'MV Wires 1', 'MV Wires 2', 'MV Wires 3']
    missing = [name for name in required if name.casefold() not in by_title]
    if missing:
        raise ValueError(f'Missing labels in {path}: {missing}')
    pole, wire, *phases = [by_title[name.casefold()] for name in required]
    if len({pole, wire, *phases}) != 5:
        raise ValueError('Duplicate IDs for required classes')
    return pole, wire, dict(enumerate(phases))


POLE_LABEL, CABLE_LABEL, SLOT_LABEL_IDS = _labels_from_catalog()


def _changes_path():
    return os.path.join(_resolve_path(CONFIG["data_dir"]), "backup", CONFIG["cable_mt_changes"])


def _state_path():
    return os.path.join(_resolve_path(CONFIG["data_dir"]), "backup", CONFIG["cable_mt_state"])


def _write_changes(server):
    """Save only edited vertices, tied to the exact source PLY fingerprint."""
    idx = np.flatnonzero(server._all_labels != server._original_labels)
    path = _changes_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, indices=idx.astype(np.int64),
                        labels=server._all_labels[idx].astype(np.int32),
                        original_labels=server._original_labels[idx].astype(np.int32),
                        source_signature=np.array(server._source_signature),
                        label_property=np.array(server._label_prop_name),
                        point_count=np.array(len(server._all_labels), dtype=np.int64))
    os.replace(tmp, path)
    return path


def _output_ply_path(input_ply_path):
    return _configured_file('data_dir', 'input_ply')


def _output_json_html_path():
    return _configured_file('others_dir', 'json_html')


def _output_json_extractor_path():
    return _configured_file('output_dir', 'json_connector')


def _poles_info_path():
    # Cable MT deliberately uses RAW metadata; json_extractor is for a later stage.
    return _configured_file('output_dir', 'json_extractor_raw')


class PLYHTTPHandler(BaseHTTPRequestHandler):
    def __init__(self, server, *a, **kw):
        self._server = server
        self._ply = server.ply_path
        super().__init__(*a, **kw)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.rstrip("/")
        if path == "/ply":
            try:
                data = self._server.visual_ply
                size = len(data)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", size)
                self.send_header(
                    "Content-Disposition", 'inline; filename="poles_label6.ply"'
                )
                self._cors()
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self.send_error(500, str(e))
                except (BrokenPipeError, ConnectionResetError):
                    pass
        else:
            self.send_error(404)

    def log_message(self, *_):
        pass


def _make_visual_ply(points, labels):
    vertex = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("class", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["red"] = 0
    vertex["green"] = 0
    vertex["blue"] = 0
    vertex["class"] = labels.astype(np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar class\nend_header\n"
    ).encode("ascii")
    return header + vertex.tobytes()


class CableLineServer:
    def __init__(self, ply_path):
        self.ply_path = ply_path
        self._centroid = None
        self._cable_points = None
        self._kdtree = None
        self._cable_preview_flat = None
        self._n_pole_pts = 0
        self.visual_ply = b""
        self._poles = []
        self._has_s_odom = False
        self.slot_points = {0: [], 1: [], 2: []}
        self.slot_labels = dict(SLOT_LABEL_IDS)
        self.pole_terminals = {}
        self._label_prop_name = None
        self._all_labels = None
        self._cable_global_indices = None
        self._active_cable = None
        self._ply_exported = False
        from cable_mt.rebuild import source_signature
        self._source_signature = source_signature(ply_path)
        self._load()

    @property
    def poles(self):
        return self._poles

    def _load_poles_from_json(self):
        json_path = _poles_info_path()
        if not os.path.exists(json_path):
            print(
                f"[WARNING] [CABLE_MT] Pole metadata file not found: {json_path}. "
                f"Run the extractor first. Pole list will be empty."
            )
            return False
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception as e:
            print(f"[WARNING] [CABLE_MT] Unable to read pole metadata file: {json_path}: {e}")
            return False

        for entry in entries:
            info = entry.get("Info", {}) if isinstance(entry, dict) else {}
            base = info.get("Base")
            if not base or len(base) < 3:
                continue
            try:
                pid = int(entry["Pole_ID"])
                cx_w = float(base[0])
                cy_w = float(base[1])
                zb_w = float(base[2])
                height = float(info.get("height", 0.0))
            except (KeyError, TypeError, ValueError):
                continue

            s_val = info.get("s_odometria_m", None)
            s_m = float(s_val) if s_val is not None else 0.0
            if s_val is not None:
                self._has_s_odom = True

            self._poles.append({
                "pid": pid,
                "name": f"P{pid}",
                "cx": cx_w - self._centroid[0],
                "cy": cy_w - self._centroid[1],
                "z_bottom": zb_w - self._centroid[2],
                "z_top": zb_w + height - self._centroid[2],
                "height": height,
                "r_bottom": 0.15,
                "r_top": 0.08,
                "n_pts": int(info.get("num_points", 0)),
                "s_m": s_m,
                "traj_d": 0.0,
            })

        print(f"[INFO] [CABLE_MT] Poles loaded: {len(self._poles)} (from {json_path})")
        if self._poles:
            print(
                f"[INFO] [CABLE_MT] First / last pole: "
                f"{self._poles[0]['name']} / {self._poles[-1]['name']}"
            )
        return True

    def _load(self):
        sep = "=" * 64
        print(f"[INFO] [CABLE_MT] Point-cloud banner: {sep}\n  Source: {self.ply_path}")
        print(f"[INFO] [CABLE_MT] Pole metadata source: {_poles_info_path()}")
        t0 = time.time()

        plydata = PlyData.read(self.ply_path)
        v = plydata["vertex"]
        label_arr = None
        for prop in v.properties:
            if prop.name.lower() in (
                "label", "labels", "classification", "scalar_label", "semantic", "class"
            ):
                self._label_prop_name = prop.name
                label_arr = np.asarray(v[prop.name], dtype=np.int32)
                print(f"[INFO] [CABLE_MT] Point-label property: {prop.name!r}")
                break
        if label_arr is None:
            raise ValueError("No label property found in PLY.")

        self._all_labels = label_arr.copy()
        self._original_labels = label_arr.copy()
        print(f"[INFO] [CABLE_MT] Total points: {len(label_arr):,} points  ({time.time() - t0:.2f}s)")
        print(f"[INFO] [CABLE_MT] Unique point labels: {np.unique(label_arr)}")

        visible_idx = np.flatnonzero(
            (label_arr == CABLE_LABEL) | (label_arr == POLE_LABEL)
        )
        visible_labels = label_arr[visible_idx]
        visible_points = np.column_stack([
            np.asarray(v["x"])[visible_idx],
            np.asarray(v["y"])[visible_idx],
            np.asarray(v["z"])[visible_idx],
        ]).astype(np.float64, copy=False)
        self._centroid = visible_points.mean(axis=0)
        cable_mask = visible_labels == CABLE_LABEL
        pole_mask = visible_labels == POLE_LABEL
        cable_idx = visible_idx[cable_mask]
        self._cable_points = visible_points[cable_mask]
        self._cable_global_indices = cable_idx
        self._active_cable = np.ones(len(self._cable_points), dtype=bool)

        n_c = len(self._cable_points)
        if n_c == 0:
            raise ValueError(f"0 points with label={CABLE_LABEL}")
        print(f"[INFO] [CABLE_MT] Cable points (label {CABLE_LABEL}) : {n_c:,} pts")
        self._kdtree = KDTree(self._cable_points)

        prev = (self._cable_points - self._centroid).astype(np.float32)
        self._cable_preview_flat = prev.flatten().tolist()
        print(f"[INFO] [CABLE_MT] Cable preview points: {len(prev):,} pts")

        pole_pts = visible_points[pole_mask]
        self._n_pole_pts = int(len(pole_pts))
        self.visual_ply = _make_visual_ply(
            pole_pts, np.full(len(pole_pts), POLE_LABEL, dtype=np.uint8)
        )
        print(
            f"[INFO] [CABLE_MT] Visualization PLY: {len(pole_pts):,} pts, "
            f"{len(self.visual_ply) / 1048576:.1f} MB (all label {POLE_LABEL})"
        )
        print(f"[INFO] [CABLE_MT] Pole points (label {POLE_LABEL}) : {len(pole_pts):,} pts")

        self._load_poles_from_json()
        print(f"{sep}\n")

    def _estimate_direction(self, anchor_pts, anchor_idxs, eps):
        knn = set()
        for idx in anchor_idxs:
            knn.update(
                self._kdtree.query_ball_point(self._cable_points[int(idx)], eps * 2.0)
            )
        pts = self._cable_points[list(knn)]
        c = pts - pts.mean(axis=0)
        if len(c) > 8000:
            c = c[:: len(c) // 8000]
        try:
            _, _, Vt = np.linalg.svd(c, full_matrices=False)
            d = Vt[0]
            n = np.linalg.norm(d)
            if n < 1e-9:
                raise ValueError()
            return d / n
        except Exception:
            if len(anchor_pts) > 1:
                d = anchor_pts[-1] - anchor_pts[0]
                n = np.linalg.norm(d)
                if n > 1e-9:
                    return d / n
            return np.array([1.0, 0.0, 0.0])

    def _trace_centerline(self, start, init_dir, perp_r, eps, gap, mom):
        path, visited = [start], {start}
        curr, d = start, init_dir.copy()
        for _ in range(20_000):
            pt = self._cable_points[curr]
            best_i, best_proj = None, -np.inf
            for radius, min_cos in [(eps, 0.60), (gap, 0.85)]:
                for nb in self._kdtree.query_ball_point(pt, radius):
                    if nb in visited:
                        continue
                    delta = self._cable_points[nb] - pt
                    dist = np.linalg.norm(delta)
                    if dist < 1e-9:
                        continue
                    proj = np.dot(delta, d)
                    if proj <= 0:
                        continue
                    if np.linalg.norm(delta - proj * d) > perp_r:
                        continue
                    if (proj / dist) < min_cos:
                        continue
                    if proj > best_proj:
                        best_proj, best_i = proj, nb
                if best_i is not None:
                    break
            if best_i is None:
                break
            delta = self._cable_points[best_i] - pt
            new_d = delta / np.linalg.norm(delta)
            d = mom * d + (1 - mom) * new_d
            n = np.linalg.norm(d)
            if n < 1e-9:
                break
            d /= n
            visited.add(best_i)
            path.append(best_i)
            curr = best_i
        return path

    def _collect_near_path(self, path_idxs, init_dir, perp_r):
        path_pts = self._cable_points[path_idxs]
        projs = np.dot(path_pts, init_dir)
        sorted_path = path_pts[np.argsort(projs)]

        dense_path = []
        for i in range(len(sorted_path) - 1):
            p1 = sorted_path[i]
            p2 = sorted_path[i + 1]
            dist = np.linalg.norm(p2 - p1)
            if 0.05 < dist < 12.0:
                steps = max(2, int(np.ceil(dist / 0.05)))
                for t in np.linspace(0, 1, steps, endpoint=False):
                    dense_path.append(p1 + t * (p2 - p1))
            else:
                dense_path.append(p1)
        dense_path.append(sorted_path[-1])
        dense_path = np.array(dense_path)

        kd = KDTree(dense_path)
        dists, _ = kd.query(self._cable_points, workers=-1)
        return np.where(dists <= perp_r)[0]

    def select_by_seeds(self, seeds_world, slot_id, perp_r=0.12, eps=0.60, gap=3.0, mom=0.85):
        if not seeds_world:
            return {"ok": False, "message": "No seeds"}
        seeds = np.asarray(seeds_world, dtype=np.float64)
        _, anchor_idxs = self._kdtree.query(seeds)
        anchor_pts = self._cable_points[anchor_idxs]
        snap_norms = (anchor_pts - self._centroid).tolist()
        init_dir = self._estimate_direction(anchor_pts, anchor_idxs, eps)

        all_path = set(anchor_idxs.tolist())
        for si in anchor_idxs.tolist():
            for sign in [1.0, -1.0]:
                all_path.update(
                    self._trace_centerline(si, sign * init_dir, perp_r, eps, gap, mom)
                )

        path_list = list(all_path)
        accepted = self._collect_near_path(path_list, init_dir, perp_r)
        accepted = accepted[self._active_cable[accepted]]

        if len(accepted) == 0:
            return {
                "ok": False,
                "message": f"0 pts with perp_r={perp_r:.3f}m",
                "snap_seeds": snap_norms,
            }

        self.slot_points[slot_id] = accepted.tolist()

        pp = self._cable_points[path_list]
        t_p = np.dot(pp - pp.mean(axis=0), init_dir)
        spine_len = float(t_p.max() - t_p.min())

        pts_norm = (self._cable_points[accepted] - self._centroid).astype(np.float32)
        if len(pts_norm) > MAX_VIZ_PTS:
            pts_norm = pts_norm[:: max(1, len(pts_norm) // MAX_VIZ_PTS)]

        pts_per_m = round(len(accepted) / max(spine_len, 0.001), 1)

        return {
            "ok": True,
            "n_total": int(len(accepted)),
            "n_path": int(len(path_list)),
            "n_viz": int(len(pts_norm)),
            "spine_length_m": round(spine_len, 1),
            "pts_per_m": pts_per_m,
            "perp_radius": perp_r,
            "snap_seeds": snap_norms,
            "points_flat": pts_norm.flatten().tolist(),
        }

    def _bounding_poles(self, anchor, direction, traced_indices=None):
        """Choose two supports of the traced cable, not adjacent pole numbers.

        A pole offset from the cable corridor is ignored even if its ID is between
        the actual supports. The trace must have samples along the proposed span.
        """
        if len(self._poles) < 2:
            return None
        dxy = np.asarray(direction[:2], dtype=np.float64).copy()
        norm = np.linalg.norm(dxy)
        if norm < 1e-9:
            return None
        dxy /= norm
        normal = np.array([-dxy[1], dxy[0]])
        pole_xy = np.asarray([[p['cx'] + self._centroid[0],
                               p['cy'] + self._centroid[1]] for p in self._poles])
        rel = pole_xy - anchor[:2]
        along = rel @ dxy
        lateral = np.abs(rel @ normal)
        # Restrict supports to the actual selected cable corridor.
        corridor = SPAN_POLE_LATERAL_TOLERANCE_M
        candidate = np.flatnonzero(lateral <= corridor)
        if len(candidate) < 2:
            return None
        if traced_indices is None or len(traced_indices) < 8:
            return None
        trace = self._cable_points[np.asarray(list(traced_indices), dtype=np.int64)]
        trace_t = (trace[:, :2] - anchor[:2]) @ dxy
        trace_u = np.abs((trace[:, :2] - anchor[:2]) @ normal)
        trace_t = trace_t[trace_u <= corridor + 0.5]
        if len(trace_t) < 8:
            return None
        lo, hi = np.percentile(trace_t, [2, 98])
        margin = SPAN_ENDPOINT_MARGIN_M
        candidates = [int(i) for i in candidate if lo-margin <= along[i] <= hi+margin]
        before = [i for i in candidates if along[i] < -0.5]
        after = [i for i in candidates if along[i] > 0.5]
        if not before or not after:
            return None
        # Favor supports at the observed ends, rather than the nearest poles
        # in numerical/trajectory order. Require cable coverage throughout.
        best = None
        for i in before:
            for j in after:
                a, b = float(along[i]), float(along[j])
                if b-a < 5.0:
                    continue
                bins = np.linspace(a, b, 9)
                occupied = np.histogram(trace_t, bins=bins)[0] > 0
                coverage = float(occupied.mean())
                if coverage < SPAN_MIN_TRACE_COVERAGE:
                    continue
                endpoint_error = abs(a-lo) + abs(b-hi)
                score = endpoint_error + 2.0*(lateral[i]+lateral[j]) + 10.0*(1.0-coverage)
                if best is None or score < best[0]:
                    best = (score, i, j)
        if best is None:
            return None
        _, i, j = best
        p0, p1 = pole_xy[i], pole_xy[j]
        chord = p1-p0
        length = float(np.linalg.norm(chord))
        return p0, p1, chord/length, length, i, j

    def _cross_section_groups(self, center, dxy, anchor_lateral):
        normal = np.array([-dxy[1], dxy[0]])
        rel = self._cable_points[:, :2] - center
        along = rel @ dxy
        lateral = rel @ normal
        candidate_idx = np.flatnonzero(
            self._active_cable
            & (np.abs(along) <= 1.5)
            & (np.abs(lateral - anchor_lateral) <= 5.0)
        )
        if len(candidate_idx) < 15:
            return []

        features = np.column_stack([
            lateral[candidate_idx],
            self._cable_points[candidate_idx, 2],
        ])
        tree = cKDTree(features)
        visited = np.zeros(len(features), dtype=bool)
        groups = []
        for start in range(len(features)):
            if visited[start]:
                continue
            queue = [start]
            visited[start] = True
            head = 0
            while head < len(queue):
                current = queue[head]
                head += 1
                for nb in tree.query_ball_point(features[current], 0.22):
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            if len(queue) >= 5:
                local = np.asarray(queue, dtype=np.int64)
                median = np.median(features[local], axis=0)
                representative = local[
                    np.argmin(np.linalg.norm(features[local] - median, axis=1))
                ]
                groups.append({
                    "idx": int(candidate_idx[representative]),
                    "n": len(queue),
                    "u": float(median[0]),
                    "z": float(median[1]),
                })
        return groups

    def _three_span_seeds(self, p0, dxy, length, anchor, top_path):
        normal = np.array([-dxy[1], dxy[0]])
        anchor_lateral = float((anchor[:2] - p0) @ normal)
        top_points = self._cable_points[list(top_path)]
        best = None

        for fraction in np.linspace(0.18, 0.82, 9):
            center = p0 + dxy * (fraction * length)
            groups = self._cross_section_groups(center, dxy, anchor_lateral)
            if len(groups) < 3:
                continue

            top_rel = top_points[:, :2] - center
            top_along = top_rel @ dxy
            near_top = np.abs(top_along) <= 2.0
            if np.any(near_top):
                top_u = np.median(top_rel[near_top] @ normal)
                top_z = np.median(top_points[near_top, 2])
            else:
                nearest = int(np.argmin(np.abs(top_along)))
                top_u = float(top_rel[nearest] @ normal)
                top_z = float(top_points[nearest, 2])
            top_feature = np.array([top_u, top_z])

            features = np.array([[g["u"], g["z"]] for g in groups])
            l1_index = int(np.argmin(np.linalg.norm(features - top_feature, axis=1)))
            l1 = groups[l1_index]
            other_indices = [i for i in range(len(groups)) if i != l1_index]
            other_indices.sort(
                key=lambda i: np.linalg.norm(features[i] - features[l1_index])
            )
            chosen = [l1, groups[other_indices[0]], groups[other_indices[1]]]
            chosen_features = np.array([[g["u"], g["z"]] for g in chosen])
            pair_distances = [
                np.linalg.norm(chosen_features[i] - chosen_features[j])
                for i in range(3) for j in range(i + 1, 3)
            ]
            min_separation = float(min(pair_distances))
            l1_error = float(np.linalg.norm(chosen_features[0] - top_feature))
            score = (
                min_separation
                - 0.35 * l1_error
                + 0.0005 * sum(g["n"] for g in chosen)
            )
            if best is None or score > best[0]:
                best = (score, chosen)

        if best is None:
            return []

        chosen = best[1]
        l1_feature = np.array([chosen[0]["u"], chosen[0]["z"]])
        chosen[1:] = sorted(
            chosen[1:],
            key=lambda g: np.linalg.norm(np.array([g["u"], g["z"]]) - l1_feature),
        )
        return chosen

    def _complete_tracks_in_span(self, tracks, p0, dxy, length):
        normal = np.array([-dxy[1], dxy[0]])
        models = []
        for indices in tracks:
            pts = self._cable_points[indices]
            rel = pts[:, :2] - p0
            t = rel @ dxy
            u = rel @ normal
            s = t / max(length, 1e-6)
            keep = (s >= 0.03) & (s <= 0.97)
            if np.count_nonzero(keep) < 10:
                keep = np.ones(len(s), dtype=bool)
            s_fit, u_fit, z_fit = s[keep], u[keep], pts[keep, 2]

            mask = np.ones(len(s_fit), dtype=bool)
            for _ in range(3):
                degree = 2 if np.count_nonzero(mask) >= 6 else 1
                cu = np.polyfit(s_fit[mask], u_fit[mask], degree)
                cz = np.polyfit(s_fit[mask], z_fit[mask], degree)
                residual = np.hypot(
                    u_fit - np.polyval(cu, s_fit),
                    z_fit - np.polyval(cz, s_fit),
                )
                new_mask = residual <= max(0.18, float(np.percentile(residual, 85)))
                if np.array_equal(mask, new_mask) or np.count_nonzero(new_mask) < 6:
                    break
                mask = new_mask
            models.append((cu, cz))

        all_rel = self._cable_points[:, :2] - p0
        all_t = all_rel @ dxy
        all_u = all_rel @ normal
        candidates = np.flatnonzero(
            self._active_cable & (all_t >= -0.5) & (all_t <= length + 0.5)
        )
        s_all = all_t[candidates] / max(length, 1e-6)
        distances = np.empty((len(candidates), 3), dtype=np.float64)
        for sid, (cu, cz) in enumerate(models):
            distances[:, sid] = np.hypot(
                all_u[candidates] - np.polyval(cu, s_all),
                self._cable_points[candidates, 2] - np.polyval(cz, s_all),
            )

        nearest = np.argmin(distances, axis=1)
        minimum = distances[np.arange(len(candidates)), nearest]
        completed = []
        for sid in range(3):
            idx = candidates[(nearest == sid) & (minimum <= 0.24)]
            completed.append(idx.astype(np.int64))
        return completed

    def paint_points(self, points_world, slot_id, radius=0.18):
        if slot_id not in (0, 1, 2) or not points_world:
            return {"ok": False, "message": "Empty paint stroke.", "slot_id": slot_id}
        painted = set(map(int, self.slot_points.get(slot_id, [])))
        for point in np.asarray(points_world, dtype=np.float64):
            for idx in self._kdtree.query_ball_point(point, radius):
                if self._active_cable[int(idx)]:
                    painted.add(int(idx))
        accepted = np.asarray(sorted(painted), dtype=np.int64)
        if len(accepted) == 0:
            return {
                "ok": False,
                "message": "No active cables under the brush.",
                "slot_id": slot_id,
            }
        self.slot_points[slot_id] = accepted.tolist()
        pts = self._cable_points[accepted]
        centered = pts - pts.mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            projection = centered @ vt[0]
            length = float(np.ptp(projection))
        except Exception:
            length = 0.0
        points_norm = (pts - self._centroid).astype(np.float32)
        return {
            "ok": True,
            "slot_id": slot_id,
            "n_total": int(len(accepted)),
            "n_path": int(len(accepted)),
            "n_viz": int(len(points_norm)),
            "spine_length_m": round(length, 1),
            "pts_per_m": round(len(accepted) / max(length, 0.001), 1),
            "perp_radius": radius,
            "points_flat": points_norm.flatten().tolist(),
        }

    def paint_indices(self, indices, slot_id):
        if slot_id not in (0, 1, 2) or not indices:
            return {"ok": False, "message": "Empty paint stroke.", "slot_id": slot_id}
        incoming = np.asarray(indices, dtype=np.int64)
        incoming = incoming[(incoming >= 0) & (incoming < len(self._cable_points))]
        incoming = incoming[self._active_cable[incoming]]
        painted = set(map(int, self.slot_points.get(slot_id, [])))
        painted.update(map(int, incoming))
        accepted = np.asarray(sorted(painted), dtype=np.int64)
        if len(accepted) == 0:
            return {
                "ok": False,
                "message": "No active cables under the brush.",
                "slot_id": slot_id,
            }
        self.slot_points[slot_id] = accepted.tolist()
        pts = self._cable_points[accepted]
        centered = pts - pts.mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            length = float(np.ptp(centered @ vt[0]))
        except Exception:
            length = 0.0
        points_norm = (pts - self._centroid).astype(np.float32)
        return {
            "ok": True,
            "slot_id": slot_id,
            "n_total": int(len(accepted)),
            "n_path": int(len(accepted)),
            "n_viz": int(len(points_norm)),
            "spine_length_m": round(length, 1),
            "pts_per_m": round(len(accepted) / max(length, 0.001), 1),
            "perp_radius": 0.0,
            "points_flat": points_norm.flatten().tolist(),
        }

    def select_three_in_span(self, seeds_world, perp_r=0.12, eps=0.60, gap=3.0, mom=0.85, pole_ids=None):
        if pole_ids is not None:
            if not isinstance(pole_ids, list) or len(pole_ids) != 2:
                return {"ok": False, "message": "Select exactly two different poles."}
            lookup = {int(p["pid"]): i for i, p in enumerate(self._poles)}
            try:
                pole0, pole1 = (lookup[int(x)] for x in pole_ids)
            except (KeyError, ValueError, TypeError):
                return {"ok": False, "message": "Selected pole not found in raw metadata."}
            if pole0 == pole1:
                return {"ok": False, "message": "Select two different poles."}
            p0 = np.array([self._poles[pole0]["cx"], self._poles[pole0]["cy"]]) + self._centroid[:2]
            p1 = np.array([self._poles[pole1]["cx"], self._poles[pole1]["cy"]]) + self._centroid[:2]
            chord = p1 - p0
            length = float(np.linalg.norm(chord))
            if length < 5.0:
                return {"ok": False, "message": "Pole span is shorter than 5 m."}
            dxy = chord / length
            normal = np.array([-dxy[1], dxy[0]])
            rel = self._cable_points[:, :2] - p0
            along = rel @ dxy
            lateral = rel @ normal
            # Only the actual corridor between the explicitly selected supports.
            eligible = np.flatnonzero(self._active_cable & (along >= 0.05 * length)
                & (along <= 0.95 * length) & (np.abs(lateral) <= 3.0))
            if len(eligible) < 30:
                return {"ok": False, "message": "Insufficient cable points between selected poles."}
            bins = np.histogram(along[eligible], bins=np.linspace(0, length, 11))[0]
            if np.count_nonzero(bins > 0) / len(bins) < SPAN_MIN_TRACE_COVERAGE:
                return {"ok": False, "message": "Cable is not continuous enough between these poles; use manual selection."}
            middle = eligible[np.abs(along[eligible] - length * 0.5).argmin()]
            # Prefer an upper conductor at the middle cross-section.
            near = eligible[np.abs(along[eligible] - length * 0.5) <= 1.5]
            if len(near):
                anchor_idx = int(near[np.argmax(self._cable_points[near, 2])])
            else:
                anchor_idx = int(middle)
            anchor = self._cable_points[anchor_idx]
            direction = np.array([dxy[0], dxy[1], 0.0])
        else:
            if not seeds_world:
                return {"ok": False, "message": "Select a point on L1."}
            seeds = np.asarray(seeds_world, dtype=np.float64)
            _, anchor_idxs = self._kdtree.query(seeds)
            anchor_idx = int(np.atleast_1d(anchor_idxs)[-1])
            anchor = self._cable_points[anchor_idx]
            anchor_pts = self._cable_points[np.atleast_1d(anchor_idxs)]
            direction = self._estimate_direction(anchor_pts, np.atleast_1d(anchor_idxs), eps)
            preliminary_path = {anchor_idx}
            for sign in (1.0, -1.0):
                preliminary_path.update(self._trace_centerline(
                    anchor_idx, sign * direction, perp_r, eps, gap, mom))
            span = self._bounding_poles(anchor, direction, preliminary_path)
            if span is None:
                return {"ok": False, "message": "No valid pole found on each side of L1."}
            p0, p1, dxy, length, pole0, pole1 = span

        top_path = {anchor_idx}
        for sign in (1.0, -1.0):
            top_path.update(
                self._trace_centerline(
                    anchor_idx, sign * direction, perp_r, eps, gap, mom
                )
            )
        seed_groups = self._three_span_seeds(p0, dxy, length, anchor, top_path)
        if len(seed_groups) != 3:
            return {
                "ok": False,
                "message": "Could not distinguish three lines in this span; use manual mode.",
            }

        direction3 = np.array([dxy[0], dxy[1], 0.0])
        normal = np.array([-dxy[1], dxy[0]])
        initial_tracks = []
        path_lengths = []
        for group in seed_groups:
            seed_idx = group["idx"]
            path = {seed_idx}
            for sign in (1.0, -1.0):
                path.update(
                    self._trace_centerline(
                        seed_idx, sign * direction3, perp_r, eps, gap, mom
                    )
                )
            path_list = list(path)
            accepted = self._collect_near_path(path_list, direction3, perp_r)

            rel = self._cable_points[accepted, :2] - p0
            along = rel @ dxy
            lateral = rel @ normal
            keep = (
                (along >= -0.5)
                & (along <= length + 0.5)
                & (np.abs(lateral - group["u"]) <= 2.0)
            )
            accepted = accepted[keep & self._active_cable[accepted]]
            if len(accepted) < 10:
                return {
                    "ok": False,
                    "message": "A line ended up with very few points inside the span.",
                }
            initial_tracks.append(accepted)
            path_lengths.append(len(path_list))

        completed_tracks = self._complete_tracks_in_span(
            initial_tracks, p0, dxy, length
        )
        selections = []
        lines_info = []

        za_list, zb_list = [], []
        for sid, accepted in enumerate(completed_tracks):
            pts = self._cable_points[accepted]
            rel = pts[:, :2] - p0
            along = rel @ dxy
            near_a = pts[along <= 0.35 * length]
            near_b = pts[along >= 0.65 * length]
            za = (
                float(np.median(near_a[:, 2]))
                if len(near_a)
                else float(pts[:, 2].mean())
            )
            zb = (
                float(np.median(near_b[:, 2]))
                if len(near_b)
                else float(pts[:, 2].mean())
            )
            za_list.append(za)
            zb_list.append(zb)

        order_a = np.argsort(-np.array(za_list))
        rank_a = np.empty(3, dtype=int)
        rank_a[order_a] = [0, 1, 2]

        order_b = np.argsort(-np.array(zb_list))
        rank_b = np.empty(3, dtype=int)
        rank_b[order_b] = [0, 1, 2]

        role_map_std = {0: "S", 1: "M", 2: "I"}
        role_map_ese = {0: "E", 1: "S", 2: "E"}
        role_map_eme = {0: "E", 1: "M", 2: "E"}

        for sid, (group, accepted) in enumerate(zip(seed_groups, completed_tracks)):
            if len(accepted) < 10:
                return {
                    "ok": False,
                    "message": f"Could not complete L{sid + 1} inside the span.",
                }

            self.slot_points[sid] = accepted.tolist()
            points_norm = (self._cable_points[accepted] - self._centroid).astype(
                np.float32
            )
            if len(points_norm) > MAX_VIZ_PTS:
                points_norm = points_norm[:: max(1, len(points_norm) // MAX_VIZ_PTS)]
            seed_idx = group["idx"]
            seed_world = self._cable_points[seed_idx]

            ra = int(rank_a[sid])
            rb = int(rank_b[sid])
            code_std = role_map_std[ra] + role_map_std[rb]
            code_ese = role_map_std[ra] + role_map_ese[rb]
            code_eme = role_map_std[ra] + role_map_eme[rb]

            info = {
                "slot_id": sid,
                "za": round(za_list[sid], 2),
                "zb": round(zb_list[sid], 2),
                "rank_a": ra,
                "rank_b": rb,
                "role_a": role_map_std[ra],
                "role_b_std": role_map_std[rb],
                "role_b_ese": role_map_ese[rb],
                "role_b_eme": role_map_eme[rb],
                "code_std": code_std,
                "code_ese": code_ese,
                "code_eme": code_eme,
            }
            lines_info.append(info)

            selections.append({
                "slot_id": sid,
                "ok": True,
                "n_total": int(len(accepted)),
                "n_path": int(path_lengths[sid]),
                "n_viz": int(len(points_norm)),
                "spine_length_m": round(length, 1),
                "pts_per_m": round(len(accepted) / max(length, 0.001), 1),
                "perp_radius": perp_r,
                "snap_seeds": [(seed_world - self._centroid).tolist()],
                "seed_world": seed_world.tolist(),
                "points_flat": points_norm.flatten().tolist(),
                "conn_code": code_std,
                "conn_info": info,
            })

        p_name_0 = self._poles[pole0]["name"]
        p_name_1 = self._poles[pole1]["name"]

        return {
            "ok": True,
            "selections": selections,
            "span_length_m": round(length, 1),
            "pole_indices": [pole0, pole1],
            "span_poles": [p_name_0, p_name_1],
            "lines_info": lines_info,
        }

    def reconstruct_cables(self):
        catenaries = []
        poles_abs_xy = (
            np.empty((0, 2))
            if not self._poles
            else np.array([
                [p["cx"] + self._centroid[0], p["cy"] + self._centroid[1]]
                for p in self._poles
            ])
        )

        for sid in range(3):
            idx_list = self.slot_points.get(sid, [])
            if len(idx_list) < 10:
                continue

            Pts = self._cable_points[idx_list]
            mean_pt = Pts.mean(axis=0)
            xy = Pts[:, :2] - mean_pt[:2]
            try:
                _, _, Vt = np.linalg.svd(xy, full_matrices=False)
                d_xy = Vt[0]
                dir_vec = np.array([d_xy[0], d_xy[1], 0.0])
                dir_vec /= np.linalg.norm(dir_vec)
            except Exception:
                dir_vec = np.array([1.0, 0.0, 0.0])

            t = np.dot(Pts - mean_pt, dir_vec)
            t_start, t_end = t.min(), t.max()
            pole_start_idx, pole_end_idx = -1, -1

            if len(poles_abs_xy) > 0:
                p_min, p_max = Pts[np.argmin(t)], Pts[np.argmax(t)]
                d_min = np.linalg.norm(poles_abs_xy - p_min[:2], axis=1)
                d_max = np.linalg.norm(poles_abs_xy - p_max[:2], axis=1)
                i_min, i_max = np.argmin(d_min), np.argmin(d_max)

                if d_min[i_min] < 15.0:
                    t_start = np.dot(
                        np.array([
                            poles_abs_xy[i_min][0],
                            poles_abs_xy[i_min][1],
                            0.0,
                        ]) - mean_pt,
                        dir_vec,
                    )
                    pole_start_idx = int(i_min)
                if d_max[i_max] < 15.0:
                    t_end = np.dot(
                        np.array([
                            poles_abs_xy[i_max][0],
                            poles_abs_xy[i_max][1],
                            0.0,
                        ]) - mean_pt,
                        dir_vec,
                    )
                    pole_end_idx = int(i_max)
                if t_start > t_end:
                    t_start, t_end = t_end, t_start
                    pole_start_idx, pole_end_idx = pole_end_idx, pole_start_idx

            A = np.vstack([t ** 2, t, np.ones(len(t))]).T
            cz, _, _, _ = np.linalg.lstsq(A, Pts[:, 2], rcond=None)
            cy, _, _, _ = np.linalg.lstsq(A, Pts[:, 1], rcond=None)
            cx, _, _, _ = np.linalg.lstsq(A, Pts[:, 0], rcond=None)

            t_lin = np.linspace(t_start, t_end, 200)
            fit_x = cx[0] * t_lin ** 2 + cx[1] * t_lin + cx[2]
            fit_y = cy[0] * t_lin ** 2 + cy[1] * t_lin + cy[2]
            fit_z = cz[0] * t_lin ** 2 + cz[1] * t_lin + cz[2]
            curve_pts = np.column_stack([fit_x, fit_y, fit_z])

            if pole_start_idx != -1:
                key = (sid, pole_start_idx)
                if key in self.pole_terminals:
                    curve_pts += (
                        self.pole_terminals[key] - curve_pts[0]
                    ) * np.linspace(1.0, 0.0, 200).reshape(-1, 1)
                else:
                    self.pole_terminals[key] = curve_pts[0].copy()

            if pole_end_idx != -1:
                key = (sid, pole_end_idx)
                if key in self.pole_terminals:
                    curve_pts += (
                        self.pole_terminals[key] - curve_pts[-1]
                    ) * np.linspace(0.0, 1.0, 200).reshape(-1, 1)
                else:
                    self.pole_terminals[key] = curve_pts[-1].copy()

            curve_norm = (curve_pts - self._centroid).astype(np.float32)
            catenaries.append({
                "slot_id": sid,
                "points_flat": curve_norm.flatten().tolist(),
            })

            self.slot_points[sid] = []

        return {"ok": True, "catenaries": catenaries}

    def label_slots(self, slot_ids, custom_labels=None):
        labeled = []
        removed = set()
        for sid in slot_ids:
            local = np.asarray(self.slot_points.get(sid, []), dtype=np.int64)
            if len(local):
                local = local[self._active_cable[local]]
            if len(local) == 0:
                continue
            if custom_labels and sid in custom_labels:
                new_label = int(custom_labels[sid])
            elif hasattr(self, "slot_labels") and sid in self.slot_labels:
                new_label = int(self.slot_labels[sid])
            else:
                new_label = SLOT_LABEL_IDS[sid]
            global_idx = self._cable_global_indices[local]
            self._all_labels[global_idx] = new_label
            self._active_cable[local] = False
            removed.update(map(int, local))
            labeled.append({
                "slot_id": sid,
                "line_index": sid + 1,
                "label": new_label,
                "n_pts": int(len(local)),
            })

        if removed:
            for sid in range(3):
                self.slot_points[sid] = [
                    i for i in self.slot_points.get(sid, []) if int(i) not in removed
                ]
        if labeled:
            _write_changes(self)
        return labeled, sorted(removed)

    def rotate_slots(self, order=(2, 0, 1)):
        p0 = list(self.slot_points.get(order[0], []))
        p1 = list(self.slot_points.get(order[1], []))
        p2 = list(self.slot_points.get(order[2], []))
        self.slot_points[0] = p0
        self.slot_points[1] = p1
        self.slot_points[2] = p2

    def init_payload(self):
        poles_data = [
            {
                "id": i,
                "pid": int(p["pid"]),
                "name": p["name"],
                "cx": float(p["cx"]),
                "cy": float(p["cy"]),
                "z_top": float(p["z_top"]),
                "z_bottom": float(p["z_bottom"]),
                "s_m": float(p.get("s_m", 0.0)),
            }
            for i, p in enumerate(self._poles)
        ]
        return json.dumps({
            "type": "init",
            "centroid": self._centroid.tolist(),
            "n_cable_pts": int(len(self._cable_points)),
            "n_pole_pts": self._n_pole_pts,
            "slot_colors": SLOT_COLORS,
            "cable_preview": self._cable_preview_flat,
            "inactive_cable_indices": np.flatnonzero(~self._active_cable).tolist(),
            "has_odometry": bool(self._has_s_odom),
            "poles": poles_data,
            "slot_label_ids": SLOT_LABEL_IDS,
            "label_poles_mt": POLE_LABEL,
            "label_wires_mt": CABLE_LABEL,
        })


_server = None
_shutdown_event = None


async def handle_client(websocket, path=None):
    addr = getattr(websocket, "remote_address", "?")
    print(f"[INFO] [CABLE_MT] WebSocket client connected: {addr}")
    try:
        payload = _server.init_payload()
        print(
            f"[INFO] [CABLE_MT] Initial payload: {len(payload) / 1024:.0f} KB  "
            f"({len(_server._poles)} poles, {len(_server._cable_points):,} cable pts)"
        )
        await websocket.send(payload)

        async for raw in websocket:
            try:
                msg = json.loads(raw)
                t = msg.get("type")

                if t == "select_seeds":
                    sid = int(msg.get("slot_id", 0))
                    t0 = time.time()
                    res = _server.select_by_seeds(
                        msg["seeds"],
                        sid,
                        float(msg.get("perp_radius", 0.12)),
                        float(msg.get("eps_grow", 0.60)),
                        float(msg.get("gap_radius", 3.0)),
                        float(msg.get("momentum", 0.85)),
                    )
                    await websocket.send(
                        json.dumps({"type": "selection", "slot_id": sid, **res})
                    )
                    if res["ok"]:
                        print(
                            f"[INFO] [CABLE_MT] Slot {sid + 1}: {res['n_total']:,} pts  "
                            f"spine={res['spine_length_m']}m "
                            f"({res['pts_per_m']} pts/m) {time.time() - t0:.2f}s"
                        )
                    else:
                        print(f"[ERROR] [CABLE_MT] Slot {sid + 1} failed: {res['message']}")

                elif t == "auto_select_span":
                    _server.slot_labels = dict(SLOT_LABEL_IDS)
                    t0 = time.time()
                    res = _server.select_three_in_span(
                        msg["seeds"],
                        float(msg.get("perp_radius", 0.12)),
                        float(msg.get("eps_grow", 0.60)),
                        float(msg.get("gap_radius", 3.0)),
                        float(msg.get("momentum", 0.85)),
                        pole_ids=msg.get("pole_ids"),
                    )
                    await websocket.send(json.dumps({"type": "auto_selection", **res}))
                    if res["ok"]:
                        counts = [s["n_total"] for s in res["selections"]]
                        print(
                            f"[INFO] [CABLE_MT] Automatically detected span: {res['span_length_m']} m | "
                            f"L1={counts[0]:,}, L2={counts[1]:,}, L3={counts[2]:,} pts "
                            f"({time.time() - t0:.2f}s)"
                        )
                    else:
                        print(f"[ERROR] [CABLE_MT] Automatic span processing failed: {res['message']}")

                elif t == "paint_indices":
                    sid = int(msg.get("slot_id", 0))
                    res = _server.paint_indices(msg.get("indices", []), sid)
                    await websocket.send(json.dumps({"type": "selection", **res}))

                elif t == "paint_points":
                    sid = int(msg.get("slot_id", 0))
                    res = _server.paint_points(
                        msg.get("points", []), sid, float(msg.get("radius", 0.18))
                    )
                    await websocket.send(json.dumps({"type": "selection", **res}))

                elif t == "reconstruct":
                    res = _server.reconstruct_cables()
                    await websocket.send(json.dumps({"type": "reconstructed", **res}))
                    print(
                        f"[INFO] [CABLE_MT] Reconstructed catenaries: {len(res.get('catenaries', []))}"
                    )

                elif t == "label_line":
                    sid = int(msg.get("slot_id", 0))
                    custom_lbl = msg.get("label")
                    c_dict = {sid: int(custom_lbl)} if custom_lbl is not None else None
                    labeled, removed = _server.label_slots([sid], custom_labels=c_dict)
                    if labeled:
                        item = labeled[0]
                        await websocket.send(
                            json.dumps({
                                "type": "labeled",
                                **item,
                                "removed_indices": removed,
                            })
                        )
                        print(
                            f"[INFO] [CABLE_MT] Assigned labels to {item['n_pts']:,} points "
                            f"to Label {item['label']} (Line {item['line_index']})."
                        )
                    else:
                        await websocket.send(
                            json.dumps({
                                "type": "error",
                                "message": "No traced points in this slot.",
                            })
                        )

                elif t == "label_all_lines":
                    custom_lbls = msg.get("labels")
                    c_dict = (
                        {int(k): int(v) for k, v in custom_lbls.items()}
                        if custom_lbls
                        else None
                    )
                    labeled, removed = _server.label_slots(
                        [0, 1, 2], custom_labels=c_dict
                    )
                    if labeled:
                        await websocket.send(
                            json.dumps({
                                "type": "labeled_all",
                                "lines": labeled,
                                "removed_indices": removed,
                            })
                        )
                        summary = ", ".join(
                            f"L{x['line_index']}={x['n_pts']:,} (Label {x['label']})"
                            for x in labeled
                        )
                        print(f"[INFO] [CABLE_MT] Labeling summary: {summary}")
                    else:
                        await websocket.send(
                            json.dumps({
                                "type": "error",
                                "message": "No traced lines to label.",
                            })
                        )

                elif t == "rotate_slots":
                    order = msg.get("order", [2, 0, 1])
                    _server.rotate_slots(order)
                    _server.slot_labels = dict(SLOT_LABEL_IDS)
                    print(
                        f"[INFO] [CABLE_MT] Slot rotation applied: L1 <- Slot {order[0] + 1}, "
                        f"L2 <- Slot {order[1] + 1}, L3 <- Slot {order[2] + 1}"
                    )

                elif t == "set_slots_mode":
                    order = msg.get("order", [0, 1, 2])
                    _server.rotate_slots(order)
                    _server.slot_labels = dict(SLOT_LABEL_IDS)
                    print(
                        f"[INFO] [CABLE_MT] Down-arrow mode applied: {order}; "
                        f"labels kept as L1/L2/L3 (L4 removed)."
                    )

                elif t == "reset_slot_labels":
                    _server.slot_labels = dict(SLOT_LABEL_IDS)

                elif t == "finalize":
                    t_exp = time.time()
                    html_report = msg.get("html_report", {})
                    extractor_report = msg.get("extractor_report", [])

                    ply_out = _output_ply_path(_server.ply_path)
                    ply_dir = os.path.dirname(ply_out)
                    if ply_dir:
                        os.makedirs(ply_dir, exist_ok=True)
                    from cable_mt.rebuild import rebuild_ply
                    changes_path = _write_changes(_server)
                    rebuild_ply(_server.ply_path, changes_path, ply_out)
                    _server._ply_exported = True
                    print(f"[INFO] [CABLE_MT] Labeled PLY exported to: {ply_out}")

                    html_path = _output_json_html_path()
                    html_dir = os.path.dirname(html_path)
                    if html_dir:
                        os.makedirs(html_dir, exist_ok=True)
                    with open(html_path, "w", encoding="utf-8") as f:
                        json.dump(html_report, f, indent=2, ensure_ascii=False)
                    print(f"[INFO] [CABLE_MT] HTML report exported to: {html_path}")

                    ext_path = _output_json_extractor_path()
                    ext_dir = os.path.dirname(ext_path)
                    if ext_dir:
                        os.makedirs(ext_dir, exist_ok=True)
                    with open(ext_path, "w", encoding="utf-8") as f:
                        json.dump(extractor_report, f, indent=2, ensure_ascii=False)
                    print(f"[INFO] [CABLE_MT] Extractor report exported to: {ext_path}")
                    state_path = _state_path()
                    os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
                    with open(state_path, "w", encoding="utf-8") as f:
                        json.dump({"schema_version": 1, "spans": msg.get("spans", []),
                                   "html_report": html_report, "extractor_report": extractor_report},
                                  f, indent=2, ensure_ascii=False)
                    print(f"[INFO] [CABLE_MT] Recovery changes: {changes_path}")

                    print(f"[INFO] [CABLE_MT] Finalization completed in {time.time() - t_exp:.2f}s")

                    await websocket.send(json.dumps({
                        "type": "finalize_ready",
                        "ply_path": os.path.abspath(ply_out),
                        "html_path": os.path.abspath(html_path),
                        "extractor_path": os.path.abspath(ext_path),
                        "ply_filename": os.path.basename(ply_out),
                        "html_filename": os.path.basename(html_path),
                        "extractor_filename": os.path.basename(ext_path),
                        "changes_path": os.path.abspath(changes_path),
                        "state_path": os.path.abspath(state_path),
                    }))

                    if _shutdown_event is not None:
                        await asyncio.sleep(0.4)
                        _shutdown_event.set()

                elif t == "clear_all":
                    _server.pole_terminals.clear()
                    for i in range(3):
                        _server.slot_points[i] = []
                    _server.slot_labels = dict(SLOT_LABEL_IDS)
                    print('[INFO] [CABLE_MT] Reconstruction cache cleared.')

                elif t == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))

            except (json.JSONDecodeError, KeyError) as e:
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print(f"[INFO] [CABLE_MT] WebSocket client disconnected: {addr}")


def _run_http(server, port):
    HTTPServer(("0.0.0.0", port), partial(PLYHTTPHandler, server)).serve_forever()


def _find_default_ply():
    return _configured_file('data_dir', 'input_ply_poles')


def _open_viewer(viewer_name: str = "cable_viewer.html") -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, viewer_name),
        os.path.join(os.getcwd(), viewer_name),
        viewer_name,
    ]
    viewer_path = next((p for p in candidates if os.path.exists(p)), None)

    if viewer_path is None:
        print(f"[WARNING] [CABLE_MT] Viewer file not found: {viewer_name}; open it manually.")
        return

    url = "file://" + os.path.abspath(viewer_path)
    try:
        webbrowser.open_new_tab(url)
        print(f"[INFO] [CABLE_MT] Viewer URL: {url}")
    except Exception as e:
        print(f"[WARNING] [CABLE_MT] Unable to open browser automatically: {e}")
        print(f"[INFO] [CABLE_MT] Open viewer manually at: {url}")


async def main():
    global _server, _shutdown_event
    args = sys.argv[1:]
    if "--rebuild" in args:
        from cable_mt.rebuild import rebuild_from_config
        print("[INFO] [CABLE_MT] Rebuilt:", rebuild_from_config(CONFIG))
        return
    ply = None
    wsp = 8765
    http = 8766

    i = 0
    rem_args = []
    while i < len(args):
        if args[i] in ("--ply", "-p") and i + 1 < len(args):
            ply = args[i + 1]
            i += 2
        else:
            rem_args.append(args[i])
            i += 1

    for a in rem_args:
        if a.endswith(".ply"):
            ply = a
        elif a.isdigit():
            if wsp == 8765:
                wsp = int(a)
            else:
                http = int(a)
        elif os.path.exists(a) and "ply" in a.lower():
            ply = a

    if ply is None:
        ply = _find_default_ply()

    if not ply or not os.path.exists(ply):
        print(f"[ERROR] [CABLE_MT] Input PLY file not found: {ply}")
        print(__doc__)
        sys.exit(1)

    print(f"[INFO] [CABLE_MT] Input point cloud: {ply}")

    _server = CableLineServer(ply)
    _shutdown_event = asyncio.Event()

    threading.Thread(target=_run_http, args=(_server, http), daemon=True).start()
    print(f"[INFO] [CABLE_MT] WebSocket endpoint: ws://localhost:{wsp}")
    print(f"[INFO] [CABLE_MT] HTTP endpoint: http://localhost:{http}/ply")

    def _delayed_open():
        time.sleep(0.6)
        _open_viewer()

    threading.Thread(target=_delayed_open, daemon=True).start()

    async with websockets.serve(handle_client, "0.0.0.0", wsp):
        await _shutdown_event.wait()


if __name__ == "__main__":
    asyncio.run(main())