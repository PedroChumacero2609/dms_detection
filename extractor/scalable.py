"""Disk-backed grouping helpers for large labelled PLY point clouds.

These helpers deliberately avoid creating a full (n_points, 3) copy per pole.
All scratch files are created inside the temporary working directory.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def cluster_statistics(points: np.ndarray, labels: np.ndarray, ids: list[int],
                       chunk_size: int = 200_000):
    """Compute count, mean XY and lowest Z by cluster, one chunk at a time."""
    keys = np.asarray(ids, dtype=np.int64)
    counts = np.zeros(len(keys), dtype=np.int64)
    x_sums = np.zeros(len(keys), dtype=np.float64)
    y_sums = np.zeros(len(keys), dtype=np.float64)
    z_min = np.full(len(keys), np.inf, dtype=np.float64)
    z_max = np.full(len(keys), -np.inf, dtype=np.float64)

    if not len(keys):
        return counts, np.empty((0, 2)), z_min, z_max

    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        lab = labels[start:stop]
        pos = np.searchsorted(keys, lab)
        safe = np.clip(pos, 0, len(keys) - 1)
        use = (lab >= 0) & (pos < len(keys)) & (keys[safe] == lab)
        if not np.any(use):
            continue
        p = points[start:stop][use]
        idx = pos[use]
        counts += np.bincount(idx, minlength=len(keys))
        x_sums += np.bincount(idx, weights=p[:, 0], minlength=len(keys))
        y_sums += np.bincount(idx, weights=p[:, 1], minlength=len(keys))
        np.minimum.at(z_min, idx, p[:, 2])
        np.maximum.at(z_max, idx, p[:, 2])

    if np.any(counts == 0):
        raise ValueError("One or more selected clusters are empty.")
    centers = np.column_stack((x_sums / counts, y_sums / counts))
    return counts, centers, z_min, z_max


def group_points_to_disk(points: np.ndarray, labels: np.ndarray, ids: list[int],
                         counts: np.ndarray, scratch_dir: Path,
                         chunk_size: int = 200_000):
    """Return a disk-mapped contiguous array and [start,end) offsets by cluster.

    Only clusters requested in ids are retained. Peak in-memory scratch is
    bounded by chunk_size, rather than by the total PLY size.
    """
    keys = np.asarray(ids, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    total = int(offsets[-1])
    if not total:
        raise ValueError("No points available for grouping.")
    path = Path(scratch_dir) / "cluster_points.npy"
    dest = np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float64,
                                     shape=(total, 3))
    cursor = offsets[:-1].copy()
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        lab = labels[start:stop]
        pos = np.searchsorted(keys, lab)
        safe = np.clip(pos, 0, len(keys) - 1)
        use = (lab >= 0) & (pos < len(keys)) & (keys[safe] == lab)
        if not np.any(use):
            continue
        bucket = pos[use]
        selected = points[start:stop][use]
        order = np.argsort(bucket, kind="stable")
        bucket = bucket[order]
        selected = selected[order]
        bounds = np.flatnonzero(np.diff(bucket)) + 1
        bounds = np.concatenate(([0], bounds, [len(bucket)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            which = int(bucket[a])
            to = int(cursor[which])
            dest[to:to + (b - a)] = selected[a:b]
            cursor[which] += b - a
    if not np.array_equal(cursor, offsets[1:]):
        raise RuntimeError("Failed to group pole points in the temporary file.")
    dest.flush()
    return dest, offsets
