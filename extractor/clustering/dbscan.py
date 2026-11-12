from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

EPS_CLUSTER = 2.0
MIN_POINTS_CLUSTER = 50
FILTER_MIN_POINTS = 100
FILTER_MIN_HEIGHT = 5.0
MAX_DBSCAN_POINTS = 500_000
DBSCAN_VOXEL_SIZE = 0.5
MAX_DBSCAN_VOXELS = 50_000


def cluster_poles(
    points: np.ndarray,
    eps: float = EPS_CLUSTER,
    min_points: int = MIN_POINTS_CLUSTER,
    max_dbscan_points: int = MAX_DBSCAN_POINTS,
    chunk_size: int = 200_000,
    scratch_dir: Path | None = None,
    voxel_size: float = DBSCAN_VOXEL_SIZE,
    max_dbscan_voxels: int = MAX_DBSCAN_VOXELS,
) -> np.ndarray:
    """Run Open3D DBSCAN on a bounded input, map labels back in chunks.

    For small files the original exact DBSCAN path is preserved. For larger
    selected point sets, a deterministic systematic sample is clustered and
    the remaining points are attached to the nearest non-noise sample within
    eps. This approximation may miss rare
    clusters whose members were inadequately represented in the sample.
    """
    n = len(points)
    if n < min_points:
        raise ValueError("Not enough points to form a cluster.")
    max_dbscan_points = int(max_dbscan_points)
    chunk_size = int(chunk_size)
    if max_dbscan_points < 1000 or chunk_size < 1000:
        raise ValueError("max_dbscan_points and chunk_size must be >= 1000.")

    if voxel_size <= 0 or max_dbscan_voxels < 1000:
        raise ValueError("voxel_size must be > 0 and max_dbscan_voxels >= 1000.")

    # Dense clusters are especially problematic for Open3D DBSCAN: the library
    # may precompute a quadratic number of neighbors. Voxel compression limits
    # both point count and local neighborhood density before running DBSCAN.
    stride = max(1, int(np.ceil(n / max_dbscan_points)))
    sample = np.ascontiguousarray(points[::stride], dtype=np.float64)
    voxelized = len(sample) > 5_000
    if voxelized:
        grid = np.floor(sample / voxel_size).astype(np.int64)
        _, first = np.unique(grid, axis=0, return_index=True)
        first.sort()
        representatives = sample[first]
        if len(representatives) > max_dbscan_voxels:
            step_voxel = int(np.ceil(len(representatives) / max_dbscan_voxels))
            representatives = representatives[::step_voxel]
        density_ratio = len(representatives) / len(sample)
        effective_min = max(3, int(round(min_points / stride * density_ratio)))
    else:
        representatives = sample
        effective_min = min_points if stride == 1 else max(3, int(round(min_points / stride)))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(representatives)
    try:
        sample_ids = np.asarray(
            cloud.cluster_dbscan(eps=eps, min_points=effective_min,
                                 print_progress=False), dtype=np.int32
        )
    finally:
        del cloud
    if stride == 1 and not voxelized:
        return sample_ids

    valid = sample_ids >= 0
    if not np.any(valid):
        raise ValueError(
            "DBSCAN found no clusters among the representatives. "
            "Increase max_dbscan_voxels or check the PLY labels/classification."
        )
    # Store the large label vector on disk if a scratch directory is provided.
    if scratch_dir is not None:
        path = Path(scratch_dir) / "cluster_labels.npy"
        labels = np.lib.format.open_memmap(str(path), mode="w+",
                                            dtype=np.int32, shape=(n,))
    else:
        labels = np.empty(n, dtype=np.int32)

    tree = cKDTree(representatives[valid])
    possible_ids = sample_ids[valid]
    del sample, representatives
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        # Workers=1 is intentional to bound RAM and avoid UI thread oversubscription.
        d, ix = tree.query(points[start:stop], k=1, distance_upper_bound=eps,
                           workers=1)
        result = np.full(stop - start, -1, dtype=np.int32)
        hit = np.isfinite(d)
        result[hit] = possible_ids[ix[hit]]
        labels[start:stop] = result
    if isinstance(labels, np.memmap):
        labels.flush()
    return labels


def eligible_cluster_ids(
    points: np.ndarray,
    cluster_ids: np.ndarray,
    min_points: int = FILTER_MIN_POINTS,
    min_height: float = FILTER_MIN_HEIGHT,
    chunk_size: int = 200_000,
) -> list[int]:
    """Validate clusters without repeating n-point boolean selections."""
    # DBSCAN produces cluster IDs 0..k-1, and -1 for noise.
    max_id = int(np.max(cluster_ids)) if len(cluster_ids) else -1
    if max_id < 0:
        raise ValueError("No clusters were found. Check the PLY label field.")
    counts = np.zeros(max_id + 1, dtype=np.int64)
    z_min = np.full(max_id + 1, np.inf)
    z_max = np.full(max_id + 1, -np.inf)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        label = cluster_ids[start:stop]
        selected = label >= 0
        if not selected.any():
            continue
        ids = label[selected]
        counts += np.bincount(ids, minlength=max_id + 1)
        z = points[start:stop, 2][selected]
        np.minimum.at(z_min, ids, z)
        np.maximum.at(z_max, ids, z)
    valid = np.flatnonzero((counts >= min_points) &
                           ((z_max - z_min) >= min_height)).tolist()
    if not valid:
        raise ValueError(
            "No cluster met the point-count and height filters. "
            "Check MV Poles in labels.json, FILTER_MIN_POINTS, and FILTER_MIN_HEIGHT."
        )
    return [int(i) for i in valid]
