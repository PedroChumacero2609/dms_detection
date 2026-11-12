from __future__ import annotations

from pathlib import Path
import numpy as np
from .clustering import dbscan as _dbscan

METHOD = "dbscan"


def extract_poles(points: np.ndarray, *,
                  chunk_size: int = 200_000,
                  scratch_dir: Path | None = None) -> tuple[np.ndarray, list[int]]:
    if METHOD == "dbscan":
        cluster_ids = _dbscan.cluster_poles(
            points, chunk_size=chunk_size, scratch_dir=scratch_dir
        )
        valid_ids = _dbscan.eligible_cluster_ids(
            points, cluster_ids, chunk_size=chunk_size
        )
        return cluster_ids, valid_ids
    raise ValueError(f"Unknown clustering method: {METHOD}")
