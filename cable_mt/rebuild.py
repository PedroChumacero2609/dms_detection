"""Reconstruct the labeled PLY from the Extractor PLY and saved point edits.

Run from project root: python -m cable_mt.rebuild
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from plyfile import PlyData


def source_signature(path):
    """Hash complete source file to reject stale vertex indices."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()



def rebuild_ply(source, changes, destination):
    source, changes, destination = map(
        str, (source, changes, destination)
    )

    if os.path.abspath(source) == os.path.abspath(destination):
        raise ValueError(
            'Input and output PLY paths must be different'
        )

    with np.load(changes, allow_pickle=False) as data:
        indices = np.asarray(data['indices'], dtype=np.int64)
        labels = np.asarray(data['labels'], dtype=np.int32)
        original = np.asarray(
            data['original_labels'], dtype=np.int32
        )
        prop = str(data['label_property'].item())
        count = int(data['point_count'].item())

    ply = PlyData.read(source)
    vertex = ply['vertex'].data

    # Comprobar estructura básica
    if len(vertex) != count or prop not in vertex.dtype.names:
        raise ValueError(
            'PLY schema/count does not match saved edits'
        )

    if (
        len(indices) != len(labels)
        or len(indices) != len(original)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError('Invalid changes file')

    if np.any(indices < 0) or np.any(indices >= count):
        raise ValueError('Vertex index out of bounds')

    # Validar únicamente los puntos editados
    current = np.asarray(
        vertex[prop][indices], dtype=np.int32
    )

    conflicts = current != original

    if np.any(conflicts):
        raise ValueError(
            f'{np.count_nonzero(conflicts)} puntos editados '
            'tienen etiquetas originales diferentes. '
            'Es posible que los índices hayan cambiado.'
        )

    # Validar rango de etiquetas
    field = vertex[prop]
    limits = np.iinfo(field.dtype)

    if len(labels) and (
        labels.min() < limits.min
        or labels.max() > limits.max
    ):
        raise ValueError(
            'Output label does not fit original PLY label type'
        )

    # Aplicar únicamente las modificaciones guardadas
    field[indices] = labels

    os.makedirs(
        os.path.dirname(os.path.abspath(destination)),
        exist_ok=True
    )

    temp = destination + '.building.ply'

    try:
        ply.write(temp)
        os.replace(temp, destination)
    finally:
        if os.path.exists(temp):
            os.remove(temp)

    return destination


def rebuild_from_config(config):
    project_root = Path(config.get('_project_root', Path(__file__).resolve().parent.parent))
    root = project_root / config['data_dir']
    backup = root / 'backup'
    return rebuild_ply(root / config['input_ply_poles'],
                       backup / config['cable_mt_changes'],
                       root / config['input_ply'])


if __name__ == '__main__':
    with open(Path(__file__).resolve().parent.parent / 'config.json', encoding='utf-8') as f:
        config = json.load(f)
    print('PLY reconstructed:', rebuild_from_config(config))
