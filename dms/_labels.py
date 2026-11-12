"""Resolve DMS class identifiers exclusively from the shared labels.json catalog."""
from pathlib import Path
import json

ORIGINAL_CLASSES = (
    "Vegetation", "LV Wires", "Ads", "Buildings", "Ground", "LV Poles",
    "Rubble", "Scaffolding", "Ladder", "Bricks", "MV Electrical Works",
)
SNAKE_CLASSES = ("Vegetation", "LV Wires", "Ground")
FIGURE_CLASSES = tuple(name for name in ORIGINAL_CLASSES if name not in SNAKE_CLASSES)
RECON_CLASSES = (
    "Reconstruction MV Poles", "Reconstruction MV Crossarms",
    "Reconstruction MV Wires", "Reconstruction MV Connecting",
)
EXCLUDED_CLASSES = (
    "Unclassified", "MV Poles", "MV Wires", "MV Wires 1",
    "MV Wires 2", "MV Wires 3",
)


def load_label_maps(cfg, config_path):
    """Build class/CP/DMS lookup tables; do not hardcode numeric semantic IDs."""
    path = Path(cfg["labels_json"]).expanduser()
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    with path.open(encoding="utf-8") as stream:
        catalog = json.load(stream)
    entries = catalog.get("labels")
    if not isinstance(entries, list):
        raise ValueError(f"Expected a 'labels' array in {path}")
    by_name = {}
    ids = set()
    for item in entries:
        name = str(item["title"]).strip()
        ident = int(item["id"])
        if name in by_name or ident in ids:
            raise ValueError(f"Duplicate label name or ID in {path}: {name}")
        by_name[name] = ident
        ids.add(ident)
    required = (
        set(ORIGINAL_CLASSES) | set(RECON_CLASSES) | set(EXCLUDED_CLASSES)
        | {f"DMS {name}" for name in ORIGINAL_CLASSES}
        | {f"CP {name}" for name in ORIGINAL_CLASSES}
    )
    missing = sorted(required - by_name.keys())
    if missing:
        raise ValueError(f"Missing labels in {path}: {', '.join(missing)}")
    original = {by_name[name]: name for name in ORIGINAL_CLASSES}
    return {
        "ids": by_name,
        "original": original,
        "dms": {by_name[name]: by_name[f"DMS {name}"] for name in ORIGINAL_CLASSES},
        "cp": {by_name[name]: by_name[f"CP {name}"] for name in ORIGINAL_CLASSES},
        "code": {by_name[name]: index for index, name in enumerate(ORIGINAL_CLASSES)},
        "recon": {by_name[name]: name for name in RECON_CLASSES},
        "excluded": {by_name[name] for name in EXCLUDED_CLASSES},
    }
