"""Rutas y nombres: únicamente desde config.json; clases desde labels_json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    return cfg

def configured_path(cfg, directory_key, filename_key):
    directory = Path(cfg[directory_key])
    if not directory.is_absolute():
        directory = ROOT / directory
    filename = Path(cfg[filename_key])
    return filename if filename.is_absolute() else directory / filename

def output_path(cfg, filename_key):
    path = configured_path(cfg, "output_dir", filename_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def label_ids(cfg, titles):
    catalog_file = Path(cfg["labels_json"])
    if not catalog_file.is_absolute():
        catalog_file = ROOT / catalog_file
    with catalog_file.open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    by_title = {str(item["title"]).strip().casefold(): int(item["id"])
                for item in catalog["labels"]}
    missing = [title for title in titles if title.casefold() not in by_title]
    if missing:
        raise ValueError(f"Clases ausentes en {catalog_file}: {missing}")
    return [by_title[title.casefold()] for title in titles]
