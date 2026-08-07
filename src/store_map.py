from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Shelf:
    shelf_id: str
    rack_id: str
    level: int
    polygon: list


@dataclass
class Camera:
    camera_id: str
    area_id: str
    reference_width: int
    reference_height: int
    shelves: list[Shelf]


class StoreMap:
    def __init__(self, market_id, cameras, section_names=None):
        self.market_id = market_id
        self.cameras = cameras
        self.section_names = section_names or ["left", "middle", "right"]

    def camera(self, camera_id):
        if camera_id not in self.cameras:
            raise KeyError(f"Kamera yapılandırmada yok: {camera_id}")
        return self.cameras[camera_id]


def load_store_map(path):
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"YAML okunamadı: {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw.get("market_id") or not isinstance(raw.get("cameras"), dict):
        raise ValueError("YAML market_id ve cameras sözlüğü içermelidir.")
    cameras = {}
    for cid, item in raw["cameras"].items():
        required = ("area_id", "reference_width", "reference_height", "shelves")
        if any(k not in item for k in required):
            raise ValueError(f"{cid} için zorunlu alan eksik.")
        shelves = []
        for s in item["shelves"]:
            if not all(k in s for k in ("shelf_id", "rack_id", "level", "polygon")):
                raise ValueError(f"{cid} raf kaydı eksik alan içeriyor.")
            if len(s["polygon"]) != 4:
                raise ValueError(f"{s['shelf_id']} poligonu dört noktalı olmalıdır.")
            shelves.append(Shelf(s["shelf_id"], s["rack_id"], int(s["level"]), s["polygon"]))
        cameras[cid] = Camera(cid, item["area_id"], int(item["reference_width"]), int(item["reference_height"]), shelves)
    return StoreMap(raw["market_id"], cameras, raw.get("section_names"))

