from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PersistentIdAllocator:
    prefix: str
    next_index: int

    def allocate(self):
        value=f"{self.prefix}-{self.next_index:04d}"
        self.next_index += 1
        return value


class ShelfMapStore:
    def __init__(self, map_dir, store_id):
        self.map_dir=Path(map_dir)
        self.store_id=store_id
        self.path=self.map_dir/"store_map.json"
        self.candidates_path=self.map_dir/"mapping_candidates.json"

    def create(self, id_prefix="SHELF", overwrite=False):
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"Mevcut harita sessizce ezilemez: {self.path}")
        return {"schema_version":3,"store_id":self.store_id,"map_uuid":str(uuid.uuid4()),
                "coordinate_frame":"map","id_prefix":id_prefix,"next_shelf_index":1,
                "retired_shelf_ids":[],"shelves":[]}

    def load(self):
        data=json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("store_id")!=self.store_id:
            raise ValueError("Harita store_id ile istenen store_id uyuşmuyor.")
        return data

    def save_atomic(self, data):
        self.map_dir.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix(f".json.{os.getpid()}.tmp")
        temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
        os.replace(temp,self.path)

    def load_candidates(self):
        return json.loads(self.candidates_path.read_text(encoding="utf-8")) if self.candidates_path.exists() else []

    def save_candidates_atomic(self, candidates):
        self.map_dir.mkdir(parents=True,exist_ok=True)
        temp=self.candidates_path.with_suffix(f".json.{os.getpid()}.tmp")
        temp.write_text(json.dumps(candidates,ensure_ascii=False,indent=2),encoding="utf-8")
        os.replace(temp,self.candidates_path)
