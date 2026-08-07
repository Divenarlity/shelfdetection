from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Iterator

import cv2

from .pose_geometry import load_frame_metadata


class PoseProvider(Protocol):
    def metadata(self, frame_id: str) -> dict: ...


@dataclass(frozen=True)
class RuntimeFrame:
    metadata_path: Path
    metadata: dict
    image_path: Path
    image: object


class OfflineFrameSource:
    """Unity and future robot adapters share this ID-free schema-v3 reader."""
    def __init__(self, source):
        self.source = Path(source)

    def __iter__(self) -> Iterator[RuntimeFrame]:
        if "ground_truth" in {p.lower() for p in self.source.parts}:
            raise ValueError("Runtime FrameSource ground_truth klasörünü açamaz.")
        paths = sorted(self.source.glob("*.json")) if self.source.is_dir() else [self.source]
        for path in paths:
            meta = load_frame_metadata(path)
            rel = meta["image_path"]
            image_path = Path(rel) if Path(rel).is_absolute() else path.parent/rel
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Görüntü okunamadı: {image_path}")
            yield RuntimeFrame(path, meta, image_path, image)
