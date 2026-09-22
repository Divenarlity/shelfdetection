"""Persistent localhost API for the Unity pose-based shelf cascade."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import re
import threading
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from src.options import add_pipeline_arguments, pipeline_config_from_args
from src.pipeline import ShelfInferencePipeline
from src.pose_geometry import load_pose_config, load_store_map, parse_frame_metadata

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("shelf_inference")
MAX_IMAGE_BYTES = 12 * 1024 * 1024


def resolve_input_path(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


class InferenceBusy(Exception):
    pass


class InferenceService:
    def __init__(self, pipeline, debug_live=False, debug_root=None):
        self.pipeline = pipeline
        self.shelf_model = pipeline.shelf_model
        self.empty_model = pipeline.empty_model
        self.lock = threading.Lock()
        self.debug_live = debug_live
        self.debug_root = Path(debug_root or ROOT / "runs" / "live_debug")

    def infer(self, image, metadata, raw_image=None):
        if not self.lock.acquire(blocking=False):
            raise InferenceBusy("Another inference request is running.")
        try:
            debug_dir = None
            if self.debug_live:
                frame_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(metadata["frame_id"]))[:80]
                debug_dir = self.debug_root / f"{frame_id}_{uuid4().hex[:12]}"
                debug_dir.mkdir(parents=True, exist_ok=False)
                if raw_image is not None:
                    suffix = ".jpg" if raw_image.startswith(b"\xff\xd8") else (
                        ".png" if raw_image.startswith(b"\x89PNG") else ".bin"
                    )
                    (debug_dir / f"incoming_frame{suffix}").write_bytes(raw_image)
                (debug_dir / "incoming_metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                LOG.info("Live debug frame %s -> %s", metadata["frame_id"], debug_dir)
            outcome = self.pipeline.infer(image, metadata, debug_dir)
            return {"success": True, **outcome.result}
        finally:
            self.lock.release()


def _load_model(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    from ultralytics import YOLO
    return YOLO(str(path))


def create_app(
    shelf_model_path=ROOT / "best.pt",
    empty_model_path=ROOT / "empty_shelf_yolo11m_best.pt",
    store_map_path=None,
    pose_config_path=ROOT / "configs" / "pose_geometry.yaml",
    pipeline_config=None,
    shelf_model=None,
    empty_model=None,
    debug_live=False,
    debug_root=None,
):
    shelf_path = resolve_input_path(shelf_model_path)
    empty_path = resolve_input_path(empty_model_path)
    map_path = resolve_input_path(store_map_path) if store_map_path else None
    pose_path = resolve_input_path(pose_config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if map_path is None:
            raise RuntimeError("--store-map is required for the Unity inference server.")
        loaded_shelf = shelf_model or _load_model(shelf_path, "Shelf model")
        loaded_empty = empty_model or _load_model(empty_path, "Empty model")
        pipeline = ShelfInferencePipeline(
            loaded_shelf,
            loaded_empty,
            load_store_map(map_path),
            load_pose_config(pose_path),
            pipeline_config,
        )
        app.state.service = InferenceService(
            pipeline,
            debug_live=debug_live,
            debug_root=resolve_input_path(debug_root) if debug_root else None,
        )
        LOG.info("Shelf inference server ready")
        try:
            yield
        finally:
            app.state.service = None

    app = FastAPI(title="Shelf Inference", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        service = getattr(request.app.state, "service", None)
        return {
            "status": "ok" if service else "unavailable",
            "shelf_model_loaded": bool(service and service.shelf_model),
            "empty_model_loaded": bool(service and service.empty_model),
        }

    @app.post("/infer")
    def infer(request: Request, image: UploadFile = File(...), metadata: str = Form(...)):
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(503, "Inference service is not ready.")
        try:
            data = parse_frame_metadata(json.loads(metadata))
            raw = image.file.read(MAX_IMAGE_BYTES + 1)
            if not raw or len(raw) > MAX_IMAGE_BYTES:
                raise ValueError("Image is empty or exceeds the 12 MiB limit.")
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Image could not be decoded.")
            height, width = frame.shape[:2]
            if data["resolution"] != [width, height]:
                raise ValueError("Metadata resolution does not match the image.")
            return service.infer(frame, data, raw)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except InferenceBusy as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            LOG.exception("Inference failed")
            raise HTTPException(500, "Inference failed; check the server log.") from exc

    return app


def build_parser():
    parser = argparse.ArgumentParser(description="Persistent Unity shelf inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug-live", action="store_true",
                        help="Save per-request diagnostics under runs/live_debug")
    parser.add_argument("--debug-root", type=Path)
    return add_pipeline_arguments(parser, ROOT)


def main():
    args = build_parser().parse_args()
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(
        create_app(
            args.shelf_model,
            args.empty_model,
            args.store_map,
            args.pose_config,
            pipeline_config_from_args(args),
            debug_live=args.debug_live,
            debug_root=args.debug_root,
        ),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
