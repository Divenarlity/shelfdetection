"""Persistent localhost API for the Unity pose-based shelf cascade."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import replace
import json
import logging
from pathlib import Path
import re
import threading
import time
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile

from src.options import add_pipeline_arguments, pipeline_config_from_args
from src.pipeline import PipelineConfig, ShelfInferencePipeline
from src.pose_geometry import load_pose_config, load_store_map, parse_frame_metadata
from src.runtime_config import resolve_runtime_configuration
from src.session_recorder import ScanSessionRecorder

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("shelf_inference")
MAX_IMAGE_BYTES = 12 * 1024 * 1024


def resolve_input_path(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


class InferenceBusy(Exception):
    pass


class VisualizationCache:
    """Small thread-safe in-memory cache; it never shares the inference lock."""

    def __init__(self, max_items=20):
        if not 10 <= max_items <= 30:
            raise ValueError("Visualization cache size must be between 10 and 30.")
        self.max_items = max_items
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def put(self, jpeg_bytes):
        visualization_id = uuid4().hex
        with self._lock:
            self._items[visualization_id] = bytes(jpeg_bytes)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return visualization_id

    def get(self, visualization_id):
        with self._lock:
            return self._items.get(visualization_id)

    def __len__(self):
        with self._lock:
            return len(self._items)


class InferenceService:
    def __init__(self, pipeline, debug_live=False, debug_root=None,
                 visualization_cache_size=20, visualization_jpeg_quality=90,
                 session_recorder=None, runtime_configuration=None,
                 warmup_iterations=0, warmup_empty_batch_size=24):
        if not 1 <= visualization_jpeg_quality <= 100:
            raise ValueError("Visualization JPEG quality must be between 1 and 100.")
        self.pipeline = pipeline
        self.shelf_model = pipeline.shelf_model
        self.empty_model = pipeline.empty_model
        self.lock = threading.Lock()
        self.visualizations = VisualizationCache(visualization_cache_size)
        self.visualization_jpeg_quality = visualization_jpeg_quality
        self.debug_live = debug_live
        self.debug_root = Path(debug_root or ROOT / "runs" / "live_debug")
        self.session_recorder = session_recorder
        self.runtime_configuration = runtime_configuration
        self.warmup_iterations = warmup_iterations
        self.warmup_empty_batch_size = warmup_empty_batch_size

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
            encode_started = time.perf_counter()
            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                outcome.annotated_image,
                [cv2.IMWRITE_JPEG_QUALITY, self.visualization_jpeg_quality],
            )
            if not encoded_ok:
                raise RuntimeError("Could not encode the annotated visualization.")
            visualization_id = self.visualizations.put(encoded.tobytes())
            payload = {"success": True, **outcome.result}
            payload["timing"] = dict(payload.get("timing", {}))
            payload["timing"]["visualization_encode_ms"] = (
                time.perf_counter() - encode_started
            ) * 1000
            payload["visualization_url"] = f"/visualization/{visualization_id}.jpg"
            return payload
        finally:
            self.lock.release()

    def infer_dual(self, left_image, left_metadata, right_image, right_metadata,
                   left_raw=None, right_raw=None, session_id=None, station_id=None):
        """Run both views while holding the inference lock exactly once."""
        if not self.lock.acquire(blocking=False):
            raise InferenceBusy("Another inference request is running.")
        try:
            debug_dirs = {"left": None, "right": None}
            if self.debug_live:
                cycle = uuid4().hex[:12]
                for camera_id, metadata, raw in (
                    ("left", left_metadata, left_raw),
                    ("right", right_metadata, right_raw),
                ):
                    frame_id = re.sub(
                        r"[^A-Za-z0-9_-]+", "_", str(metadata["frame_id"])
                    )[:80]
                    debug_dir = self.debug_root / f"{frame_id}_{cycle}_{camera_id}"
                    debug_dir.mkdir(parents=True, exist_ok=False)
                    if raw is not None:
                        suffix = ".jpg" if raw.startswith(b"\xff\xd8") else (
                            ".png" if raw.startswith(b"\x89PNG") else ".bin"
                        )
                        (debug_dir / f"incoming_frame{suffix}").write_bytes(raw)
                    (debug_dir / "incoming_metadata.json").write_text(
                        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    debug_dirs[camera_id] = debug_dir
            outcome = self.pipeline.infer_dual(
                left_image, left_metadata, right_image, right_metadata,
                debug_dirs["left"], debug_dirs["right"],
            )
            encode_started = time.perf_counter()
            sides = {}
            annotated_bytes = {}
            for camera_id, side in (("left", outcome.left), ("right", outcome.right)):
                encoded_ok, encoded = cv2.imencode(
                    ".jpg", side.annotated_image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.visualization_jpeg_quality],
                )
                if not encoded_ok:
                    raise RuntimeError(
                        f"Could not encode the {camera_id} annotated visualization."
                    )
                annotated_bytes[camera_id] = encoded.tobytes()
                visualization_id = self.visualizations.put(annotated_bytes[camera_id])
                sides[camera_id] = {
                    "success": True, **side.result,
                    "visualization_url": f"/visualization/{visualization_id}.jpg",
                }
            timing = dict(outcome.timing)
            timing["visualization_encode_ms"] = (
                time.perf_counter() - encode_started
            ) * 1000
            payload = {
                "success": True,
                "mode": "dual_camera_batch",
                "left": sides["left"],
                "right": sides["right"],
                "timing": timing,
                "counts": outcome.counts,
            }
            persistence_started = time.perf_counter()
            if self.session_recorder is None:
                persistence = {"enabled": False, "saved": False}
            elif not session_id or not station_id:
                persistence = {
                    "enabled": True,
                    "saved": False,
                    "warning": "session_id and station_id were not supplied; scan was not archived.",
                }
            else:
                try:
                    persistence = {
                        "enabled": True,
                        **self.session_recorder.record_station(
                            session_id,
                            station_id,
                            left_raw=left_raw,
                            right_raw=right_raw,
                            left_annotated=annotated_bytes["left"],
                            right_annotated=annotated_bytes["right"],
                            dual_result=payload,
                        ),
                    }
                except Exception as exc:
                    LOG.exception(
                        "Could not persist session=%r station=%r", session_id, station_id
                    )
                    persistence = {
                        "enabled": True,
                        "saved": False,
                        "warning": f"Scan persistence failed: {type(exc).__name__}: {exc}",
                    }
            persistence["save_ms"] = (
                time.perf_counter() - persistence_started
            ) * 1000
            payload["persistence"] = persistence
            return payload
        finally:
            self.lock.release()

    def complete_session(self, session_id):
        if self.session_recorder is None:
            return {"enabled": False, "saved": False}
        started = time.perf_counter()
        try:
            result = {"enabled": True, **self.session_recorder.complete_session(session_id)}
        except Exception as exc:
            LOG.exception("Could not complete session=%r", session_id)
            result = {
                "enabled": True,
                "saved": False,
                "warning": f"Session completion persistence failed: {type(exc).__name__}: {exc}",
            }
        result["save_ms"] = (time.perf_counter() - started) * 1000
        return result


def _decode_request_view(upload, metadata_json):
    """Apply the exact single-view request protections to one multipart view."""
    data = parse_frame_metadata(json.loads(metadata_json))
    raw = upload.file.read(MAX_IMAGE_BYTES + 1)
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is empty or exceeds the 12 MiB limit.")
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Image could not be decoded.")
    height, width = frame.shape[:2]
    if data["resolution"] != [width, height]:
        raise ValueError("Metadata resolution does not match the image.")
    return frame, data, raw


def _load_model(path, label, task):
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    from ultralytics import YOLO
    # Engine filenames do not reliably communicate the task before the backend
    # is initialized. Supply the known production contract so segmentation
    # outputs cannot be interpreted as detection-only results.
    model = YOLO(str(path), task=task)
    LOG.info(
        "%s:\npath=%s\ntask=%s\nclasses=%s",
        label, path, model.task, model.names,
    )
    return model


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
    visualization_cache_size=20,
    visualization_jpeg_quality=90,
    save_scan_sessions=True,
    scan_session_root=ROOT / "runs" / "scan_sessions",
    warmup_iterations=0,
    warmup_empty_batch_size=24,
    cudnn_benchmark=False,
):
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations cannot be negative.")
    if warmup_empty_batch_size <= 0:
        raise ValueError("warmup_empty_batch_size must be positive.")
    shelf_path = resolve_input_path(shelf_model_path)
    empty_path = resolve_input_path(empty_model_path)
    map_path = resolve_input_path(store_map_path) if store_map_path else None
    pose_path = resolve_input_path(pose_config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if map_path is None:
            raise RuntimeError("--store-map is required for the Unity inference server.")
        configured = pipeline_config or PipelineConfig()
        runtime = resolve_runtime_configuration(
            configured.device,
            configured.precision,
            cudnn_benchmark,
            (shelf_path, empty_path),
        )
        configured = replace(configured, device=runtime.predict_device)
        loaded_shelf = shelf_model or _load_model(shelf_path, "Shelf model", "segment")
        loaded_empty = empty_model or _load_model(empty_path, "Empty model", "detect")
        pipeline = ShelfInferencePipeline(
            loaded_shelf,
            loaded_empty,
            load_store_map(map_path),
            load_pose_config(pose_path),
            configured,
        )
        if shelf_model is not None:
            LOG.info(
                "Shelf model:\npath=%s\ntask=%s\nclasses=%s",
                shelf_path, pipeline.shelf_task, pipeline.shelf_names,
            )
        if empty_model is not None:
            LOG.info(
                "Empty shelf model:\npath=%s\ntask=%s\nclasses=%s",
                empty_path, pipeline.empty_task, pipeline.empty_names,
            )
        shelf_backend = "TensorRT" if shelf_path.suffix.lower() == ".engine" else "PyTorch"
        empty_backend = "TensorRT" if empty_path.suffix.lower() == ".engine" else "PyTorch"
        LOG.info(
            "Shelf model runtime:\nbackend=%s\ndevice=%s\nprecision=%s",
            shelf_backend, runtime.device, runtime.precision,
        )
        LOG.info(
            "Empty model runtime:\nbackend=%s\ndevice=%s\nprecision=%s",
            empty_backend, runtime.device, runtime.precision,
        )
        LOG.info(
            "Inference runtime:\nbackend=%s\ndevice=%s\nprecision=%s\n"
            "torch=%s\ntorch_cuda=%s\ngpu=%s\ncudnn_benchmark=%s",
            runtime.backend, runtime.device, runtime.precision,
            runtime.torch_version, runtime.torch_cuda_version,
            runtime.gpu_name or "none", runtime.cudnn_benchmark,
        )
        if warmup_iterations:
            started = time.perf_counter()
            pipeline.warmup(warmup_iterations, warmup_empty_batch_size)
            LOG.info(
                "Startup warmup complete: iterations=%d empty_batch=%d elapsed_ms=%.1f",
                warmup_iterations, warmup_empty_batch_size,
                (time.perf_counter() - started) * 1000,
            )
        session_recorder = None
        if save_scan_sessions:
            config = pipeline.config
            session_recorder = ScanSessionRecorder(
                resolve_input_path(scan_session_root),
                shelf_model=shelf_path,
                empty_model=empty_path,
                thresholds={
                    "shelf_conf": config.shelf_conf,
                    "empty_conf": config.empty_conf,
                    "min_mask_overlap": config.min_mask_overlap,
                    "dedup_iou": config.dedup_iou,
                },
            )
        app.state.service = InferenceService(
            pipeline,
            debug_live=debug_live,
            debug_root=resolve_input_path(debug_root) if debug_root else None,
            visualization_cache_size=visualization_cache_size,
            visualization_jpeg_quality=visualization_jpeg_quality,
            session_recorder=session_recorder,
            runtime_configuration=runtime,
            warmup_iterations=warmup_iterations,
            warmup_empty_batch_size=warmup_empty_batch_size,
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
        payload = {
            "status": "ok" if service else "unavailable",
            "shelf_model_loaded": bool(service and service.shelf_model),
            "empty_model_loaded": bool(service and service.empty_model),
        }
        if service and service.runtime_configuration:
            payload.update(service.runtime_configuration.health_fields())
            payload["warmup_iterations"] = service.warmup_iterations
            payload["warmup_empty_batch_size"] = service.warmup_empty_batch_size
        return payload

    @app.post("/infer")
    def infer(request: Request, image: UploadFile = File(...), metadata: str = Form(...)):
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(503, "Inference service is not ready.")
        try:
            frame, data, raw = _decode_request_view(image, metadata)
            return service.infer(frame, data, raw)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except InferenceBusy as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            LOG.exception("Inference failed")
            raise HTTPException(500, "Inference failed; check the server log.") from exc

    @app.post("/infer/dual")
    def infer_dual(
        request: Request,
        left_image: UploadFile = File(...),
        left_metadata: str = Form(...),
        right_image: UploadFile = File(...),
        right_metadata: str = Form(...),
        session_id: str | None = Form(None),
        station_id: str | None = Form(None),
    ):
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(503, "Inference service is not ready.")
        try:
            left_frame, left_data, left_raw = _decode_request_view(
                left_image, left_metadata
            )
            right_frame, right_data, right_raw = _decode_request_view(
                right_image, right_metadata
            )
            if left_data["frame_id"] == right_data["frame_id"]:
                raise ValueError("Dual view frame_id values must be distinct.")
            return service.infer_dual(
                left_frame, left_data, right_frame, right_data, left_raw, right_raw,
                session_id, station_id,
            )
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except InferenceBusy as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            LOG.exception("Dual inference failed")
            raise HTTPException(500, "Dual inference failed; check the server log.") from exc

    @app.post("/session/{session_id}/complete")
    def complete_session(request: Request, session_id: str):
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(503, "Inference service is not ready.")
        return service.complete_session(session_id)

    @app.get("/visualization/{visualization_id}.jpg")
    def visualization(request: Request, visualization_id: str):
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(503, "Inference service is not ready.")
        if re.fullmatch(r"[0-9a-f]{32}", visualization_id) is None:
            raise HTTPException(404, "Visualization not found.")
        jpeg_bytes = service.visualizations.get(visualization_id)
        if jpeg_bytes is None:
            raise HTTPException(404, "Visualization not found.")
        return Response(
            content=jpeg_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    return app


def build_parser():
    parser = argparse.ArgumentParser(description="Persistent Unity shelf inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug-live", action="store_true",
                        help="Save per-request diagnostics under runs/live_debug")
    parser.add_argument("--debug-root", type=Path)
    parser.add_argument("--visualization-cache-size", type=int, default=20,
                        choices=range(10, 31))
    parser.add_argument("--visualization-jpeg-quality", type=int, default=90,
                        choices=range(1, 101))
    parser.add_argument(
        "--save-scan-sessions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist production dual scans (default: enabled; use --no-save-scan-sessions to disable)",
    )
    parser.add_argument(
        "--scan-session-root", type=Path, default=Path("runs/scan_sessions"),
        help="Persistent scan-session directory (default: runs/scan_sessions)",
    )
    parser.add_argument(
        "--warmup-iterations", type=int, default=0,
        help="Model-only startup warmup iterations (default: 0)",
    )
    parser.add_argument(
        "--warmup-empty-batch-size", type=int, default=24,
        help="Representative empty-ROI warmup batch size (default: 24)",
    )
    parser.add_argument(
        "--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False,
        help="Enable cuDNN algorithm benchmarking for repeated input shapes",
    )
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
            visualization_cache_size=args.visualization_cache_size,
            visualization_jpeg_quality=args.visualization_jpeg_quality,
            save_scan_sessions=args.save_scan_sessions,
            scan_session_root=args.scan_session_root,
            warmup_iterations=args.warmup_iterations,
            warmup_empty_batch_size=args.warmup_empty_batch_size,
            cudnn_benchmark=args.cudnn_benchmark,
        ),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
