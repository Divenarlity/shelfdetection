"""Benchmark the unchanged production dual pipeline on a captured real station."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import tempfile
import time

import cv2
import numpy as np

from inference_server import ROOT, resolve_input_path
from src.pipeline import PipelineConfig, ShelfInferencePipeline
from src.pose_geometry import load_pose_config, load_store_map, parse_frame_metadata
from src.runtime_config import resolve_runtime_configuration
from src.session_recorder import ScanSessionRecorder


def _view(directory):
    directory = Path(directory)
    images = list(directory.glob("incoming_frame.*"))
    if len(images) != 1:
        raise ValueError(f"Expected one incoming_frame file in {directory}; found {len(images)}.")
    raw = images[0].read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {images[0]}.")
    metadata = parse_frame_metadata(json.loads(
        (directory / "incoming_metadata.json").read_text(encoding="utf-8")
    ))
    return image, metadata, raw


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _stats(values):
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": _percentile(values, .95),
    }


def _semantic_side(result):
    shelves = []
    for shelf in result["shelves"]:
        shelves.append({
            "parent_shelf_id": shelf.get("parent_shelf_id"),
            "shelf_level_id": shelf.get("shelf_level_id"),
            "level_number": shelf.get("level_number"),
            "status": shelf.get("status"),
            "bbox": shelf.get("global_bbox_xyxy"),
            "confidence": shelf.get("confidence"),
            "sections": shelf.get("sections"),
            "detections": [{
                "section": item.get("section"),
                "bbox": item.get("global_bbox_xyxy"),
                "confidence": item.get("confidence"),
                "mask_overlap_ratio": item.get("mask_overlap_ratio"),
            } for item in shelf.get("detections", [])],
        })
    return {
        "counts": result["counts"],
        "shelves": shelves,
        "rejected_detections": result.get("rejected_detections", []),
    }


def _payload(outcome, left_url="/visualization/benchmark_left.jpg",
             right_url="/visualization/benchmark_right.jpg"):
    return {
        "success": True,
        "mode": "dual_camera_batch",
        "left": {"success": True, **deepcopy(outcome.left.result),
                 "visualization_url": left_url},
        "right": {"success": True, **deepcopy(outcome.right.result),
                  "visualization_url": right_url},
        "timing": deepcopy(outcome.timing),
        "counts": deepcopy(outcome.counts),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-debug-dir", type=Path, required=True)
    parser.add_argument("--right-debug-dir", type=Path, required=True)
    parser.add_argument("--store-map", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, default=ROOT / "configs/pose_geometry.yaml")
    parser.add_argument("--shelf-model", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--empty-model", type=Path,
                        default=ROOT / "empty_shelf_yolo11m_best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive.")
    from ultralytics import YOLO
    import torch

    shelf_path = resolve_input_path(args.shelf_model)
    empty_path = resolve_input_path(args.empty_model)
    runtime = resolve_runtime_configuration(
        args.device, args.precision, args.cudnn_benchmark,
        (shelf_path, empty_path),
    )
    left_image, left_metadata, left_raw = _view(args.left_debug_dir)
    right_image, right_metadata, right_raw = _view(args.right_debug_dir)
    pipeline = ShelfInferencePipeline(
        YOLO(str(shelf_path), task="segment"),
        YOLO(str(empty_path), task="detect"),
        load_store_map(resolve_input_path(args.store_map)),
        load_pose_config(resolve_input_path(args.pose_config)),
        PipelineConfig(
            device=runtime.predict_device,
            precision=runtime.precision,
            synchronize_cuda_timing=True,
        ),
    )

    for _ in range(args.warmup):
        pipeline.infer_dual(left_image, left_metadata, right_image, right_metadata)
    if runtime.device != "cpu":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    stage_values = {name: [] for name in (
        "shelf_batch_inference_ms", "shelf_postprocess_ms", "mapping_ms",
        "roi_preparation_ms", "empty_batch_inference_ms", "postprocessing_ms",
        "visualization_ms", "pipeline_total_ms", "jpeg_encode_ms",
        "session_save_ms", "server_total_ms",
    )}
    last_outcome = None
    with tempfile.TemporaryDirectory(prefix="shelf_gpu_benchmark_") as temp_root:
        recorder = ScanSessionRecorder(Path(temp_root) / "sessions")
        for index in range(args.iterations):
            if runtime.device != "cpu":
                torch.cuda.synchronize()
            server_started = time.perf_counter()
            outcome = pipeline.infer_dual(
                left_image, left_metadata, right_image, right_metadata
            )
            timing = outcome.timing
            encode_started = time.perf_counter()
            encoded = {}
            for side_name, side in (("left", outcome.left), ("right", outcome.right)):
                ok, jpeg = cv2.imencode(
                    ".jpg", side.annotated_image,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError(f"Could not encode {side_name} benchmark image.")
                encoded[side_name] = jpeg.tobytes()
            jpeg_ms = (time.perf_counter() - encode_started) * 1000
            save_started = time.perf_counter()
            recorder.record_station(
                "benchmark_session", f"iteration_{index:02d}",
                left_raw=left_raw, right_raw=right_raw,
                left_annotated=encoded["left"], right_annotated=encoded["right"],
                dual_result=_payload(outcome),
            )
            save_ms = (time.perf_counter() - save_started) * 1000
            server_ms = (time.perf_counter() - server_started) * 1000

            stage_values["shelf_batch_inference_ms"].append(
                timing["shelf_batch_inference_ms"]
            )
            stage_values["shelf_postprocess_ms"].append(
                timing["left_shelf_postprocess_ms"] + timing["right_shelf_postprocess_ms"]
            )
            stage_values["mapping_ms"].append(
                timing["left_mapping_ms"] + timing["right_mapping_ms"]
            )
            stage_values["roi_preparation_ms"].append(
                timing["left_roi_preparation_ms"] + timing["right_roi_preparation_ms"] +
                timing["descriptor_preparation_ms"]
            )
            stage_values["empty_batch_inference_ms"].append(
                timing["empty_batch_inference_ms"]
            )
            stage_values["postprocessing_ms"].append(
                timing["left_association_ms"] + timing["right_association_ms"]
            )
            stage_values["visualization_ms"].append(
                timing["left_visualization_render_ms"] +
                timing["right_visualization_render_ms"]
            )
            stage_values["pipeline_total_ms"].append(timing["dual_total_ms"])
            stage_values["jpeg_encode_ms"].append(jpeg_ms)
            stage_values["session_save_ms"].append(save_ms)
            stage_values["server_total_ms"].append(server_ms)
            if outcome.counts["shelf_model_predict_calls"] != 1 or \
                    outcome.counts["shelf_model_inputs"] != 2 or \
                    outcome.counts["empty_model_predict_calls"] not in {0, 1}:
                raise RuntimeError("Production dual batching contract changed during benchmark.")
            last_outcome = outcome

    memory = None
    if runtime.device != "cpu":
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        memory = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "free_bytes": free,
            "total_bytes": total,
        }
    report = {
        "mode": (
            f"TENSORRT_{runtime.precision.upper()}"
            if runtime.backend == "tensorrt" else
            "CPU_PT_FP32" if runtime.device == "cpu" else
            f"CUDA_PT_{runtime.precision.upper()}"
        ),
        "runtime": runtime.health_fields(),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "input": {
            "left_frame_id": left_metadata["frame_id"],
            "right_frame_id": right_metadata["frame_id"],
            "resolution": left_metadata["resolution"],
        },
        "counts": last_outcome.counts,
        "timing_ms": {name: _stats(values) for name, values in stage_values.items()},
        "cuda_memory": memory,
        "semantic_signature": {
            "left": _semantic_side(last_outcome.left.result),
            "right": _semantic_side(last_outcome.right.result),
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
