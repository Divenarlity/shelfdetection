from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.live_debug import LiveDebugRecorder, low_confidence_records, stage_counts
from src.model_contracts import validate_model_contract
from src.pose_geometry import UNKNOWN_SHELF, assign_pose_ids, generate_candidates
from src.postprocessing import (
    best_shelf_for_box,
    contour_polygon,
    deduplicate,
    local_to_global,
    mask_bbox,
    mask_box_relation,
    padded_roi,
    roi_boxes,
    section_for_box,
)


class EmptyInferenceMode(str, Enum):
    FULL_FRAME_GATED = "full_frame_gated"
    SHELF_ROI = "shelf_roi"
    SHELF_LEVEL_ROI = "shelf_level_roi"


@dataclass(frozen=True)
class PipelineConfig:
    shelf_conf: float = 0.25
    empty_conf: float = 0.10
    shelf_imgsz: int = 960
    empty_imgsz: int = 640
    empty_inference_mode: EmptyInferenceMode = EmptyInferenceMode.SHELF_LEVEL_ROI
    roi_padding_ratio: float = 0.02
    shelf_level_roi_padding: float = 0.02
    empty_roi_mode: str = "full_shelf"
    min_mask_overlap: float = 0.30
    dedup_iou: float = 0.50
    max_shelf_rois: int = 12
    tile_size: int = 384
    tile_overlap: float = 0.25
    device: str = "cpu"
    precision: str = "fp32"
    synchronize_cuda_timing: bool = False

    def __post_init__(self):
        try:
            mode = EmptyInferenceMode(self.empty_inference_mode)
        except ValueError as exc:
            values = ", ".join(item.value for item in EmptyInferenceMode)
            raise ValueError(f"empty_inference_mode must be one of: {values}") from exc
        object.__setattr__(self, "empty_inference_mode", mode)
        for name in ("shelf_conf", "empty_conf", "min_mask_overlap", "dedup_iou"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.roi_padding_ratio < 0 or self.shelf_level_roi_padding < 0:
            raise ValueError("ROI padding ratios cannot be negative")
        if self.empty_roi_mode not in {"full_shelf", "tiles"}:
            raise ValueError("empty_roi_mode must be full_shelf or tiles")
        if self.tile_size <= 0 or not 0 <= self.tile_overlap < 1:
            raise ValueError("tile_size must be positive and tile_overlap must be in [0, 1)")
        if self.max_shelf_rois <= 0:
            raise ValueError("max_shelf_rois must be positive")
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError("precision must be fp32 or fp16")
        if self.precision == "fp16" and str(self.device).lower() == "cpu":
            raise ValueError("FP16 inference requires a CUDA device")


@dataclass
class PipelineOutcome:
    result: dict
    annotated_image: np.ndarray
    native_shelf_plot: np.ndarray


@dataclass
class DualPipelineOutcome:
    left: PipelineOutcome
    right: PipelineOutcome
    timing: dict
    counts: dict


def extract_shelves(result, image_shape, shelf_names, shelf_task="segment"):
    """Parse supported Ultralytics shelf outputs without confusing boxes and masks."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    if shelf_task not in {"detect", "segment"}:
        raise RuntimeError(f"Unsupported shelf model task: {shelf_task!r}")
    if shelf_task == "segment" and getattr(result, "masks", None) is None:
        if len(boxes) == 0:
            return []
        raise RuntimeError("Shelf segmentation produced boxes without masks.")
    h, w = image_shape[:2]
    confidences = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    raw_masks = (
        result.masks.data.cpu().numpy() if shelf_task == "segment" else None
    )
    raw_boxes = boxes.xyxy.cpu().tolist() if shelf_task == "detect" else None
    geometry_count = len(raw_masks) if raw_masks is not None else len(raw_boxes)
    if geometry_count != len(confidences) or geometry_count != len(classes):
        raise RuntimeError("Shelf mask and box counts differ.")
    shelves = []
    geometry = raw_masks if raw_masks is not None else raw_boxes
    for raw_geometry, confidence, class_id in zip(geometry, confidences, classes):
        if shelf_names.get(int(class_id)) != "shelves":
            continue
        if shelf_task == "segment":
            mask = cv2.resize(raw_geometry, (w, h), interpolation=cv2.INTER_NEAREST) >= 0.5
            bbox = mask_bbox(mask)
        else:
            x1, y1, x2, y2 = raw_geometry
            bbox = [
                max(0, int(np.floor(x1))), max(0, int(np.floor(y1))),
                min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2))),
            ]
            mask = np.zeros((h, w), dtype=bool)
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = True
        if bbox is None:
            continue
        ys, xs = np.where(mask)
        shelves.append({
            "mask": mask,
            "bbox": bbox,
            "centroid": [float(xs.mean()), float(ys.mean())],
            "segmentation_confidence": float(confidence),
            "confidence": float(confidence),
            "model_task": shelf_task,
            "has_segmentation_mask": shelf_task == "segment",
        })
    shelves.sort(key=lambda shelf: shelf["confidence"], reverse=True)
    for index, shelf in enumerate(shelves):
        shelf["shelf_index"] = index
    return shelves


def _predict(model, images, conf, imgsz, device=None, precision="fp32"):
    if not images:
        return []
    kwargs = {"source": images if len(images) > 1 else images[0], "conf": conf,
              "imgsz": imgsz, "verbose": False}
    if device:
        kwargs["device"] = device
    if precision == "fp16":
        kwargs["quantize"] = 16
    results = model.predict(**kwargs)
    return list(results) if isinstance(results, (list, tuple)) else [results]


def _synchronize_cuda(device):
    if str(device).lower() == "cpu":
        return
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize(int(device))


def build_regions(shelves, image, config: PipelineConfig, mode=None):
    """Create empty-model inputs for shelf-model proposals selected by the ROI cap."""
    selected_mode = mode or config.empty_roi_mode
    regions = []
    for fallback_index, shelf in enumerate(shelves):
        if not shelf.get("empty_inference_selected", True):
            continue
        selections = roi_boxes(
            shelf["roi_bbox"], image.shape, selected_mode,
            config.tile_size, config.tile_overlap,
        )
        for index, (kind, bbox) in enumerate(selections, 1):
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue
            prefix = shelf.get("shelf_level_id") or (
                f"SHELF-{shelf.get('shelf_index', fallback_index):02d}"
            )
            region_id = (
                f"{prefix}-FULL" if kind == "full_shelf" else f"{prefix}-TILE-{index:02d}"
            )
            regions.append({
                "shelf": shelf, "region_id": region_id, "tile_id": region_id,
                "kind": kind, "bbox": bbox, "image": crop,
            })
    return regions


def _prediction_records(result, region, image_shape, empty_names, min_mask_overlap):
    accepted, rejected, raw = [], [], []
    if result.boxes is None:
        return raw, accepted, rejected
    for local_box, confidence, class_id in zip(
        result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
    ):
        if empty_names.get(int(class_id)) != "empty_shelf":
            continue
        global_box = local_to_global(local_box, region["bbox"], image_shape)
        passes, overlap, center_inside = mask_box_relation(
            region["shelf"]["mask"], global_box, min_mask_overlap,
        )
        local_box = [float(value) for value in local_box]
        record = {
            "confidence": float(confidence),
            "roi_bbox_xyxy": local_box,
            "global_bbox_xyxy": global_box,
            "assigned_shelf_id": region["shelf"]["shelf_id"],
            "parent_shelf_id": region["shelf"].get("parent_shelf_id", UNKNOWN_SHELF),
            "shelf_level_id": region["shelf"].get("shelf_level_id"),
            "assigned_shelf_index": region["shelf"]["shelf_index"],
            "class_id": int(class_id),
            "class_name": empty_names[int(class_id)],
            "region_id": region["region_id"],
            "tile_id": region["tile_id"],
            "inference_source": region.get("inference_source", "shelf_roi"),
            "mask_overlap_ratio": overlap,
            "center_inside_mask": center_inside,
            "touches_tile_edge": bool(
                region["kind"] == "tile" and (
                    local_box[0] <= 1 or local_box[1] <= 1
                    or local_box[2] >= region["image"].shape[1] - 1
                    or local_box[3] >= region["image"].shape[0] - 1
                )
            ),
        }
        raw.append(record)
        if passes:
            accepted.append(record)
        else:
            rejected.append({**record, "reason": "outside_assigned_shelf_mask"})
    return raw, accepted, rejected


def _full_frame_prediction_records(
    result, shelves, image_shape, empty_names, min_mask_overlap,
):
    """Associate global empty boxes to exactly one shelf using mask containment.

    Center inclusion is retained as evidence and a tie-breaker, but it cannot
    override the configured overlap threshold in full-frame mode.
    """
    raw, accepted, rejected = [], [], []
    if result.boxes is None:
        return raw, accepted, rejected
    full_frame = [0, 0, image_shape[1], image_shape[0]]
    for box, confidence, class_id in zip(
        result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
    ):
        class_id = int(class_id)
        if empty_names.get(class_id) != "empty_shelf":
            continue
        global_box = local_to_global(box, full_frame, image_shape)
        shelf, overlap, center_inside = best_shelf_for_box(shelves, global_box)
        passes = shelf is not None and overlap >= min_mask_overlap
        record = {
            "confidence": float(confidence),
            "roi_bbox_xyxy": None,
            "global_bbox_xyxy": global_box,
            "assigned_shelf_id": shelf["shelf_id"] if passes else None,
            "parent_shelf_id": shelf.get("parent_shelf_id") if passes else None,
            "shelf_level_id": shelf.get("shelf_level_id") if passes else None,
            "assigned_shelf_index": shelf["shelf_index"] if passes else None,
            "candidate_shelf_id": shelf["shelf_id"] if shelf is not None else None,
            "candidate_shelf_index": shelf["shelf_index"] if shelf is not None else None,
            "class_id": class_id,
            "class_name": empty_names[class_id],
            "region_id": "FULL-FRAME",
            "tile_id": None,
            "inference_source": "full_frame",
            "mask_overlap_ratio": overlap,
            "center_inside_mask": center_inside,
            "touches_tile_edge": False,
        }
        raw.append(record)
        if passes:
            accepted.append(record)
        else:
            rejected.append({**record, "reason": "outside_all_shelf_masks"})
    return raw, accepted, rejected


def _label_font(size):
    for candidate in (
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_visualization(shelf_result, image, detections):
    """Reuse the production shelf result and add only final accepted empty boxes."""
    native_plot = shelf_result.plot(img=image.copy())
    if not isinstance(native_plot, np.ndarray) or native_plot.shape != image.shape:
        raise RuntimeError("Shelf result plot returned an invalid visualization image.")
    canvas = Image.fromarray(cv2.cvtColor(native_plot, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font = _label_font(max(16, round(image.shape[1] / 64)))
    for detection in detections:
        x1, y1, x2, y2 = map(lambda value: int(round(value)), detection["global_bbox_xyxy"])
        x1 = max(0, min(image.shape[1] - 1, x1))
        y1 = max(0, min(image.shape[0] - 1, y1))
        x2 = max(0, min(image.shape[1] - 1, x2))
        y2 = max(0, min(image.shape[0] - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        semantic_id = detection.get("shelf_level_id") or detection.get("assigned_shelf_id", "")
        section = detection.get("section", "")
        prefix = " ".join(value for value in (semantic_id, section) if value)
        label = (
            f"{prefix} | BOŞLUK " if prefix else "BOŞLUK "
        ) + f"%{round(max(0.0, min(1.0, detection['confidence'])) * 100)}"
        bounds = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        label_width = bounds[2] - bounds[0] + 10
        label_height = bounds[3] - bounds[1] + 8
        label_top = max(0, y1 - label_height)
        label_left = max(0, min(x1, image.shape[1] - label_width))
        draw.rectangle((label_left, label_top,
                        min(image.shape[1] - 1, label_left + label_width), y1),
                       fill=(255, 0, 0))
        draw.text((label_left + 5, label_top + 3), label, font=font, fill=(255, 255, 255),
                  stroke_width=1, stroke_fill=(255, 0, 0))
    annotated = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    return native_plot, annotated


class ShelfInferencePipeline:
    """Shared single/dual shelf cascade with one model call per batch stage."""

    def __init__(self, shelf_model, empty_model, store_map, pose_config,
                 config: PipelineConfig | None = None):
        self.shelf_model = shelf_model
        self.empty_model = empty_model
        self.store_map = store_map
        self.pose_config = pose_config
        self.config = config or PipelineConfig()
        self.shelf_task = shelf_model.task
        self.empty_task = empty_model.task
        self.shelf_names = validate_model_contract(
            shelf_model, {"detect", "segment"}, "shelves", "Shelf model"
        )
        self.empty_names = validate_model_contract(
            empty_model, "detect", "empty_shelf", "Empty model"
        )

    def infer(self, image, metadata, debug_dir: Path | None = None):
        outcomes, _, _ = self._infer_views([
            {"camera_id": "single", "image": image, "metadata": metadata,
             "debug_dir": debug_dir},
        ])
        return outcomes[0]

    def infer_dual(self, left_image, left_metadata, right_image, right_metadata,
                   left_debug_dir: Path | None = None,
                   right_debug_dir: Path | None = None):
        outcomes, timing, counts = self._infer_views([
            {"camera_id": "left", "image": left_image, "metadata": left_metadata,
             "debug_dir": left_debug_dir},
            {"camera_id": "right", "image": right_image, "metadata": right_metadata,
             "debug_dir": right_debug_dir},
        ])
        return DualPipelineOutcome(outcomes[0], outcomes[1], timing, counts)

    def warmup(self, iterations=3, empty_batch_size=24):
        """Warm model kernels only; no semantics, debug evidence, or sessions are emitted."""
        if iterations < 0:
            raise ValueError("warmup iterations cannot be negative")
        if empty_batch_size <= 0:
            raise ValueError("warmup empty batch size must be positive")
        if iterations <= 0:
            return
        shelf_inputs = [np.zeros((720, 1280, 3), np.uint8) for _ in range(2)]
        # Alternate representative ROI aspect ratios. Ultralytics then uses its
        # square batch preprocessing path, as it does for real mixed shelf-level
        # crops, instead of warming only one rectangular shape.
        empty_inputs = [
            np.zeros((256 if index % 2 == 0 else 320,
                      640 if index % 2 == 0 else 600, 3), np.uint8)
            for index in range(empty_batch_size)
        ]
        for _ in range(iterations):
            _predict(
                self.shelf_model, shelf_inputs, self.config.shelf_conf,
                self.config.shelf_imgsz, self.config.device, self.config.precision,
            )
            _predict(
                self.empty_model, empty_inputs, self.config.empty_conf,
                self.config.empty_imgsz, self.config.device, self.config.precision,
            )
        _synchronize_cuda(self.config.device)

    def _infer_views(self, specs):
        if self.config.synchronize_cuda_timing:
            _synchronize_cuda(self.config.device)
        dual_started = time.perf_counter()
        views = [self._prepare_view(spec) for spec in specs]

        if self.config.synchronize_cuda_timing:
            _synchronize_cuda(self.config.device)
        shelf_started = time.perf_counter()
        shelf_results = _predict(
            self.shelf_model, [view["image"] for view in views],
            self.config.shelf_conf, self.config.shelf_imgsz, self.config.device,
            self.config.precision,
        )
        if self.config.synchronize_cuda_timing:
            _synchronize_cuda(self.config.device)
        shelf_done = time.perf_counter()
        if len(shelf_results) != len(views):
            raise RuntimeError(
                f"Shelf-model result count {len(shelf_results)} does not match "
                f"input count {len(views)}."
            )

        for view, shelf_result in zip(views, shelf_results):
            view["shelf_result"] = shelf_result
            extraction_started = time.perf_counter()
            shelves = extract_shelves(
                shelf_result, view["image"].shape, self.shelf_names, self.shelf_task
            )
            view["segmentation_done"] = time.perf_counter()
            view["shelf_postprocess_ms"] = (
                view["segmentation_done"] - extraction_started
            ) * 1000
            if view["recorder"]:
                view["recorder"].segmentation(view["image"], shelves)
            mapping_started = time.perf_counter()
            view["shelves"] = assign_pose_ids(
                shelves, view["candidates"], view["image"].shape, self.pose_config
            )
            view["mapping_started"] = mapping_started
            view["mapping_done"] = time.perf_counter()
            if view["recorder"]:
                view["recorder"].matching(
                    view["image"], view["shelves"], view["candidates"]
                )
            roi_started = time.perf_counter()
            self._prepare_empty_regions(view)
            view["roi_preparation_ms"] = (time.perf_counter() - roi_started) * 1000

        descriptor_started = time.perf_counter()
        descriptors = []
        for view_index, view in enumerate(views):
            if view["full_frame_mode"]:
                descriptors.append({
                    "camera_id": view["camera_id"], "view_index": view_index,
                    "kind": "full_frame", "region": None, "image": view["image"],
                })
            else:
                for region in view["regions"]:
                    region["camera_id"] = view["camera_id"]
                    descriptors.append({
                        "camera_id": view["camera_id"], "view_index": view_index,
                        "kind": "region", "region": region, "image": region["image"],
                    })
        descriptor_preparation_ms = (time.perf_counter() - descriptor_started) * 1000

        if self.config.synchronize_cuda_timing:
            _synchronize_cuda(self.config.device)
        empty_started = time.perf_counter()
        empty_results = _predict(
            self.empty_model, [item["image"] for item in descriptors],
            self.config.empty_conf, self.config.empty_imgsz, self.config.device,
            self.config.precision,
        )
        if self.config.synchronize_cuda_timing:
            _synchronize_cuda(self.config.device)
        empty_done = time.perf_counter()
        if len(empty_results) != len(descriptors):
            raise RuntimeError(
                f"Empty-model result count {len(empty_results)} does not match "
                f"combined input count {len(descriptors)}."
            )
        for view in views:
            view["empty_pairs"] = []
        for descriptor, result in zip(descriptors, empty_results):
            views[descriptor["view_index"]]["empty_pairs"].append((descriptor, result))

        outcomes = [
            self._finalize_view(view, shelf_done - shelf_started,
                                empty_done - empty_started)
            for view in views
        ]
        dual_done = time.perf_counter()
        total_empty_inputs = len(descriptors)
        timing = {
            "dual_total_ms": (dual_done - dual_started) * 1000,
            "shelf_batch_inference_ms": (shelf_done - shelf_started) * 1000,
            "shelf_batch_size": len(views),
            "left_shelf_postprocess_ms": self._metric(views, 0, "shelf_postprocess_ms"),
            "right_shelf_postprocess_ms": self._metric(views, 1, "shelf_postprocess_ms"),
            "left_mapping_ms": self._metric(views, 0, "mapping_ms"),
            "right_mapping_ms": self._metric(views, 1, "mapping_ms"),
            "left_roi_preparation_ms": self._metric(views, 0, "roi_preparation_ms"),
            "right_roi_preparation_ms": self._metric(views, 1, "roi_preparation_ms"),
            "descriptor_preparation_ms": descriptor_preparation_ms,
            "empty_batch_inference_ms": (empty_done - empty_started) * 1000,
            "empty_batch_size": total_empty_inputs,
            "left_association_ms": self._metric(views, 0, "association_ms"),
            "right_association_ms": self._metric(views, 1, "association_ms"),
            "left_visualization_render_ms": self._metric(views, 0, "visualization_ms"),
            "right_visualization_render_ms": self._metric(views, 1, "visualization_ms"),
        }
        counts = {
            "shelf_model_predict_calls": 1,
            "shelf_model_inputs": len(views),
            "empty_model_predict_calls": 1 if descriptors else 0,
            "left_empty_inputs": self._view_input_count(views, 0),
            "right_empty_inputs": self._view_input_count(views, 1),
            "total_empty_inputs": total_empty_inputs,
            "left_final_empty_spaces": self._outcome_count(outcomes, 0),
            "right_final_empty_spaces": self._outcome_count(outcomes, 1),
        }
        return outcomes, timing, counts

    @staticmethod
    def _metric(views, index, key):
        return views[index][key] if len(views) > index else 0.0

    @staticmethod
    def _view_input_count(views, index):
        return len(views[index]["empty_pairs"]) if len(views) > index else 0

    @staticmethod
    def _outcome_count(outcomes, index):
        return (outcomes[index].result["counts"]["final_empty_spaces"]
                if len(outcomes) > index else 0)

    def _prepare_view(self, spec):
        image, metadata = spec["image"], spec["metadata"]
        started = time.perf_counter()
        width, height = image.shape[1], image.shape[0]
        if metadata["resolution"] != [width, height]:
            raise ValueError("Metadata resolution does not match the image.")
        recorder = (
            LiveDebugRecorder(Path(spec["debug_dir"])) if spec.get("debug_dir") else None
        )
        candidates = generate_candidates(self.store_map, metadata, self.pose_config)
        if recorder:
            recorder.candidates(image, candidates)
        return {
            **spec, "started": started, "width": width, "height": height,
            "recorder": recorder, "candidates": candidates,
            "geometry_done": time.perf_counter(),
        }

    def _prepare_empty_regions(self, view):
        image, shelves, recorder = view["image"], view["shelves"], view["recorder"]
        full_frame_mode = self.config.empty_inference_mode == EmptyInferenceMode.FULL_FRAME_GATED
        shelf_level_mode = self.config.empty_inference_mode == EmptyInferenceMode.SHELF_LEVEL_ROI
        eligible = [shelf for shelf in shelves if shelf["shelf_id"] != UNKNOWN_SHELF]
        selected_indices = set()
        for shelf in shelves:
            shelf["roi_bbox"] = None
            shelf["empty_roi_mode"] = None
            shelf["empty_inference_source"] = (
                "full_frame" if full_frame_mode else self.config.empty_inference_mode.value
            )
            shelf["empty_inference_selected"] = (
                full_frame_mode and shelf["shelf_id"] != UNKNOWN_SHELF
            )
        regions = []
        if not full_frame_mode:
            selected = eligible if shelf_level_mode else eligible[:self.config.max_shelf_rois]
            selected_indices = {shelf["shelf_index"] for shelf in selected}
            for shelf in shelves:
                shelf["empty_inference_selected"] = shelf["shelf_index"] in selected_indices
                if shelf["empty_inference_selected"]:
                    padding = (self.config.shelf_level_roi_padding if shelf_level_mode
                               else self.config.roi_padding_ratio)
                    shelf["roi_bbox"] = padded_roi(shelf["bbox"], image.shape, padding)
                    if recorder:
                        recorder.shelf(image, shelf)
            regions = build_regions(
                shelves, image, self.config, "full_shelf" if shelf_level_mode else None
            )
            for region in regions:
                region["shelf"]["empty_roi_mode"] = region["kind"]
                region["inference_source"] = self.config.empty_inference_mode.value
                region["camera_id"] = view["camera_id"]
                if recorder:
                    recorder.production_input(region)
        view.update({
            "full_frame_mode": full_frame_mode, "shelf_level_mode": shelf_level_mode,
            "eligible_shelves": eligible, "selected_indices": selected_indices,
            "regions": regions,
        })

    def _finalize_view(self, view, shelf_elapsed, empty_elapsed):
        association_started = time.perf_counter()
        raw, accepted, rejected, per_region = [], [], [], []
        pairs = view["empty_pairs"]
        if view["full_frame_mode"]:
            if len(pairs) != 1:
                raise RuntimeError("Full-frame empty inference must return exactly one result per view.")
            raw, accepted, rejected = _full_frame_prediction_records(
                pairs[0][1], view["eligible_shelves"], view["image"].shape,
                self.empty_names, self.config.min_mask_overlap,
            )
        else:
            if len(pairs) != len(view["regions"]):
                raise RuntimeError("Demultiplexed empty result count does not match view ROI count.")
            for descriptor, result in pairs:
                region = descriptor["region"]
                if descriptor["camera_id"] != view["camera_id"]:
                    raise RuntimeError("Empty-model result was routed to the wrong camera.")
                region_raw, region_accepted, region_rejected = _prediction_records(
                    result, region, view["image"].shape, self.empty_names,
                    self.config.min_mask_overlap,
                )
                raw.extend(region_raw)
                accepted.extend(region_accepted)
                rejected.extend(region_rejected)
                per_region.append({
                    "region_id": region["region_id"],
                    "shelf_id": region["shelf"]["shelf_id"],
                    "parent_shelf_id": region["shelf"].get("parent_shelf_id"),
                    "shelf_level_id": region["shelf"].get("shelf_level_id"),
                    "roi_dimensions": [region["image"].shape[1], region["image"].shape[0]],
                    "production_predictions": region_raw,
                    "raw_predictions_conf001": [],
                })
        association_done = time.perf_counter()

        # This dedup call is deliberately per view: image coordinates never cross cameras.
        final, duplicates_removed = deduplicate(accepted, self.config.dedup_iou)
        shelves_by_index = {shelf["shelf_index"]: shelf for shelf in view["shelves"]}
        detections_by_shelf_index = {index: [] for index in shelves_by_index}
        for detection in final:
            shelf = shelves_by_index[detection["assigned_shelf_index"]]
            detection["section"] = section_for_box(
                detection["global_bbox_xyxy"], shelf["bbox"]
            )
            detections_by_shelf_index[shelf["shelf_index"]].append(detection)
        semantic_done = time.perf_counter()

        output_shelves = []
        for shelf in view["shelves"]:
            detections = detections_by_shelf_index[shelf["shelf_index"]]
            sections = {name: sum(item["section"] == name for item in detections)
                        for name in ("SOL", "ORTA", "SAĞ")}
            status = (
                "UNLOCALIZED" if shelf["shelf_id"] == UNKNOWN_SHELF else
                "ROI_LIMITED" if (not view["full_frame_mode"]
                                  and not shelf["empty_inference_selected"]) else
                "EMPTY_SPACE_DETECTED" if detections else "NO_EMPTY_SPACE"
            )
            polygon = contour_polygon(shelf["mask"]) if shelf["has_segmentation_mask"] else []
            output_shelves.append({
                "shelf_index": shelf["shelf_index"], "shelf_id": shelf["shelf_id"],
                "parent_shelf_id": shelf.get("parent_shelf_id", UNKNOWN_SHELF),
                "shelf_level_id": shelf.get("shelf_level_id"),
                "level_number": shelf.get("level_number"), "model_task": shelf["model_task"],
                "confidence": shelf["confidence"],
                "segmentation_confidence": shelf["segmentation_confidence"],
                "mapping_confidence": shelf.get("mapping_confidence", 0.0),
                "mapping_scores": shelf.get("mapping_scores", {}), "status": status,
                "mask_polygon": [{"x": int(point[0]), "y": int(point[1])}
                                 for point in polygon],
                "bbox_xyxy": shelf["bbox"], "global_bbox_xyxy": shelf["bbox"],
                "roi_bbox_xyxy": shelf["roi_bbox"],
                "empty_roi_mode": shelf["empty_roi_mode"],
                "empty_inference_selected": shelf["empty_inference_selected"],
                "empty_inference_source": shelf["empty_inference_source"],
                "empty_space_count": len(detections), "sections": sections,
                "detections": detections,
            })
        output_shelves.sort(key=lambda shelf: (
            shelf["parent_shelf_id"] == UNKNOWN_SHELF, shelf["parent_shelf_id"],
            shelf["level_number"] if shelf["level_number"] is not None else 1_000_000,
            shelf["shelf_index"],
        ))

        visualization_started = time.perf_counter()
        native_plot, annotated = render_visualization(
            view["shelf_result"], view["image"], final
        )
        visualization_done = time.perf_counter()
        view["mapping_ms"] = (view["mapping_done"] - view["mapping_started"]) * 1000
        view["association_ms"] = (association_done - association_started) * 1000
        view["visualization_ms"] = (visualization_done - visualization_started) * 1000
        input_count = len(pairs)
        result = {
            "frame_id": view["metadata"]["frame_id"],
            "image": {"width": view["width"], "height": view["height"]},
            "strategy": ("full_frame_empty_mask_gated_with_pose_mapping"
                         if view["full_frame_mode"] else
                         "detected_shelf_level_roi_batch_with_pose_mapping"
                         if view["shelf_level_mode"] else
                         "shelf_model_roi_cascade_with_pose_mapping"),
            "model_tasks": {"shelf": self.shelf_task, "empty_shelf": self.empty_task},
            "empty_inference_mode": self.config.empty_inference_mode.value,
            "empty_roi_mode": (None if view["full_frame_mode"] else "full_shelf"
                               if view["shelf_level_mode"] else self.config.empty_roi_mode),
            "thresholds": {
                "shelf_conf": self.config.shelf_conf, "empty_conf": self.config.empty_conf,
                "min_mask_overlap": self.config.min_mask_overlap,
                "dedup_iou": self.config.dedup_iou,
                "max_shelf_rois": self.config.max_shelf_rois,
                "shelf_level_roi_padding": self.config.shelf_level_roi_padding,
            },
            "timing": {
                "geometry_seconds": view["geometry_done"] - view["started"],
                "shelf_segmentation_seconds": shelf_elapsed,
                "mapping_seconds": view["mapping_done"] - view["mapping_started"],
                "empty_detection_seconds": semantic_done - view["mapping_done"],
                "total_seconds": visualization_done - view["started"],
                "shelf_inference_ms": shelf_elapsed * 1000,
                "empty_shelf_total_inference_ms": empty_elapsed * 1000,
                "empty_batch_inference_ms": (empty_elapsed * 1000
                                             if not view["full_frame_mode"] and input_count else 0.0),
                "empty_full_frame_inference_ms": (empty_elapsed * 1000
                                                  if view["full_frame_mode"] else 0.0),
                "parent_association_ms": view["mapping_ms"],
                "empty_association_ms": view["association_ms"],
                "association_ms": view["mapping_ms"] + view["association_ms"],
                "visualization_render_ms": view["visualization_ms"],
                "total_pipeline_ms": (visualization_done - view["started"]) * 1000,
            },
            "counts": {
                "visible_candidates": len(view["candidates"]),
                "segmented_shelves": len(view["shelves"]),
                "detected_shelves": len(view["shelves"]),
                "matched_shelves": sum(s["shelf_id"] != UNKNOWN_SHELF for s in view["shelves"]),
                "matched_parent_regions": len({s["parent_shelf_id"] for s in view["shelves"]
                                               if s.get("parent_shelf_id") != UNKNOWN_SHELF}),
                "unknown_shelves": sum(s["shelf_id"] == UNKNOWN_SHELF for s in view["shelves"]),
                "empty_inference_calls": 1 if input_count else 0,
                "empty_model_predict_calls": 1 if input_count else 0,
                "empty_inference_inputs": input_count,
                "roi_inferences": len(view["regions"]),
                "shelf_rois_selected": len(view["selected_indices"]),
                "shelf_rois_limited": (0 if view["full_frame_mode"] or view["shelf_level_mode"]
                                       else max(0, len(view["eligible_shelves"])
                                                - len(view["selected_indices"]))),
                "raw_empty_predictions": len(raw), "mask_rejected": len(rejected),
                "duplicates_removed": duplicates_removed, "final_empty_spaces": len(final),
            },
            "shelves": output_shelves, "rejected_detections": rejected,
        }
        if view["recorder"]:
            self._record_debug(view, result, native_plot, annotated, raw, accepted,
                               rejected, final, per_region)
        return PipelineOutcome(result, annotated, native_plot)

    def _record_debug(self, view, result, native_plot, annotated, raw, accepted,
                      rejected, final, per_region):
        recorder, image = view["recorder"], view["image"]
        recorder.visualization(image, native_plot, annotated)
        diagnostics = _predict(
            self.empty_model, [region["image"] for region in view["regions"]],
            0.01, self.config.empty_imgsz, self.config.device,
        )
        production_results = [pair[1] for pair in view["empty_pairs"]]
        for index, (region, diagnostic, production) in enumerate(
                zip(view["regions"], diagnostics, production_results)):
            low = low_confidence_records(
                diagnostic, region, image.shape, self.config.empty_conf,
                self.config.min_mask_overlap, self.empty_names,
            )
            per_region[index]["raw_predictions_conf001"] = low
            region_accepted = [item for item in accepted if item["region_id"] == region["region_id"]
                               and item["assigned_shelf_id"] == region["shelf"]["shelf_id"]]
            region_rejected = [item for item in rejected if item["region_id"] == region["region_id"]
                               and item["assigned_shelf_id"] == region["shelf"]["shelf_id"]]
            recorder.region_predictions(region, low, per_region[index]["production_predictions"],
                                        region_accepted, region_rejected)
        full_frame = recorder.full_frame_control(
            image, self.empty_model, .01, self.config.empty_imgsz,
            self.config.device, self.empty_names,
        )
        comparison_shelves = []
        for shelf in view["shelves"]:
            comparison = dict(shelf)
            comparison["empty_inference_selected"] = True
            comparison["roi_bbox"] = padded_roi(
                shelf["bbox"], image.shape, self.config.roi_padding_ratio
            )
            comparison_shelves.append(comparison)
            recorder.shelf(image, comparison)
        full_regions = build_regions(comparison_shelves, image, self.config, "full_shelf")
        full_shelf = [item for region in full_regions for item in recorder.full_shelf_control(
            region, self.empty_model, .01, self.config.empty_imgsz, self.config.device,
            self.empty_names, image.shape,
        )]
        tile_low, tile_production = [], []
        for region in build_regions(comparison_shelves, image, self.config, "tiles"):
            low_result = _predict(self.empty_model, [region["image"]], .01,
                                  self.config.empty_imgsz, self.config.device)[0]
            production_result = _predict(
                self.empty_model, [region["image"]], self.config.empty_conf,
                self.config.empty_imgsz, self.config.device,
            )[0]
            low, production = recorder.tile_control(
                region, low_result, production_result, self.config.empty_conf,
                self.empty_names, image.shape,
            )
            tile_low.extend(low)
            tile_production.extend(production)
        failure_stage = ("none" if final else "shelf_mask_gate_rejected" if raw else
                         "empty_model_no_prediction" if view["full_frame_mode"]
                         or view["regions"] else "no_shelf_roi")
        debug = {
            "debug_only": True, "directory": str(recorder.directory),
            "failure_stage": failure_stage, "full_frame_raw": len(full_frame),
            "full_shelf_roi_raw": len(full_shelf), "tile_raw": len(tile_low),
            "tile_production_raw": len(tile_production), "production_raw": len(raw),
            "mask_rejected": len(rejected), "final": len(final),
            "per_shelf": stage_counts(view["shelves"], view["regions"], per_region,
                                      accepted, rejected, final),
        }
        result["debug"] = debug
        recorder.summary({**debug, "frame_id": view["metadata"]["frame_id"],
                          "production_regions": per_region, "final_detections": final})
        recorder.production_final(image, view["shelves"], raw, accepted,
                                  rejected, final, result)
