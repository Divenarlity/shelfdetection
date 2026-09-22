from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.live_debug import LiveDebugRecorder, low_confidence_records, stage_counts
from src.model_contracts import validate_model_contract
from src.pose_geometry import UNKNOWN_SHELF, assign_pose_ids, generate_candidates
from src.postprocessing import (
    contour_polygon,
    deduplicate,
    local_to_global,
    mask_bbox,
    mask_box_relation,
    padded_roi,
    roi_boxes,
    section_for_box,
)


@dataclass(frozen=True)
class PipelineConfig:
    shelf_conf: float = 0.25
    empty_conf: float = 0.10
    shelf_imgsz: int = 960
    empty_imgsz: int = 640
    roi_padding_ratio: float = 0.02
    empty_roi_mode: str = "full_shelf"
    min_mask_overlap: float = 0.30
    dedup_iou: float = 0.50
    tile_size: int = 384
    tile_overlap: float = 0.25
    device: str | None = None

    def __post_init__(self):
        for name in ("shelf_conf", "empty_conf", "min_mask_overlap", "dedup_iou"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.roi_padding_ratio < 0:
            raise ValueError("roi_padding_ratio cannot be negative")
        if self.empty_roi_mode not in {"full_shelf", "tiles"}:
            raise ValueError("empty_roi_mode must be full_shelf or tiles")
        if self.tile_size <= 0 or not 0 <= self.tile_overlap < 1:
            raise ValueError("tile_size must be positive and tile_overlap must be in [0, 1)")


@dataclass
class PipelineOutcome:
    result: dict
    annotated_image: np.ndarray


def extract_shelves(result, image_shape, shelf_names):
    if result.masks is None:
        if getattr(result, "boxes", None) is not None and len(result.boxes) == 0:
            return []
        raise RuntimeError("Shelf segmentation produced no masks; box fallback is forbidden.")
    if result.boxes is None:
        raise RuntimeError("Shelf masks are missing class/confidence boxes.")
    h, w = image_shape[:2]
    masks = result.masks.data.cpu().numpy()
    confidences = result.boxes.conf.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    if len(masks) != len(confidences) or len(masks) != len(classes):
        raise RuntimeError("Shelf mask and box counts differ.")
    shelves = []
    for raw_mask, confidence, class_id in zip(masks, confidences, classes):
        if shelf_names.get(int(class_id)) != "shelves":
            continue
        mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST) >= 0.5
        bbox = mask_bbox(mask)
        if bbox is None:
            continue
        ys, xs = np.where(mask)
        shelves.append({
            "mask": mask,
            "bbox": bbox,
            "centroid": [float(xs.mean()), float(ys.mean())],
            "segmentation_confidence": float(confidence),
        })
    return shelves


def _predict(model, images, conf, imgsz, device=None):
    if not images:
        return []
    kwargs = {"source": images if len(images) > 1 else images[0], "conf": conf,
              "imgsz": imgsz, "verbose": False}
    if device:
        kwargs["device"] = device
    results = model.predict(**kwargs)
    return list(results) if isinstance(results, (list, tuple)) else [results]


def build_regions(shelves, image, config: PipelineConfig, mode=None):
    """Create empty-model inputs for matched shelves only."""
    selected_mode = mode or config.empty_roi_mode
    regions = []
    for shelf in shelves:
        if shelf["shelf_id"] == UNKNOWN_SHELF:
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
            region_id = "FULL-SHELF" if kind == "full_shelf" else f"TILE-{index:02d}"
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
            "region_id": region["region_id"],
            "tile_id": region["tile_id"],
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


def _render(image, shelves, detections):
    canvas, overlay = image.copy(), image.copy()
    for shelf in shelves:
        color = (110, 110, 110) if shelf["shelf_id"] == UNKNOWN_SHELF else (0, 150, 0)
        overlay[shelf["mask"]] = color
        x1, y1, x2, y2 = shelf["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, shelf["shelf_id"], (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
    canvas = cv2.addWeighted(canvas, .72, overlay, .28, 0)
    for detection in detections:
        x1, y1, x2, y2 = map(lambda value: int(round(value)), detection["global_bbox_xyxy"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"{detection['assigned_shelf_id']} {detection['section']} {detection['confidence']:.2f}"
        cv2.putText(canvas, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 255), 2, cv2.LINE_AA)
    return canvas


class ShelfInferencePipeline:
    """Single authoritative Unity pose -> shelf mask -> matched shelf ROI pipeline."""

    def __init__(self, shelf_model, empty_model, store_map, pose_config,
                 config: PipelineConfig | None = None):
        self.shelf_model = shelf_model
        self.empty_model = empty_model
        self.store_map = store_map
        self.pose_config = pose_config
        self.config = config or PipelineConfig()
        self.shelf_names = validate_model_contract(shelf_model, "segment", "shelves", "Shelf model")
        self.empty_names = validate_model_contract(empty_model, "detect", "empty_shelf", "Empty model")

    def infer(self, image, metadata, debug_dir: Path | None = None):
        started = time.perf_counter()
        width, height = image.shape[1], image.shape[0]
        if metadata["resolution"] != [width, height]:
            raise ValueError("Metadata resolution does not match the image.")
        recorder = LiveDebugRecorder(Path(debug_dir)) if debug_dir else None

        candidates = generate_candidates(self.store_map, metadata, self.pose_config)
        if recorder:
            recorder.candidates(image, candidates)
        geometry_done = time.perf_counter()

        shelf_kwargs = {"source": image, "conf": self.config.shelf_conf,
                        "imgsz": self.config.shelf_imgsz, "verbose": False}
        if self.config.device:
            shelf_kwargs["device"] = self.config.device
        shelf_result = self.shelf_model.predict(**shelf_kwargs)[0]
        shelves = extract_shelves(shelf_result, image.shape, self.shelf_names)
        if recorder:
            recorder.segmentation(image, shelves)
        segmentation_done = time.perf_counter()

        shelves = assign_pose_ids(shelves, candidates, image.shape, self.pose_config)
        if recorder:
            recorder.matching(image, shelves, candidates)
        mapping_done = time.perf_counter()

        for shelf in shelves:
            shelf["roi_bbox"] = None
            shelf["empty_roi_mode"] = None
            shelf["status"] = "UNLOCALIZED" if shelf["shelf_id"] == UNKNOWN_SHELF else "PROCESSED"
            if shelf["shelf_id"] != UNKNOWN_SHELF:
                shelf["roi_bbox"] = padded_roi(
                    shelf["bbox"], image.shape, self.config.roi_padding_ratio,
                )
                if recorder:
                    recorder.shelf(image, shelf)

        regions = build_regions(shelves, image, self.config)
        for region in regions:
            region["shelf"]["empty_roi_mode"] = region["kind"]
            if recorder:
                recorder.production_input(region)
        empty_results = _predict(
            self.empty_model, [region["image"] for region in regions],
            self.config.empty_conf, self.config.empty_imgsz, self.config.device,
        )
        if len(empty_results) != len(regions):
            raise RuntimeError("Empty-model result count does not match ROI count.")

        raw, accepted, rejected = [], [], []
        per_region = []
        for region, result in zip(regions, empty_results):
            region_raw, region_accepted, region_rejected = _prediction_records(
                result, region, image.shape, self.empty_names, self.config.min_mask_overlap,
            )
            raw.extend(region_raw)
            accepted.extend(region_accepted)
            rejected.extend(region_rejected)
            per_region.append({
                "region_id": region["region_id"],
                "shelf_id": region["shelf"]["shelf_id"],
                "production_predictions": region_raw,
                "raw_predictions_conf001": [],
            })
        final, duplicates_removed = deduplicate(accepted, self.config.dedup_iou)
        for detection in final:
            shelf = next(item for item in shelves if item["shelf_id"] == detection["assigned_shelf_id"])
            detection["section"] = section_for_box(detection["global_bbox_xyxy"], shelf["bbox"])
        empty_done = time.perf_counter()

        output_shelves = []
        for shelf in shelves:
            detections = [item for item in final if item["assigned_shelf_id"] == shelf["shelf_id"]]
            sections = {name: sum(item["section"] == name for item in detections)
                        for name in ("SOL", "ORTA", "SAĞ")}
            status = (
                "UNLOCALIZED" if shelf["shelf_id"] == UNKNOWN_SHELF else
                "EMPTY_SPACE_DETECTED" if detections else
                "NO_EMPTY_SPACE"
            )
            output_shelves.append({
                "shelf_id": shelf["shelf_id"],
                "segmentation_confidence": shelf["segmentation_confidence"],
                "mapping_confidence": shelf.get("mapping_confidence", 0.0),
                "mapping_scores": shelf.get("mapping_scores", {}),
                "status": status,
                "mask_polygon": contour_polygon(shelf["mask"]),
                "bbox_xyxy": shelf["bbox"],
                "roi_bbox_xyxy": shelf["roi_bbox"],
                "empty_roi_mode": shelf["empty_roi_mode"],
                "empty_space_count": len(detections),
                "sections": sections,
                "detections": detections,
            })

        result = {
            "frame_id": metadata["frame_id"],
            "image": {"width": width, "height": height},
            "strategy": "pose_matched_shelf_roi_cascade",
            "empty_roi_mode": self.config.empty_roi_mode,
            "thresholds": {
                "shelf_conf": self.config.shelf_conf,
                "empty_conf": self.config.empty_conf,
                "min_mask_overlap": self.config.min_mask_overlap,
                "dedup_iou": self.config.dedup_iou,
            },
            "timing": {
                "geometry_seconds": geometry_done - started,
                "shelf_segmentation_seconds": segmentation_done - geometry_done,
                "mapping_seconds": mapping_done - segmentation_done,
                "empty_detection_seconds": empty_done - mapping_done,
                "total_seconds": empty_done - started,
            },
            "counts": {
                "visible_candidates": len(candidates),
                "segmented_shelves": len(shelves),
                "matched_shelves": sum(shelf["shelf_id"] != UNKNOWN_SHELF for shelf in shelves),
                "unknown_shelves": sum(shelf["shelf_id"] == UNKNOWN_SHELF for shelf in shelves),
                "roi_inferences": len(regions),
                "raw_empty_predictions": len(raw),
                "mask_rejected": len(rejected),
                "duplicates_removed": duplicates_removed,
                "final_empty_spaces": len(final),
            },
            "shelves": output_shelves,
            "rejected_detections": rejected,
        }

        if recorder:
            diagnostics = _predict(
                self.empty_model, [region["image"] for region in regions],
                0.01, self.config.empty_imgsz, self.config.device,
            )
            for index, (region, diagnostic, production) in enumerate(zip(regions, diagnostics, empty_results)):
                low = low_confidence_records(
                    diagnostic, region, image.shape, self.config.empty_conf,
                    self.config.min_mask_overlap, self.empty_names,
                )
                per_region[index]["raw_predictions_conf001"] = low
                region_accepted = [item for item in accepted if item["region_id"] == region["region_id"]
                                   and item["assigned_shelf_id"] == region["shelf"]["shelf_id"]]
                region_rejected = [item for item in rejected if item["region_id"] == region["region_id"]
                                   and item["assigned_shelf_id"] == region["shelf"]["shelf_id"]]
                recorder.region_predictions(
                    region, low, per_region[index]["production_predictions"],
                    region_accepted, region_rejected,
                )
            full_frame = recorder.full_frame_control(
                image, self.empty_model, .01, self.config.empty_imgsz,
                self.config.device, self.empty_names,
            )
            full_shelf_regions = build_regions(shelves, image, self.config, "full_shelf")
            full_shelf = [
                item
                for region in full_shelf_regions
                for item in recorder.full_shelf_control(
                    region, self.empty_model, .01, self.config.empty_imgsz,
                    self.config.device, self.empty_names, image.shape,
                )
            ]
            tile_regions = build_regions(shelves, image, self.config, "tiles")
            tile_low, tile_production = [], []
            for region in tile_regions:
                low_result = _predict(self.empty_model, [region["image"]], .01,
                                      self.config.empty_imgsz, self.config.device)[0]
                prod_result = _predict(self.empty_model, [region["image"]], self.config.empty_conf,
                                       self.config.empty_imgsz, self.config.device)[0]
                low, production = recorder.tile_control(
                    region, low_result, prod_result, self.config.empty_conf,
                    self.empty_names, image.shape,
                )
                tile_low.extend(low)
                tile_production.extend(production)
            failure_stage = (
                "none" if final else
                "empty_model_context_sensitivity" if full_frame or full_shelf else
                "empty_model_no_prediction" if regions else
                "no_matched_shelf_roi"
            )
            debug = {
                "debug_only": True,
                "directory": str(recorder.directory),
                "failure_stage": failure_stage,
                "full_frame_raw": len(full_frame),
                "full_shelf_roi_raw": len(full_shelf),
                "tile_raw": len(tile_low),
                "tile_production_raw": len(tile_production),
                "production_raw": len(raw),
                "mask_rejected": len(rejected),
                "final": len(final),
                "per_shelf": stage_counts(shelves, regions, per_region, accepted, rejected, final),
            }
            result["debug"] = debug
            recorder.summary({**debug, "frame_id": metadata["frame_id"],
                              "production_regions": per_region,
                              "final_detections": final})
            recorder.production_final(image, shelves, raw, accepted, rejected, final, result)

        return PipelineOutcome(result, _render(image, shelves, final))
