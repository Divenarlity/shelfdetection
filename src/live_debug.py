"""Opt-in visual and JSON evidence for the live shelf cascade."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.postprocessing import local_to_global, mask_box_relation


def low_confidence_records(result, region, image_shape, production_conf, min_mask_overlap,
                           empty_names):
    """Describe diagnostic predictions without changing production decisions."""
    if result.boxes is None:
        return []
    boxes = result.boxes
    records = []
    for box, confidence, class_id in zip(boxes.xyxy.cpu().tolist(),
                                          boxes.conf.cpu().tolist(),
                                          boxes.cls.cpu().tolist()):
        if empty_names.get(int(class_id)) != "empty_shelf":
            continue
        global_box = local_to_global(box, region["bbox"], image_shape)
        passes_mask, overlap, center_inside = mask_box_relation(
            region["shelf"]["mask"], global_box, min_mask_overlap)
        passes_confidence = float(confidence) >= production_conf
        reason = ("below_production_confidence" if not passes_confidence else
                  "center_outside_and_mask_overlap_below_threshold" if not passes_mask else
                  "passes_diagnostic_checks")
        records.append({
            "confidence": float(confidence),
            "local_bbox_xyxy": [float(v) for v in box],
            "global_bbox_xyxy": global_box,
            "assigned_shelf_id": region["shelf"]["shelf_id"],
            "parent_shelf_id": region["shelf"].get("parent_shelf_id"),
            "shelf_level_id": region["shelf"].get("shelf_level_id"),
            "region_id": region["region_id"],
            "tile_id": region["tile_id"],
            "center_inside_mask": center_inside,
            "mask_overlap_ratio": overlap,
            "passes_confidence": passes_confidence,
            "passes_mask_filter": passes_mask,
            "reason": reason,
        })
    return records


def stage_counts(shelves, regions, diagnostic_regions, accepted, rejected, final):
    """Count the actual production and diagnostic stages per matched shelf."""
    result = {}
    for shelf in shelves:
        shelf_id = shelf["shelf_id"]
        if shelf_id == "UNKNOWN_SHELF":
            continue
        before = sum(item["assigned_shelf_id"] == shelf_id for item in accepted)
        after = sum(item["assigned_shelf_id"] == shelf_id for item in final)
        result[shelf_id] = {
            "tiles": sum(region["shelf"] is shelf and region["kind"] == "tile"
                         for region in regions),
            "raw_conf001": sum(len(item["raw_predictions_conf001"])
                               for item in diagnostic_regions if item["shelf_id"] == shelf_id),
            "raw_production_conf": sum(len(item["production_predictions"])
                                       for item in diagnostic_regions if item["shelf_id"] == shelf_id),
            "rejected_by_mask": sum(item["assigned_shelf_id"] == shelf_id for item in rejected),
            "accepted_before_dedup": before,
            "duplicates_removed": before - after,
            "final": after,
        }
    return result


def _write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save live debug image: {path}")


def _label(image, text, position, color=(255, 255, 255)):
    x, y = map(int, position)
    cv2.putText(image, text, (max(2, x), max(18, y)), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (max(2, x), max(18, y)), cv2.FONT_HERSHEY_SIMPLEX,
                .55, color, 1, cv2.LINE_AA)


def _box(image, coords, color, text=None):
    x1, y1, x2, y2 = [int(round(v)) for v in coords]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    if text:
        _label(image, text, (x1, y1 - 5), color)


class LiveDebugRecorder:
    def __init__(self, directory: Path):
        self.directory = directory

    def candidates(self, image, candidates):
        canvas = image.copy()
        for candidate in candidates:
            polygon = np.rint(candidate["polygon"]).astype(np.int32)
            cv2.polylines(canvas, [polygon], True, (0, 255, 255), 2)
            x, y = polygon[0]
            _label(canvas, f"{candidate['shelf_id']} front={candidate['front_alignment']:.2f}",
                   (x, y - 5), (0, 255, 255))
        _write_image(self.directory / "geometric_candidates.jpg", canvas)

    def segmentation(self, image, shelves):
        canvas, overlay = image.copy(), image.copy()
        for index, shelf in enumerate(shelves):
            color = ((0, 255, 0), (255, 180, 0), (255, 0, 255))[index % 3]
            overlay[shelf["mask"]] = color
        canvas = cv2.addWeighted(canvas, .65, overlay, .35, 0)
        for index, shelf in enumerate(shelves):
            color = ((0, 255, 0), (255, 180, 0), (255, 0, 255))[index % 3]
            _box(canvas, shelf["bbox"], color,
                 f"mask {index+1} conf={shelf['segmentation_confidence']:.2f}")
        _write_image(self.directory / "shelf_segmentation.jpg", canvas)

    def matching(self, image, shelves, candidates):
        canvas, overlay = image.copy(), image.copy()
        for shelf in shelves:
            overlay[shelf["mask"]] = (100, 100, 100) if shelf["shelf_id"] == "UNKNOWN_SHELF" else (0, 180, 0)
        canvas = cv2.addWeighted(canvas, .65, overlay, .35, 0)
        for candidate in candidates:
            cv2.polylines(canvas, [np.rint(candidate["polygon"]).astype(np.int32)],
                          True, (0, 255, 255), 2)
        for shelf in shelves:
            _box(canvas, shelf["bbox"], (255, 255, 255),
                 f"{shelf['shelf_id']} map={shelf.get('mapping_confidence', 0):.2f}")
        _write_image(self.directory / "shelf_matching.jpg", canvas)

    def visualization(self, input_image, native_shelf_plot, final_annotated):
        """Save the exact three-stage production visualization for inspection."""
        _write_image(self.directory / "01_input.jpg", input_image)
        _write_image(self.directory / "02_native_shelf_plot.jpg", native_shelf_plot)
        _write_image(self.directory / "03_final_annotated.jpg", final_annotated)
        # Stable names are convenient for tools that do not sort numbered stages.
        if not (self.directory / "incoming_frame.jpg").exists():
            _write_image(self.directory / "incoming_frame.jpg", input_image)
        _write_image(self.directory / "native_shelf_plot.jpg", native_shelf_plot)
        _write_image(self.directory / "final_annotated.jpg", final_annotated)

    def shelf(self, image, shelf):
        directory = self.directory / shelf["shelf_id"]
        directory.mkdir(exist_ok=True)
        x1, y1, x2, y2 = shelf["roi_bbox"]
        crop = image[y1:y2, x1:x2]
        _write_image(directory / "full_shelf_crop.jpg", crop)
        # PNG is the lossless proof of the exact array supplied to YOLO.
        _write_image(directory / "full_shelf_crop.png", crop)
        _write_image(directory / "shelf_mask.png",
                     shelf["mask"][y1:y2, x1:x2].astype(np.uint8) * 255)

    def tile_input(self, region):
        directory = self.directory / region["shelf"]["shelf_id"]
        stem = region["tile_id"].lower().replace("-", "_")
        _write_image(directory / f"{stem}_input.jpg", region["image"])
        # PNG preserves the exact BGR array passed to YOLO; JPEG is for quick viewing.
        _write_image(directory / f"{stem}_input.png", region["image"])
        x1, y1, x2, y2 = region["bbox"]
        _write_image(directory / f"{stem}_mask.png",
                     region["shelf"]["mask"][y1:y2, x1:x2].astype(np.uint8) * 255)

    def production_input(self, region):
        shelf_level_id = region["shelf"].get("shelf_level_id")
        if shelf_level_id:
            _write_image(self.directory / f"{shelf_level_id}_input.jpg", region["image"])
        if region["kind"] == "tile":
            self.tile_input(region)

    def region_predictions(self, region, low_confidence, production_raw, accepted, rejected):
        directory = self.directory / region["shelf"]["shelf_id"]
        stem = region["region_id"].lower().replace("-", "_")
        raw_image = region["image"].copy()
        accepted_image = region["image"].copy()
        rejected_image = region["image"].copy()
        for item in low_confidence:
            _box(raw_image, item["local_bbox_xyxy"], (255, 0, 255),
                 f"{item['confidence']:.2f}")
        for item in production_raw:
            _box(raw_image, item["roi_bbox_xyxy"], (0, 165, 255),
                 f"production {item['confidence']:.2f}")
        for item in accepted:
            _box(accepted_image, item["roi_bbox_xyxy"], (0, 220, 0),
                 f"{item['confidence']:.2f}")
        for item in rejected:
            _box(rejected_image, item["roi_bbox_xyxy"], (0, 0, 255),
                 f"{item['confidence']:.2f} mask reject")
        prefix = "production_" if region["kind"] == "full_shelf" else ""
        _write_image(directory / f"{prefix}{stem}_raw_conf001.jpg", raw_image)
        _write_image(directory / f"{prefix}{stem}_accepted.jpg", accepted_image)
        _write_image(directory / f"{prefix}{stem}_rejected.jpg", rejected_image)
        shelf_level_id = region["shelf"].get("shelf_level_id")
        if shelf_level_id:
            _write_image(self.directory / f"{shelf_level_id}_empty_result.jpg", raw_image)
        (directory / f"{stem}_predictions.json").write_text(
            json.dumps({"region_id": region["region_id"],
                        "parent_shelf_id": region["shelf"].get("parent_shelf_id"),
                        "shelf_level_id": shelf_level_id,
                        "roi_dimensions": [region["image"].shape[1], region["image"].shape[0]],
                        "raw_predictions_conf001": low_confidence,
                        "raw_predictions_production_conf": production_raw,
                        "accepted_before_dedup": accepted,
                        "rejected_by_mask": rejected}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    @staticmethod
    def prediction_records(result, origin, image_shape, empty_names):
        records = []
        if result.boxes is None:
            return records
        for box, confidence, class_id in zip(result.boxes.xyxy.cpu().tolist(),
                                              result.boxes.conf.cpu().tolist(),
                                              result.boxes.cls.cpu().tolist()):
            if empty_names.get(int(class_id)) != "empty_shelf":
                continue
            records.append({"confidence": float(confidence),
                            "local_bbox_xyxy": [float(v) for v in box],
                            "global_bbox_xyxy": local_to_global(box, origin, image_shape)})
        return records

    @staticmethod
    def _predict_one(model, image, conf, imgsz, device):
        kwargs = {"source": image, "conf": conf, "imgsz": imgsz, "verbose": False}
        if device:
            kwargs["device"] = device
        return model.predict(**kwargs)[0]

    def full_frame_control(self, image, model, conf, imgsz, device, empty_names):
        result = self._predict_one(model, image, conf, imgsz, device)
        canvas = image.copy()
        records = self.prediction_records(
            result, [0, 0, image.shape[1], image.shape[0]], image.shape, empty_names)
        for item in records:
            _box(canvas, item["local_bbox_xyxy"], (255, 0, 255), f"{item['confidence']:.2f}")
        _write_image(self.directory / "control_full_frame.jpg", canvas)
        (self.directory / "control_full_frame.json").write_text(
            json.dumps({"debug_only": True, "confidence_threshold": conf,
                         "predictions": records}, indent=2), encoding="utf-8")
        return records

    def full_shelf_control(self, region, model, conf, imgsz, device, empty_names, image_shape):
        result = self._predict_one(model, region["image"], conf, imgsz, device)
        records = self.prediction_records(result, region["bbox"], image_shape, empty_names)
        directory = self.directory / region["shelf"]["shelf_id"]
        canvas = region["image"].copy()
        for item in records:
            _box(canvas, item["local_bbox_xyxy"], (255, 0, 255), f"{item['confidence']:.2f}")
        _write_image(directory / "control_full_shelf_roi.jpg", region["image"])
        _write_image(directory / "control_full_shelf_roi.png", region["image"])
        _write_image(directory / "control_full_shelf_roi_predictions.jpg", canvas)
        (directory / "control_full_shelf_roi.json").write_text(json.dumps({
            "debug_only": True, "confidence_threshold": conf,
            "roi_bbox_xyxy": region["bbox"], "predictions": records,
        }, indent=2), encoding="utf-8")
        return records

    def tile_control(self, region, low_result, production_result, production_conf,
                     empty_names, image_shape):
        self.tile_input(region)
        low = self.prediction_records(low_result, region["bbox"], image_shape, empty_names)
        production = self.prediction_records(
            production_result, region["bbox"], image_shape, empty_names)
        directory = self.directory / region["shelf"]["shelf_id"]
        stem = region["tile_id"].lower().replace("-", "_")
        canvas = region["image"].copy()
        for item in low:
            _box(canvas, item["local_bbox_xyxy"], (255, 0, 255), f"{item['confidence']:.2f}")
        _write_image(directory / f"{stem}_raw_conf001.jpg", canvas)
        (directory / f"{stem}_predictions.json").write_text(json.dumps({
            "tile_id": region["tile_id"], "raw_predictions_conf001": low,
            "production_confidence_threshold": production_conf,
            "raw_predictions_production_conf": production,
        }, indent=2), encoding="utf-8")
        return low, production

    def production_final(self, image, shelves, raw, accepted, rejected, final, payload):
        canvas, overlay = image.copy(), image.copy()
        for shelf in shelves:
            overlay[shelf["mask"]] = (100, 100, 100) if shelf["shelf_id"] == "UNKNOWN_SHELF" else (0, 120, 0)
            if shelf.get("roi_bbox"):
                _box(canvas, shelf["roi_bbox"], (0, 255, 255), f"ROI {shelf['shelf_id']}")
        canvas = cv2.addWeighted(canvas, .72, overlay, .28, 0)
        _label(canvas, "ROI yellow | raw magenta | accepted orange | rejected red | FINAL green",
               (8, 22), (255, 255, 255))
        for item in raw:
            _box(canvas, item["global_bbox_xyxy"], (255, 0, 255))
        for item in rejected:
            _box(canvas, item["global_bbox_xyxy"], (0, 0, 255), item.get("reason", "rejected"))
        for item in accepted:
            _box(canvas, item["global_bbox_xyxy"], (0, 165, 255))
        for item in final:
            _box(canvas, item["global_bbox_xyxy"], (0, 255, 0),
                 f"FINAL {item['confidence']:.3f}")
        _write_image(self.directory / "production_final.jpg", canvas)
        (self.directory / "production_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def summary(self, summary):
        (self.directory / "debug_counts.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
