"""Create standalone vs shelf-ROI vs full-frame-gated real-model evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2

from src.pipeline import (
    _full_frame_prediction_records,
    _predict,
    _prediction_records,
    build_regions,
    extract_shelves,
    PipelineConfig,
)
from src.postprocessing import deduplicate, padded_roi

ROOT = Path(__file__).resolve().parent


def _resolve(path: Path):
    return (path if path.is_absolute() else ROOT / path).resolve()


def _write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def _draw_box(image, box, color, label):
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (max(2, x1), max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, (max(2, x1), max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)


def _draw_shelves(image, shelves):
    canvas, overlay = image.copy(), image.copy()
    for shelf in shelves:
        overlay[shelf["mask"]] = (255, 220, 0)
    canvas = cv2.addWeighted(canvas, .82, overlay, .18, 0)
    for shelf in shelves:
        _draw_box(canvas, shelf["bbox"], (255, 220, 0),
                  f"SHELF {shelf['shelf_index']} {shelf['confidence']:.2f}")
    return canvas


def _standalone_records(result, image_shape, names):
    records = []
    if result.boxes is None:
        return records
    h, w = image_shape[:2]
    for box, confidence, class_id in zip(
        result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
    ):
        class_id = int(class_id)
        if names.get(class_id) != "empty_shelf":
            continue
        records.append({
            "confidence": float(confidence),
            "class_id": class_id,
            "class_name": names[class_id],
            "global_bbox_xyxy": [
                max(0.0, min(float(w), float(box[0]))),
                max(0.0, min(float(h), float(box[1]))),
                max(0.0, min(float(w), float(box[2]))),
                max(0.0, min(float(h), float(box[3]))),
            ],
        })
    return records


def _serializable_detection(record):
    return {
        key: value for key, value in record.items()
        if key not in {"touches_tile_edge"}
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Compare empty inference input strategies.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shelf-model", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--empty-model", type=Path,
                        default=ROOT / "empty_shelf_yolo11m_best.pt")
    parser.add_argument("--shelf-conf", type=float, default=.25)
    parser.add_argument("--empty-conf", type=float, default=.10)
    parser.add_argument("--shelf-imgsz", type=int, default=960)
    parser.add_argument("--empty-imgsz", type=int, default=640)
    parser.add_argument("--roi-padding-ratio", type=float, default=.02)
    parser.add_argument("--min-mask-overlap", type=float, default=.30)
    parser.add_argument("--dedup-iou", type=float, default=.50)
    parser.add_argument("--max-shelf-rois", type=int, default=12)
    parser.add_argument("--device")
    return parser


def run(args):
    source = _resolve(args.source)
    image = cv2.imread(str(source))
    if image is None:
        raise RuntimeError(f"Image could not be read: {source}")
    output = _resolve(args.output_dir) if args.output_dir else (
        ROOT / "runs" /
        f"full_frame_gated_validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output.mkdir(parents=True, exist_ok=False)

    from ultralytics import YOLO
    shelf_model = YOLO(str(_resolve(args.shelf_model)))
    empty_model = YOLO(str(_resolve(args.empty_model)))
    config = PipelineConfig(
        shelf_conf=args.shelf_conf,
        empty_conf=args.empty_conf,
        shelf_imgsz=args.shelf_imgsz,
        empty_imgsz=args.empty_imgsz,
        roi_padding_ratio=args.roi_padding_ratio,
        min_mask_overlap=args.min_mask_overlap,
        dedup_iou=args.dedup_iou,
        max_shelf_rois=args.max_shelf_rois,
        device=args.device,
    )
    shelf_names = shelf_model.names
    empty_names = empty_model.names

    shelf_started = time.perf_counter()
    shelf_result = _predict(
        shelf_model, [image], config.shelf_conf, config.shelf_imgsz, config.device,
    )[0]
    shelves = extract_shelves(shelf_result, image.shape, shelf_names, shelf_model.task)
    shelf_ms = (time.perf_counter() - shelf_started) * 1000
    shelves = shelves[:config.max_shelf_rois]
    for shelf in shelves:
        # Raw-image comparison has no pose metadata. These labels are explicit
        # detected-shelf identities for the A/B/C tool, not semantic map IDs.
        shelf["shelf_id"] = f"DETECTED_SHELF_{shelf['shelf_index']:02d}"
        shelf["roi_bbox"] = padded_roi(
            shelf["bbox"], image.shape, config.roi_padding_ratio,
        )
        shelf["empty_inference_selected"] = True

    full_started = time.perf_counter()
    full_result = _predict(
        empty_model, [image], config.empty_conf, config.empty_imgsz, config.device,
    )[0]
    full_ms = (time.perf_counter() - full_started) * 1000
    standalone = _standalone_records(full_result, image.shape, empty_names)

    association_started = time.perf_counter()
    gated_raw, gated_accepted, gated_rejected = _full_frame_prediction_records(
        full_result, shelves, image.shape, empty_names, config.min_mask_overlap,
    )
    gated_final, gated_duplicates = deduplicate(gated_accepted, config.dedup_iou)
    association_ms = (time.perf_counter() - association_started) * 1000

    regions = build_regions(shelves, image, config, "full_shelf")
    roi_started = time.perf_counter()
    roi_results = _predict(
        empty_model, [region["image"] for region in regions],
        config.empty_conf, config.empty_imgsz, config.device,
    )
    roi_ms = (time.perf_counter() - roi_started) * 1000
    roi_raw, roi_accepted, roi_rejected = [], [], []
    for region, result in zip(regions, roi_results):
        raw, accepted, rejected = _prediction_records(
            result, region, image.shape, empty_names, config.min_mask_overlap,
        )
        roi_raw.extend(raw)
        roi_accepted.extend(accepted)
        roi_rejected.extend(rejected)
    roi_final, roi_duplicates = deduplicate(roi_accepted, config.dedup_iou)

    _write_image(output / "01_source.jpg", image)
    _write_image(output / "02_shelves.jpg", _draw_shelves(image, shelves))
    standalone_image = image.copy()
    for item in standalone:
        _draw_box(standalone_image, item["global_bbox_xyxy"], (0, 0, 255),
                  f"FULL {item['confidence']:.2f}")
    _write_image(output / "03_empty_full_frame_raw.jpg", standalone_image)
    roi_image = _draw_shelves(image, shelves)
    for item in roi_final:
        _draw_box(roi_image, item["global_bbox_xyxy"], (0, 0, 255),
                  f"ROI {item['confidence']:.2f}")
    _write_image(output / "04_old_shelf_roi_cascade.jpg", roi_image)
    gated_image = _draw_shelves(image, shelves)
    for item in gated_final:
        _draw_box(gated_image, item["global_bbox_xyxy"], (0, 0, 255),
                  f"GATED {item['confidence']:.2f}")
    _write_image(output / "05_full_frame_gated.jpg", gated_image)

    comparison = {
        "source": str(source),
        "image": {"width": image.shape[1], "height": image.shape[0]},
        "models": {
            "shelf": {"task": shelf_model.task, "classes": shelf_names},
            "empty_shelf": {"task": empty_model.task, "classes": empty_names},
        },
        "parameters": {
            "shelf_imgsz": config.shelf_imgsz,
            "empty_imgsz": config.empty_imgsz,
            "shelf_conf": config.shelf_conf,
            "empty_conf": config.empty_conf,
            "min_mask_overlap": config.min_mask_overlap,
            "dedup_iou": config.dedup_iou,
            "roi_padding_ratio": config.roi_padding_ratio,
        },
        "shelf_inference_ms": shelf_ms,
        "shelf_count": len(shelves),
        "standalone": {
            "empty_model_predict_calls": 1,
            "empty_inference_inputs": 1,
            "empty_inference_ms": full_ms,
            "detections": standalone,
        },
        "shelf_roi": {
            "empty_model_predict_calls": 1 if regions else 0,
            "empty_inference_inputs": len(regions),
            "empty_inference_ms": roi_ms,
            "raw_count": len(roi_raw),
            "rejected_count": len(roi_rejected),
            "duplicates_removed": roi_duplicates,
            "detections": [_serializable_detection(item) for item in roi_final],
        },
        "full_frame_gated": {
            "empty_model_predict_calls": 1,
            "empty_inference_inputs": 1,
            "empty_inference_ms": full_ms,
            "association_ms": association_ms,
            "raw_count": len(gated_raw),
            "rejected_count": len(gated_rejected),
            "duplicates_removed": gated_duplicates,
            "detections": [_serializable_detection(item) for item in gated_final],
        },
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "result.json").write_text(json.dumps({
        "empty_inference_mode": "full_frame_gated",
        "image": comparison["image"],
        "shelves": [{
            "shelf_index": shelf["shelf_index"],
            "confidence": shelf["confidence"],
            "global_bbox_xyxy": shelf["bbox"],
        } for shelf in shelves],
        "detections": comparison["full_frame_gated"]["detections"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "shelves": len(shelves),
        "standalone": len(standalone),
        "shelf_roi": len(roi_final),
        "full_frame_gated": len(gated_final),
    }, ensure_ascii=False))
    return comparison, output


def main():
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
