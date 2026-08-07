from __future__ import annotations

import numpy as np
from .geometry import box_mask, classify_section, mask_iou, overlap_over_detection, polygon_mask, scale_polygon


def map_gap(box, image_shape, camera, shelf_predictions=None, min_overlap=0.25,
            shelf_match_iou=0.10, section_names=("left", "middle", "right"), boundary_margin=0.02):
    h, w = image_shape[:2]
    gap = box_mask(box, image_shape)
    predictions = shelf_predictions or []
    candidates = []
    for shelf in camera.shelves:
        poly = scale_polygon(shelf.polygon, (camera.reference_width, camera.reference_height), (w, h))
        manual = polygon_mask(poly, image_shape)
        best_prediction, best_iou = None, 0.0
        for prediction in predictions:
            iou = mask_iou(manual, prediction["mask"])
            if iou > best_iou:
                best_iou, best_prediction = iou, prediction
        if best_prediction is not None and best_iou >= shelf_match_iou:
            valid = (manual > 0) & (best_prediction["mask"] > 0)
            source = "shelf_mask_and_manual_id"
            shelf_conf = best_prediction.get("confidence")
        else:
            valid = manual
            source = "manual_roi_fallback"
            shelf_conf = None
        overlap = overlap_over_detection(gap, valid)
        candidates.append((overlap, shelf, poly, source, shelf_conf))
    if not candidates:
        return {"shelf_id": "unknown_shelf", "section": "unknown_section", "gap_shelf_overlap": 0.0,
                "mapping_source": "manual_roi_fallback"}
    overlap, shelf, poly, source, shelf_conf = max(candidates, key=lambda x: x[0])
    if overlap < min_overlap:
        return {"shelf_id": "unknown_shelf", "rack_id": "", "level": "", "section": "unknown_section",
                "gap_shelf_overlap": overlap, "mapping_source": source, "shelf_confidence": shelf_conf}
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return {"shelf_id": shelf.shelf_id, "rack_id": shelf.rack_id, "level": shelf.level,
            "section": classify_section(center, poly, section_names, boundary_margin),
            "gap_shelf_overlap": overlap, "mapping_source": source, "shelf_confidence": shelf_conf}

