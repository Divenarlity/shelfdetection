from __future__ import annotations

import cv2
import numpy as np


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def padded_roi(bbox, image_shape, ratio: float):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = (x2 - x1) * ratio, (y2 - y1) * ratio
    return [
        max(0, int(np.floor(x1 - px))),
        max(0, int(np.floor(y1 - py))),
        min(w, int(np.ceil(x2 + px))),
        min(h, int(np.ceil(y2 + py))),
    ]


def context_tiles(bbox, image_shape, tile_size=384, overlap=0.25):
    """Cover a shelf ROI with fixed-size image-context tiles."""
    h, w = image_shape[:2]
    size = min(int(tile_size), h, w)
    if size == h == w:
        size -= 1
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("tile_size must be positive and tile_overlap must be in [0, 1)")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return []

    def starts(first, last, limit):
        if last - first <= size:
            return [max(0, min(limit - size, int(round((first + last - size) / 2))))]
        begin = max(0, min(limit - size, first))
        end = max(0, min(limit - size, last - size))
        step = max(1, int(round(size * (1 - overlap))))
        values = list(range(begin, end + 1, step))
        if values[-1] != end:
            values.append(end)
        return values

    x_starts = starts(x1, x2, w)
    y_starts = starts(y1, y2, h)
    return [[x, y, x + size, y + size] for y in y_starts for x in x_starts]


def roi_boxes(roi, image_shape, mode="full_shelf", tile_size=384, overlap=0.25):
    if mode not in {"full_shelf", "tiles"}:
        raise ValueError("empty ROI mode must be full_shelf or tiles")
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        return []
    if mode == "tiles":
        return [("tile", box) for box in context_tiles(roi, image_shape, tile_size, overlap)]
    return [("full_shelf", list(roi))]


def local_to_global(box, roi, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box
    rx1, ry1, _, _ = roi
    return [
        float(np.clip(x1 + rx1, 0, w)),
        float(np.clip(y1 + ry1, 0, h)),
        float(np.clip(x2 + rx1, 0, w)),
        float(np.clip(y2 + ry1, 0, h)),
    ]


def mask_box_metrics(mask, box):
    h, w = mask.shape
    x1, y1, x2, y2 = box
    ix1, iy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    ix2, iy2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0, False
    cx = int(np.clip((x1 + x2) / 2, 0, w - 1))
    cy = int(np.clip((y1 + y2) / 2, 0, h - 1))
    center_inside = bool(mask[cy, cx])
    overlap = float(
        np.count_nonzero(mask[iy1:iy2, ix1:ix2]) / ((ix2 - ix1) * (iy2 - iy1))
    )
    return overlap, center_inside


def mask_box_relation(mask, box, min_overlap: float):
    """Legacy ROI validation: center inclusion remains a secondary acceptance path."""
    overlap, center_inside = mask_box_metrics(mask, box)
    return center_inside or overlap >= min_overlap, overlap, center_inside


def best_shelf_for_box(shelves, box):
    """Return the single shelf with the greatest bbox-to-mask containment ratio."""
    ranked = []
    for shelf in shelves:
        overlap, center_inside = mask_box_metrics(shelf["mask"], box)
        ranked.append((
            overlap,
            int(center_inside),
            float(shelf.get("confidence", shelf.get("segmentation_confidence", 0.0))),
            -int(shelf.get("shelf_index", 0)),
            shelf,
            center_inside,
        ))
    if not ranked:
        return None, 0.0, False
    overlap, _, _, _, shelf, center_inside = max(ranked, key=lambda item: item[:4])
    return shelf, float(overlap), bool(center_inside)


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def intersection_over_smaller(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller = min(
        max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]),
        max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]),
    )
    return intersection / smaller if smaller else 0.0


def deduplicate(detections: list[dict], threshold: float):
    """Class-aware global NMS; overlapping shelf ROIs cannot return the same gap twice."""
    remaining = sorted(
        detections,
        key=lambda item: (
            -item["mask_overlap_ratio"],
            -int(item["center_inside_mask"]),
            int(item.get("touches_tile_edge", False)),
            -item["confidence"],
        ),
    )
    kept = []
    while remaining:
        seed = remaining.pop(0)
        rest = []
        for candidate in remaining:
            a, b = seed["global_bbox_xyxy"], candidate["global_bbox_xyxy"]
            same_class = seed.get("class_id", 0) == candidate.get("class_id", 0)
            overlaps = box_iou(a, b) >= threshold or intersection_over_smaller(a, b) >= threshold
            if not (same_class and overlaps):
                rest.append(candidate)
        remaining = rest
        kept.append(seed)
    return kept, len(detections) - len(kept)


def section_for_box(box, shelf_bbox):
    center_x = (box[0] + box[2]) / 2
    x1, _, x2, _ = shelf_bbox
    position = (center_x - x1) / max(1, x2 - x1)
    return "SOL" if position < 1 / 3 else "ORTA" if position < 2 / 3 else "SAĞ"


def contour_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(int).tolist()
