from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def validate_model_contract(model, expected_task: str, required_class: str, label: str):
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    names = {int(key): str(value) for key, value in names.items()}
    if model.task != expected_task:
        raise RuntimeError(f"{label} görevi {expected_task!r} olmalı; bulunan: {model.task!r}")
    if required_class not in names.values():
        raise RuntimeError(f"{label} sınıflarında {required_class!r} yok; bulunan: {list(names.values())}")
    return names


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
    """Raf bbox'ını tamamen örten, görüntü içindeki sabit boyutlu bağlam tile'ları üretir."""
    h, w = image_shape[:2]
    size = min(int(tile_size), h, w)
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("tile_size pozitif, tile overlap [0,1) aralığında olmalıdır.")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return []
    if x2 - x1 <= size:
        x_starts = [max(0, min(w-size, int(round((x1+x2-size)/2))))]
    else:
        first=max(0,min(w-size,x1)); last=max(0,min(w-size,x2-size))
        step=max(1,int(round(size*(1-overlap))))
        x_starts=list(range(first,last+1,step))
        if x_starts[-1] != last: x_starts.append(last)
    if y2 - y1 <= size:
        y_starts = [max(0, min(h-size, int(round((y1+y2-size)/2))))]
    else:
        first=max(0,min(h-size,y1)); last=max(0,min(h-size,y2-size))
        step=max(1,int(round(size*(1-overlap))))
        y_starts=list(range(first,last+1,step))
        if y_starts[-1] != last: y_starts.append(last)
    return [[x, y, x+size, y+size] for y in y_starts for x in x_starts]


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


def mask_box_relation(mask, box, min_overlap: float):
    h, w = mask.shape
    x1, y1, x2, y2 = box
    ix1, iy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    ix2, iy2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return False, 0.0, False
    cx = int(np.clip((x1 + x2) / 2, 0, w - 1))
    cy = int(np.clip((y1 + y2) / 2, 0, h - 1))
    center_inside = bool(mask[cy, cx])
    overlap = float(np.count_nonzero(mask[iy1:iy2, ix1:ix2]) / ((ix2 - ix1) * (iy2 - iy1)))
    return center_inside or overlap >= min_overlap, overlap, center_inside


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def intersection_over_smaller(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
    smaller = min(max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1]),
                  max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1]))
    return intersection/smaller if smaller else 0.0


def deduplicate(detections: list[dict], threshold: float):
    remaining = list(detections)
    kept = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        rest = []
        for candidate in remaining:
            a,b=seed["global_bbox_xyxy"],candidate["global_bbox_xyxy"]
            if box_iou(a,b) >= threshold or intersection_over_smaller(a,b) >= threshold:
                cluster.append(candidate)
            else:
                rest.append(candidate)
        remaining = rest
        cluster.sort(
            key=lambda item: (
                -item["mask_overlap_ratio"],
                -int(item["center_inside_mask"]),
                int(item.get("touches_tile_edge", False)),
                -item["confidence"],
                item["assigned_shelf_id"],
            )
        )
        kept.append(cluster[0])
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


def normalized_polygon(mask):
    h, w = mask.shape
    return [[round(x / w, 6), round(y / h, 6)] for x, y in contour_polygon(mask)]


def polygon_to_mask(points, shape):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    polygon = np.asarray([[round(x * w), round(y * h)] for x, y in points], np.int32)
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def mask_iou(a, b):
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return intersection / union if union else 0.0


def deterministic_shelf_ids(masks: list[dict], camera_id: str):
    ordered = sorted(masks, key=lambda item: (item["centroid"][1], item["centroid"][0]))
    for index, item in enumerate(ordered, 1):
        item["shelf_id"] = f"{camera_id}-RAF-{index:02d}"
    return ordered


def assign_shelf_ids(masks, camera_id, map_path: Path, image_shape, rebuild=False, match_iou=0.30):
    masks = deterministic_shelf_ids(masks, camera_id)
    if map_path.exists() and not rebuild:
        saved = json.loads(map_path.read_text(encoding="utf-8"))
        old = [
            (entry["shelf_id"], polygon_to_mask(entry["normalized_mask_polygon"], image_shape[:2]))
            for entry in saved.get("shelves", [])
        ]
        pairs = sorted(
            ((mask_iou(item["mask"], old_mask), index, shelf_id)
             for index, item in enumerate(masks) for shelf_id, old_mask in old),
            reverse=True,
        )
        used_new, used_old = set(), set()
        for score, index, shelf_id in pairs:
            if score < match_iou or index in used_new or shelf_id in used_old:
                continue
            masks[index]["shelf_id"] = shelf_id
            used_new.add(index)
            used_old.add(shelf_id)
        existing_numbers = [int(sid.rsplit("-", 1)[-1]) for sid, _ in old if sid.rsplit("-", 1)[-1].isdigit()]
        next_number = max(existing_numbers, default=0) + 1
        for index, item in enumerate(masks):
            if index not in used_new:
                item["shelf_id"] = f"{camera_id}-RAF-{next_number:02d}"
                next_number += 1
    map_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = image_shape[:2]
    payload = {
        "camera_id": camera_id, "image_width": w, "image_height": h,
        "shelves": [{
            "shelf_id": item["shelf_id"],
            "normalized_bbox": [round(item["bbox"][0]/w, 6), round(item["bbox"][1]/h, 6),
                                round(item["bbox"][2]/w, 6), round(item["bbox"][3]/h, 6)],
            "normalized_mask_polygon": normalized_polygon(item["mask"]),
            "centroid": [round(item["centroid"][0]/w, 6), round(item["centroid"][1]/h, 6)],
        } for item in masks],
    }
    map_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return sorted(masks, key=lambda item: item["shelf_id"])
