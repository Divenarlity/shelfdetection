from __future__ import annotations

import cv2
import numpy as np


def order_quad(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Poligon tam olarak dört adet [x, y] noktası içermelidir.")
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    ordered = np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])
    if abs(cv2.contourArea(ordered)) < 1:
        raise ValueError("Poligon dejenere veya sıfır alanlı.")
    return ordered


def scale_polygon(points, reference_size, target_size):
    rw, rh = reference_size
    tw, th = target_size
    if min(rw, rh, tw, th) <= 0:
        raise ValueError("Görüntü boyutları pozitif olmalıdır.")
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] *= tw / rw
    pts[:, 1] *= th / rh
    return pts


def polygon_mask(points, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask


def box_mask(box, shape):
    x1, y1, x2, y2 = np.rint(box).astype(int)
    x1, x2 = sorted((max(0, x1), min(shape[1], x2)))
    y1, y2 = sorted((max(0, y1), min(shape[0], y2)))
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def overlap_over_detection(detection_mask, shelf_mask):
    area = int(np.count_nonzero(detection_mask))
    return 0.0 if not area else float(np.count_nonzero((detection_mask > 0) & (shelf_mask > 0)) / area)


def mask_iou(a, b):
    inter = np.count_nonzero((a > 0) & (b > 0))
    union = np.count_nonzero((a > 0) | (b > 0))
    return 0.0 if not union else float(inter / union)


def normalized_position(point, quad):
    q = order_quad(quad)
    width = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    height = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    matrix = cv2.getPerspectiveTransform(q, dst)
    p = cv2.perspectiveTransform(np.asarray([[point]], np.float32), matrix)[0, 0]
    return float(p[0] / width), float(p[1] / height)


def classify_section(point, quad, names=("left", "middle", "right"), boundary_margin=0.02):
    x, y = normalized_position(point, quad)
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return "unknown_section"
    if abs(x - 1 / 3) <= boundary_margin or abs(x - 2 / 3) <= boundary_margin:
        return "unknown_section"
    return names[0] if x < 1 / 3 else names[1] if x < 2 / 3 else names[2]

