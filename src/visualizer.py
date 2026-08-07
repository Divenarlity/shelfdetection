from __future__ import annotations

import cv2
import numpy as np
from .geometry import scale_polygon


def draw(image, camera, mappings, shelf_predictions):
    out = image.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    for pred in shelf_predictions:
        overlay[pred["mask"] > 0] = (0, 180, 255)
    out = cv2.addWeighted(out, .72, overlay, .28, 0)
    for shelf in camera.shelves:
        poly = scale_polygon(shelf.polygon, (camera.reference_width, camera.reference_height), (w, h)).astype(np.int32)
        cv2.polylines(out, [poly], True, (255, 180, 0), 2)
        cv2.putText(out, shelf.shelf_id, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,180,0), 2)
    for item in mappings:
        x1,y1,x2,y2 = map(int, item["box"])
        label = f"{item['shelf_id']} {item['section']} {item['gap_confidence']:.2f} {item['mapping_source']}"
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(out, label, (x1,max(15,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,0,255), 1)
    return out

