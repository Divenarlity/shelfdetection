from __future__ import annotations

import cv2
import numpy as np


class ModelRunner:
    def __init__(self, shelf_path, gap_path):
        from ultralytics import YOLO
        self.shelf_model, self.gap_model = YOLO(str(shelf_path)), YOLO(str(gap_path))

    @staticmethod
    def shelf_predictions(result, shape):
        found = []
        if result.masks is None:
            return found
        masks = result.masks.data.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(masks))
        for mask, conf in zip(masks, confs):
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = [c.reshape(-1, 2).tolist() for c in contours if len(c) >= 3]
            found.append({"mask": mask, "polygons": polygons, "confidence": float(conf)})
        return found

    @staticmethod
    def gap_predictions(result):
        if result.boxes is None:
            return []
        boxes = result.boxes
        return [{"box": xyxy.tolist(), "confidence": float(conf), "class_id": int(cls)}
                for xyxy, conf, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())]

    def run(self, image, shelf_conf=0.25, gap_conf=0.25, imgsz=640, device=None):
        kwargs = {"verbose": False, "imgsz": imgsz}
        if device is not None:
            kwargs["device"] = device
        sr = self.shelf_model.predict(image, conf=shelf_conf, **kwargs)[0]
        gr = self.gap_model.predict(image, conf=gap_conf, **kwargs)[0]
        return self.shelf_predictions(sr, image.shape), self.gap_predictions(gr)

