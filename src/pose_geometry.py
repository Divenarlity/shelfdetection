from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

UNKNOWN_SHELF = "UNKNOWN_SHELF"
FORBIDDEN_METADATA_KEYS = {
    "shelf_id", "expected_shelf_id", "visible_shelf_ids", "ground_truth",
    "label", "labels", "raycast_shelf_id", "center_shelf_id",
    "expected_shelf_ids", "ground_truth_ids", "shelf_ids", "visible_shelves",
}


@dataclass
class PoseConfig:
    min_visible_area: float = 200.0
    min_front_dot: float = 0.05
    min_score: float = 0.25
    min_pose_quality: float = 0.70
    min_mask_polygon_overlap: float = 0.05
    weights: dict | None = None

    def __post_init__(self):
        self.weights = self.weights or {
            "iou": .25, "mask_coverage": .18, "polygon_coverage": .12,
            "centroid_score": .15, "area_score": .10,
            "segmentation_confidence": .05, "distance_score": .05,
            "front_facing_score": .10,
        }


def load_pose_config(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Pose configuration must be a YAML object.")
    return PoseConfig(**data)


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def load_frame_metadata(path):
    path = Path(path)
    if "ground_truth" in {part.lower() for part in path.parts}:
        raise ValueError("Inference metadata ground_truth klasöründen okunamaz.")
    return parse_frame_metadata(json.loads(path.read_text(encoding="utf-8")))


def parse_frame_metadata(data):
    """Validate and normalize the same ID-free Unity metadata from disk or HTTP."""
    if not isinstance(data, dict):
        raise ValueError("Frame metadata must be a JSON object.")
    data = dict(data)
    present = FORBIDDEN_METADATA_KEYS.intersection(_walk_keys(data))
    if present:
        raise ValueError(f"Inference metadata cannot contain identity/ground-truth fields: {sorted(present)}")
    if int(data.get("schema_version", 0)) != 3:
        raise ValueError("Inference metadata must use schema_version 3.")
    if not {"frame_id", "image", "camera", "pose"}.issubset(data):
        raise ValueError("Schema v3 metadata requires frame_id, image, camera, and pose.")
    camera, image, pose = data["camera"], data["image"], data["pose"]
    position_key = "position_map" if "position_map" in camera else "position_world"
    if not {position_key, "rotation_xyzw", "intrinsics"}.issubset(camera):
        raise ValueError("Schema v3 camera pose or intrinsics are missing.")
    if not isinstance(pose, dict) or not {
        "quality", "relocalized", "scale_initialized"
    }.issubset(pose):
        raise ValueError("Schema v3 requires pose quality, relocalized, and scale_initialized.")
    if not 0 <= float(pose["quality"]) <= 1 or not all(
        isinstance(pose[key], bool) for key in ("relocalized", "scale_initialized")
    ):
        raise ValueError("Schema v3 pose quality or flags are invalid.")
    if int(image["width"]) <= 0 or int(image["height"]) <= 0:
        raise ValueError("Image dimensions must be positive.")
    intrinsics = camera["intrinsics"]
    if not {"fx", "fy", "cx", "cy"}.issubset(intrinsics) or (
        float(intrinsics["fx"]) <= 0 or float(intrinsics["fy"]) <= 0
    ):
        raise ValueError("Camera intrinsics are missing or invalid.")
    data["resolution"] = [image["width"], image["height"]]
    data["camera_position_world"] = camera[position_key]
    data["camera_rotation_xyzw"] = camera["rotation_xyzw"]
    data["intrinsics"] = intrinsics
    data["image_path"] = data.get("image_path", f"{data['frame_id']}.png")
    return data


def load_store_map(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("shelves"), list):
        raise ValueError("store_map.json shelves listesi içermelidir.")
    ids = [s.get("shelf_id") for s in data["shelves"] if s.get("active", True)]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("Aktif raf ID'leri dolu ve benzersiz olmalıdır.")
    return data


def _quaternion_rotation(q):
    x, y, z, w = np.asarray(q, float) / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def project_world_points(points, metadata):
    """Unity left-handed/Y-up world coordinates -> OpenCV top-left pixels."""
    pts = np.asarray(points, float)
    rotation = _quaternion_rotation(metadata["camera_rotation_xyzw"])
    camera = (rotation.T @ (pts - np.asarray(metadata["camera_position_world"])).T).T
    intrinsics = metadata["intrinsics"]
    z = camera[:, 2]
    valid = z > 1e-6
    pixels = np.c_[
        intrinsics["fx"] * camera[:, 0] / z + intrinsics["cx"],
        intrinsics["cy"] - intrinsics["fy"] * camera[:, 1] / z,
    ]
    return pixels, valid


def _corners(shelf):
    value = shelf["corners_world"]
    if isinstance(value, dict):
        return [value[k] for k in ("bottom_left", "top_left", "top_right", "bottom_right")]
    if len(value) != 4:
        raise ValueError(f"{shelf.get('shelf_id')}: dört sıralı köşe gerekli.")
    return value


def polygon_area(points):
    return abs(float(cv2.contourArea(np.asarray(points, np.float32))))


def _clip_polygon(points, width, height):
    image = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)
    area, poly = cv2.intersectConvexConvex(np.asarray(points, np.float32), image)
    return poly.reshape(-1, 2) if area > 0 and poly is not None else np.empty((0, 2))


def generate_candidates(store_map, metadata, config=PoseConfig()):
    pose=metadata.get("pose",{})
    if pose and (not pose.get("relocalized",True) or
                 float(pose.get("quality",1.0)) < config.min_pose_quality or
                 not pose.get("scale_initialized",True)):
        return []
    camera = np.asarray(metadata["camera_position_world"], float)
    width, height = metadata["resolution"]
    candidates = []
    for shelf in store_map["shelves"]:
        if not shelf.get("active", True):
            continue
        points = _corners(shelf)
        pixels, valid = project_world_points(points, metadata)
        if not valid.all():
            continue
        center = np.asarray(shelf["center_world"], float)
        to_camera = camera - center
        distance = float(np.linalg.norm(to_camera))
        front = np.asarray(shelf["front_normal_world"], float)
        front /= max(np.linalg.norm(front), 1e-9)
        front_dot = float(np.dot(front, to_camera / max(distance, 1e-9)))
        if distance <= 1e-9 or front_dot < config.min_front_dot:
            continue
        clipped = _clip_polygon(pixels, width, height)
        area = polygon_area(clipped) if len(clipped) >= 3 else 0
        if area < config.min_visible_area:
            continue
        candidates.append({"shelf_id": shelf["shelf_id"], "polygon": clipped,
                           "distance": distance, "front_alignment": front_dot,
                           "visible_area": area, "parent_surface_id": shelf.get("parent_surface_id")})
    return candidates


def polygon_mask(points, shape):
    out = np.zeros(shape[:2], np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(out, [np.rint(points).astype(np.int32)], 1)
    return out.astype(bool)


def score_pair(shelf, candidate, shape, config):
    mask, poly = shelf["mask"].astype(bool), polygon_mask(candidate["polygon"], shape)
    inter = float(np.count_nonzero(mask & poly))
    ma, pa = float(np.count_nonzero(mask)), float(np.count_nonzero(poly))
    if "centroid" in shelf:
        mc = np.asarray(shelf["centroid"], float)
    else:
        ys, xs = np.where(mask)
        mc = np.array([xs.mean(), ys.mean()]) if len(xs) else np.zeros(2)
    pc = np.asarray(candidate["polygon"], float).mean(axis=0)
    diagonal = float(np.hypot(shape[1], shape[0]))
    parts = {
        "iou": inter / max(ma + pa - inter, 1),
        "mask_coverage": inter / max(ma, 1),
        "polygon_coverage": inter / max(pa, 1),
        "centroid_score": max(0.0, 1.0-float(np.linalg.norm(mc-pc))/max(diagonal, 1)),
        "area_score": min(ma, pa) / max(ma, pa, 1),
        "segmentation_confidence": float(shelf["segmentation_confidence"]),
        "distance_score": 1.0/(1.0+candidate["distance"]),
        "front_facing_score": max(0.0, candidate["front_alignment"]),
    }
    score = sum(config.weights.get(k, 0)*v for k, v in parts.items())
    if min(parts["mask_coverage"], parts["polygon_coverage"]) < config.min_mask_polygon_overlap:
        score = 0.0
    return score, parts


def _global_assignment(scores):
    """Exact maximum-weight one-to-one assignment with optional unmatched rows."""
    rows, cols = scores.shape
    if cols > 20:
        raise ValueError("Global assignment en fazla 20 görünür harita adayını destekler.")
    @lru_cache(None)
    def solve(row, used):
        if row == rows:
            return 0.0, ()
        best_score, best = solve(row+1, used)
        best = (-1,) + best
        for col in range(cols):
            if used & (1 << col):
                continue
            tail_score, tail = solve(row+1, used | (1 << col))
            value = float(scores[row, col]) + tail_score
            if value > best_score:
                best_score, best = value, (col,) + tail
        return best_score, best
    return solve(0, 0)[1]


def assign_pose_ids(shelves, candidates, shape, config=PoseConfig()):
    if not shelves:
        return shelves
    scores = np.zeros((len(shelves), len(candidates)), float)
    parts = {}
    for si, shelf in enumerate(shelves):
        for ci, candidate in enumerate(candidates):
            scores[si, ci], parts[si, ci] = score_pair(shelf, candidate, shape, config)
    assignment = _global_assignment(np.where(scores >= config.min_score, scores, 0.0))
    for si, ci in enumerate(assignment):
        if ci < 0 or scores[si, ci] < config.min_score:
            best_ci = int(np.argmax(scores[si])) if len(candidates) else None
            shelves[si].update(shelf_id=UNKNOWN_SHELF, mapping_confidence=0.0,
                               mapping_scores=parts[si, best_ci] if best_ci is not None else {},
                               projected_candidate=None)
        else:
            shelves[si].update(shelf_id=candidates[ci]["shelf_id"],
                               mapping_confidence=float(scores[si, ci]),
                               mapping_scores=parts[si, ci],
                               projected_candidate=candidates[ci])
    return shelves
