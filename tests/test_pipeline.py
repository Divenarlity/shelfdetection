from types import SimpleNamespace

import numpy as np
import pytest

from src.model_contracts import validate_model_contract
from src.pipeline import PipelineConfig, ShelfInferencePipeline, build_regions, extract_shelves
from src.pose_geometry import PoseConfig
from src.postprocessing import (
    context_tiles, deduplicate, local_to_global, mask_bbox, mask_box_relation,
    padded_roi, roi_boxes, section_for_box,
)


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value
    def tolist(self): return self.value.tolist()


class Boxes:
    def __init__(self, confidences, classes, xyxy=None):
        self.conf, self.cls = Tensor(confidences), Tensor(classes)
        if xyxy is not None:
            self.xyxy = Tensor(xyxy)
    def __len__(self): return len(self.conf.value)


class ShelfModel:
    task, names = "segment", {0: "shelves"}
    def __init__(self): self.calls = []
    def predict(self, **kwargs):
        self.calls.append(kwargs)
        masks = np.zeros((2, 80, 120), np.float32)
        masks[0, 20:60, 40:80] = 1
        masks[1, 0:10, 0:10] = 1
        return [SimpleNamespace(masks=SimpleNamespace(data=Tensor(masks)),
                                boxes=Boxes([.9, .8], [0, 0]))]


class EmptyModel:
    task, names = "detect", {0: "empty_shelf"}
    def __init__(self, boxes=None):
        self.calls, self.boxes = [], boxes
    def predict(self, **kwargs):
        self.calls.append(kwargs)
        count = len(kwargs["source"]) if isinstance(kwargs["source"], list) else 1
        boxes = None if self.boxes is None else Boxes([.8] * len(self.boxes),
                                                       [0] * len(self.boxes), self.boxes)
        return [SimpleNamespace(boxes=boxes) for _ in range(count)]


def metadata():
    return {
        "frame_id": "frame_000001", "resolution": [120, 80],
        "camera_position_world": [0, 0, 0], "camera_rotation_xyzw": [0, 0, 0, 1],
        "intrinsics": {"fx": 80, "fy": 80, "cx": 60, "cy": 40},
        "pose": {"quality": 1, "relocalized": True, "scale_initialized": True},
    }


def store_map():
    return {"shelves": [{
        "shelf_id": "A-L-01", "active": True,
        "corners_world": [[-.5, -.5, 2], [-.5, .5, 2], [.5, .5, 2], [.5, -.5, 2]],
        "center_world": [0, 0, 2], "front_normal_world": [0, 0, -1],
    }]}


def test_model_contract_fails_fast():
    assert validate_model_contract(ShelfModel(), "segment", "shelves", "shelf")[0] == "shelves"
    wrong = ShelfModel()
    wrong.task = "detect"
    with pytest.raises(RuntimeError, match="segment"):
        validate_model_contract(wrong, "segment", "shelves", "shelf")


def test_segmenter_requires_masks_but_allows_zero_detections():
    with pytest.raises(RuntimeError, match="fallback"):
        extract_shelves(SimpleNamespace(masks=None, boxes=Boxes([.8], [0])), (20, 20, 3), {0:"shelves"})
    assert extract_shelves(SimpleNamespace(masks=None, boxes=Boxes([], [])),
                           (20, 20, 3), {0:"shelves"}) == []


def test_roi_and_mask_primitives():
    mask = np.zeros((20, 30), bool)
    mask[4:10, 6:16] = True
    assert mask_bbox(mask) == [6, 4, 16, 10]
    assert padded_roi([0, 0, 10, 10], (20, 30, 3), .5) == [0, 0, 15, 15]
    assert local_to_global([1, 2, 5, 6], [10, 20, 30, 40], (100, 100, 3)) == [11., 22., 15., 26.]
    accepted, overlap, center = mask_box_relation(mask, [7, 5, 12, 9], .3)
    assert accepted and center and overlap == 1
    assert section_for_box([110, 0, 120, 10], [100, 0, 400, 30]) == "SOL"
    assert section_for_box([245, 0, 255, 10], [100, 0, 400, 30]) == "ORTA"
    assert section_for_box([380, 0, 390, 10], [100, 0, 400, 30]) == "SAĞ"


def test_full_shelf_is_default_and_tiles_are_explicit():
    roi = [20, 20, 100, 70]
    assert roi_boxes(roi, (100, 200, 3)) == [("full_shelf", roi)]
    tiles = context_tiles([0, 20, 200, 70], (100, 200, 3), 64, .25)
    assert len(tiles) > 1
    assert roi_boxes([0, 20, 200, 70], (100, 200, 3), "tiles", 64, .25) == [
        ("tile", tile) for tile in tiles
    ]
    with pytest.raises(ValueError, match="full_shelf or tiles"):
        roi_boxes(roi, (100, 200, 3), "auto")


def test_deduplication_never_merges_different_shelves():
    base = {"confidence": .8, "mask_overlap_ratio": .8, "center_inside_mask": True,
            "global_bbox_xyxy": [1, 1, 10, 10]}
    kept, removed = deduplicate([
        {**base, "assigned_shelf_id": "A"},
        {**base, "assigned_shelf_id": "B"},
    ], .5)
    assert removed == 0 and len(kept) == 2
    kept, removed = deduplicate([
        {**base, "assigned_shelf_id": "A", "touches_tile_edge": False},
        {**base, "assigned_shelf_id": "A", "touches_tile_edge": True, "confidence": .9},
    ], .5)
    assert removed == 1 and not kept[0]["touches_tile_edge"]


def test_pipeline_only_processes_pose_matched_shelves_without_debug_side_effects():
    shelf_model, empty_model = ShelfModel(), EmptyModel()
    pipeline = ShelfInferencePipeline(
        shelf_model, empty_model, store_map(),
        PoseConfig(min_visible_area=1, min_score=.25), PipelineConfig(),
    )
    image = np.zeros((80, 120, 3), np.uint8)
    outcome = pipeline.infer(image, metadata())
    assert outcome.result["counts"] == {
        "visible_candidates": 1, "segmented_shelves": 2, "matched_shelves": 1,
        "unknown_shelves": 1, "roi_inferences": 1, "raw_empty_predictions": 0,
        "mask_rejected": 0, "duplicates_removed": 0, "final_empty_spaces": 0,
    }
    assert len(shelf_model.calls) == len(empty_model.calls) == 1
    assert shelf_model.calls[0]["source"].shape == image.shape
    assert empty_model.calls[0]["source"].shape != image.shape
    unknown = next(item for item in outcome.result["shelves"] if item["shelf_id"] == "UNKNOWN_SHELF")
    assert unknown["status"] == "UNLOCALIZED" and unknown["roi_bbox_xyxy"] is None
    assert "debug" not in outcome.result


def test_explicit_tile_mode_remains_matched_shelf_only():
    image = np.zeros((80, 120, 3), np.uint8)
    mask = np.ones((80, 120), bool)
    config = PipelineConfig(empty_roi_mode="tiles", tile_size=32)
    regions = build_regions([
        {"shelf_id": "KNOWN", "roi_bbox": [20, 20, 100, 60], "mask": mask},
        {"shelf_id": "UNKNOWN_SHELF", "roi_bbox": [0, 0, 120, 80], "mask": mask},
    ], image, config)
    assert len(regions) > 1
    assert all(region["shelf"]["shelf_id"] == "KNOWN" for region in regions)
    assert all(region["kind"] == "tile" and region["image"].shape[:2] == (32, 32)
               for region in regions)


def test_debug_controls_are_strictly_opt_in(tmp_path):
    shelf_model, empty_model = ShelfModel(), EmptyModel()
    pipeline = ShelfInferencePipeline(
        shelf_model, empty_model, store_map(),
        PoseConfig(min_visible_area=1, min_score=.25), PipelineConfig(tile_size=32),
    )
    result = pipeline.infer(np.zeros((80, 120, 3), np.uint8), metadata(), tmp_path).result
    assert result["debug"]["debug_only"] is True
    assert len(empty_model.calls) > result["counts"]["roi_inferences"]
    assert (tmp_path / "control_full_frame.jpg").is_file()
    assert (tmp_path / "production_final.jpg").is_file()
