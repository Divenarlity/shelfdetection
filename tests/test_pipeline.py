from types import SimpleNamespace

import numpy as np
import pytest

from src.model_contracts import validate_model_contract
from src.pipeline import (
    EmptyInferenceMode, PipelineConfig, ShelfInferencePipeline,
    _full_frame_prediction_records, _prediction_records, build_regions, extract_shelves,
    render_visualization,
)
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
    def __init__(self): self.calls, self.plot_calls = [], 0
    def predict(self, **kwargs):
        self.calls.append(kwargs)
        count = len(kwargs["source"]) if isinstance(kwargs["source"], list) else 1
        results = []
        for _ in range(count):
            masks = np.zeros((2, 80, 120), np.float32)
            masks[0, 20:60, 40:80] = 1
            masks[1, 0:10, 0:10] = 1
            result = SimpleNamespace(masks=SimpleNamespace(data=Tensor(masks)),
                                     boxes=Boxes([.9, .8], [0, 0]))
            def plot(img=None, **_):
                self.plot_calls += 1
                canvas = img.copy()
                canvas[20:60, 40:80] = (255, 0, 0)
                return canvas
            result.plot = plot
            results.append(result)
        return results


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
    with pytest.raises(RuntimeError, match="without masks"):
        extract_shelves(SimpleNamespace(masks=None, boxes=Boxes([.8], [0])), (20, 20, 3), {0:"shelves"})
    assert extract_shelves(SimpleNamespace(masks=None, boxes=Boxes([], [])),
                           (20, 20, 3), {0:"shelves"}) == []


def test_detection_shelf_output_uses_boxes_and_rectangular_filter_mask():
    result = SimpleNamespace(
        masks=None,
        boxes=Boxes([.91], [0], [[3.2, 4.1, 15.8, 17.2]]),
    )
    shelves = extract_shelves(result, (20, 30, 3), {0: "shelves"}, "detect")
    assert len(shelves) == 1
    assert shelves[0]["bbox"] == [3, 4, 16, 18]
    assert shelves[0]["model_task"] == "detect"
    assert not shelves[0]["has_segmentation_mask"]
    assert shelves[0]["mask"][5, 5] and not shelves[0]["mask"][0, 0]


def test_roi_and_mask_primitives():
    mask = np.zeros((20, 30), bool)
    mask[4:10, 6:16] = True
    assert mask_bbox(mask) == [6, 4, 16, 10]
    assert padded_roi([0, 0, 10, 10], (20, 30, 3), .5) == [0, 0, 15, 15]
    assert local_to_global([1, 2, 5, 6], [10, 20, 30, 40], (100, 100, 3)) == [11., 22., 15., 26.]
    assert local_to_global([-20, -30, 500, 600], [10, 20, 30, 40], (100, 100, 3)) == [0., 0., 100., 100.]
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


def test_global_deduplication_assigns_overlap_to_best_shelf():
    base = {"confidence": .8, "mask_overlap_ratio": .8, "center_inside_mask": True,
            "global_bbox_xyxy": [1, 1, 10, 10]}
    kept, removed = deduplicate([
        {**base, "assigned_shelf_id": "A"},
        {**base, "assigned_shelf_id": "B"},
    ], .5)
    assert removed == 1 and len(kept) == 1
    kept, removed = deduplicate([
        {**base, "assigned_shelf_id": "A", "touches_tile_edge": False},
        {**base, "assigned_shelf_id": "A", "touches_tile_edge": True, "confidence": .9},
    ], .5)
    assert removed == 1 and not kept[0]["touches_tile_edge"]


def test_default_pipeline_runs_one_shelf_level_roi_input_for_only_matched_shelf():
    shelf_model, empty_model = ShelfModel(), EmptyModel()
    pipeline = ShelfInferencePipeline(
        shelf_model, empty_model, store_map(),
        PoseConfig(min_visible_area=1, min_score=.25), PipelineConfig(),
    )
    image = np.zeros((80, 120, 3), np.uint8)
    outcome = pipeline.infer(image, metadata())
    counts = outcome.result["counts"]
    assert counts["visible_candidates"] == 1
    assert counts["segmented_shelves"] == 2
    assert counts["matched_shelves"] == counts["unknown_shelves"] == 1
    assert outcome.result["empty_inference_mode"] == "shelf_level_roi"
    assert counts["roi_inferences"] == counts["shelf_rois_selected"] == 1
    assert counts["empty_model_predict_calls"] == counts["empty_inference_inputs"] == 1
    assert counts["shelf_rois_limited"] == counts["final_empty_spaces"] == 0
    assert len(shelf_model.calls) == len(empty_model.calls) == 1
    assert shelf_model.plot_calls == 1
    assert shelf_model.calls[0]["source"].shape == image.shape
    assert empty_model.calls[0]["source"].shape != image.shape
    unknown = next(item for item in outcome.result["shelves"] if item["shelf_id"] == "UNKNOWN_SHELF")
    assert unknown["status"] == "UNLOCALIZED" and unknown["roi_bbox_xyxy"] is None
    assert unknown["empty_inference_source"] == "shelf_level_roi"
    assert not unknown["empty_inference_selected"]
    assert next(
        item for item in outcome.result["shelves"] if item["shelf_id"] == "A-L-01-01"
    )["empty_inference_selected"]
    known = next(item for item in outcome.result["shelves"] if item["shelf_id"] == "A-L-01-01")
    assert known["parent_shelf_id"] == "A-L-01"
    assert known["shelf_level_id"] == "A-L-01-01" and known["level_number"] == 1
    assert len(unknown["mask_polygon"]) >= 3
    assert all(
        set(point) == {"x", "y"}
        and isinstance(point["x"], int)
        and isinstance(point["y"], int)
        for point in unknown["mask_polygon"]
    )
    assert "debug" not in outcome.result


def test_visualization_reuses_native_plot_and_draws_only_final_accepted_boxes():
    image = np.zeros((80, 120, 3), np.uint8)
    calls = []
    shelf_result = SimpleNamespace(plot=lambda img: (calls.append(img.copy()) or
                                                      np.full_like(img, (10, 20, 30))))
    accepted = [{"global_bbox_xyxy": [40, 30, 70, 55], "confidence": .876}]
    native, annotated = render_visualization(shelf_result, image, accepted)
    assert len(calls) == 1
    assert np.array_equal(native[5, 5], [10, 20, 30])
    assert np.array_equal(annotated[30, 40], [0, 0, 255])
    # A non-final candidate location is untouched by the custom annotation pass.
    assert np.array_equal(annotated[70, 100], native[70, 100])


def test_explicit_tile_mode_skips_unlocalized_shelf_proposals():
    image = np.zeros((80, 120, 3), np.uint8)
    mask = np.ones((80, 120), bool)
    config = PipelineConfig(
        empty_inference_mode="shelf_roi", empty_roi_mode="tiles", tile_size=32,
    )
    regions = build_regions([
        {"shelf_id": "KNOWN", "roi_bbox": [20, 20, 100, 60], "mask": mask,
         "empty_inference_selected": True},
        {"shelf_id": "UNKNOWN_SHELF", "roi_bbox": [0, 0, 120, 80], "mask": mask,
         "empty_inference_selected": False},
    ], image, config)
    assert len(regions) > 1
    assert {region["shelf"]["shelf_id"] for region in regions} == {"KNOWN"}
    assert all(region["kind"] == "tile" and region["image"].shape[:2] == (32, 32)
               for region in regions)


def test_empty_detection_outside_segmentation_mask_is_rejected():
    mask = np.zeros((40, 60), bool)
    mask[10:20, 10:20] = True
    shelf = {"shelf_id": "A", "shelf_index": 0, "mask": mask}
    region = {
        "shelf": shelf, "bbox": [0, 0, 60, 40], "region_id": "SHELF-00-FULL",
        "tile_id": "SHELF-00-FULL", "kind": "full_shelf",
        "image": np.zeros((40, 60, 3), np.uint8),
    }
    result = SimpleNamespace(boxes=Boxes([.8], [0], [[30, 20, 50, 35]]))
    raw, accepted, rejected = _prediction_records(
        result, region, (40, 60, 3), {0: "empty_shelf"}, .5,
    )
    assert len(raw) == len(rejected) == 1 and accepted == []


def test_no_shelf_rejects_all_full_frame_empties_and_roi_limit_is_roi_only():
    class NoShelfModel(ShelfModel):
        def predict(self, **kwargs):
            self.calls.append(kwargs)
            result = SimpleNamespace(masks=None, boxes=Boxes([], []))
            result.plot = lambda img=None, **_: img.copy()
            return [result]

    empty = EmptyModel()
    no_shelf = ShelfInferencePipeline(
        NoShelfModel(), empty, store_map(), PoseConfig(min_visible_area=1), PipelineConfig(),
    ).infer(np.zeros((80, 120, 3), np.uint8), metadata()).result
    assert no_shelf["shelves"] == [] and len(empty.calls) == 0
    assert no_shelf["counts"]["empty_model_predict_calls"] == 0
    assert no_shelf["counts"]["empty_inference_inputs"] == 0
    assert no_shelf["counts"]["final_empty_spaces"] == 0

    empty = EmptyModel()
    limited = ShelfInferencePipeline(
        ShelfModel(), empty, store_map(), PoseConfig(min_visible_area=1, min_score=.25),
        PipelineConfig(empty_inference_mode="shelf_roi", max_shelf_rois=1),
    ).infer(np.zeros((80, 120, 3), np.uint8), metadata()).result
    assert len(limited["shelves"]) == 2
    assert limited["counts"]["roi_inferences"] == 1
    assert limited["counts"]["shelf_rois_limited"] == 0
    assert sum(shelf["empty_inference_selected"] for shelf in limited["shelves"]) == 1


def test_unlocalized_shelf_mask_cannot_produce_a_final_empty_detection():
    result = ShelfInferencePipeline(
        ShelfModel(), EmptyModel([[45, 25, 50, 30], [1, 1, 5, 5]]), store_map(),
        PoseConfig(min_visible_area=1, min_score=.25), PipelineConfig(),
    ).infer(np.zeros((80, 120, 3), np.uint8), metadata()).result
    assert len(result["shelves"]) == 2
    assert result["counts"]["final_empty_spaces"] == 1
    known = next(shelf for shelf in result["shelves"] if shelf["shelf_id"] == "A-L-01-01")
    unknown = next(shelf for shelf in result["shelves"] if shelf["shelf_id"] == "UNKNOWN_SHELF")
    assert known["empty_space_count"] == 1
    assert unknown["empty_space_count"] == 0
    assert unknown["status"] == "UNLOCALIZED"


def test_full_frame_gate_uses_overlap_as_primary_and_assigns_one_best_shelf():
    left = np.zeros((40, 60), bool)
    right = np.zeros((40, 60), bool)
    left[5:35, 5:35] = True
    right[5:35, 25:55] = True
    shelves = [
        {"shelf_id": "LEFT", "shelf_index": 0, "mask": left, "confidence": .8},
        {"shelf_id": "RIGHT", "shelf_index": 1, "mask": right, "confidence": .9},
    ]
    result = SimpleNamespace(boxes=Boxes(
        [.9, .8, .7, .6], [0, 0, 0, 0],
        [[8, 8, 18, 18], [28, 8, 48, 18], [2, 2, 12, 12], [0, 36, 10, 40]],
    ))
    raw, accepted, rejected = _full_frame_prediction_records(
        result, shelves, (40, 60, 3), {0: "empty_shelf"}, .30,
    )
    assert len(raw) == 4 and len(accepted) == 3 and len(rejected) == 1
    assert accepted[0]["assigned_shelf_id"] == "LEFT"
    assert accepted[1]["assigned_shelf_id"] == "RIGHT"
    assert accepted[2]["mask_overlap_ratio"] >= .30
    assert rejected[0]["assigned_shelf_id"] is None
    assert all(item["roi_bbox_xyxy"] is None for item in raw)


def test_shelf_roi_mode_remains_explicit_and_maps_local_boxes_to_global():
    image = np.zeros((80, 120, 3), np.uint8)
    empty = EmptyModel([[1, 1, 5, 5]])
    result = ShelfInferencePipeline(
        ShelfModel(), empty, store_map(), PoseConfig(min_visible_area=1, min_score=.25),
        PipelineConfig(empty_inference_mode=EmptyInferenceMode.SHELF_ROI),
    ).infer(image, metadata()).result
    assert result["empty_inference_mode"] == "shelf_roi"
    assert result["counts"]["roi_inferences"] == 1
    assert not isinstance(empty.calls[0]["source"], list)
    assert empty.calls[0]["source"].shape != image.shape
    assert all(
        detection["inference_source"] == "shelf_roi"
        for shelf in result["shelves"] for detection in shelf["detections"]
    )


def test_shelf_level_mode_batches_all_levels_and_numbers_top_to_bottom():
    class MultiShelfModel(ShelfModel):
        def predict(self, **kwargs):
            self.calls.append(kwargs)
            masks = np.zeros((3, 80, 120), np.float32)
            masks[0, 44:54, 42:78] = 1
            masks[1, 22:32, 42:78] = 1
            masks[2, 0:5, 0:5] = 1
            result = SimpleNamespace(
                masks=SimpleNamespace(data=Tensor(masks)),
                boxes=Boxes([.95, .85, .75], [0, 0, 0]),
            )
            result.plot = lambda img=None, **_: img.copy()
            return [result]

    shelf_model, empty_model = MultiShelfModel(), EmptyModel()
    outcome = ShelfInferencePipeline(
        shelf_model, empty_model, store_map(),
        PoseConfig(min_visible_area=1, min_score=.20), PipelineConfig(),
    ).infer(np.zeros((80, 120, 3), np.uint8), metadata())
    result = outcome.result
    known = [item for item in result["shelves"] if item["parent_shelf_id"] == "A-L-01"]
    assert {item["shelf_level_id"] for item in known} == {"A-L-01-01", "A-L-01-02"}
    assert next(item for item in known if item["level_number"] == 1)["global_bbox_xyxy"][1] == 22
    assert result["counts"]["empty_model_predict_calls"] == 1
    assert result["counts"]["empty_inference_calls"] == 1
    assert result["counts"]["empty_inference_inputs"] == 2
    assert result["counts"]["roi_inferences"] == 2
    assert isinstance(empty_model.calls[0]["source"], list)
    assert len(empty_model.calls[0]["source"]) == 2
    assert result["timing"]["empty_batch_inference_ms"] >= 0
    assert len(shelf_model.calls) == 1


def test_semantic_detection_fields_and_section_use_level_geometry():
    image_shape = (40, 90, 3)
    mask = np.zeros(image_shape[:2], bool)
    mask[10:30, 30:60] = True
    shelf = {
        "shelf_id": "A-L-02-03", "parent_shelf_id": "A-L-02",
        "shelf_level_id": "A-L-02-03", "shelf_index": 7, "mask": mask,
    }
    region = {
        "shelf": shelf, "bbox": [28, 8, 62, 32], "region_id": "A-L-02-03-FULL",
        "tile_id": "A-L-02-03-FULL", "kind": "full_shelf",
        "inference_source": "shelf_level_roi",
        "image": np.zeros((24, 34, 3), np.uint8),
    }
    raw, accepted, rejected = _prediction_records(
        SimpleNamespace(boxes=Boxes([.8], [0], [[3, 4, 10, 15]])),
        region, image_shape, {0: "empty_shelf"}, .3,
    )
    assert not rejected and raw == accepted
    detection = accepted[0]
    assert detection["global_bbox_xyxy"] == [31., 12., 38., 23.]
    assert detection["parent_shelf_id"] == "A-L-02"
    assert detection["shelf_level_id"] == "A-L-02-03"
    assert section_for_box(detection["global_bbox_xyxy"], [30, 10, 60, 30]) == "SOL"


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
    assert (tmp_path / "A-L-01-01_input.jpg").is_file()
    assert (tmp_path / "A-L-01-01_empty_result.jpg").is_file()


def test_dual_pipeline_batches_models_once_and_matches_single_view_semantics():
    image = np.zeros((80, 120, 3), np.uint8)
    left_metadata = metadata()
    right_metadata = {**metadata(), "frame_id": "frame_000001_right"}

    single_pipeline = ShelfInferencePipeline(
        ShelfModel(), EmptyModel(), store_map(),
        PoseConfig(min_visible_area=1, min_score=.25), PipelineConfig(),
    )
    expected_left = single_pipeline.infer(image, left_metadata).result
    expected_right = single_pipeline.infer(image, right_metadata).result

    shelf_model, empty_model = ShelfModel(), EmptyModel()
    dual = ShelfInferencePipeline(
        shelf_model, empty_model, store_map(),
        PoseConfig(min_visible_area=1, min_score=.25), PipelineConfig(),
    ).infer_dual(image, left_metadata, image.copy(), right_metadata)

    assert len(shelf_model.calls) == 1
    assert isinstance(shelf_model.calls[0]["source"], list)
    assert len(shelf_model.calls[0]["source"]) == 2
    assert len(empty_model.calls) == 1
    assert isinstance(empty_model.calls[0]["source"], list)
    assert len(empty_model.calls[0]["source"]) == 2
    assert dual.counts == {
        "shelf_model_predict_calls": 1, "shelf_model_inputs": 2,
        "empty_model_predict_calls": 1, "left_empty_inputs": 1,
        "right_empty_inputs": 1, "total_empty_inputs": 2,
        "left_final_empty_spaces": 0, "right_final_empty_spaces": 0,
    }
    for actual, expected in ((dual.left.result, expected_left),
                             (dual.right.result, expected_right)):
        assert actual["shelves"] == expected["shelves"]
        assert actual["rejected_detections"] == expected["rejected_detections"]
        assert actual["counts"] == expected["counts"]


def test_dual_pipeline_keeps_zero_roi_views_valid_and_dedup_separate():
    class MixedShelfModel(ShelfModel):
        def predict(self, **kwargs):
            self.calls.append(kwargs)
            results = []
            for image in kwargs["source"]:
                if image[0, 0, 0] == 0:
                    result = SimpleNamespace(masks=None, boxes=Boxes([], []))
                else:
                    masks = np.zeros((1, 80, 120), np.float32)
                    masks[0, 20:60, 40:80] = 1
                    result = SimpleNamespace(
                        masks=SimpleNamespace(data=Tensor(masks)),
                        boxes=Boxes([.9], [0]),
                    )
                result.plot = lambda img=None, **_: img.copy()
                results.append(result)
            return results

    left, right = np.zeros((80, 120, 3), np.uint8), np.ones((80, 120, 3), np.uint8)
    shelf_model, empty_model = MixedShelfModel(), EmptyModel([[1, 1, 5, 5]])
    dual = ShelfInferencePipeline(
        shelf_model, empty_model, store_map(),
        PoseConfig(min_visible_area=1, min_score=.25), PipelineConfig(),
    ).infer_dual(left, metadata(), right, {**metadata(), "frame_id": "right"})
    assert dual.left.result["shelves"] == []
    assert dual.counts["left_empty_inputs"] == 0
    assert dual.counts["right_empty_inputs"] == dual.counts["total_empty_inputs"] == 1
    assert dual.counts["empty_model_predict_calls"] == 1
    assert dual.right.result["counts"]["final_empty_spaces"] == 1

    both_empty_model = EmptyModel()
    both_zero = ShelfInferencePipeline(
        MixedShelfModel(), both_empty_model, store_map(),
        PoseConfig(min_visible_area=1, min_score=.25), PipelineConfig(),
    ).infer_dual(left, metadata(), left.copy(), {**metadata(), "frame_id": "right"})
    assert both_zero.counts["empty_model_predict_calls"] == 0
    assert both_zero.counts["total_empty_inputs"] == 0
    assert both_empty_model.calls == []


def test_dual_pipeline_validates_batched_result_counts():
    class ShortShelfModel(ShelfModel):
        def predict(self, **kwargs):
            return super().predict(**{**kwargs, "source": kwargs["source"][0]})

    pipeline = ShelfInferencePipeline(
        ShortShelfModel(), EmptyModel(), store_map(),
        PoseConfig(min_visible_area=1, min_score=.25), PipelineConfig(),
    )
    with pytest.raises(RuntimeError, match="Shelf-model result count"):
        pipeline.infer_dual(
            np.zeros((80, 120, 3), np.uint8), metadata(),
            np.zeros((80, 120, 3), np.uint8), {**metadata(), "frame_id": "right"},
        )

    class ShortEmptyModel(EmptyModel):
        def predict(self, **kwargs):
            self.calls.append(kwargs)
            return [SimpleNamespace(boxes=None)]

    pipeline = ShelfInferencePipeline(
        ShelfModel(), ShortEmptyModel(), store_map(),
        PoseConfig(min_visible_area=1, min_score=.25), PipelineConfig(),
    )
    with pytest.raises(RuntimeError, match="Empty-model result count"):
        pipeline.infer_dual(
            np.zeros((80, 120, 3), np.uint8), metadata(),
            np.zeros((80, 120, 3), np.uint8), {**metadata(), "frame_id": "right"},
        )


def test_dual_deduplication_is_isolated_per_camera():
    image = np.zeros((80, 120, 3), np.uint8)
    dual = ShelfInferencePipeline(
        ShelfModel(), EmptyModel([[5, 5, 10, 10]]), store_map(),
        PoseConfig(min_visible_area=1, min_score=.25), PipelineConfig(),
    ).infer_dual(image, metadata(), image.copy(), {**metadata(), "frame_id": "right"})
    assert dual.left.result["counts"]["final_empty_spaces"] == 1
    assert dual.right.result["counts"]["final_empty_spaces"] == 1
    assert dual.counts["left_final_empty_spaces"] == 1
    assert dual.counts["right_final_empty_spaces"] == 1


def test_fp16_uses_ultralytics_quantize_without_changing_dual_batch_calls():
    image = np.zeros((80, 120, 3), np.uint8)
    shelf_model, empty_model = ShelfModel(), EmptyModel()
    dual = ShelfInferencePipeline(
        shelf_model, empty_model, store_map(),
        PoseConfig(min_visible_area=1, min_score=.25),
        PipelineConfig(device="0", precision="fp16"),
    ).infer_dual(image, metadata(), image.copy(), {**metadata(), "frame_id": "right"})
    assert len(shelf_model.calls) == len(empty_model.calls) == 1
    assert len(shelf_model.calls[0]["source"]) == 2
    assert shelf_model.calls[0]["quantize"] == 16
    assert empty_model.calls[0]["quantize"] == 16
    assert dual.counts["shelf_model_predict_calls"] == 1
    assert dual.counts["empty_model_predict_calls"] == 1


def test_fp16_rejects_cpu_and_invalid_precision():
    with pytest.raises(ValueError, match="requires a CUDA"):
        PipelineConfig(device="cpu", precision="fp16")
    with pytest.raises(ValueError, match="fp32 or fp16"):
        PipelineConfig(device="0", precision="int8")


def test_warmup_validates_shape_configuration_without_running_models():
    pipeline = ShelfInferencePipeline(
        ShelfModel(), EmptyModel(), store_map(),
        PoseConfig(min_visible_area=1, min_score=.25), PipelineConfig(),
    )
    with pytest.raises(ValueError, match="iterations"):
        pipeline.warmup(-1, 20)
    with pytest.raises(ValueError, match="batch size"):
        pipeline.warmup(0, 0)
