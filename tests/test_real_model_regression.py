from pathlib import Path

import cv2
import pytest

from src.pipeline import PipelineConfig, ShelfInferencePipeline
from src.pose_geometry import load_frame_metadata, load_pose_config, load_store_map


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "a_l_02_regression"


@pytest.mark.integration
def test_a_l_02_live_false_negative_regression_with_real_models():
    shelf_weights = ROOT / "best.pt"
    empty_weights = ROOT / "empty_shelf_yolo11m_best.pt"
    if not shelf_weights.is_file() or not empty_weights.is_file():
        pytest.skip("Local production weights are not available.")

    from ultralytics import YOLO

    image = cv2.imread(str(DATA / "frame.jpg"))
    assert image is not None
    shelf_model = YOLO(str(shelf_weights))
    empty_model = YOLO(str(empty_weights))
    assert shelf_model.task == "segment" and shelf_model.names == {0: "shelves"}
    assert empty_model.task == "detect" and empty_model.names == {0: "empty_shelf"}
    pipeline = ShelfInferencePipeline(
        shelf_model,
        empty_model,
        load_store_map(DATA / "store_map.json"),
        load_pose_config(ROOT / "configs" / "pose_geometry.yaml"),
        PipelineConfig(empty_conf=.10),
    )
    result = pipeline.infer(image, load_frame_metadata(DATA / "metadata.json")).result
    shelf = next(item for item in result["shelves"] if item["shelf_id"] == "A-L-02-01")
    assert result["counts"]["matched_shelves"] == 1
    assert shelf["parent_shelf_id"] == "A-L-02"
    assert shelf["shelf_level_id"] == "A-L-02-01" and shelf["level_number"] == 1
    assert result["empty_inference_mode"] == "shelf_level_roi"
    assert result["counts"]["roi_inferences"] == 1
    assert result["counts"]["empty_model_predict_calls"] == 1
    assert result["counts"]["empty_inference_inputs"] == 1
    assert result["counts"]["final_empty_spaces"] == 1
    assert shelf["empty_space_count"] == 1
    assert shelf["detections"][0]["confidence"] >= .10
