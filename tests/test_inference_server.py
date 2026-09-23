import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import cv2
import numpy as np
from fastapi.testclient import TestClient
import pytest

import inference_server
from inference_server import create_app


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value
    def tolist(self): return self.value.tolist()


class Boxes:
    def __init__(self, confidences, classes):
        self.conf, self.cls = Tensor(confidences), Tensor(classes)
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
    def __init__(self): self.calls = []
    def predict(self, **kwargs):
        self.calls.append(kwargs)
        count = len(kwargs["source"]) if isinstance(kwargs["source"], list) else 1
        return [SimpleNamespace(boxes=None) for _ in range(count)]


def setup(tmp_path, shelf_model=None, empty_model=None):
    pose_path = tmp_path / "pose_geometry.yaml"
    pose_path.write_text("min_visible_area: 1\nmin_score: 0.25\n", encoding="utf-8")
    map_path = tmp_path / "store_map.json"
    map_path.write_text(json.dumps({"shelves": [{
        "shelf_id": "A-L-01", "active": True,
        "corners_world": [[-.5,-.5,2],[-.5,.5,2],[.5,.5,2],[.5,-.5,2]],
        "center_world": [0,0,2], "front_normal_world": [0,0,-1],
    }]}), encoding="utf-8")
    shelf_model, empty_model = shelf_model or ShelfModel(), empty_model or EmptyModel()
    app = create_app(tmp_path / "best.pt", tmp_path / "empty.pt", map_path, pose_path,
                     shelf_model=shelf_model, empty_model=empty_model)
    metadata = {
        "schema_version": 3, "frame_id": "frame_000001",
        "image": {"width": 120, "height": 80},
        "camera": {"position_map": [0,0,0], "rotation_xyzw": [0,0,0,1],
                   "intrinsics": {"fx":80,"fy":80,"cx":60,"cy":40}},
        "pose": {"quality":1,"relocalized":True,"scale_initialized":True},
    }
    ok, encoded = cv2.imencode(".jpg", np.zeros((80,120,3), np.uint8))
    assert ok
    return app, shelf_model, empty_model, metadata, encoded.tobytes()


def post(client, metadata, image):
    return client.post("/infer", data={"metadata": json.dumps(metadata)},
                       files={"image": ("frame.jpg", image, "image/jpeg")})


def test_server_imports_shared_pipeline_not_offline_cli():
    assert "run_shelf_gap_cascade" not in inspect.getsource(inference_server)


def test_health_and_valid_request_use_one_full_frame_empty_inference(tmp_path):
    app, shelf_model, empty_model, metadata, image = setup(tmp_path)
    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok", "shelf_model_loaded": True, "empty_model_loaded": True,
        }
        response = post(client, metadata, image)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] and body["frame_id"] == "frame_000001"
    assert body["image"] == {"width": 120, "height": 80}
    assert body["counts"]["segmented_shelves"] == 2
    assert body["counts"]["matched_shelves"] == body["counts"]["unknown_shelves"] == 1
    assert body["empty_inference_mode"] == "full_frame_gated"
    assert body["counts"]["roi_inferences"] == 0
    assert body["counts"]["empty_model_predict_calls"] == 1
    assert body["counts"]["empty_inference_inputs"] == 1
    assert body["counts"]["final_empty_spaces"] == 0
    assert "debug" not in body
    assert len(shelf_model.calls) == len(empty_model.calls) == 1
    assert shelf_model.calls[0]["source"].shape == (80, 120, 3)
    assert empty_model.calls[0]["source"].shape == (80, 120, 3)
    matched = next(shelf for shelf in body["shelves"] if shelf["shelf_id"] == "A-L-01")
    assert matched["empty_roi_mode"] is None
    assert matched["empty_inference_source"] == "full_frame"


def test_invalid_metadata_ground_truth_and_resolution_are_rejected_before_models(tmp_path):
    app, shelf_model, empty_model, metadata, image = setup(tmp_path)
    with TestClient(app) as client:
        assert client.post("/infer", data={"metadata": "{"},
                           files={"image": ("frame.jpg", image, "image/jpeg")}).status_code == 422
        metadata["visible_shelf_ids"] = ["A-L-01"]
        assert post(client, metadata, image).status_code == 422
        del metadata["visible_shelf_ids"]
        metadata["image"]["width"] = 999
        assert post(client, metadata, image).status_code == 422
    assert shelf_model.calls == empty_model.calls == []


def test_requests_do_not_run_models_concurrently(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class SlowShelf(ShelfModel):
        def predict(self, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().predict(**kwargs)
    app, _, _, metadata, image = setup(tmp_path, shelf_model=SlowShelf())
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(post, client, metadata, image)
        assert entered.wait(5)
        second = post(client, metadata, image)
        assert second.status_code == 503
        release.set()
        assert first.result(timeout=10).status_code == 200


def test_detection_shelf_model_is_supported_and_unsupported_task_fails_startup(tmp_path):
    wrong = ShelfModel()
    wrong.task = "pose"
    app, _, _, _, _ = setup(tmp_path, shelf_model=wrong)
    with pytest.raises(RuntimeError, match="detect.*segment"):
        with TestClient(app):
            pass
