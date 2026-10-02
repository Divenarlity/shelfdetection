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
                return img.copy()
            result.plot = plot
            results.append(result)
        return results


class EmptyModel:
    task, names = "detect", {0: "empty_shelf"}
    def __init__(self): self.calls = []
    def predict(self, **kwargs):
        self.calls.append(kwargs)
        count = len(kwargs["source"]) if isinstance(kwargs["source"], list) else 1
        return [SimpleNamespace(boxes=None) for _ in range(count)]


def setup(tmp_path, shelf_model=None, empty_model=None, visualization_cache_size=20,
          save_scan_sessions=False, scan_session_root=None):
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
                     shelf_model=shelf_model, empty_model=empty_model,
                     visualization_cache_size=visualization_cache_size,
                     save_scan_sessions=save_scan_sessions,
                     scan_session_root=scan_session_root or tmp_path / "scan_sessions")
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


def post_dual(client, left_metadata, right_metadata, image,
              session_id=None, station_id=None):
    data = {"left_metadata": json.dumps(left_metadata),
            "right_metadata": json.dumps(right_metadata)}
    if session_id is not None:
        data["session_id"] = session_id
    if station_id is not None:
        data["station_id"] = station_id
    return client.post(
        "/infer/dual",
        data=data,
        files={"left_image": ("left.jpg", image, "image/jpeg"),
               "right_image": ("right.jpg", image, "image/jpeg")},
    )


def test_server_imports_shared_pipeline_not_offline_cli():
    assert "run_shelf_gap_cascade" not in inspect.getsource(inference_server)


def test_model_loader_supplies_explicit_production_task(tmp_path, monkeypatch):
    path = tmp_path / "model.engine"
    path.write_bytes(b"engine")
    calls = []

    class Loaded:
        task = "segment"
        names = {0: "shelves"}

    def fake_yolo(model_path, task):
        calls.append((model_path, task))
        return Loaded()

    monkeypatch.setattr("ultralytics.YOLO", fake_yolo)
    assert inference_server._load_model(path, "Shelf model", "segment").task == "segment"
    assert calls == [(str(path), "segment")]


def test_health_and_valid_request_use_default_shelf_level_roi_inference(tmp_path):
    app, shelf_model, empty_model, metadata, image = setup(tmp_path)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["shelf_model_loaded"] and health["empty_model_loaded"]
        assert health["device"] == "cpu"
        assert health["precision"] == "fp32"
        assert health["backend"] == "pytorch"
        assert isinstance(health["cuda_available"], bool)
        assert health["warmup_iterations"] == 0
        response = post(client, metadata, image)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] and body["frame_id"] == "frame_000001"
    assert body["image"] == {"width": 120, "height": 80}
    assert body["counts"]["segmented_shelves"] == 2
    assert body["counts"]["matched_shelves"] == body["counts"]["unknown_shelves"] == 1
    assert body["empty_inference_mode"] == "shelf_level_roi"
    assert body["counts"]["roi_inferences"] == 1
    assert body["counts"]["empty_model_predict_calls"] == 1
    assert body["counts"]["empty_inference_inputs"] == 1
    assert body["counts"]["final_empty_spaces"] == 0
    assert body["visualization_url"].startswith("/visualization/")
    assert body["visualization_url"].endswith(".jpg")
    assert body["timing"]["visualization_render_ms"] >= 0
    assert body["timing"]["visualization_encode_ms"] >= 0
    assert "debug" not in body
    assert len(shelf_model.calls) == len(empty_model.calls) == 1
    assert shelf_model.plot_calls == 1
    assert shelf_model.calls[0]["source"].shape == (80, 120, 3)
    assert empty_model.calls[0]["source"].shape != (80, 120, 3)
    matched = next(shelf for shelf in body["shelves"] if shelf["shelf_id"] == "A-L-01-01")
    assert matched["parent_shelf_id"] == "A-L-01"
    assert matched["shelf_level_id"] == "A-L-01-01"
    assert matched["level_number"] == 1
    assert matched["empty_roi_mode"] == "full_shelf"
    assert matched["empty_inference_source"] == "shelf_level_roi"


def test_visualization_url_returns_jpeg_and_unknown_id_is_404(tmp_path):
    app, _, _, metadata, image = setup(tmp_path)
    with TestClient(app) as client:
        body = post(client, metadata, image).json()
        response = client.get(body["visualization_url"])
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["cache-control"] == "no-store"
        decoded = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (80, 120, 3)
        assert client.get("/visualization/" + "0" * 32 + ".jpg").status_code == 404
        assert client.get("/visualization/not-an-id.jpg").status_code == 404


def test_visualization_cache_is_bounded_and_get_does_not_use_inference_lock(tmp_path):
    app, _, _, metadata, image = setup(tmp_path, visualization_cache_size=10)
    with TestClient(app) as client:
        urls = [post(client, metadata, image).json()["visualization_url"] for _ in range(11)]
        assert len(app.state.service.visualizations) == 10
        app.state.service.lock.acquire()
        try:
            assert client.get(urls[-1]).status_code == 200
        finally:
            app.state.service.lock.release()
        assert client.get(urls[0]).status_code == 404


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


def test_dual_endpoint_uses_one_two_image_shelf_batch_and_one_combined_roi_batch(tmp_path):
    app, shelf_model, empty_model, left_metadata, image = setup(tmp_path)
    right_metadata = json.loads(json.dumps(left_metadata))
    right_metadata["frame_id"] = "frame_000001_right"
    with TestClient(app) as client:
        response = post_dual(client, left_metadata, right_metadata, image)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] and body["mode"] == "dual_camera_batch"
        assert body["left"]["frame_id"] == "frame_000001"
        assert body["right"]["frame_id"] == "frame_000001_right"
        assert body["counts"] == {
            "shelf_model_predict_calls": 1, "shelf_model_inputs": 2,
            "empty_model_predict_calls": 1, "left_empty_inputs": 1,
            "right_empty_inputs": 1, "total_empty_inputs": 2,
            "left_final_empty_spaces": 0, "right_final_empty_spaces": 0,
        }
        assert body["timing"]["shelf_batch_size"] == 2
        assert body["timing"]["empty_batch_size"] == 2
        assert body["timing"]["visualization_encode_ms"] >= 0
        assert body["left"]["visualization_url"] != body["right"]["visualization_url"]
        assert client.get(body["left"]["visualization_url"]).status_code == 200
        assert client.get(body["right"]["visualization_url"]).status_code == 200
    assert len(shelf_model.calls) == len(empty_model.calls) == 1
    assert len(shelf_model.calls[0]["source"]) == 2
    assert len(empty_model.calls[0]["source"]) == 2
    assert shelf_model.plot_calls == 2


def test_dual_endpoint_rejects_either_bad_view_before_running_models(tmp_path):
    app, shelf_model, empty_model, left_metadata, image = setup(tmp_path)
    right_metadata = json.loads(json.dumps(left_metadata))
    right_metadata["frame_id"] = "right"
    with TestClient(app) as client:
        bad_left = client.post(
            "/infer/dual",
            data={"left_metadata": "{", "right_metadata": json.dumps(right_metadata)},
            files={"left_image": ("left.jpg", image, "image/jpeg"),
                   "right_image": ("right.jpg", image, "image/jpeg")},
        )
        assert bad_left.status_code == 422
        bad_right_image = client.post(
            "/infer/dual",
            data={"left_metadata": json.dumps(left_metadata),
                  "right_metadata": json.dumps(right_metadata)},
            files={"left_image": ("left.jpg", image, "image/jpeg"),
                   "right_image": ("right.jpg", b"not-an-image", "image/jpeg")},
        )
        assert bad_right_image.status_code == 422
        bad_left_image = client.post(
            "/infer/dual",
            data={"left_metadata": json.dumps(left_metadata),
                  "right_metadata": json.dumps(right_metadata)},
            files={"left_image": ("left.jpg", b"not-an-image", "image/jpeg"),
                   "right_image": ("right.jpg", image, "image/jpeg")},
        )
        assert bad_left_image.status_code == 422
        wrong_resolution = json.loads(json.dumps(right_metadata))
        wrong_resolution["image"]["width"] = 999
        assert post_dual(client, left_metadata, wrong_resolution, image).status_code == 422
        forbidden = json.loads(json.dumps(right_metadata))
        forbidden["visible_shelf_ids"] = ["A-L-01"]
        assert post_dual(client, left_metadata, forbidden, image).status_code == 422
        same_id = json.loads(json.dumps(left_metadata))
        assert post_dual(client, left_metadata, same_id, image).status_code == 422
    assert shelf_model.calls == empty_model.calls == []


def test_dual_endpoint_busy_is_503_and_single_endpoint_remains_unchanged(tmp_path):
    app, _, _, left_metadata, image = setup(tmp_path)
    right_metadata = json.loads(json.dumps(left_metadata))
    right_metadata["frame_id"] = "right"
    with TestClient(app) as client:
        app.state.service.lock.acquire()
        try:
            assert post_dual(client, left_metadata, right_metadata, image).status_code == 503
        finally:
            app.state.service.lock.release()
        single = post(client, left_metadata, image)
        assert single.status_code == 200
        assert single.json()["frame_id"] == "frame_000001"
        assert "mode" not in single.json()


def test_dual_persistence_is_optional_and_failure_does_not_fail_inference(tmp_path):
    app, shelf_model, empty_model, left_metadata, image = setup(
        tmp_path, save_scan_sessions=True
    )
    right_metadata = json.loads(json.dumps(left_metadata))
    right_metadata["frame_id"] = "right"
    with TestClient(app) as client:
        missing = post_dual(client, left_metadata, right_metadata, image)
        assert missing.status_code == 200
        assert missing.json()["persistence"]["saved"] is False
        assert not (tmp_path / "scan_sessions").exists()

        class BrokenRecorder:
            def record_station(self, *_args, **_kwargs):
                raise OSError("disk full")
            def complete_session(self, *_args, **_kwargs):
                raise OSError("disk full")

        app.state.service.session_recorder = BrokenRecorder()
        response = post_dual(
            client, left_metadata, right_metadata, image,
            "session_20261002_140512", "A-01",
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["persistence"]["enabled"] is True
        assert body["persistence"]["saved"] is False
        assert "disk full" in body["persistence"]["warning"]
        completed = client.post("/session/session_20261002_140512/complete")
        assert completed.status_code == 200
        assert completed.json()["saved"] is False
    assert len(shelf_model.calls) == len(empty_model.calls) == 2


def test_dual_endpoint_records_exact_bytes_and_completion(tmp_path):
    app, shelf_model, empty_model, left_metadata, image = setup(
        tmp_path, save_scan_sessions=True
    )
    right_metadata = json.loads(json.dumps(left_metadata))
    right_metadata["frame_id"] = "right"
    with TestClient(app) as client:
        response = post_dual(
            client, left_metadata, right_metadata, image,
            "session_20261002_140512", "A-01",
        )
        assert response.status_code == 200, response.text
        persistence = response.json()["persistence"]
        assert persistence["saved"] is True and persistence["save_ms"] >= 0
        session_dir = tmp_path / "scan_sessions" / persistence["session_id"]
        station_dir = session_dir / "A-01"
        assert (station_dir / "left_raw.jpg").read_bytes() == image
        assert (station_dir / "right_raw.jpg").read_bytes() == image
        assert (station_dir / "left_annotated.jpg").read_bytes() == client.get(
            response.json()["left"]["visualization_url"]
        ).content
        assert (station_dir / "right_annotated.jpg").read_bytes() == client.get(
            response.json()["right"]["visualization_url"]
        ).content
        archived = json.loads((station_dir / "dual_result.json").read_text("utf-8"))
        assert archived["left"]["frame_id"] == left_metadata["frame_id"]
        assert archived["right"]["frame_id"] == right_metadata["frame_id"]
        assert "persistence" not in archived
        complete = client.post("/session/session_20261002_140512/complete")
        assert complete.status_code == 200 and complete.json()["saved"] is True
    summary = json.loads((session_dir / "session_summary.json").read_text("utf-8"))
    assert summary["status"] == "completed" and summary["completed_at"]
    assert len(shelf_model.calls) == len(empty_model.calls) == 1


def test_scan_session_cli_defaults_enabled_and_can_be_disabled():
    parser = inference_server.build_parser()
    defaults = parser.parse_args(["--store-map", "map.json"])
    assert defaults.save_scan_sessions is True
    assert defaults.scan_session_root.as_posix() == "runs/scan_sessions"
    assert defaults.device == "cpu" and defaults.precision == "fp32"
    assert defaults.warmup_iterations == 0
    assert defaults.warmup_empty_batch_size == 24
    assert parser.parse_args([
        "--store-map", "map.json", "--no-save-scan-sessions"
    ]).save_scan_sessions is False


def test_gpu_startup_options_parse_without_changing_batch_configuration():
    args = inference_server.build_parser().parse_args([
        "--store-map", "map.json", "--device", "0", "--half",
        "--warmup-iterations", "3", "--warmup-empty-batch-size", "20",
        "--cudnn-benchmark",
    ])
    config = inference_server.pipeline_config_from_args(args)
    assert config.device == "0" and config.precision == "fp16"
    assert config.shelf_imgsz == 960 and config.empty_imgsz == 640
    assert args.warmup_iterations == 3 and args.warmup_empty_batch_size == 20
    assert args.cudnn_benchmark is True


def test_startup_warmup_configuration_rejects_invalid_values(tmp_path):
    with pytest.raises(ValueError, match="warmup_iterations"):
        create_app(store_map_path=tmp_path / "map.json", warmup_iterations=-1)
    with pytest.raises(ValueError, match="warmup_empty_batch_size"):
        create_app(store_map_path=tmp_path / "map.json", warmup_empty_batch_size=0)
