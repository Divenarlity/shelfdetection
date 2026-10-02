import json

import pytest

from src.session_recorder import ScanSessionRecorder, sanitize_identifier


def detection(parent, level, level_number, section, confidence, bbox, overlap=.75):
    return {
        "parent_shelf_id": parent,
        "shelf_level_id": level,
        "section": section,
        "confidence": confidence,
        "global_bbox_xyxy": bbox,
        "mask_overlap_ratio": overlap,
    }


def dual_result():
    return {
        "success": True,
        "mode": "dual_camera_batch",
        "left": {
            "success": True,
            "frame_id": "a-01_left_001",
            "shelves": [
                {
                    "shelf_id": "A-L-02-03",
                    "parent_shelf_id": "A-L-02",
                    "shelf_level_id": "A-L-02-03",
                    "status": "EMPTY_SPACE",
                    "level_number": 3,
                    "detections": [
                        detection("A-L-02", "A-L-02-03", 3, "SAĞ", .268,
                                  [412.3, 388.1, 525.6, 470.9]),
                        detection("A-L-02", "A-L-02-03", 3, "SOL", .31,
                                  [300.0, 388.1, 390.0, 470.9]),
                        detection("A-L-02", "A-L-02-03", 3, "ORTA", .29,
                                  [390.0, 388.1, 412.0, 470.9]),
                    ],
                },
                {
                    "shelf_id": "UNKNOWN_SHELF",
                    "status": "UNLOCALIZED",
                    "level_number": 0,
                    "detections": [
                        detection("INVENTED", "INVENTED-01", 1, "SOL", .99,
                                  [0, 0, 1, 1])
                    ],
                },
            ],
            "rejected_detections": [{"confidence": .999, "section": "SAĞ"}],
            "visualization_url": "/visualization/left.jpg",
        },
        "right": {
            "success": True,
            "frame_id": "a-01_right_002",
            "shelves": [],
            "rejected_detections": [{"confidence": .888, "section": "SOL"}],
            "visualization_url": "/visualization/right.jpg",
        },
        "timing": {"dual_total_ms": 123.4},
        "counts": {"shelf_model_predict_calls": 1, "shelf_model_inputs": 2},
    }


def record(tmp_path, result=None, session="session_20261002_140512", station="A-01"):
    recorder = ScanSessionRecorder(
        tmp_path / "scan_sessions",
        thresholds={"shelf_conf": .25, "empty_conf": .10},
    )
    info = recorder.record_station(
        session,
        station,
        left_raw=b"left raw jpeg",
        right_raw=b"right raw jpeg",
        left_annotated=b"left annotated jpeg",
        right_annotated=b"right annotated jpeg",
        dual_result=result or dual_result(),
    )
    return recorder, info, tmp_path / "scan_sessions" / info["session_id"]


def test_session_and_station_files_preserve_exact_bytes_and_json(tmp_path):
    result = dual_result()
    recorder, info, session_dir = record(tmp_path, result)
    station = session_dir / "A-01"
    assert info["saved"] and session_dir.is_dir() and station.is_dir()
    assert (station / "left_raw.jpg").read_bytes() == b"left raw jpeg"
    assert (station / "right_raw.jpg").read_bytes() == b"right raw jpeg"
    assert (station / "left_annotated.jpg").read_bytes() == b"left annotated jpeg"
    assert (station / "right_annotated.jpg").read_bytes() == b"right annotated jpeg"
    assert json.loads((station / "dual_result.json").read_text("utf-8")) == result
    assert json.loads((station / "left_result.json").read_text("utf-8")) == result["left"]
    assert json.loads((station / "right_result.json").read_text("utf-8")) == result["right"]
    assert (session_dir / "session_log.txt").is_file()
    assert (session_dir / "session_summary.json").is_file()
    assert recorder is not None


def test_log_uses_only_final_localized_detections_in_stable_order(tmp_path):
    _, _, session_dir = record(tmp_path)
    text = (session_dir / "session_log.txt").read_text("utf-8")
    assert text.count("SESSION START") == 1
    assert "Station: A-01" in text
    assert "Camera: LEFT" in text and "Camera: RIGHT" in text
    assert "Parent shelf: A-L-02" in text
    assert "Shelf level: A-L-02-03" in text
    assert "Level number: 3" in text
    assert "Gap position: SOL" in text
    assert "Gap position: ORTA" in text
    assert "Gap position: SAĞ" in text
    assert "Confidence: 0.310" in text and "Confidence: 0.268" in text
    assert "BBox: [412.3, 388.1, 525.6, 470.9]" in text
    assert "Mask overlap: 0.750" in text
    assert text.index("Gap position: SOL") < text.index("Gap position: ORTA")
    assert text.index("Gap position: ORTA") < text.index("Gap position: SAĞ")
    assert "Confidence: 0.999" not in text
    assert "Confidence: 0.888" not in text
    assert "Parent shelf: INVENTED" not in text
    assert "excluded from production gap logging" in text
    assert "Result: NO_EMPTY_SPACE" in text
    assert "--- Station A-01 Summary ---" in text
    assert "LEFT gaps: 3" in text and "RIGHT gaps: 0" in text and "TOTAL: 3" in text


def test_summary_updates_and_completion_appends_once(tmp_path):
    recorder, _, session_dir = record(tmp_path)
    summary = json.loads((session_dir / "session_summary.json").read_text("utf-8"))
    assert summary["status"] == "in_progress" and summary["completed_at"] is None
    assert summary["stations_processed"] == summary["dual_scans"] == 1
    assert summary["camera_results"] == 2
    assert summary["total_empty_spaces"] == 3
    assert summary["by_camera"] == {"left": 3, "right": 0}
    assert summary["by_parent_shelf"] == {"A-L-02": 3}
    assert summary["by_section"] == {"SOL": 1, "ORTA": 1, "SAĞ": 1}

    first = recorder.complete_session("session_20261002_140512")
    second = recorder.complete_session("session_20261002_140512")
    assert first["saved"] and not first["already_completed"]
    assert second["saved"] and second["already_completed"]
    summary = json.loads((session_dir / "session_summary.json").read_text("utf-8"))
    assert summary["status"] == "completed" and summary["completed_at"]
    text = (session_dir / "session_log.txt").read_text("utf-8")
    assert text.count("SESSION COMPLETE") == 1
    assert "Stations processed: 1" in text
    assert "Camera scans: 2" in text
    assert "Total gaps: 3" in text
    assert "A-L-02: 3" in text


def test_repeated_station_is_idempotent_and_does_not_duplicate_log(tmp_path):
    recorder, _, session_dir = record(tmp_path)
    repeat = recorder.record_station(
        "session_20261002_140512", "A-01",
        left_raw=b"different", right_raw=b"different",
        left_annotated=b"different", right_annotated=b"different",
        dual_result=dual_result(),
    )
    assert repeat["duplicate_ignored"] is True
    assert (session_dir / "A-01" / "left_raw.jpg").read_bytes() == b"left raw jpeg"
    text = (session_dir / "session_log.txt").read_text("utf-8")
    assert text.count("--- Station A-01 Summary ---") == 1
    summary = json.loads((session_dir / "session_summary.json").read_text("utf-8"))
    assert summary["stations_processed"] == 1


def test_names_are_sanitized_and_collision_never_overwrites(tmp_path):
    assert sanitize_identifier("../session:bad/name") == "_session_bad_name"
    assert sanitize_identifier("CON") == "_CON"
    with pytest.raises(ValueError):
        sanitize_identifier("..")
    root = tmp_path / "scan_sessions"
    (root / "session_20261002_140512").mkdir(parents=True)
    recorder, info, session_dir = record(tmp_path, station="A:01/..")
    assert info["session_id"].startswith("session_20261002_140512_")
    assert info["station_id"] == "A_01_"
    assert session_dir.parent == root.resolve()
    assert (session_dir / "A_01_").is_dir()
