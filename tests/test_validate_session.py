import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from validate_session import validate_session

@pytest.fixture
def session_path(tmp_path):
    session_dir = tmp_path / "test_session"
    input_dir = session_dir / "input"
    input_dir.mkdir(parents=True)
    return session_dir

def create_frame(input_dir, frame_id, pos, quat=[0,0,0,1], image=True):
    meta = {
        "schema_version": 3,
        "session_id": "test_session",
        "store_id": "MARKET-001",
        "frame_id": str(frame_id),
        "image_path": f"{frame_id}.png",
        "image": {"width": 100, "height": 80},
        "camera": {
            "position_map": pos,
            "rotation_xyzw": quat,
            "intrinsics": {"fx": 90, "fy": 90, "cx": 50, "cy": 40}
        },
        "pose": {"quality": 0.9, "relocalized": True, "scale_initialized": True}
    }
    (input_dir / f"{frame_id}.json").write_text(json.dumps(meta))
    if image:
        (input_dir / f"{frame_id}.png").touch()

def test_valid_session(session_path, capsys):
    input_dir = session_path / "input"
    create_frame(input_dir, 1, [0, 0, 0])
    create_frame(input_dir, 2, [1, 0, 0])
    
    validate_session(session_path, is_mapping=False)
    captured = capsys.readouterr()
    
    assert "Session: test_session" in captured.out
    assert "Frames: 2" in captured.out
    assert "Schema: OK" in captured.out
    assert "Images: OK" in captured.out
    assert "Maximum camera baseline: 1.00 m" in captured.out
    assert "Usable frames: 2" in captured.out
    assert "Pose rejected: 0" in captured.out

def test_missing_image(session_path, capsys):
    input_dir = session_path / "input"
    create_frame(input_dir, 1, [0, 0, 0], image=False)
    
    validate_session(session_path, is_mapping=False)
    captured = capsys.readouterr()
    
    assert "Image not found for frame 1" in captured.out

def test_duplicate_frame_id(session_path, capsys):
    input_dir = session_path / "input"
    create_frame(input_dir, 1, [0, 0, 0])
    
    meta = {
        "schema_version": 3,
        "session_id": "test_session",
        "store_id": "MARKET-001",
        "frame_id": "1",
        "image_path": "2.png",
        "image": {"width": 100, "height": 80},
        "camera": {
            "position_map": [1, 0, 0],
            "rotation_xyzw": [0,0,0,1],
            "intrinsics": {"fx": 90, "fy": 90, "cx": 50, "cy": 40}
        },
        "pose": {"quality": 0.9, "relocalized": True, "scale_initialized": True}
    }
    (input_dir / "2.json").write_text(json.dumps(meta))
    (input_dir / "2.png").touch()

    validate_session(session_path, is_mapping=False)
    captured = capsys.readouterr()

    assert "Duplicate frame_id found: 1" in captured.out

def test_mapping_ready(session_path, capsys):
    input_dir = session_path / "input"
    for i in range(5):
        create_frame(input_dir, i, [i * 0.1, 0, 0])
    
    validate_session(session_path, is_mapping=True)
    captured = capsys.readouterr()
    assert "Mapping readiness: READY" in captured.out

def test_mapping_not_ready_baseline(session_path, capsys):
    input_dir = session_path / "input"
    for i in range(5):
        create_frame(input_dir, i, [i * 0.05, 0, 0])

    validate_session(session_path, is_mapping=True)
    captured = capsys.readouterr()
    assert "Mapping readiness: NOT READY" in captured.out
    assert "baseline 0.20 m < 0.30 m" in captured.out

def test_mapping_not_ready_observations(session_path, capsys):
    input_dir = session_path / "input"
    for i in range(4):
        create_frame(input_dir, i, [i * 0.1, 0, 0])

    validate_session(session_path, is_mapping=True)
    captured = capsys.readouterr()
    assert "Mapping readiness: NOT READY" in captured.out
    assert "4/5 usable frames" in captured.out

def test_ground_truth_leakage_rejection(session_path, capsys):
    input_dir = session_path / "input"
    meta = {
        "schema_version": 3, "frame_id": "1", "image_path": "1.png",
        "image": {"width": 100, "height": 80},
        "camera": {"position_map": [0,0,0], "rotation_xyzw": [0,0,0,1], "intrinsics": {"fx": 90, "fy": 90, "cx": 50, "cy": 40}},
        "shelf_id": "A-L-01"
    }
    (input_dir / "1.json").write_text(json.dumps(meta))
    (input_dir / "1.png").touch()
    
    validate_session(session_path, is_mapping=False)
    captured = capsys.readouterr()

    assert "Inference metadata ID/ground-truth alanı içeremez" in captured.out
