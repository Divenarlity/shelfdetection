from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import yaml

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pose_geometry import load_frame_metadata


def validate_session(session_dir: Path, is_mapping: bool = True) -> bool:
    session_dir = Path(session_dir).resolve()
    input_dir = session_dir / "input" if (session_dir / "input").exists() else session_dir

    if not input_dir.exists():
        print(f"ERROR: Session directory does not exist: {input_dir}")
        return False

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print("ERROR: No JSON metadata files found in session directory.")
        return False

    # Read mapping config if required or available
    config_path = Path(__file__).resolve().parent / "configs" / "mapping.yaml"
    min_pose_quality = 0.70
    min_observations = 5
    min_baseline_m = 0.30
    if config_path.exists():
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            min_pose_quality = config.get("min_pose_quality", min_pose_quality)
            min_observations = config.get("min_observations", min_observations)
            min_baseline_m = config.get("min_baseline_m", min_baseline_m)
        except Exception:
            pass

    frame_ids = set()
    session_ids = set()
    store_ids = set()
    resolutions = set()
    camera_positions = []
    usable_frames = 0
    pose_rejected = 0
    errors = []

    all_metadata = []
    for f in json_files:
        try:
            meta = load_frame_metadata(f)
            all_metadata.append(meta)

            fid = meta.get("frame_id")
            if fid in frame_ids:
                errors.append(f"Duplicate frame_id found: {fid}")
            frame_ids.add(fid)

            if "session_id" in meta:
                session_ids.add(meta["session_id"])
            if "store_id" in meta:
                store_ids.add(meta["store_id"])

            res = meta.get("resolution")
            if not res or len(res) != 2 or res[0] <= 0 or res[1] <= 0:
                errors.append(f"Frame {fid}: Invalid image dimensions {res}")
            else:
                resolutions.add(tuple(res))

            # Intrinsics check
            intr = meta.get("intrinsics", {})
            for k in ("fx", "fy", "cx", "cy"):
                if k not in intr or intr[k] <= 0:
                    errors.append(f"Frame {fid}: Invalid intrinsic parameter {k}={intr.get(k)}")

            # Image existence check
            rel_image = meta.get("image_path")
            img_path = input_dir / rel_image
            if not img_path.exists():
                errors.append(f"Image not found for frame {fid}: {img_path}")

            # Quaternion check
            quat = meta.get("camera_rotation_xyzw")
            if not quat or len(quat) != 4 or not np.all(np.isfinite(quat)):
                errors.append(f"Frame {fid}: Invalid camera rotation quaternion {quat}")

            # Position check
            pos = meta.get("camera_position_world")
            if pos is None or len(pos) != 3 or not np.all(np.isfinite(pos)):
                errors.append(f"Frame {fid}: Non-finite or invalid camera position {pos}")
            else:
                camera_positions.append(np.asarray(pos, float))

            # Pose quality & flags check
            pose = meta.get("pose", {})
            pq = float(pose.get("quality", 1.0))
            reloc = bool(pose.get("relocalized", True))
            scale_init = bool(pose.get("scale_initialized", True))

            if pq >= min_pose_quality and reloc and scale_init:
                usable_frames += 1
            else:
                pose_rejected += 1

        except Exception as e:
            errors.append(f"Failed to process {f.name}: {e}")

    session_name = session_ids.pop() if len(session_ids) == 1 else session_dir.name
    print(f"Session: {session_name}")
    print(f"Frames: {len(json_files)}")
    print(f"Usable frames: {usable_frames}")
    print(f"Pose rejected: {pose_rejected}")

    if len(camera_positions) > 1:
        pos_arr = np.array(camera_positions)
        dists = np.linalg.norm(pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :], axis=-1)
        max_baseline = float(np.max(dists))
    else:
        max_baseline = 0.0

    print(f"Maximum camera baseline: {max_baseline:.2f} m")

    if is_mapping:
        print(f"Required observations: {min_observations}")
        print(f"Required baseline: {min_baseline_m:.2f} m")

    if errors:
        print("Schema: FAIL")
        print("Images: FAIL" if any("Image not found" in e for e in errors) else "Images: OK")
        for err in errors:
            print(f"ERROR: {err}")
        if is_mapping:
            print("Mapping readiness: NOT READY")
        return False

    print("Schema: OK")
    print("Images: OK")

    if len(session_ids) > 1:
        print(f"WARNING: Multiple session_ids found: {session_ids}")
    if len(store_ids) > 1:
        print(f"WARNING: Multiple store_ids found: {store_ids}")
    if len(resolutions) > 1:
        print(f"WARNING: Multiple image resolutions found: {resolutions}")

    reasons = []
    if usable_frames < min_observations:
        reasons.append(f"{usable_frames}/{min_observations} usable frames")
    if max_baseline < min_baseline_m:
        reasons.append(f"baseline {max_baseline:.2f} m < {min_baseline_m:.2f} m")

    if is_mapping:
        if reasons:
            print(f"Mapping readiness: NOT READY ({', '.join(reasons)})")
            return False
        else:
            print("Mapping readiness: READY")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a Unity recording session.")
    parser.add_argument("session_dir", type=Path, help="Path to the session directory.")
    parser.add_argument("--mapping", action="store_true", default=True, help="Validate for mapping readiness.")
    args = parser.parse_args()

    if not args.session_dir.exists():
        print(f"ERROR: Session directory not found: {args.session_dir}")
        sys.exit(1)
    else:
        success = validate_session(args.session_dir, is_mapping=args.mapping)
        sys.exit(0 if success else 1)
