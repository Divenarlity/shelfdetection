"""Run the production pose-based cascade against one saved Unity frame."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

from src.options import add_pipeline_arguments, pipeline_config_from_args
from src.pipeline import ShelfInferencePipeline
from src.pose_geometry import load_frame_metadata, load_pose_config, load_store_map

ROOT = Path(__file__).resolve().parent


def resolve(path: Path):
    return (path if path.is_absolute() else ROOT / path).resolve()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run shelf segmentation, pose/map matching, and matched-shelf empty detection."
    )
    parser.add_argument("--source", type=Path, required=True, help="Saved Unity RGB frame")
    parser.add_argument("--frame-metadata", type=Path, required=True, help="ID-free Unity metadata JSON")
    parser.add_argument("--output-dir", type=Path, help="Output directory; defaults under runs/offline")
    parser.add_argument("--debug-live", action="store_true", help="Save diagnostic control evidence")
    parser.add_argument("--show", action="store_true")
    return add_pipeline_arguments(parser, ROOT)


def _load_model(path, label):
    path = resolve(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    from ultralytics import YOLO
    return YOLO(str(path))


def run(args, shelf_model=None, empty_model=None):
    source = resolve(args.source)
    metadata_path = resolve(args.frame_metadata)
    image = cv2.imread(str(source))
    if image is None:
        raise RuntimeError(f"Image could not be read: {source}")
    metadata = load_frame_metadata(metadata_path)
    output_dir = resolve(args.output_dir) if args.output_dir else (
        ROOT / "runs" / "offline" /
        f"{metadata['frame_id']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    debug_dir = output_dir / "debug" if args.debug_live else None
    if debug_dir:
        debug_dir.mkdir()

    shelf_model = shelf_model or _load_model(args.shelf_model, "Shelf model")
    empty_model = empty_model or _load_model(args.empty_model, "Empty model")
    pipeline = ShelfInferencePipeline(
        shelf_model,
        empty_model,
        load_store_map(resolve(args.store_map)),
        load_pose_config(resolve(args.pose_config)),
        pipeline_config_from_args(args),
    )
    outcome = pipeline.infer(image, metadata, debug_dir)
    (output_dir / "result.json").write_text(
        json.dumps(outcome.result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not cv2.imwrite(str(output_dir / "annotated.jpg"), outcome.annotated_image):
        raise OSError("Could not save annotated image.")
    print(json.dumps(outcome.result["counts"], ensure_ascii=False))
    print(f"Output: {output_dir}")
    if args.show:
        cv2.imshow("shelf inference", outcome.annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return outcome, output_dir


def main():
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
