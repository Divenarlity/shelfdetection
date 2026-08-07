from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from ultralytics import YOLO

from run_shelf_gap_cascade import build_parser, run


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Bir klasördeki görüntüler için shelf-gap cascade çalıştırır.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--camera-id", default="TEST-BATCH")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    images = sorted(
        path for path in args.source.resolve().iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    shelf_model = YOLO("best.pt")
    empty_model = YOLO("empty_shelf_yolo11m_best.pt")
    records: list[dict] = []
    started = time.time()

    for index, path in enumerate(images, 1):
        item_started = time.time()
        try:
            cascade_args = build_parser().parse_args([
                "--source", str(path),
                "--camera-id", args.camera_id,
                "--device", args.device,
                "--save-annotated",
                "--rebuild-shelf-map",
            ])
            payload, output_dir = run(
                cascade_args,
                shelf_model=shelf_model,
                empty_model=empty_model,
            )
            record = {
                "index": index,
                "source": str(path),
                "status": "ok",
                "output": str(output_dir),
                "shelves": payload["detected_shelf_count"],
                "empty_spaces": payload["final_empty_space_count"],
                "seconds": round(time.time() - item_started, 3),
            }
        except Exception as exc:
            traceback.print_exc()
            record = {
                "index": index,
                "source": str(path),
                "status": "error",
                "error": str(exc),
                "seconds": round(time.time() - item_started, 3),
            }

        records.append(record)
        summary = {
            "source": str(args.source.resolve()),
            "total": len(images),
            "processed": len(records),
            "ok": sum(item["status"] == "ok" for item in records),
            "errors": sum(item["status"] == "error" for item in records),
            "total_shelves": sum(item.get("shelves", 0) for item in records),
            "total_empty_spaces": sum(item.get("empty_spaces", 0) for item in records),
            "elapsed_seconds": round(time.time() - started, 3),
            "records": records,
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "BATCH_PROGRESS",
            index,
            "/",
            len(images),
            record["status"],
            path.name,
            "seconds",
            record["seconds"],
            flush=True,
        )

    print(
        "BATCH_COMPLETE",
        json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
