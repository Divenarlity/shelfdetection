from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

FIELDS = ["timestamp","source","frame_index","market_id","camera_id","area_id","rack_id","shelf_id",
          "level","section","event","gap_confidence","shelf_confidence","gap_shelf_overlap",
          "mapping_source","status","evidence_image"]


def unique_output_path(folder, stem, suffix):
    """Var olan bir çıktının üzerine yazmadan yeni bir yol döndürür."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    candidate = folder / f"{stem}{suffix}"
    index = 1
    while candidate.exists():
        candidate = folder / f"{stem}_{index}{suffix}"
        index += 1
    return candidate


class Reporter:
    def __init__(self, root):
        self.root = Path(root)
        self.rows = []

    def add(self, row):
        self.rows.append({k: row.get(k, "") for k in FIELDS})

    def save(self):
        reports = self.root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        with (reports / "detections.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, FIELDS); writer.writeheader(); writer.writerows(self.rows)
        (reports / "detections.json").write_text(json.dumps(self.rows, ensure_ascii=False, indent=2), encoding="utf-8")
        counts = Counter(r["shelf_id"] for r in self.rows)
        with (reports / "shelf_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f); writer.writerow(["shelf_id","detection_count"]); writer.writerows(counts.items())
        unknown = [r for r in self.rows if r["shelf_id"] == "unknown_shelf" or r["section"] == "unknown_section"]
        (reports / "review_unknown.json").write_text(json.dumps(unknown, ensure_ascii=False, indent=2), encoding="utf-8")


def human_message(row):
    sections = {"left":"sol", "middle":"orta", "right":"sağ", "unknown_section":"bilinmeyen"}
    return f"{row['market_id']}, {row['area_id']} bölgesi, {row['shelf_id']} numaralı rafın {sections.get(row['section'], row['section'])} bölümünde boşluk tespit edildi."
