"""Evaluate saved inference results against separately captured Unity ground truth."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from src.pose_geometry import UNKNOWN_SHELF


def evaluate(predictions, ground_truth):
    rows = []
    per_shelf = defaultdict(lambda: [0, 0])
    for prediction_path in sorted(Path(predictions).glob("**/result.json")):
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        frame_id = str(prediction.get("frame_id"))
        ground_truth_path = Path(ground_truth) / (
            f"{frame_id}.json" if frame_id.startswith("frame_") else f"frame_{frame_id}.json"
        )
        if not ground_truth_path.exists():
            continue
        truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        expected = set(truth.get("visible_shelf_ids", []))
        shelves = prediction.get("shelves", [])
        actual = {
            shelf["shelf_id"] for shelf in shelves
            if shelf.get("shelf_id") != UNKNOWN_SHELF
        }
        unknown = sum(shelf.get("shelf_id") == UNKNOWN_SHELF for shelf in shelves)
        true_positive = len(actual & expected)
        false_positive = len(actual - expected)
        false_negative = len(expected - actual)
        for shelf_id in expected:
            per_shelf[shelf_id][1] += 1
            per_shelf[shelf_id][0] += int(shelf_id in actual)
        rows.append({
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "exact": actual == expected,
            "unknown": unknown,
            "shelves": len(shelves),
        })

    true_positive = sum(row["tp"] for row in rows)
    false_positive = sum(row["fp"] for row in rows)
    false_negative = sum(row["fn"] for row in rows)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "frames": len(rows),
        "shelf_id_accuracy": true_positive / max(
            true_positive + false_positive + false_negative, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "exact_frame_match": sum(row["exact"] for row in rows) / max(len(rows), 1),
        "wrong_id_rate": false_positive / max(true_positive + false_positive, 1),
        "unknown_shelf_rate": sum(row["unknown"] for row in rows) / max(
            sum(row["shelves"] for row in rows), 1),
        "per_shelf": {
            shelf_id: {
                "correct": values[0],
                "total": values[1],
                "accuracy": values[0] / values[1],
            }
            for shelf_id, values in sorted(per_shelf.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved results against separate Unity ground truth."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation_metrics.json"))
    args = parser.parse_args()
    payload = evaluate(args.predictions, args.ground_truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
