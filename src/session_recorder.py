"""Persistent, production-only recording for dual-camera scan sessions."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import threading
from uuid import uuid4


_SAFE_CHARACTER = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_SECTION_ORDER = {"SOL": 0, "ORTA": 1, "SAĞ": 2}


def sanitize_identifier(value, label="identifier", max_length=80):
    """Return a safe single Windows path component without trusting client text."""
    if value is None:
        raise ValueError(f"{label} is required.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is required.")
    safe = _SAFE_CHARACTER.sub("_", text).strip(" .")[:max_length].rstrip(" .")
    if not safe or safe in {".", ".."}:
        raise ValueError(f"{label} does not contain a safe filename component.")
    if safe.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        safe = "_" + safe
    return safe


def _now():
    return datetime.now().astimezone()


def _timestamp(value):
    return value.isoformat(timespec="seconds")


def _display_timestamp(value):
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _atomic_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    _atomic_write_bytes(path, encoded)


def _append_text(path: Path, value: str):
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass
class _SessionState:
    requested_id: str
    session_id: str
    directory: Path
    summary: dict
    processed_stations: set[str]


class ScanSessionRecorder:
    """Owns session folders without changing or re-running inference."""

    def __init__(self, root, *, shelf_model="best.pt",
                 empty_model="empty_shelf_yolo11m_best.pt", mode="dual_camera_batch",
                 thresholds=None):
        self.root = Path(root).resolve()
        self.shelf_model = Path(shelf_model).name
        self.empty_model = Path(empty_model).name
        self.mode = mode
        self.thresholds = dict(thresholds or {})
        self._lock = threading.RLock()
        self._sessions: dict[str, _SessionState] = {}

    def _new_state(self, requested_session_id):
        safe_requested = sanitize_identifier(requested_session_id, "session_id")
        self.root.mkdir(parents=True, exist_ok=True)
        session_id = safe_requested
        directory = self.root / session_id
        while True:
            try:
                directory.mkdir(exist_ok=False)
                break
            except FileExistsError:
                session_id = f"{safe_requested}_{uuid4().hex[:8]}"
                directory = self.root / session_id

        started = _now()
        summary = {
            "session_id": session_id,
            "requested_session_id": safe_requested,
            "status": "in_progress",
            "started_at": _timestamp(started),
            "completed_at": None,
            "stations_processed": 0,
            "station_ids": [],
            "dual_scans": 0,
            "camera_results": 0,
            "total_empty_spaces": 0,
            "by_camera": {"left": 0, "right": 0},
            "by_parent_shelf": {},
            "by_section": {"SOL": 0, "ORTA": 0, "SAĞ": 0},
            "stations": [],
        }
        state = _SessionState(
            safe_requested, session_id, directory, summary, set()
        )
        _atomic_write_json(directory / "session_summary.json", summary)
        separator = "=" * 50
        threshold_text = ", ".join(
            f"{key}={value}" for key, value in sorted(self.thresholds.items())
        ) or "reported in per-station JSON"
        _append_text(
            directory / "session_log.txt",
            f"{separator}\n"
            "SESSION START\n"
            f"{separator}\n"
            f"Timestamp: {_display_timestamp(started)}\n"
            f"Session ID: {session_id}\n"
            f"Mode: {self.mode}\n"
            f"Models: {self.shelf_model}, {self.empty_model}\n"
            f"Production thresholds: {threshold_text}\n"
            + "=" * 50 + "\n\n",
        )
        self._sessions[safe_requested] = state
        return state

    def _state(self, requested_session_id):
        safe_requested = sanitize_identifier(requested_session_id, "session_id")
        return self._sessions.get(safe_requested) or self._new_state(safe_requested)

    @staticmethod
    def _gaps(side_result):
        gaps = []
        unlocalized = False
        for shelf in side_result.get("shelves", []) or []:
            if not isinstance(shelf, dict):
                continue
            shelf_id = shelf.get("shelf_id")
            if shelf_id == "UNKNOWN_SHELF" or shelf.get("status") == "UNLOCALIZED":
                unlocalized = True
                continue
            for index, detection in enumerate(shelf.get("detections", []) or []):
                if not isinstance(detection, dict):
                    continue
                parent = detection.get("parent_shelf_id") or shelf.get("parent_shelf_id")
                level = detection.get("shelf_level_id") or shelf.get("shelf_level_id")
                if not parent or not level or parent == "UNKNOWN_SHELF" or level == "UNKNOWN_SHELF":
                    unlocalized = True
                    continue
                gaps.append({
                    "parent_shelf_id": parent,
                    "shelf_level_id": level,
                    "level_number": shelf.get("level_number"),
                    "section": detection.get("section"),
                    "confidence": detection.get("confidence"),
                    "global_bbox_xyxy": detection.get("global_bbox_xyxy"),
                    "mask_overlap_ratio": detection.get("mask_overlap_ratio"),
                    "_index": index,
                })
        gaps.sort(key=lambda gap: (
            str(gap["parent_shelf_id"]),
            gap["level_number"] if isinstance(gap["level_number"], int) else 10**9,
            _SECTION_ORDER.get(gap["section"], 99),
            -(float(gap["confidence"]) if gap["confidence"] is not None else -1.0),
            gap["_index"],
        ))
        return gaps, unlocalized

    @staticmethod
    def _format_value(value):
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def _station_log(self, timestamp, station_id, dual_result):
        blocks = []
        gap_counts = {}
        all_gaps = {}
        for camera in ("left", "right"):
            side = dual_result.get(camera, {})
            frame = sanitize_identifier(side.get("frame_id", "missing_frame"), "frame_id")
            gaps, unlocalized = self._gaps(side)
            gap_counts[camera] = len(gaps)
            all_gaps[camera] = gaps
            if not gaps:
                blocks.append(
                    f"[{_display_timestamp(timestamp)}]\n"
                    f"Station: {station_id}\n"
                    f"Camera: {camera.upper()}\n"
                    f"Frame: {frame}\n"
                    "Result: NO_EMPTY_SPACE\n\n"
                )
            else:
                for gap in gaps:
                    bbox = gap["global_bbox_xyxy"]
                    bbox_text = json.dumps(bbox, ensure_ascii=False) if bbox is not None else "null"
                    block = (
                        f"[{_display_timestamp(timestamp)}]\n"
                        f"Station: {station_id}\n"
                        f"Camera: {camera.upper()}\n"
                        f"Frame: {frame}\n"
                        f"Parent shelf: {gap['parent_shelf_id']}\n"
                        f"Shelf level: {gap['shelf_level_id']}\n"
                        f"Level number: {self._format_value(gap['level_number'])}\n"
                        f"Gap position: {self._format_value(gap['section'])}\n"
                        f"Confidence: {self._format_value(gap['confidence'])}\n"
                        f"BBox: {bbox_text}\n"
                    )
                    if gap["mask_overlap_ratio"] is not None:
                        block += (
                            "Mask overlap: "
                            f"{self._format_value(gap['mask_overlap_ratio'])}\n"
                        )
                    blocks.append(block + "\n")
            if unlocalized:
                blocks.append(
                    f"[{_display_timestamp(timestamp)}]\n"
                    f"Station: {station_id}\n"
                    f"Camera: {camera.upper()}\n"
                    "Warning: Unlocalized shelf detected; excluded from production gap logging.\n\n"
                )
        blocks.append(
            f"--- Station {station_id} Summary ---\n"
            f"LEFT gaps: {gap_counts['left']}\n"
            f"RIGHT gaps: {gap_counts['right']}\n"
            f"TOTAL: {gap_counts['left'] + gap_counts['right']}\n\n"
        )
        return "".join(blocks), gap_counts, all_gaps

    def record_station(self, session_id, station_id, *, left_raw, right_raw,
                       left_annotated, right_annotated, dual_result):
        """Save one accepted dual response. Repeated station IDs are idempotent."""
        with self._lock:
            state = self._state(session_id)
            safe_station = sanitize_identifier(station_id, "station_id")
            if state.summary["status"] == "completed":
                raise ValueError(f"Session {state.session_id} is already completed.")
            if safe_station in state.processed_stations:
                return {
                    "saved": True,
                    "duplicate_ignored": True,
                    "session_id": state.session_id,
                    "station_id": safe_station,
                    "session_directory": str(state.directory),
                }

            station_directory = state.directory / safe_station
            station_directory.mkdir(exist_ok=False)
            _atomic_write_bytes(station_directory / "left_raw.jpg", bytes(left_raw))
            _atomic_write_bytes(station_directory / "right_raw.jpg", bytes(right_raw))
            _atomic_write_bytes(
                station_directory / "left_annotated.jpg", bytes(left_annotated)
            )
            _atomic_write_bytes(
                station_directory / "right_annotated.jpg", bytes(right_annotated)
            )
            archived = deepcopy(dual_result)
            _atomic_write_json(station_directory / "dual_result.json", archived)
            _atomic_write_json(station_directory / "left_result.json", archived["left"])
            _atomic_write_json(station_directory / "right_result.json", archived["right"])

            recorded_at = _now()
            log_text, gap_counts, all_gaps = self._station_log(
                recorded_at, safe_station, archived
            )
            _append_text(state.directory / "session_log.txt", log_text)

            total = gap_counts["left"] + gap_counts["right"]
            state.processed_stations.add(safe_station)
            summary = state.summary
            summary["stations_processed"] += 1
            summary["station_ids"].append(safe_station)
            summary["dual_scans"] += 1
            summary["camera_results"] += 2
            summary["total_empty_spaces"] += total
            summary["by_camera"]["left"] += gap_counts["left"]
            summary["by_camera"]["right"] += gap_counts["right"]
            station_sections = {"SOL": 0, "ORTA": 0, "SAĞ": 0}
            station_parents = {}
            for gaps in all_gaps.values():
                for gap in gaps:
                    parent = gap["parent_shelf_id"]
                    section = gap["section"]
                    summary["by_parent_shelf"][parent] = (
                        summary["by_parent_shelf"].get(parent, 0) + 1
                    )
                    station_parents[parent] = station_parents.get(parent, 0) + 1
                    if section in summary["by_section"]:
                        summary["by_section"][section] += 1
                        station_sections[section] += 1
            summary["stations"].append({
                "station_id": safe_station,
                "recorded_at": _timestamp(recorded_at),
                "left_frame_id": archived["left"].get("frame_id"),
                "right_frame_id": archived["right"].get("frame_id"),
                "left_gaps": gap_counts["left"],
                "right_gaps": gap_counts["right"],
                "total_gaps": total,
                "by_parent_shelf": station_parents,
                "by_section": station_sections,
            })
            _atomic_write_json(state.directory / "session_summary.json", summary)
            return {
                "saved": True,
                "duplicate_ignored": False,
                "session_id": state.session_id,
                "station_id": safe_station,
                "session_directory": str(state.directory),
            }

    def complete_session(self, session_id):
        with self._lock:
            state = self._state(session_id)
            if state.summary["status"] == "completed":
                return {
                    "saved": True, "already_completed": True,
                    "session_id": state.session_id,
                    "session_directory": str(state.directory),
                }
            completed = _now()
            summary = state.summary
            summary["status"] = "completed"
            summary["completed_at"] = _timestamp(completed)
            _atomic_write_json(state.directory / "session_summary.json", summary)
            separator = "=" * 50
            shelf_lines = "".join(
                f"{shelf}: {count}\n"
                for shelf, count in sorted(summary["by_parent_shelf"].items())
            ) or "(none)\n"
            _append_text(
                state.directory / "session_log.txt",
                f"{separator}\n"
                "SESSION COMPLETE\n"
                f"{separator}\n"
                f"Completed at: {_display_timestamp(completed)}\n"
                f"Stations processed: {summary['stations_processed']}\n"
                f"Camera scans: {summary['camera_results']}\n"
                f"Total gaps: {summary['total_empty_spaces']}\n\n"
                f"SOL: {summary['by_section']['SOL']}\n"
                f"ORTA: {summary['by_section']['ORTA']}\n"
                f"SAĞ: {summary['by_section']['SAĞ']}\n\n"
                "By shelf:\n" + shelf_lines + "\n"
                f"Session directory: {state.directory}\n"
                f"{separator}\n",
            )
            return {
                "saved": True, "already_completed": False,
                "session_id": state.session_id,
                "session_directory": str(state.directory),
            }
