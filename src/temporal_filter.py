from __future__ import annotations

from dataclasses import dataclass


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0


@dataclass
class Track:
    box: list
    first_frame: int
    last_frame: int
    last_reported: int = -10**9


class TemporalFilter:
    def __init__(self, min_frames=3, cooldown_frames=90, iou=0.3):
        self.min_frames, self.cooldown_frames, self.iou = min_frames, cooldown_frames, iou
        self.tracks = []

    def update(self, detections, frame_index):
        confirmed = []
        for det in detections:
            matches = [t for t in self.tracks if box_iou(t.box, det["box"]) >= self.iou and frame_index-t.last_frame <= 1]
            track = max(matches, key=lambda t: box_iou(t.box, det["box"])) if matches else Track(det["box"], frame_index, frame_index)
            if not matches:
                self.tracks.append(track)
            track.box, track.last_frame = det["box"], frame_index
            if frame_index-track.first_frame+1 >= self.min_frames and frame_index-track.last_reported >= self.cooldown_frames:
                track.last_reported = frame_index
                confirmed.append(det)
        self.tracks = [t for t in self.tracks if frame_index-t.last_frame <= self.min_frames]
        return confirmed

