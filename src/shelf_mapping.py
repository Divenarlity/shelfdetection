from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import numpy as np

from .persistent_map import PersistentIdAllocator


@dataclass
class ShelfObservation:
    frame_id: str
    session_id: str
    center_map: list[float]
    corners_map: list[list[float]]
    front_normal_map: list[float]
    camera_position_map: list[float]
    pose_quality: float
    scale_initialized: bool


@dataclass
class ShelfTrack:
    observations: list[ShelfObservation]=field(default_factory=list)
    status: str="candidate"

    @property
    def center(self):
        return np.median([o.center_map for o in self.observations],axis=0)

    @property
    def normal(self):
        n=np.mean([o.front_normal_map for o in self.observations],axis=0)
        return n/max(np.linalg.norm(n),1e-9)

    @property
    def baseline(self):
        cameras=np.asarray([o.camera_position_map for o in self.observations])
        return max((np.linalg.norm(a-b) for a in cameras for b in cameras),default=0.0)


class ShelfMapBuilder:
    def __init__(self, config):
        self.config=config
        self.tracks:list[ShelfTrack]=[]
        self.rejected_pose_count=0
        self.duplicate_merged_count=0

    def add(self, observation):
        if observation.pose_quality < self.config["min_pose_quality"] or not observation.scale_initialized:
            self.rejected_pose_count+=1
            return None
        center=np.asarray(observation.center_map)
        candidates=[]
        for track in self.tracks:
            distance=float(np.linalg.norm(center-track.center))
            alignment=float(np.dot(np.asarray(observation.front_normal_map),track.normal))
            if distance<=self.config["association_distance_m"] and alignment>=self.config["min_normal_dot"]:
                candidates.append((distance,track))
        track=min(candidates,key=lambda x:x[0])[1] if candidates else ShelfTrack()
        if not candidates:self.tracks.append(track)
        track.observations.append(observation)
        return track

    def merge_duplicates(self):
        kept=[]
        for track in self.tracks:
            match=next((x for x in kept if np.linalg.norm(track.center-x.center)<=self.config["duplicate_distance_m"]
                        and np.dot(track.normal,x.normal)>=self.config["min_normal_dot"]),None)
            if match:
                match.observations.extend(track.observations);self.duplicate_merged_count+=1
            else:kept.append(track)
        self.tracks=kept

    def confirm_into(self, shelf_map, session_id, allow_existing_update=False):
        self.merge_duplicates()
        allocator=PersistentIdAllocator(shelf_map["id_prefix"],shelf_map["next_shelf_index"])
        created=[]
        for track in self.tracks:
            unique_frames=len({o.frame_id for o in track.observations})
            if unique_frames<self.config["min_observations"] or track.baseline<self.config["min_baseline_m"]:
                continue
            corners=np.median([o.corners_map for o in track.observations],axis=0)
            center=track.center;normal=track.normal
            width=float((np.linalg.norm(corners[3]-corners[0])+np.linalg.norm(corners[2]-corners[1]))/2)
            height=float((np.linalg.norm(corners[1]-corners[0])+np.linalg.norm(corners[2]-corners[3]))/2)
            confidence=min(1.0,unique_frames/self.config["target_observations"])
            existing=None
            if allow_existing_update:
                existing=next((s for s in shelf_map["shelves"]
                    if np.linalg.norm(np.asarray(s["center_map"])-center)<=self.config["association_distance_m"]
                    and np.dot(np.asarray(s["front_normal_map"]),normal)>=self.config["min_normal_dot"]),None)
            if existing:
                existing.update(center_map=center.tolist(),corners_map=corners.tolist(),
                    front_normal_map=normal.tolist(),width=width,height=height,
                    observation_count=int(existing.get("observation_count",0))+len(track.observations),
                    mapping_confidence=confidence,last_seen_session=session_id)
                track.status="confirmed";continue
            sid=allocator.allocate()
            record={"shelf_id":sid,"status":"confirmed","center_map":center.tolist(),
                    "corners_map":corners.tolist(),"front_normal_map":normal.tolist(),
                    "width":width,"height":height,"observation_count":len(track.observations),
                    "mapping_confidence":confidence,"first_seen_session":session_id,
                    "last_seen_session":session_id}
            shelf_map["shelves"].append(record);created.append(record);track.status="confirmed"
        shelf_map["next_shelf_index"]=allocator.next_index
        return created

    def serialize_candidates(self):
        return [{"status":t.status,"observations":[asdict(o) for o in t.observations]} for t in self.tracks if t.status!="confirmed"]

    def resume(self, payload):
        self.tracks=[ShelfTrack([ShelfObservation(**o) for o in item["observations"]],item.get("status","candidate"))
                     for item in payload]
