from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from .multi_view_geometry import triangulate_mask_pair
from .persistent_map import ShelfMapStore
from .pose_geometry import _global_assignment
from .runtime_io import OfflineFrameSource
from .shelf_mapping import ShelfMapBuilder, ShelfObservation


def _segment(model, frame, args):
    from run_shelf_gap_cascade import extract_shelves
    kwargs={"source":frame.image,"conf":args.shelf_conf,"imgsz":args.shelf_imgsz,"verbose":False}
    if args.device:kwargs["device"]=args.device
    return extract_shelves(model.predict(**kwargs)[0],frame.image.shape)


def run_mapping(args, update=False):
    from ultralytics import YOLO
    from src.cascade import validate_model_contract
    config=yaml.safe_load(args.mapping_config.read_text(encoding="utf-8"))
    store=ShelfMapStore(args.map_dir,args.store_id)
    if update:
        shelf_map=store.load()
    elif args.resume:
        shelf_map=store.load() if store.path.exists() else store.create()
    else:
        shelf_map=store.create(overwrite=args.overwrite_map)
    builder=ShelfMapBuilder(config)
    if args.resume:builder.resume(store.load_candidates())
    model=YOLO(str(args.shelf_model));validate_model_contract(model,"segment","shelves","Raf modeli")
    previous=None;frame_count=0
    for frame in OfflineFrameSource(args.source):
        pose=frame.metadata["pose"]
        masks=_segment(model,frame,args);frame_count+=1
        if previous is not None and pose["quality"]>=config["min_pose_quality"]:
            old_frame,old_masks=previous
            proposals={}
            scores=np.zeros((len(old_masks),len(masks)),float)
            for oi,old in enumerate(old_masks):
                for ni,new in enumerate(masks):
                    geometry=triangulate_mask_pair(old["mask"],old_frame.metadata,new["mask"],frame.metadata)
                    if geometry:
                        proposals[oi,ni]=geometry
                        scores[oi,ni]=1.0/(1.0+geometry["reprojection_error_px"])
            # Exact global one-to-one; mask ordering never supplies identity.
            for oi,ni in enumerate(_global_assignment(scores)):
                if ni<0 or (oi,ni) not in proposals:continue
                geometry=proposals[oi,ni]
                builder.add(ShelfObservation(frame.metadata["frame_id"],frame.metadata.get("session_id","mapping"),
                    geometry["center_map"],geometry["corners_map"],geometry["front_normal_map"],
                    frame.metadata["camera_position_world"],float(pose["quality"]),bool(pose["scale_initialized"])))
        previous=(frame,masks)
        store.save_candidates_atomic(builder.serialize_candidates())
    created=builder.confirm_into(shelf_map,args.session_id,allow_existing_update=update)
    store.save_atomic(shelf_map);store.save_candidates_atomic(builder.serialize_candidates())
    print(f"Sistem modu: {'mapping_update' if update else 'mapping'}")
    print(f"İşlenen kare: {frame_count}\nOnaylanan kalıcı raf: {len(created)}")
    for item in created:print(item["shelf_id"])
    print(f"Harita başarıyla kaydedildi: {store.path.resolve()}")
    return {"frame_count":frame_count,"created":created,"map_path":str(store.path),
            "candidate_track_count":len(builder.tracks),"pose_rejected_count":builder.rejected_pose_count,
            "duplicate_merged_count":builder.duplicate_merged_count}


def persistent_map_as_geometry(data):
    return {"schema_version":3,"coordinate_system":"map","shelves":[{
        "shelf_id":s["shelf_id"],"corners_world":s["corners_map"],"center_world":s["center_map"],
        "front_normal_world":s["front_normal_map"],"active":s.get("status")=="confirmed"}
        for s in data["shelves"]]}
