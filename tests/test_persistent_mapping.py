import hashlib
import json
from pathlib import Path

import pytest

from src.persistent_map import ShelfMapStore
from src.persistent_runner import persistent_map_as_geometry
from src.shelf_mapping import ShelfMapBuilder, ShelfObservation


CFG={"min_pose_quality":.7,"min_observations":5,"target_observations":20,
     "min_baseline_m":.3,"association_distance_m":.6,
     "duplicate_distance_m":.25,"min_normal_dot":.9}


def obs(frame,center=(0,1,3),camera=(0,1,0),quality=.95,scale=True,session="pass1"):
    x,y,z=center
    return ShelfObservation(str(frame),session,[x,y,z],
        [[x-1,y-1,z],[x-1,y+1,z],[x+1,y+1,z],[x+1,y-1,z]],
        [0,0,-1],list(camera),quality,scale)


def test_same_shelf_30_frames_gets_one_persistent_id():
    builder=ShelfMapBuilder(CFG)
    for i in range(30):builder.add(obs(i,camera=(i*.05,1,0)))
    data=ShelfMapStore("unused","MARKET").create()
    created=builder.confirm_into(data,"pass1")
    assert len(created)==1 and created[0]["shelf_id"]=="SHELF-0001"
    assert created[0]["observation_count"]==30


def test_two_physical_shelves_even_same_parent_concept_get_two_ids():
    builder=ShelfMapBuilder(CFG)
    for i in range(10):
        builder.add(obs(f"a{i}",center=(-1,1,3),camera=(i*.1,1,0)))
        builder.add(obs(f"b{i}",center=(1,1,3),camera=(i*.1,1,0)))
    data=ShelfMapStore("unused","MARKET").create()
    assert [x["shelf_id"] for x in builder.confirm_into(data,"pass1")]==["SHELF-0001","SHELF-0002"]


def test_disappear_and_reappear_does_not_create_second_track():
    builder=ShelfMapBuilder(CFG)
    for i in list(range(5))+list(range(20,25)):builder.add(obs(i,camera=(i*.05,1,0)))
    assert len(builder.tracks)==1


def test_single_view_scale_and_low_pose_do_not_confirm():
    builder=ShelfMapBuilder(CFG)
    builder.add(obs(1))
    builder.add(obs(2,quality=.2))
    builder.add(obs(3,scale=False))
    data=ShelfMapStore("unused","MARKET").create()
    assert builder.confirm_into(data,"pass1")==[]
    assert builder.rejected_pose_count==2


def test_duplicate_candidate_tracks_merge():
    builder=ShelfMapBuilder(CFG)
    for i in range(5):builder.add(obs(i,center=(0,1,3),camera=(i*.1,1,0)))
    # Force a second candidate close enough for duplicate merge but outside association gate.
    from src.shelf_mapping import ShelfTrack
    second=ShelfTrack([obs(f"x{i}",center=(.1,1,3),camera=(i*.1,1,0)) for i in range(5)])
    builder.tracks.append(second);builder.merge_duplicates()
    assert len(builder.tracks)==1 and builder.duplicate_merged_count==1


def test_map_atomic_save_reload_counter_and_no_overwrite(tmp_path):
    store=ShelfMapStore(tmp_path/"maps"/"MARKET","MARKET")
    data=store.create();data["next_shelf_index"]=7;store.save_atomic(data)
    assert store.load()["next_shelf_index"]==7
    with pytest.raises(FileExistsError):store.create()


def test_resume_candidates_preserves_track(tmp_path):
    store=ShelfMapStore(tmp_path,"MARKET");builder=ShelfMapBuilder(CFG)
    for i in range(3):builder.add(obs(i,camera=(i*.1,1,0)))
    store.save_candidates_atomic(builder.serialize_candidates())
    resumed=ShelfMapBuilder(CFG);resumed.resume(store.load_candidates())
    assert len(resumed.tracks)==1 and len(resumed.tracks[0].observations)==3


def test_persistent_ids_convert_to_localization_without_new_id():
    data={"shelves":[{"shelf_id":"SHELF-0042","status":"confirmed","center_map":[0,1,3],
      "corners_map":[[-1,0,3],[-1,2,3],[1,2,3],[1,0,3]],"front_normal_map":[0,0,-1]}]}
    converted=persistent_map_as_geometry(data)
    assert [x["shelf_id"] for x in converted["shelves"]]==["SHELF-0042"]
