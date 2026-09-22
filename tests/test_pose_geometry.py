import json
import numpy as np
import pytest
from src.pose_geometry import PoseConfig, assign_pose_ids, generate_candidates, load_frame_metadata, project_world_points


def metadata():
    return {"frame_id":"1","image_path":"x.png","resolution":[100,100],
            "camera_position_world":[0,0,0],
            "camera_rotation_xyzw":[0,0,0,1],
            "intrinsics":{"fx":50,"fy":50,"cx":50,"cy":50}}


def shelf(sid="A-L-01", z=2, normal=(0,0,-1)):
    return {"shelf_id":sid,"corners_world":{"bottom_left":[-1,-1,z],"top_left":[-1,1,z],
            "top_right":[1,1,z],"bottom_right":[1,-1,z]},"center_world":[0,0,z],
            "front_normal_world":normal}


def test_center_projects_to_center_pixel_and_axis_is_top_left():
    pixels, valid=project_world_points([[0,0,2],[0,1,2]],metadata())
    assert valid.all() and np.allclose(pixels[0],[50,50])
    assert pixels[1,1] < pixels[0,1]


def test_behind_camera_and_backface_are_rejected():
    assert generate_candidates({"shelves":[shelf(z=-2)]},metadata()) == []
    assert generate_candidates({"shelves":[shelf(normal=(0,0,1))]},metadata()) == []
    far_right=shelf()
    for point in far_right["corners_world"].values(): point[0] += 100
    far_right["center_world"][0] += 100
    assert generate_candidates({"shelves":[far_right]},metadata()) == []


def test_assignment_is_one_to_one_and_unknown_on_low_score():
    m=np.zeros((100,100),bool); m[25:75,25:75]=1
    shelves=[{"mask":m.copy(),"segmentation_confidence":.9},
             {"mask":np.zeros_like(m),"segmentation_confidence":.9}]
    candidates=generate_candidates({"shelves":[shelf()]},metadata(),PoseConfig(min_visible_area=1))
    result=assign_pose_ids(shelves,candidates,m.shape,PoseConfig(min_score=.2))
    assert result[0]["shelf_id"]=="A-L-01"
    assert result[1]["shelf_id"]=="UNKNOWN_SHELF"


def test_same_pixels_different_pose_selects_different_static_id():
    m=np.ones((100,100),bool)
    s1={"mask":m,"segmentation_confidence":1}
    cfg=PoseConfig(min_visible_area=1,min_score=.01)
    a=generate_candidates({"shelves":[shelf("A-L-01")]},metadata(),cfg)
    b=generate_candidates({"shelves":[shelf("B-R-02")]},metadata(),cfg)
    assert assign_pose_ids([s1.copy()],a,m.shape,cfg)[0]["shelf_id"]=="A-L-01"
    assert assign_pose_ids([s1.copy()],b,m.shape,cfg)[0]["shelf_id"]=="B-R-02"


def test_two_regions_same_parent_are_distinct_and_order_independent():
    left=np.zeros((100,100),bool); left[20:80,5:45]=1
    right=np.zeros((100,100),bool); right[20:80,55:95]=1
    shelves=[{"mask":left,"centroid":[25,50],"segmentation_confidence":1},
             {"mask":right,"centroid":[75,50],"segmentation_confidence":1}]
    candidates=[
        {"shelf_id":"KNOWN-LEFT","parent_surface_id":"PHOTO","polygon":np.array([[5,20],[5,80],[45,80],[45,20]]),
         "distance":2,"front_alignment":1,"visible_area":2400},
        {"shelf_id":"KNOWN-RIGHT","parent_surface_id":"PHOTO","polygon":np.array([[55,20],[55,80],[95,80],[95,20]]),
         "distance":2,"front_alignment":1,"visible_area":2400},
    ]
    cfg=PoseConfig(min_score=.25)
    a=assign_pose_ids([dict(x) for x in shelves],candidates,(100,100),cfg)
    b=assign_pose_ids([dict(x) for x in reversed(shelves)],candidates,(100,100),cfg)
    assert [x["shelf_id"] for x in a]==["KNOWN-LEFT","KNOWN-RIGHT"]
    assert [x["shelf_id"] for x in b]==["KNOWN-RIGHT","KNOWN-LEFT"]
    assert len({x["shelf_id"] for x in a})==2


def test_two_masks_one_known_region_leaves_one_unknown():
    mask=np.ones((50,50),bool)
    shelves=[{"mask":mask,"centroid":[25,25],"segmentation_confidence":1} for _ in range(2)]
    candidate={"shelf_id":"ONLY","polygon":np.array([[0,0],[0,49],[49,49],[49,0]]),
               "distance":1,"front_alignment":1,"visible_area":2401}
    ids=[x["shelf_id"] for x in assign_pose_ids(shelves,[candidate],mask.shape,PoseConfig(min_score=.1))]
    assert ids.count("ONLY")==1 and ids.count("UNKNOWN_SHELF")==1


def test_only_schema_v3_is_accepted_and_nested_identity_is_rejected(tmp_path):
    valid={"schema_version":3,"frame_id":"frame_1","image_path":"frame_1.png",
           "image":{"width":100,"height":80},"camera":{"position_world":[0,0,0],
           "rotation_xyzw":[0,0,0,1],"intrinsics":{"fx":90,"fy":90,"cx":50,"cy":40}},
           "pose":{"quality":1,"relocalized":True,"scale_initialized":True}}
    p=tmp_path/"frame.json"; p.write_text(json.dumps(valid),encoding="utf-8")
    assert load_frame_metadata(p)["resolution"]==[100,80]
    valid["camera"]["shelf_id"]="LEAK"; p.write_text(json.dumps(valid),encoding="utf-8")
    with pytest.raises(ValueError,match="identity/ground-truth"): load_frame_metadata(p)
    del valid["camera"]["shelf_id"]
    valid["expected_shelf_ids"]=["A-L-01"]
    p.write_text(json.dumps(valid),encoding="utf-8")
    with pytest.raises(ValueError,match="identity/ground-truth"): load_frame_metadata(p)
    del valid["expected_shelf_ids"]
    valid["schema_version"]=2
    p.write_text(json.dumps(valid),encoding="utf-8")
    with pytest.raises(ValueError,match="schema_version 3"): load_frame_metadata(p)


def test_schema_v3_requires_pose_state(tmp_path):
    data={"schema_version":3,"frame_id":"frame_1","image":{"width":100,"height":80},
          "camera":{"position_map":[0,0,0],"rotation_xyzw":[0,0,0,1],
                    "intrinsics":{"fx":90,"fy":90,"cx":50,"cy":40}}}
    path=tmp_path/"frame_1.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="requires.*pose"): load_frame_metadata(path)


def test_intrinsics_projection_uses_unity_forward_and_top_left_y():
    meta={"resolution":[100,80],"camera_position_world":[0,0,0],"camera_rotation_xyzw":[0,0,0,1],
          "intrinsics":{"fx":100,"fy":100,"cx":50,"cy":40}}
    pixels,valid=project_world_points([[0,0,2],[0,1,2],[0,0,-2]],meta)
    assert valid.tolist()==[True,True,False]
    assert np.allclose(pixels[0],[50,40]) and pixels[1,1]<40


def test_failed_relocalization_or_low_pose_quality_has_no_candidates():
    meta=metadata();meta["pose"]={"quality":.95,"relocalized":False,"scale_initialized":True}
    assert generate_candidates({"shelves":[shelf()]},meta,PoseConfig(min_visible_area=1))==[]
    meta["pose"]={"quality":.2,"relocalized":True,"scale_initialized":True}
    assert generate_candidates({"shelves":[shelf()]},meta,PoseConfig(min_visible_area=1))==[]
