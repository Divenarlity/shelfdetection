import json
from pathlib import Path

import numpy as np

from run_shelf_gap_cascade import extract_shelves, predict_empty_rois, put_text
from src.cascade import (
    assign_shelf_ids, context_tiles, deduplicate, local_to_global, mask_bbox, mask_box_relation,
    padded_roi, section_for_box, validate_model_contract,
)


class FakeModel:
    def __init__(self, task, names):
        self.task, self.names, self.calls = task, names, []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return ["result"] * len(kwargs["source"])


def test_segmenter_without_masks_is_explicit_error():
    class Result: masks=None
    try:
        extract_shelves(Result(),(20,20,3))
        assert False
    except RuntimeError as exc:
        assert "fallback yasaktır" in str(exc)


def test_model_task_and_class_contract():
    assert validate_model_contract(FakeModel("segment",{0:"shelves"}),"segment","shelves","raf")[0]=="shelves"
    try:
        validate_model_contract(FakeModel("detect",{0:"other"}),"detect","empty_shelf","boşluk")
        assert False
    except RuntimeError:
        pass


def test_mask_to_roi():
    mask=np.zeros((20,30),bool); mask[4:10,6:16]=1
    assert mask_bbox(mask)==[6,4,16,10]


def test_padding_clipped_to_image():
    assert padded_roi([0,0,10,10],(20,30,3),.5)==[0,0,15,15]


def test_local_to_global_coordinates():
    assert local_to_global([1,2,5,6],[10,20,30,40],(100,100,3))==[11.0,22.0,15.0,26.0]


def test_inside_mask_accepted():
    mask=np.zeros((20,20),bool); mask[5:15,5:15]=1
    accepted,overlap,center=mask_box_relation(mask,[6,6,12,12],.3)
    assert accepted and center and overlap==1


def test_outside_mask_rejected():
    mask=np.zeros((20,20),bool); mask[5:15,5:15]=1
    accepted,overlap,center=mask_box_relation(mask,[0,0,4,4],.3)
    assert not accepted and not center and overlap==0


def test_duplicate_cleanup_and_single_shelf_assignment():
    base={"confidence":.8,"mask_overlap_ratio":.8,"center_inside_mask":True}
    items=[
        {**base,"global_bbox_xyxy":[1,1,10,10],"assigned_shelf_id":"CAM-02-RAF-02"},
        {**base,"global_bbox_xyxy":[1,1,10,10],"assigned_shelf_id":"CAM-02-RAF-01","mask_overlap_ratio":.9},
    ]
    kept,removed=deduplicate(items,.5)
    assert removed==1 and len(kept)==1 and kept[0]["assigned_shelf_id"]=="CAM-02-RAF-01"


def test_tile_edge_clipped_duplicate_keeps_complete_box():
    common={"assigned_shelf_id":"CAM-02-RAF-04","mask_overlap_ratio":.95,
            "center_inside_mask":True}
    complete={**common,"confidence":.78,"global_bbox_xyxy":[195,247,301,315],
              "touches_tile_edge":False}
    clipped={**common,"confidence":.86,"global_bbox_xyxy":[256,247,304,315],
             "touches_tile_edge":True}
    kept,removed=deduplicate([complete,clipped],.5)
    assert removed==1 and kept==[complete]


def test_sections_use_shelf_width():
    shelf=[100,0,400,30]
    assert section_for_box([110,0,120,10],shelf)=="SOL"
    assert section_for_box([245,0,255,10],shelf)=="ORTA"
    assert section_for_box([380,0,390,10],shelf)=="SAĞ"


def test_camera_based_shelf_ids(tmp_path):
    masks=[]
    for y in (30,10):
        mask=np.zeros((50,50),bool); mask[y:y+5,5:20]=1
        masks.append({"mask":mask,"bbox":mask_bbox(mask),"centroid":[12,y+2],"segmentation_confidence":.9})
    result=assign_shelf_ids(masks,"CAM-02",tmp_path/"CAM-02.json",(50,50,3),True)
    assert [x["shelf_id"] for x in result]==["CAM-02-RAF-01","CAM-02-RAF-02"]
    saved=json.loads((tmp_path/"CAM-02.json").read_text(encoding="utf-8"))
    assert saved["camera_id"]=="CAM-02"


def test_no_shelf_means_empty_model_not_called():
    model=FakeModel("detect",{0:"empty_shelf"})
    assert predict_empty_rois(model,[],.1,640)==[]
    assert model.calls==[]


def test_detector_receives_only_roi_crops():
    model=FakeModel("detect",{0:"empty_shelf"})
    full=np.zeros((640,640,3),np.uint8)
    crops=[full[10:394,20:404].copy(),full[200:584,100:484].copy()]
    predict_empty_rois(model,crops,.1,640)
    sent=model.calls[0]["source"]
    assert [x.shape[:2] for x in sent]==[(384,384),(384,384)]
    assert all(x.shape!=full.shape for x in sent)


def test_heterogeneous_batch_is_rejected():
    model=FakeModel("detect",{0:"empty_shelf"})
    try:
        predict_empty_rois(model,[np.zeros((20,30,3),np.uint8),np.zeros((30,20,3),np.uint8)],.1,640)
        assert False
    except ValueError:
        assert model.calls==[]


def test_context_tiles_stay_inside_and_cover_thin_shelf():
    bbox=[0,248,640,326]
    tiles=context_tiles(bbox,(640,640,3),384,.25)
    assert all(0<=x1<x2<=640 and 0<=y1<y2<=640 for x1,y1,x2,y2 in tiles)
    coverage=np.zeros((640,640),bool)
    for x1,y1,x2,y2 in tiles: coverage[y1:y2,x1:x2]=1
    assert coverage[bbox[1]:bbox[3],bbox[0]:bbox[2]].all()


def test_tiles_and_annotations_do_not_modify_source():
    source=np.random.default_rng(3).integers(0,256,(640,640,3),dtype=np.uint8)
    before=source.copy()
    tile_box=context_tiles([0,248,640,326],source.shape,384,.25)[0]
    x1,y1,x2,y2=tile_box
    tile=source[y1:y2,x1:x2].copy()
    put_text(tile,[((5,5),"Boşluk",(220,0,0))])
    assert np.array_equal(source,before)
    assert np.array_equal(tile,before[y1:y2,x1:x2])
