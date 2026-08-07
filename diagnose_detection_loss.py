"""RAF-04 tespit kaybını karşılaştırmalı olarak ölçen tanı aracı."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.cascade import box_iou


TARGET = [196.0, 251.0, 302.0, 313.0]
ROIS = {
    "CAM-02-RAF-01":[94,94,600,171], "CAM-02-RAF-02":[0,166,640,253],
    "CAM-02-RAF-03":[370,245,640,327], "CAM-02-RAF-04":[0,248,640,326],
    "CAM-02-RAF-05":[0,316,640,389], "CAM-02-RAF-06":[0,382,640,453],
    "CAM-02-RAF-07":[0,446,640,516], "CAM-02-RAF-08":[0,509,640,580],
}


def predictions(result, offset=(0,0)):
    if result.boxes is None:
        return []
    found=[]
    for box,conf,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()):
        global_box=[box[0]+offset[0],box[1]+offset[1],box[2]+offset[0],box[3]+offset[1]]
        found.append({"confidence":float(conf),"class_id":int(cls),"local_bbox":box,
                      "global_bbox":global_box,"target_iou":box_iou(global_box,TARGET)})
    return found


def square_tiles(image, roi, size, overlap=.25):
    h,w=image.shape[:2]; x1,y1,x2,y2=roi
    step=max(1,int(size*(1-overlap)))
    starts=list(range(x1,max(x1+1,x2-size+1),step))
    last=max(0,min(w-size,x2-size))
    if not starts or starts[-1]!=last: starts.append(last)
    cy=(y1+y2)/2; ty1=max(0,min(h-size,int(round(cy-size/2)))); ty2=min(h,ty1+size)
    return [([sx,ty1,min(w,sx+size),ty2],image[ty1:ty2,sx:min(w,sx+size)].copy()) for sx in starts]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True); p.add_argument("--model",type=Path,default=Path("empty_shelf_yolo11m_best.pt"))
    p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    image=cv2.imread(str(args.source))
    if image is None: raise SystemExit("Görüntü okunamadı")
    model=YOLO(str(args.model))
    out={"target_bbox":TARGET,"experiments":[]}
    def record(name,inputs,origins,conf,rect,batch):
        results=model.predict(source=inputs,conf=conf,imgsz=640,rect=rect,batch=batch,verbose=False)
        all_preds=[]
        for result,origin in zip(results,origins):
            all_preds.extend(predictions(result,origin))
        nearest=max(all_preds,key=lambda x:x["target_iou"],default=None)
        out["experiments"].append({"strategy":name,"conf":conf,"rect":rect,"batch":batch,
            "input_shapes":[list(x.shape[:2]) for x in inputs],"input_count":len(inputs),
            "raw_count":len(all_preds),"predictions":all_preds,"nearest_target":nearest})
    for conf in (.10,.01):
        record("full_image_diagnostic",[image.copy()],[(0,0)],conf,True,1)
    crops=[]; origins=[]
    for roi in ROIS.values():
        x1,y1,x2,y2=roi; crops.append(image[y1:y2,x1:x2].copy()); origins.append((x1,y1))
    raf4=crops[3]
    for conf in (.10,.01):
        record("raf04_single_rect",[raf4],[(0,248)],conf,True,1)
        record("heterogeneous_batch",crops,origins,conf,True,8)
        for index,(crop,origin) in enumerate(zip(crops,origins)):
            record(f"sequential_{index+1:02d}",[crop],[origin],conf,True,1)
    for size in (192,256,320,384):
        tiles=square_tiles(image,ROIS["CAM-02-RAF-04"],size,.25)
        for conf in (.10,.01):
            record(f"raf04_square_tiles_{size}",[x[1] for x in tiles],
                   [(x[0][0],x[0][1]) for x in tiles],conf,True,len(tiles))
            out["experiments"][-1]["tile_boxes"]=[x[0] for x in tiles]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    for e in out["experiments"]:
        if e["strategy"] in ("full_image_diagnostic","raf04_single_rect","heterogeneous_batch") or "tiles" in e["strategy"]:
            nearest=e["nearest_target"]
            print(e["strategy"],"conf",e["conf"],"raw",e["raw_count"],
                  "nearest",None if nearest is None else (round(nearest["confidence"],4),round(nearest["target_iou"],4),nearest["global_bbox"]))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
