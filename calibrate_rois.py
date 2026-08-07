from __future__ import annotations
import argparse, shutil
from datetime import datetime
from pathlib import Path
import cv2, yaml
from src.geometry import order_quad

parser=argparse.ArgumentParser(description="Dört noktalı raf ROI kalibrasyonu")
parser.add_argument("--image",required=True,type=Path); parser.add_argument("--camera-id",required=True)
parser.add_argument("--market-id",default="DEMO-MARKET-01"); parser.add_argument("--area-id",default="A")
parser.add_argument("--output",type=Path,default=Path("configs/store_map.yaml"))
args=parser.parse_args()
image=cv2.imread(str(args.image))
if image is None: raise SystemExit(f"Görüntü okunamadı: {args.image}")
points=[]; shelves=[]
def click(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN and len(points)<4: points.append([x,y])
cv2.namedWindow("calibrate_rois"); cv2.setMouseCallback("calibrate_rois",click)
print("4 köşe tıklayın. n=rafı ekle, u=son noktayı geri al, d=son rafı sil, r=çizimi sıfırla, s=kaydet, q=çık.")
while True:
    view=image.copy()
    for s in shelves:
        p=order_quad(s["polygon"]).astype(int); cv2.polylines(view,[p],True,(0,255,0),2)
        cv2.putText(view,s["shelf_id"],tuple(p[0]),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
    for p in points: cv2.circle(view,tuple(p),5,(0,0,255),-1)
    if len(points)>1: cv2.polylines(view,[__import__("numpy").array(points)],False,(0,0,255),2)
    cv2.imshow("calibrate_rois",view); key=cv2.waitKey(30)&0xFF
    if key==ord("u") and points: points.pop()
    elif key==ord("r"): points.clear()
    elif key==ord("d") and shelves: shelves.pop()
    elif key==ord("n"):
        if len(points)!=4: print("Önce dört nokta seçin."); continue
        sid=input("Raf ID: ").strip(); rack=input("Rack ID: ").strip()
        try: level=int(input("Seviye: "))
        except ValueError: print("Seviye tam sayı olmalı."); continue
        shelves.append({"shelf_id":sid,"rack_id":rack,"level":level,"polygon":order_quad(points).tolist()}); points.clear()
    elif key==ord("s"):
        raw={"market_id":args.market_id,"section_names":["left","middle","right"],"cameras":{}}
        if args.output.exists():
            raw=yaml.safe_load(args.output.read_text(encoding="utf-8")) or raw
            backup=args.output.with_suffix(f".{datetime.now():%Y%m%d_%H%M%S}.bak.yaml")
            shutil.copy2(args.output,backup); print(f"Yedek: {backup}")
        raw["market_id"]=args.market_id; raw.setdefault("cameras",{})[args.camera_id]={
            "area_id":args.area_id,"reference_width":image.shape[1],"reference_height":image.shape[0],"shelves":shelves}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(yaml.safe_dump(raw,sort_keys=False,allow_unicode=True),encoding="utf-8")
        print(f"Kaydedildi: {args.output}")
    elif key in (ord("q"),27): break
cv2.destroyAllWindows()

