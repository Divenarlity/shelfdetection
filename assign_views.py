from __future__ import annotations
import argparse, csv
from pathlib import Path
import cv2

EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
parser = argparse.ArgumentParser(description="Görüntülere sabit kamera/görünüm kimliği atar.")
parser.add_argument("--source", required=True, type=Path)
parser.add_argument("--output", type=Path, default=Path("manifests/view_manifest.csv"))
parser.add_argument("--market-id", default="DEMO-MARKET-01")
parser.add_argument("--area-id", default="A")
args = parser.parse_args()
images = [args.source] if args.source.is_file() else sorted(p for p in args.source.rglob("*") if p.suffix.lower() in EXTS)
if not images:
    raise SystemExit("Görüntü bulunamadı.")
rows = []
print("Her görüntü için terminale camera_id yazın. q=çıkış, s=atla.")
for path in images:
    image = cv2.imread(str(path))
    if image is None: continue
    view = image.copy()
    cv2.putText(view, path.name, (15,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow("assign_views", view); cv2.waitKey(1)
    cid = input(f"{path.name} camera_id: ").strip()
    if cid.lower() == "q": break
    if not cid or cid.lower() == "s": continue
    calibration = input("Kalibrasyon için kullanılsın mı? [e/H]: ").strip().lower() in ("e","evet","y","yes")
    rows.append({"source":str(path.resolve()),"camera_id":cid,"market_id":args.market_id,
                 "area_id":args.area_id,"use_for_calibration":str(calibration).lower()})
cv2.destroyAllWindows()
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,["source","camera_id","market_id","area_id","use_for_calibration"]); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} atama yazıldı: {args.output}")

