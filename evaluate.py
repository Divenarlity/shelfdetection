from __future__ import annotations
import argparse,csv,json
from pathlib import Path
parser=argparse.ArgumentParser(); parser.add_argument("--ground-truth",type=Path); parser.add_argument("--predictions",type=Path,default=Path("outputs/reports/detections.csv")); parser.add_argument("--empty-labels",type=Path)
args=parser.parse_args()
if args.ground_truth:
    with args.ground_truth.open(encoding="utf-8-sig") as f: gt=list(csv.DictReader(f))
    with args.predictions.open(encoding="utf-8-sig") as f: pred=list(csv.DictReader(f))
    by_source={}
    for p in pred: by_source.setdefault(Path(p["source"]).name,[]).append(p)
    total=len(gt); shelf=section=both=unknown=wrong=count=0
    for g in gt:
        ps=by_source.get(Path(g["source"]).name,[]); count+=len(ps)
        if not ps: unknown+=1; continue
        p=ps[0]; s=p["shelf_id"]==g["expected_shelf_id"]; q=p["section"]==g["expected_section"]
        shelf+=s; section+=q; both+=s and q; unknown+=p["shelf_id"]=="unknown_shelf"; wrong+=p["shelf_id"] not in (g["expected_shelf_id"],"unknown_shelf")
    metrics={"samples":total,"shelf_accuracy":shelf/total if total else 0,"section_accuracy":section/total if total else 0,
             "end_to_end_accuracy":both/total if total else 0,"unknown_shelf_rate":unknown/total if total else 0,
             "detections_per_image":count/total if total else 0,"wrong_location_assignments":wrong}
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
else: print("Konum değerlendirmesi için --ground-truth sağlayın.")
if not args.empty_labels or not args.empty_labels.exists():
    print("Gerçek empty_shelf ground-truth etiketleri bulunmadığı için detection metrikleri hesaplanmadı.")
else:
    print("Empty-shelf etiketleri bulundu. Detection metrikleri için Ultralytics veri YAML'sı ve val akışı ayrıca çalıştırılmalıdır; bu araç sahte metrik üretmez.")
