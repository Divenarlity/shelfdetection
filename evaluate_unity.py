import argparse, json
from collections import defaultdict
from pathlib import Path

UNKNOWN="UNKNOWN_SHELF"


def evaluate(predictions, ground_truth):
    rows=[]; per=defaultdict(lambda:[0,0])
    for pred_path in sorted(Path(predictions).glob("**/results.json")):
        pred=json.loads(pred_path.read_text(encoding="utf-8")); fid=str(pred.get("frame_id"))
        gt_name=f"{fid}.json" if fid.startswith("frame_") else f"frame_{fid}.json"
        gt_path=Path(ground_truth)/gt_name
        if not gt_path.exists(): continue
        gt=json.loads(gt_path.read_text(encoding="utf-8"))
        expected=set(gt.get("visible_shelf_ids",[]))
        predicted=pred.get("shelves",pred.get("predicted_shelves",[]))
        actual={x["shelf_id"] for x in predicted if x.get("shelf_id")!=UNKNOWN}
        unknown=sum(x.get("shelf_id")==UNKNOWN for x in predicted)
        tp=len(actual&expected);fp=len(actual-expected);fn=len(expected-actual)
        for sid in expected:per[sid][1]+=1;per[sid][0]+=int(sid in actual)
        rows.append((tp,fp,fn,actual==expected,unknown,len(predicted)))
    tp=sum(x[0] for x in rows);fp=sum(x[1] for x in rows);fn=sum(x[2] for x in rows)
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    return {"frames":len(rows),"shelf_id_accuracy":tp/max(tp+fp+fn,1),"precision":precision,
      "recall":recall,"f1":2*precision*recall/max(precision+recall,1e-12),
      "exact_frame_match":sum(x[3] for x in rows)/max(len(rows),1),
      "wrong_id_rate":fp/max(tp+fp,1),"unknown_shelf_rate":sum(x[4] for x in rows)/max(sum(x[5] for x in rows),1),
      "per_shelf":{k:{"correct":v[0],"total":v[1],"accuracy":v[0]/v[1]} for k,v in sorted(per.items())}}


def main():
    p=argparse.ArgumentParser(description="Tahminleri yalnızca ayrı Unity ground truth klasörüyle değerlendirir.")
    p.add_argument("--predictions",type=Path,required=True);p.add_argument("--ground-truth",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("evaluation_metrics.json"));a=p.parse_args()
    payload=evaluate(a.predictions,a.ground_truth);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2),encoding="utf-8");print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
