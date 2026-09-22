import json
from evaluate_unity import evaluate


def test_multiple_visible_ids_and_unknown_rate(tmp_path):
    pred=tmp_path/"pred"/"run";gt=tmp_path/"gt";pred.mkdir(parents=True);gt.mkdir()
    (pred/"result.json").write_text(json.dumps({"frame_id":"frame_1","shelves":[
        {"shelf_id":"A"},{"shelf_id":"B"},{"shelf_id":"UNKNOWN_SHELF"}]}),encoding="utf-8")
    (gt/"frame_1.json").write_text(json.dumps({"visible_shelf_ids":["A","B"]}),encoding="utf-8")
    result=evaluate(tmp_path/"pred",gt)
    assert result["frames"]==1 and result["precision"]==result["recall"]==result["shelf_id_accuracy"]==1
    assert result["unknown_shelf_rate"]==1/3
