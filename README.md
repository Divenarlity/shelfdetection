# Shelf detection

This repository contains the Python side of the live Unity shelf-empty-space system. The production path is deliberately singular:

`Unity RGB frame + camera pose → projected known shelves → shelf segmentation → one-to-one ID matching → matched full-shelf ROI → empty-space detection → spatial validation → API response`

The persistent server loads both models and the Unity-exported store map once. `UNKNOWN_SHELF` masks are reported as unlocalized and never reach the empty-space model. Full-frame empty inference is diagnostic only.

## Model contracts

- `best.pt`: Ultralytics `segment` model containing class `shelves`
- `empty_shelf_yolo11m_best.pt`: Ultralytics `detect` model containing class `empty_shelf`

Startup fails if either contract is wrong. Weight files remain local and are ignored by Git.

## Live Unity workflow

Create a virtual environment, install `requirements.txt`, then start the server from this repository:

```powershell
python inference_server.py --store-map "<path-to-shelfsimulation>\ShelfSystemData\store_map.json"
```

Then open `Market_01` in Unity and enter Play Mode. The scene's `ShelfInferenceClient` posts to `http://127.0.0.1:8000`, displays current status, and only emits a state-change notification when a shelf result changes.

The default empty input strategy is `full_shelf` with 2% padding and an empty confidence threshold of `0.10`. Fixed 384×384 tiles are available only through the explicit `--empty-roi-mode tiles` option; they are not selected automatically because the A-L-02 regression demonstrates a tile-context false negative.

### Unity responsibilities

- `KnownShelfRegion`: serialized static shelf ID and BL/TL/TR/BR world geometry
- `ShelfLocationBridge`: camera capture, ID-free pose metadata, map export, and offline frame capture
- `ShelfInferenceClient`: serialized non-overlapping HTTP loop and status UI
- `ShelfInferenceResponse`: response parsing and state-change suppression
- `UnitySequenceRecorder`: optional offline dataset/evaluation sequence capture

`ShelfSystemData/store_map.json`, exported from the scene, is the authoritative known-shelf geometry for the live workflow. Ground truth is written only beside offline captures and is rejected if supplied as inference metadata.

## HTTP contract

- `GET /health` reports model readiness.
- `POST /infer` accepts multipart fields `image` and `metadata`.

The response includes `frame_id`, `strategy`, `empty_roi_mode`, `counts`, `shelves`, and `rejected_detections`. Each shelf contains its identity/mapping confidence, status, ROI, SOL/ORTA/SAĞ counts, and final detections. `counts.roi_inferences` is the number of matched-shelf model inputs.

The server serializes requests with one inference lock for GPU safety. A concurrent request receives HTTP 503. Invalid images, dimensions, pose metadata, or forbidden identity/ground-truth fields receive HTTP 422.

## Debug mode

Add `--debug-live` to save diagnostic evidence under `runs/live_debug/`:

```powershell
python inference_server.py --store-map "<path-to-shelfsimulation>\ShelfSystemData\store_map.json" --debug-live
```

Debug mode performs additional confidence-0.01, full-frame, full-shelf, and tile control passes. Without this flag there are no control model passes, debug images, debug JSON, or debug filesystem writes.

## Offline inference

The offline CLI calls the exact same `ShelfInferencePipeline` as the API:

```powershell
python run_shelf_gap_cascade.py `
  --source ShelfSystemData\input\frame_000001.png `
  --frame-metadata ShelfSystemData\input\frame_000001.json `
  --store-map ShelfSystemData\store_map.json
```

It writes `result.json` and `annotated.jpg` under `runs/offline/`. Use `--debug-live` only when control evidence is needed. `evaluate_unity.py` compares offline `result.json` files with a separate ground-truth directory; evaluation data is never an inference input.

## Validation

```powershell
python -m compileall inference_server.py run_shelf_gap_cascade.py evaluate_unity.py src tests
python -m pytest -q
python inference_server.py --help
python run_shelf_gap_cascade.py --help
```

The real-model regression test uses the small saved A-L-02 frame in `tests/data/a_l_02_regression`. It is skipped only when the local `.pt` files are absent.

## Troubleshooting

- Startup contract error: confirm each model task and required class above.
- Unity waits for the server: check `GET http://127.0.0.1:8000/health` and the client URL.
- No known shelf match: validate/export the Unity scene map and check pose quality, relocalization, scale initialization, camera intrinsics, and scene geometry.
- Shelf matches but no empty space: reproduce with the offline CLI; enable `--debug-live` only for diagnosis.
- HTTP 503: a previous inference is still running; Unity intentionally does not overlap requests.
