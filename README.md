# Shelf detection

This repository contains the Python side of the live Unity shelf-empty-space system. The production path is deliberately singular:

`Unity RGB frame + camera pose → shelf model and empty model on the same full frame → shelf-mask association → optional one-to-one known-ID mapping → global NMS → API response`

The persistent server loads both models and the Unity-exported store map once. The default `full_frame_gated` mode runs each model once, then assigns each empty box to the eligible pose/map-matched shelf mask with the greatest containment ratio. Proposals that cannot be mapped retain `UNKNOWN_SHELF` and remain visible as unlocalized shelf proposals, but do not produce production empty-gap detections.

## Model contracts

- `best.pt`: currently an Ultralytics `segment` model containing class `shelves`; the pipeline explicitly supports `segment` and `detect` shelf weights
- `empty_shelf_yolo11m_best.pt`: Ultralytics `detect` model containing class `empty_shelf`

Startup fails if either contract is wrong. Weight files remain local and are ignored by Git.

## Live Unity workflow

Create a virtual environment, install `requirements.txt`, then start the server from this repository:

```powershell
python inference_server.py --store-map "<path-to-shelfsimulation>\ShelfSystemData\store_map.json"
```

Then open `Market_01` in Unity and enter Play Mode. The scene's `ShelfInferenceClient` posts to `http://127.0.0.1:8000`, displays current status, and only emits a state-change notification when a shelf result changes.

The default empty inference mode is `full_frame_gated`, with shelf confidence `0.25`, empty confidence `0.10`, mask containment `0.30`, and global dedup IoU `0.50`. The former crop path remains available only through `--empty-inference-mode shelf_roi`; its 2% padding, 12-ROI cap, and optional fixed tiles apply only in that explicit comparison/debug mode.

### Unity responsibilities

- `KnownShelfRegion`: serialized static shelf ID and BL/TL/TR/BR world geometry
- `ShelfLocationBridge`: camera capture, ID-free pose metadata, map export, and offline frame capture
- `ShelfInferenceClient`: serialized non-overlapping HTTP loop and result dispatch
- `ShelfInferenceHud`: 1920×1080 reference-scaled top-right status card
- `DetectionOverlayManager`: pooled cyan shelf boxes and red empty-space boxes using one resolution/aspect mapping path
- `ShelfInferenceResponse`: response parsing and state-change suppression
- `UnitySequenceRecorder`: optional offline dataset/evaluation sequence capture

`ShelfSystemData/store_map.json`, exported from the scene, is the authoritative known-shelf geometry for the live workflow. Ground truth is written only beside offline captures and is rejected if supplied as inference metadata.

## HTTP contract

- `GET /health` reports model readiness.
- `POST /infer` accepts multipart fields `image` and `metadata`.

The response includes `empty_inference_mode`, `frame_id`, `image`, `model_tasks`, `strategy`, timing, counts, shelves, and rejected detections. Each shelf carries `empty_inference_source`; full-frame detections preserve their original global coordinates and receive exactly one `assigned_shelf_index` only after mask gating. `roi_inferences` and ROI-selection counts are zero in the default mode, while `empty_model_predict_calls` and `empty_inference_inputs` are one.

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

For a raw-image A/B/C comparison without pose metadata:

```powershell
python compare_empty_inference_modes.py --source "<image.jpg>"
```

This writes source, shelf masks, standalone full-frame detections, legacy shelf-ROI detections, gated detections, and `comparison.json` under `runs/full_frame_gated_validation_*`.

## Validation

```powershell
python -m compileall inference_server.py run_shelf_gap_cascade.py compare_empty_inference_modes.py evaluate_unity.py src tests
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
