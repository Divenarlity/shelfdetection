# Shelf detection

This repository is the Python half of the live Unity shelf-monitoring system. The production pipeline is:

`Unity frame + camera metadata → shelf segmentation → pose/map parent matching → shelf-level ROIs → one empty-model batch → level-mask validation → global deduplication → SOL / ORTA / SAĞ → JSON + annotated JPEG`

The server loads both models and the Unity-exported store map once. It serializes inference requests with one lock so GPU work never overlaps.

## Model contracts

- `best.pt`: Ultralytics `segment`, class `shelves`, confidence `0.25`, image size `960`
- `empty_shelf_yolo11m_best.pt`: Ultralytics `detect`, class `empty_shelf`, confidence `0.10`, image size `640`

Startup fails when either model violates its contract. Weight files remain local and are ignored by Git.

## Production inference

`shelf_level_roi` is the default and production mode. Shelf masks are associated with known parent regions from the camera pose and `store_map.json`. Levels belonging to the same parent are numbered deterministically from top to bottom, for example `A-L-02-01`, `A-L-02-02`, and so on.

Every eligible known level contributes one padded mask-bounding-box ROI. All ROIs are passed to the empty model in one batch. Local detections are translated to global image coordinates, validated against the exact originating level mask, deduplicated globally, and classified as `SOL`, `ORTA`, or `SAĞ` relative to that level.

`UNKNOWN_SHELF` proposals remain `UNLOCALIZED`. They receive no invented level ID and are excluded from production empty inference.

Two explicit non-production modes remain for diagnosis and comparison:

- `full_frame_gated`: one full-frame empty-model input followed by shelf-mask association
- `shelf_roi`: the earlier shelf-ROI path, including optional fixed-size tiles and the ROI cap

Their mode-specific CLI settings remain supported because the comparison tool and regression tests exercise them. They do not complicate the default `shelf_level_roi` route.

## Start the server

Create a virtual environment, install `requirements.txt`, and run:

```powershell
python inference_server.py `
  --host 127.0.0.1 `
  --port 8000 `
  --shelf-model best.pt `
  --empty-model empty_shelf_yolo11m_best.pt `
  --store-map "<Unity project>\ShelfSystemData\store_map.json"
```

Quote the complete `--store-map` path when it contains spaces. `GET /health` reports readiness. `POST /infer` accepts multipart `image` and `metadata`; invalid uploads or metadata return 422, concurrent/not-ready requests return 503, and unexpected inference failures return a generic 500 while details stay in server logs.

The API enforces a 12 MiB upload limit, schema-v3 pose metadata, matching image dimensions, and rejection of ground-truth identity fields.

## Visualization

`PythonAnnotatedFrame` is the Unity production display mode. Python reuses the shelf result already computed for inference and calls native Ultralytics `result.plot()` once; it does not run the shelf model again. Only final accepted empty detections are added as red boxes. The JPEG is kept in a bounded in-memory cache and returned through `visualization_url`.

Unity downloads that URL and displays the JPEG in an aspect-preserving `RawImage` under a scene-root `ScreenSpaceOverlay` canvas. The semantic HUD remains separate. The tested `UnityGeometryOverlay` polygon/bbox renderer is retained only as an explicit debug/fallback display mode.

## Unity workflow and patrol

Open `Market_01` in the paired Unity project and enter Play Mode with the server ready. The final patrol is Enter-gated:

1. The current camera pose settles and is scanned once without movement.
2. The result and annotated image remain visible in `WaitingForInput`.
3. Enter or Numpad Enter advances exactly once through any transit points to the next scan point.
4. `RobotRig` moves horizontally at fixed height; the mounted camera local pose remains fixed.
5. At the scan point the rig aligns only around Y, settles, stays stationary during inference, displays the result, and returns to `WaitingForInput`.
6. The route traverses scan points in PingPong order. Enter presses while moving, settling, or inferring are ignored rather than queued.

The route setup computes observation distance from shelf size and camera FOV so the complete shelf remains framed. `ShelfLocationBridge` owns capture/metadata/map export, `ShelfInferenceClient` owns HTTP lifecycle, `ShelfInferenceHud` owns presentation, and `ShelfCameraPatrolController` owns movement/input/scan sequencing.

## Debug and offline tools

Production requests do not write files. Add `--debug-live` to save explicit per-request diagnostic evidence under `runs/live_debug/`; this also enables additional low-confidence/full-frame/full-shelf/tile control passes.

Run the same pipeline on a saved Unity frame:

```powershell
python run_shelf_gap_cascade.py `
  --source ShelfSystemData\input\frame_000001.png `
  --frame-metadata ShelfSystemData\input\frame_000001.json `
  --store-map ShelfSystemData\store_map.json
```

Use `compare_empty_inference_modes.py --source <image.jpg>` for a raw-image A/B comparison of the retained diagnostic strategies. `evaluate_unity.py` compares offline results with separately stored ground truth; ground truth is never an inference input. Generated `runs/`, caches, local environments, weights, and evaluation output are ignored by Git.

## Validation

```powershell
python -m compileall inference_server.py run_shelf_gap_cascade.py compare_empty_inference_modes.py evaluate_unity.py src tests
python -m pytest -q
python inference_server.py --help
python run_shelf_gap_cascade.py --help
python compare_empty_inference_modes.py --help
python evaluate_unity.py --help
```

The real-model regression test uses the checked-in A-L-02 frame and is skipped only if the local weights are absent. Unity provides scene validation, inference-contract checks, and patrol checks under `Tools/Shelf Location`; these cover camera-state restoration, response parsing, UI ownership, texture replacement, initial scan, fresh/busy Enter behavior, fixed pose, framing, inference waiting, retries, and PingPong boundaries.

Performance is dominated by the two model stages. Production performs one shelf-model call and, when known levels exist, one batched empty-model call. Debug mode intentionally performs extra control inference and should not be used for normal operation.
