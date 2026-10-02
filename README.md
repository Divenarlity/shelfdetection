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

## GPU acceleration

The validated Windows environment uses the repository `.venv`, PyTorch
`2.6.0+cu124`, torchvision `0.21.0+cu124`, and the official CUDA 12.4 wheel
index. Keep this platform-specific choice out of `requirements.txt`; install it
explicitly when provisioning the RTX machine:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps `
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 `
  --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The selected production mode is CUDA FP32. It preserved
the CPU FP32 semantic result on the real A02 dual capture and reduced median
Python pipeline latency from 6414 ms to 551 ms (11.64x, 91.4% lower):

| Mode | Shelf median | Empty median | Pipeline median | CPU semantic match |
| --- | ---: | ---: | ---: | --- |
| CPU PT FP32 | 1819 ms | 4222 ms | 6414 ms | baseline |
| CUDA PT FP32 | 87 ms | 258 ms | 551 ms | yes |
| CUDA PT FP16 | 51 ms | 152 ms | 408 ms | no |
| TensorRT FP16 | 29 ms | 102 ms | 391 ms | no |

```powershell
.\.venv\Scripts\python.exe inference_server.py `
  --store-map "C:\Users\yusuf\OneDrive\Belgeler\Unity Projects\My project\ShelfSystemData\store_map.json" `
  --device 0 --precision fp32 `
  --warmup-iterations 3 --warmup-empty-batch-size 24
```

CPU fallback remains explicit and uses the unchanged `.pt` models:

```powershell
.\.venv\Scripts\python.exe inference_server.py `
  --store-map "C:\Users\yusuf\OneDrive\Belgeler\Unity Projects\My project\ShelfSystemData\store_map.json" `
  --device cpu --precision fp32
```

`--device auto` is also supported and reports the resolved device in startup
logs and `/health`. CUDA FP16 (`--device 0 --precision fp16`, or `--half`) is
functional, but is not the production selection because one threshold-adjacent
empty detection changed on the regression capture. Derived TensorRT FP16 engines
are loadable with the same pipeline and dynamic batches:

```powershell
.\.venv\Scripts\python.exe inference_server.py `
  --shelf-model best_fp16.engine `
  --empty-model empty_shelf_yolo11m_fp16.engine `
  --store-map "C:\Users\yusuf\OneDrive\Belgeler\Unity Projects\My project\ShelfSystemData\store_map.json" `
  --device 0 --precision fp16 --warmup-iterations 3 --warmup-empty-batch-size 20
```

TensorRT is a comparison path, not production: although its median was 391 ms,
it changed final detections and left too little VRAM headroom while Unity shared
the 8 GB GPU. Original `.pt` files remain authoritative; `.onnx` and `.engine`
artifacts are ignored by Git.

cuDNN autotuning is available through `--cudnn-benchmark`, but is intentionally
off in production. It saved only about 8 ms in the isolated repeated-input test,
while previously unseen variable ROI batch shapes caused multi-second live
tuning spikes. The 24-ROI mixed-aspect startup warmup covers the configured
dual-camera maximum without writing sessions or debug artifacts.

Unity presentation rendering is capped at 60 FPS by
`DualShelfScanController` (`vSyncCount=0`, configurable `targetFrameRate=60`) so
the Player and CUDA inference can share the laptop GPU. In the validated full
six-station route, both displays initialized and delivered every annotated
frame; mean server-side dual processing was 743 ms and mean Unity-observed
capture-to-result latency was 0.917 s. The previous CPU route averaged 9.966 s
end-to-end, so the measured live improvement is 10.86x (90.8% lower latency).
The uncapped GPU comparison saturated the GPU and averaged 13.9 s, which is why
the cap is part of the production setup.

Use `benchmark_dual_inference.py` with a real saved left/right debug capture for
repeatable 3-warmup/5-or-more-iteration CPU, CUDA, and TensorRT comparisons. It
uses the production pipeline, synchronizes CUDA only for benchmark timing,
asserts the 2-image/combined-ROI batching contract, and reports stage timing,
session-save cost, semantic output, and labeled CUDA allocator statistics.

`POST /infer/dual` accepts the same two images/metadata plus optional operational
`session_id` and `station_id` fields. These identify saved scans only; expected shelf IDs
and other ground truth are never sent. `POST /session/{session_id}/complete` finalizes a
route session.

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

## Persistent scan sessions

Production dual-scan recording is enabled by default. Each route is written under
`runs/scan_sessions/session_YYYYMMDD_HHMMSS_<suffix>/`; disable it with
`--no-save-scan-sessions`, or choose another root with `--scan-session-root <path>`.
Each station contains the exact request JPEGs (`left_raw.jpg`, `right_raw.jpg`), the exact
JPEG bytes placed in the visualization cache, readable per-view JSON, and
`dual_result.json`. The recorder consumes the already-final production response and never
runs a model or post-processing stage again.

One append-only `session_log.txt` records final accepted gaps in LEFT-then-RIGHT order,
including station, frame, parent shelf, shelf level, level number, SOL/ORTA/SAĞ,
confidence, bbox, and mask overlap. A processed camera with no gap receives
`NO_EMPTY_SPACE`; unlocalized shelves are warnings, never physical alarms. Each station
ends with a count summary. `session_summary.json` is atomically refreshed after every
station with status, frames, aggregate camera/shelf/section counts, and becomes
`completed` only after the Unity route completion call. Interrupted sessions remain
usable with status `in_progress`. A top-level `persistence.save_ms` reports disk time
separately from model/pipeline timing, and a disk error returns a warning without turning
a successful inference into a failure.

## Debug and offline tools

Persistent production recording and `--debug-live` are independent. Session recording
saves only normal production inputs/results and makes zero additional model calls.
`--debug-live` instead saves explicit per-request diagnostic evidence under
`runs/live_debug/` and enables additional low-confidence/full-frame/full-shelf/tile
control passes.

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
