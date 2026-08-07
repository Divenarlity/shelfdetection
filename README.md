# Empty Shelf Detection

Varsayılan `main.py` akışı gerçek iki aşamalı cascade kullanır:

```text
ana görüntü
→ best.pt shelves instance segmentation
→ gerçek raf maskelerinden ayrı ROI crop'ları
→ empty_shelf_yolo11m_best.pt ile yalnızca ROI'lerde detection
→ maske filtresi, global koordinat dönüşümü ve duplicate temizliği
→ raf ID, SOL/ORTA/SAĞ raporu
```

Boşluk modeli ana görüntünün tamamında çalıştırılmaz. `run_empty_shelf_test.py` önceki bağımsız tek-model test modu olarak korunmuştur.

## Kurulum ve test

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q tests
```

## Cascade çalıştırma

```powershell
.\.venv\Scripts\python.exe .\main.py `
  --source "C:\veri\market.jpg" `
  --camera-id CAM-02 `
  --shelf-model ".\best.pt" `
  --empty-model ".\empty_shelf_yolo11m_best.pt" `
  --shelf-conf 0.25 `
  --empty-conf 0.10 `
  --roi-padding-ratio 0.02 `
  --min-mask-overlap 0.30 `
  --dedup-iou 0.50 `
  --save-annotated `
  --save-rois
```

Diğer seçenekler: `--shelf-imgsz`, `--empty-imgsz`, `--tile-size`, `--tile-overlap`,
`--device`, `--show` ve `--rebuild-shelf-map`. Örnek eşikler ve tile değerleri
başlangıç doğrulama değerleridir; optimize edilmiş oldukları iddia edilmez.

Boşluk modeli çok ince raf şeritleri yerine varsayılan olarak rafı tamamen örten
`384×384` bağlam tile'larında çalışır. Tile'lar %25 yatay örtüşür, yalnızca ham
kaynak görüntüden çıkarılır ve eşit boyutlu tek batch halinde işlenir. Bu davranış
`--tile-size 384 --tile-overlap 0.25` ile değiştirilebilir. Tile sınırında kırpılmış
tekrarlar global koordinatta edge-aware duplicate filtresiyle temizlenir. Boşluk
modeline ana görüntünün tamamı gönderilmez.

İlk çalışmada kamera raf referansları `configs/shelf_maps/<camera_id>.json` dosyasına yazılır. Sonraki sabit kamera görüntülerinde normalize maske IoU eşleştirmesi ID sürekliliğini korumaya çalışır. `--rebuild-shelf-map` yalnızca verilen kameranın haritasını yeniden kurar. CAM-01 ve CAM-02 haritaları ayrıdır.

Her çalışma eski çıktıyı ezmeyen benzersiz bir klasöre yazılır:

```text
runs/shelf_gap_cascade/<camera_id>_<görsel_adı>/
├── annotated.jpg
├── results.json
├── diagnostics.json
├── shelf_summary.csv
└── rois/
    ├── <shelf_id>_<tile_id>_input.jpg
    ├── <shelf_id>_<tile_id>_mask.png
    ├── <shelf_id>_<tile_id>_raw_conf001.jpg
    ├── <shelf_id>_<tile_id>_accepted.jpg
    └── <shelf_id>_<tile_id>_rejected.jpg
```

## Tek-model test modu

Eski bağımsız empty-shelf testi hâlâ kullanılabilir:

```powershell
.\.venv\Scripts\python.exe .\run_empty_shelf_test.py `
  --source "C:\veri\test.jpg" `
  --camera-id CAM-01 `
  --conf 0.10
```

Bu mod cascade değildir ve yalnızca açıkça çağrıldığında çalışır.
# Persistent robot shelf mapping

The robot workflow has three explicit modes: `mapping`, `localization`, and
`mapping_update`. `legacy_roi` and the manually prepared `pose_geometry` mode
remain available for regression and evaluation.

Runtime schema v3 defines `camera.position_map` and `camera.rotation_xyzw` as
`T_map_camera`: the camera pose expressed in the persistent SLAM map frame.
Quaternions are always `xyzw`. Unity uses a left-handed, Y-up world with camera
forward along local +Z. Projection converts to OpenCV optical coordinates
(X right, Y down, Z forward); pixels have a top-left origin. Serialized
compatibility matrices are row-major. Intrinsics are `fx, fy, cx, cy`.

`mapping` requires multiple ID-free RGB frames with initialized metric scale,
adequate pose quality, and camera baseline. Mask quads are associated globally
between views and triangulated from the known poses. A landmark is confirmed
only after the thresholds in `configs/mapping.yaml` pass. Map writes are
atomic. `localization` loads that map read-only and never allocates an ID.
Unity Inspector shelf identities are written only to the separate evaluation
ground-truth directory.

```powershell
.\run_mapping.ps1 -Source "<ShelfSystemData\sessions\mapping_pass_01\input>"
.\run_localization.ps1 -Source "<ShelfSystemData\sessions\localization_pass_01\input>"
```
