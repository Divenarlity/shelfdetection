# Otomatik Test Raporu

Test zamanı: 25 Temmuz 2026, 19:21 (Europe/Istanbul)  
Test çalışma klasörü: `outputs/test_runs/20260725_192156`  
İşletim sistemi: Microsoft Windows 11 Home 64 bit, 10.0.26200  
Terminal: PowerShell 7.6.4 Core

## Sonuç

- Başarılı: model bütünlüğü, iki checkpoint yükleme/metadata, JSON raporu, Python derleme, 12 birim testi, 5 CLI yardım testi, örnek YAML doğrulaması, mask entegrasyon statik kontrolü.
- Başarısız: yok.
- Atlanan: gerçek görüntü inference, gerçek kalibrasyon kontrolü, tek görüntü uçtan uca test, tüm veri seti testi ve gerçek ground-truth değerlendirmesi.
- Atlanma nedeni: projede gerçek test görüntüsü, `configs/store_map.yaml`, `manifests/view_manifest.csv`, konum ground-truth CSV veya gerçek `empty_shelf` kutu etiketleri yoktur. Örnek dosyalar gerçek kalibrasyon olarak kullanılmamıştır.

## Ortam

| Bileşen | Sürüm |
|---|---|
| Python | 3.11.9 |
| Ultralytics | 8.4.104 |
| PyTorch | 2.13.0+cpu |
| OpenCV | 5.0.0 |
| NumPy | 2.4.6 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |

`pytest` başlangıçta eksikti ve mevcut `requirements.txt` üzerinden kuruldu. PyTorch, Ultralytics veya CUDA yükseltilmedi. Klasör bir Git deposu olmadığı için `git status` uygulanamadı.

## Model bütünlüğü ve metadata

Başlangıç ve bitiş hashleri aynıdır ve kullanıcı tarafından verilen değerlerle eşleşir:

| Model | Başlangıç SHA-256 | Bitiş SHA-256 | Görev | Sınıf |
|---|---|---|---|---|
| `best.pt` | `c1f8eb697eba0f86c11d84a2527058bb55f10d44e7efad3f63eb928966e8938a` | aynı | `segment` | `shelves` |
| `empty_shelf_yolo11m_best.pt` | `bd395db93eb47f01655fd91f932dfd82316a3b102ea8ce3a2a3a3e929488a680` | aynı | `detect` | `empty_shelf` |

`best.pt`, `SegmentationModel` ve `Segment26` başlığıyla yüklendi. `empty_shelf_yolo11m_best.pt`, `DetectionModel` olarak yüklendi. Zaman damgalı ayrıntılı rapor `outputs/test_runs/20260725_192156/model_report.json` dosyasındadır. `src/model_runner.py`, segmentasyon çıktısını `result.masks.data` üzerinden okuyup maskeyi ve poligonları korur. Gerçek görüntü bulunmadığı için çalışma zamanında sıfırdan maske üretimi denenmemiştir.

## Testler

İlk ve düzeltme sonrası testlerde aşağıdaki 12 davranış başarıyla doğrulandı:

1. Dörtgen noktalarının sıralanması
2. ROI çözünürlük ölçekleme
3. Tespit-raf kesişimi
4. En uygun rafın seçilmesi
5. Eşik altı sonucun `unknown_shelf` olması
6. Homografiyle `left`, `middle`, `right`
7. Sınırda `unknown_section`
8. Eksik kamera kimliği
9. Bozuk YAML
10. Eksik YAML alanları
11. Raf maskesi yokken `manual_roi_fallback`
12. Var olan çıktının üzerine yazılmaması

`python -m compileall .` başarılıdır. `python -m pytest -q` sonucu: `12 passed`.

Şu komutların tümü çıkış kodu 0 ile `--help` üretti:

- `inspect_models.py`
- `assign_views.py`
- `calibrate_rois.py`
- `main.py`
- `evaluate.py`

README'deki temel seçenekler gerçek CLI seçenekleriyle tutarlıdır.

## Bulunan sorunlar ve düzeltmeler

### Üretim davranışını sınamayan çıktı çakışması testi

Kök neden: `test_unique_output_policy`, üretimdeki işlevi çağırmak yerine test içinde benzer bir yerel fonksiyon tanımlıyordu. Bu nedenle üretim kodundaki olası bir gerilemeyi yakalayamazdı.

Düzeltme:

- Gerçek davranış `src.reporter.unique_output_path` içine taşındı.
- `main.py` işaretlenmiş görüntü, video ve kanıt yolları için bu işlevi kullanacak şekilde güncellendi.
- Birim testi doğrudan üretim işlevini test edecek şekilde değiştirildi.

### Ground-truth yok mesajı

`evaluate.py` mesajı istenen açık ifadeyle değiştirildi:

`Gerçek empty_shelf ground-truth etiketleri bulunmadığı için detection metrikleri hesaplanmadı.`

Düzeltmelerden sonra bütün derleme, pytest ve CLI testleri yeniden çalıştırıldı ve geçti.

## Çalıştırılan komutlar

```powershell
git status --short
py -3.11 --version
py -3.11 -m pip install -r requirements.txt
Get-FileHash -Algorithm SHA256 *.pt
py -3.11 inspect_models.py --output "outputs\test_runs\20260725_192156\model_report.json"
py -3.11 -m compileall .
py -3.11 -m pytest -q
py -3.11 inspect_models.py --help
py -3.11 assign_views.py --help
py -3.11 calibrate_rois.py --help
py -3.11 main.py --help
py -3.11 evaluate.py --help
py -3.11 evaluate.py
Get-FileHash -Algorithm SHA256 *.pt
```

Komutların ayrıntılı çıktıları `outputs/test_runs/20260725_192156` altındadır.

## Atlanan testler

- Bağımsız detection ve segmentation inference: gerçek görüntü yok.
- Gerçek maskenin çalışma zamanında üretilmesi ve çizilmesi: gerçek görüntü yok.
- Kalibrasyon sınırları ve manifest-kamera çapraz kontrolü: yalnızca örnek dosyalar var.
- Tek görüntü uçtan uca test: gerçek görüntü ve gerçek kalibrasyon yok.
- Tüm veri seti testi: gerçek veri ve manifest yok.
- Konum doğruluğu: ground-truth CSV yok.
- Precision, Recall, F1 ve mAP: gerçek `empty_shelf` ground-truth kutuları yok. `Product` etiketleri kullanılmadı.

## Kullanıcının sonraki işlemi

Önce gerçek market test görüntülerinin bulunduğu klasörle görünüm atamasını başlatın:

```powershell
py -3.11 assign_views.py --source "C:\GERCEK\MARKET\GORUNTULERI"
```

Bu işlemden sonra gerçek bir referans görüntü için `calibrate_rois.py` ile manuel raf kalibrasyonu yapılabilir.
