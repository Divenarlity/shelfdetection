from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "empty_shelf_yolo11m_best.pt"
DEFAULT_DATASET = (
    PROJECT_ROOT.parent
    / "Veri setleri"
    / "Empty shelf detection data sets"
    / "Merged_Empty_Shelf_YOLOv11"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def select_labeled_test_image(dataset_root: Path = DEFAULT_DATASET) -> tuple[Path, Path, int]:
    """Alfabetik ilk, boş olmayan YOLO etiketi bulunan test görüntüsünü seçer."""
    images_dir = dataset_root / "test" / "images"
    labels_dir = dataset_root / "test" / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Veri setinin test/images veya test/labels klasörü bulunamadı: {dataset_root}"
        )
    for image_path in sorted(
        (path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    ):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        rows = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            return image_path.resolve(), label_path.resolve(), len(rows)
    raise FileNotFoundError("test/images içinde boş olmayan eş etiket dosyasına sahip görüntü bulunamadı.")


def validate_model(model_path: Path):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    class_names = {int(index): str(name) for index, name in names.items()}
    if model.task != "detect":
        raise RuntimeError(
            f"Geçersiz model görevi: {model.task!r}. Yalnızca 'detect' modeli destekleniyor."
        )
    if "empty_shelf" not in class_names.values():
        raise RuntimeError(
            f"Model sınıflarında 'empty_shelf' yok. Bulunan sınıflar: {list(class_names.values())}"
        )
    return model, class_names


def find_unicode_font(size: int = 24) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\DejaVuSans.ttf"),
        Path(__file__).resolve().parent / "DejaVuSans.ttf",
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError as exc:
        raise RuntimeError("Türkçe metin için Unicode destekli TrueType font bulunamadı.") from exc


def annotate(image_bgr, detections: list[dict]):
    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = find_unicode_font()
    if not detections:
        text = "Boş raf alanı tespit edilmedi"
        box = draw.textbbox((12, 12), text, font=font)
        draw.rectangle((8, 8, box[2] + 8, box[3] + 8), fill=(255, 255, 255))
        draw.text((12, 12), text, font=font, fill=(220, 0, 0))
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        text = f"Burada boşluk var ({detection['confidence']:.2f})"
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=4)
        text_box = draw.textbbox((x1, y1), text, font=font)
        text_height = text_box[3] - text_box[1]
        text_y = max(0, y1 - text_height - 8)
        background = draw.textbbox((x1, text_y), text, font=font)
        draw.rectangle((background[0] - 3, background[1] - 3, background[2] + 3, background[3] + 3),
                       fill=(255, 255, 255))
        draw.text((x1, text_y), text, font=font, fill=(220, 0, 0))
    return cv2.cvtColor(__import__("numpy").asarray(image), cv2.COLOR_RGB2BGR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tek empty_shelf modeliyle tek görüntü inference.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--camera-id", default="CAM-01")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", help="Örnek: cpu, 0. Verilmezse Ultralytics otomatik seçer.")
    parser.add_argument("--show", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf 0 ile 1 arasında olmalıdır.")
    source = args.source.resolve() if args.source else None
    if source is None:
        source, label, gt_count = select_labeled_test_image()
        print(f"Seçilen test görüntüsü: {source}")
        print(f"Etiket dosyası: {label}")
        print(f"Ground-truth kutu sayısı: {gt_count}")
    if not source.is_file():
        raise FileNotFoundError(f"Kaynak görüntü bulunamadı: {source}")
    image = cv2.imread(str(source))
    if image is None:
        raise RuntimeError(f"Kaynak görüntü OpenCV ile okunamadı: {source}")

    model_path = args.model.resolve()
    print(f"Yüklenen tek model: {model_path}")
    model, class_names = validate_model(model_path)
    print(f"Model görevi: {model.task}; sınıflar: {list(class_names.values())}")

    predict_args = {"source": image, "conf": args.conf, "imgsz": args.imgsz, "verbose": False}
    if args.device:
        predict_args["device"] = args.device
    result = model.predict(**predict_args)[0]  # Tek inference çağrısı.

    detections = []
    if result.boxes is not None:
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            name = class_names[int(class_id)]
            if name != "empty_shelf":
                continue
            detections.append(
                {
                    "class_name": name,
                    "confidence": float(confidence),
                    "bbox_xyxy": [float(value) for value in box],
                    "message": "Burada boşluk var",
                }
            )

    annotated = annotate(image, detections)
    output_dir = PROJECT_ROOT / "runs" / "empty_shelf_single" / args.camera_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image = output_dir / f"annotated_{source.stem}.jpg"
    results_path = output_dir / "results.json"
    if not cv2.imwrite(str(output_image), annotated):
        raise RuntimeError(f"İşaretlenmiş görüntü yazılamadı: {output_image}")
    payload = {
        "camera_id": args.camera_id,
        "source_image": str(source),
        "model": str(model_path),
        "confidence_threshold": args.conf,
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "detection_count": len(detections),
        "detections": detections,
    }
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Tahmin edilen boşluk sayısı: {len(detections)}")
    print(f"İşaretlenmiş görüntü: {output_image.resolve()}")
    print(f"Sonuç JSON: {results_path.resolve()}")

    # TODO: Yalnızca ileride gerçekten kesişen kameralar sağlanırsa kameralar arası eşleştirme değerlendirilecek.
    if args.show:
        cv2.imshow(f"empty_shelf - {args.camera_id}", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return payload


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

