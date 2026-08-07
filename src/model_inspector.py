from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json


def _names(value):
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {str(i): str(v) for i, v in enumerate(value)}
    return {}


def inspect_checkpoint(path):
    from ultralytics import YOLO
    import ultralytics
    import torch
    path = Path(path)
    model = YOLO(str(path))
    names = _names(getattr(model, "names", {}))
    ckpt = getattr(model, "ckpt", None) or {}
    train_args = ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {}
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "task": getattr(model, "task", None),
        "class_names": names,
        "model_class": type(getattr(model, "model", None)).__name__,
        "ultralytics_version": ultralytics.__version__,
        "pytorch_version": torch.__version__,
        "device": str(next(model.model.parameters()).device),
        "cuda_available": torch.cuda.is_available(),
        "checkpoint_version": ckpt.get("version") if isinstance(ckpt, dict) else None,
        "model_yaml": getattr(getattr(model, "model", None), "yaml", None),
        "train_args": {k: train_args.get(k) for k in ("model", "data", "epochs", "imgsz") if k in train_args},
    }


def classify_roles(reports):
    shelf_words = ("shelf", "shelves", "raf")
    gap_words = ("empty", "gap", "space", "bos", "boş")
    shelf, gap = [], []
    for report in reports:
        labels = " ".join(report["class_names"].values()).lower()
        if report["task"] == "segment" and any(w in labels for w in shelf_words):
            shelf.append(report["filename"])
        if report["task"] in ("detect", "segment") and any(w in labels for w in gap_words):
            gap.append(report["filename"])
    if len(shelf) != 1 or len(gap) != 1 or shelf[0] == gap[0]:
        raise RuntimeError(f"Model rolleri güvenilir biçimde ayrılamadı. shelf={shelf}, gap={gap}")
    return {"shelf_model": shelf[0], "gap_model": gap[0]}


def inspect_directory(root, output):
    root, output = Path(root), Path(output)
    paths = sorted(root.glob("*.pt"))
    if len(paths) != 2:
        raise RuntimeError(f"Tam olarak iki .pt dosyası bekleniyor; bulunan: {len(paths)}")
    reports = [inspect_checkpoint(p) for p in paths]
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "models": reports}
    payload["roles"] = classify_roles(reports)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return payload
