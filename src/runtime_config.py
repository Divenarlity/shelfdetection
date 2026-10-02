"""Resolve explicit inference device and precision without hiding fallbacks."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RuntimeConfiguration:
    requested_device: str
    predict_device: str
    device: str
    precision: str
    backend: str
    cuda_available: bool
    gpu_name: str | None
    torch_version: str
    torch_cuda_version: str | None
    cudnn_benchmark: bool

    def health_fields(self):
        value = asdict(self)
        value.pop("predict_device")
        value.pop("requested_device")
        value["cuda_version"] = value.pop("torch_cuda_version")
        return value


def resolve_runtime_configuration(device="cpu", precision="fp32",
                                  cudnn_benchmark=False, model_paths=()):
    import torch

    requested = str(device or "cpu").strip().lower()
    if requested == "auto":
        predict_device = "0" if torch.cuda.is_available() else "cpu"
    elif requested.startswith("cuda:"):
        predict_device = requested.split(":", 1)[1]
    else:
        predict_device = requested

    if predict_device == "cpu":
        display_device = "cpu"
    else:
        try:
            index = int(predict_device)
        except ValueError as exc:
            raise ValueError("device must be cpu, auto, a CUDA index, or cuda:<index>.") from exc
        if not torch.cuda.is_available():
            raise ValueError(
                f"CUDA device {index} was requested but torch.cuda.is_available() is False."
            )
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {index} is outside the available range "
                f"0..{torch.cuda.device_count() - 1}."
            )
        predict_device = str(index)
        display_device = f"cuda:{index}"

    precision = str(precision).lower()
    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16.")
    if precision == "fp16" and display_device == "cpu":
        raise ValueError("FP16 inference requires CUDA; use --device 0 or --precision fp32.")

    suffixes = {str(path).lower().rsplit(".", 1)[-1] for path in model_paths}
    backend = "tensorrt" if suffixes == {"engine"} else (
        "pytorch" if not suffixes or suffixes == {"pt"} else "mixed"
    )
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark and display_device != "cpu")
    index = int(predict_device) if display_device != "cpu" else None
    return RuntimeConfiguration(
        requested_device=requested,
        predict_device=predict_device,
        device=display_device,
        precision=precision,
        backend=backend,
        cuda_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(index) if index is not None else None,
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
    )
