import pytest

from src.runtime_config import resolve_runtime_configuration


def test_explicit_cpu_runtime_is_stable_even_when_cuda_exists():
    runtime = resolve_runtime_configuration(
        "cpu", "fp32", False, ("best.pt", "empty.pt")
    )
    assert runtime.predict_device == runtime.device == "cpu"
    assert runtime.precision == "fp32" and runtime.backend == "pytorch"
    assert runtime.gpu_name is None and runtime.cudnn_benchmark is False


def test_cpu_fp16_is_rejected_cleanly():
    with pytest.raises(ValueError, match="FP16 inference requires CUDA"):
        resolve_runtime_configuration("cpu", "fp16")


def test_auto_resolution_is_explicit_and_engine_backend_is_reported():
    runtime = resolve_runtime_configuration(
        "auto", "fp32", False, ("shelf.engine", "empty.engine")
    )
    assert runtime.device in {"cpu", "cuda:0"}
    assert runtime.backend == "tensorrt"
