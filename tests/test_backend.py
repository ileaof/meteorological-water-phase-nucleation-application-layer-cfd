"""Tests for meteorological_flow.backend: the CPU/GPU compute-backend
resolution state machine (get_backend).

GPU-hardware tests are skipped automatically in environments without a
working CUDA/CuPy stack, so this file passes on CPU-only CI unchanged.
"""
import sys

import pytest

from meteorological_flow.backend import (
    BackendError,
    MissingLibraryError,
    get_backend,
)


def _gpu_available() -> bool:
    try:
        import cupy
        _ = cupy.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def test_device_cpu_always_succeeds():
    b = get_backend("cpu")
    assert b.name == "cpu"
    assert b.xp.__name__ == "numpy"
    assert b.fallback_reason is None
    assert b.to_cpu(b.zeros((2, 2))) is not None


def test_device_invalid_raises_value_error():
    with pytest.raises(ValueError):
        get_backend("quantum")


def test_device_auto_falls_back_when_cupy_absent(monkeypatch):
    # `sys.modules["cupy"] = None` makes a subsequent `import cupy` raise
    # ImportError, regardless of whether cupy is really installed here.
    monkeypatch.setitem(sys.modules, "cupy", None)
    logged = []
    b = get_backend("auto", log=logged.append)
    assert b.name == "cpu"
    assert b.fallback_reason is not None
    assert "missing_library" in b.fallback_reason
    assert logged and "missing_library" in logged[0]


def test_device_gpu_raises_when_cupy_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(MissingLibraryError) as exc_info:
        get_backend("gpu")
    assert exc_info.value.category == "missing_library"


def test_device_gpu_never_silently_returns_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(BackendError):
        b = get_backend("gpu")
        assert b.name != "cpu"   # unreachable if the raise above fires, by design


@pytest.mark.skipif(not _gpu_available(), reason="no working CUDA/CuPy GPU in this environment")
def test_gpu_backend_resolves_and_initializes():
    b = get_backend("gpu")
    assert b.name == "gpu"
    assert b.xp.__name__ == "cupy"
    assert b.sparse is not None and b.sparse_linalg is not None
    label = b.device_info()["label"]
    assert "GPU" in label
    z = b.zeros((4, 4))
    assert float(b.to_cpu(z).sum()) == 0.0
    b.synchronize()
    b.free_memory()


@pytest.mark.skipif(not _gpu_available(), reason="no working CUDA/CuPy GPU in this environment")
def test_gpu_auto_prefers_gpu_when_available():
    logged = []
    b = get_backend("auto", log=logged.append)
    assert b.name == "gpu"
    assert b.fallback_reason is None
    assert not logged   # no fallback message: GPU resolved cleanly on the first try


@pytest.mark.skipif(not _gpu_available(), reason="no working CUDA/CuPy GPU in this environment")
def test_gpu_memory_probe_reports_positive_free_gb():
    import cupy
    free_b, _total_b = cupy.cuda.Device(0).mem_info
    assert free_b > 0
