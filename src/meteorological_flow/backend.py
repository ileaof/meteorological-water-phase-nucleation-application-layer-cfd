"""Compute backend abstraction: CPU (NumPy/SciPy) vs GPU (CuPy), selected via
``get_backend(device="auto"|"cpu"|"gpu")``.

``Grid``, ``FlowState`` and the per-step operators (advection, diffusion,
buoyancy, boundary conditions, thermodynamics, the pressure Poisson solve,
and the two-way ``precip_microphysics`` coupling) are backend-aware: they
read ``grid.xp``/``grid.backend`` (or, for ``precip_microphysics``, a plain
``xp`` module reference threaded in without a dependency on this module --
see ``precip_microphysics/state.py``) rather than hardcoding NumPy, so a
GPU-resolved ``Backend`` keeps the large 3-D fields resident on the device
across the whole time loop. The nucleation-lookup interpolation layer stays
CPU-side by design (documented, bounded scope -- see docs/architecture.md).

The detection state machine genuinely probes for a working CUDA/CuPy stack
and a fitting VRAM budget before returning a GPU backend. ``--device gpu``
fails loudly on any failure (never silently runs CPU math while claiming
GPU); ``--device auto`` falls back to CPU with a logged reason.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class BackendError(Exception):
    """Base class for backend-resolution failures.

    Each subclass names one of the required fail-loudly categories for
    ``--device gpu``: missing library, incompatible driver, CUDA unavailable,
    no GPU detected, insufficient memory, unsupported kernel.
    """
    category = "unknown"


class MissingLibraryError(BackendError):
    category = "missing_library"


class DriverIncompatibleError(BackendError):
    category = "driver_incompatible"


class CudaUnavailableError(BackendError):
    category = "cuda_unavailable"


class NoGpuDetectedError(BackendError):
    category = "no_gpu_detected"


class InsufficientMemoryError(BackendError):
    category = "insufficient_memory"


class UnsupportedKernelError(BackendError):
    category = "unsupported_kernel"
    """Raised from a specific GPU call site (not the up-front probe) when a
    particular operation/dtype/shape combination isn't supported -- must
    never be swallowed into a silent wrong result."""


@dataclass
class Backend:
    """Thin array-module wrapper threaded through Grid/FlowState/solvers.

    ``xp`` is ``numpy`` for the CPU backend and the ``cupy`` module for the
    GPU backend, so call sites that already hold a ``grid``/``state`` reach
    ``grid.xp``/``state.grid.xp`` and use it exactly like ``numpy``.
    """
    name: str                        # "cpu" | "gpu"
    xp: Any                          # numpy, or the cupy module
    sparse: Any = None               # scipy.sparse, or cupyx.scipy.sparse
    sparse_linalg: Any = None        # scipy.sparse.linalg, or cupyx.scipy.sparse.linalg
    fallback_reason: str | None = None   # set only when auto fell back gpu->cpu

    def asarray(self, a, dtype=None):
        return self.xp.asarray(a, dtype=dtype)

    def zeros(self, shape, dtype=float):
        return self.xp.zeros(shape, dtype=dtype)

    def to_cpu(self, a):
        """Return a plain NumPy array regardless of which backend produced it."""
        if self.name == "cpu":
            return a
        return a.get() if hasattr(a, "get") else np.asarray(a)

    def synchronize(self) -> None:
        if self.name == "gpu":
            self.xp.cuda.Stream.null.synchronize()

    def free_memory(self) -> None:
        """Release cached GPU memory pools. No-op on CPU."""
        if self.name == "gpu":
            self.xp.get_default_memory_pool().free_all_blocks()
            self.xp.get_default_pinned_memory_pool().free_all_blocks()

    def device_info(self) -> dict:
        if self.name == "cpu":
            return {
                "label": "CPU (%s, %d logical cores)" % (
                    platform.processor() or platform.machine(), os.cpu_count() or 1),
                "cpu_count": os.cpu_count(),
            }
        dev = self.xp.cuda.Device(0)
        free_b, total_b = dev.mem_info
        props = self.xp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props["name"]
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        return {
            "label": "GPU (%s)" % gpu_name,
            "gpu_name": gpu_name,
            "free_gb": free_b / 1e9,
            "total_gb": total_b / 1e9,
        }


def _cpu_backend() -> Backend:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    return Backend(name="cpu", xp=np, sparse=sp, sparse_linalg=spla)


def _probe_gpu():
    """Run the real GPU detection state machine.

    Returns the imported ``cupy`` module on success; raises a categorized
    :class:`BackendError` describing exactly which check failed otherwise.
    Order: library import -> device count -> driver/kernel probe -> (memory
    is checked separately by the caller, since it needs the requested size).
    """
    try:
        import cupy
    except ImportError as e:
        raise MissingLibraryError(
            "cupy is not installed. Install the optional GPU extra: "
            "pip install \"met_water_nucleation[gpu]\" (or `pip install "
            "cupy-cuda12x` directly) -- requires an NVIDIA GPU and driver.") from e

    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except Exception as e:
        raise CudaUnavailableError(
            "the CUDA runtime could not be queried (%s); the NVIDIA driver may "
            "be missing, disabled, or incompatible with the installed CuPy "
            "build." % e) from e
    if count == 0:
        raise NoGpuDetectedError(
            "cupy is installed and CUDA responded, but no CUDA device was "
            "found (getDeviceCount() == 0).")

    try:
        dev = cupy.cuda.Device(0)
        _ = dev.compute_capability
        _ = (cupy.zeros(4) + 1).sum()   # trivial kernel: forces context + init
        cupy.cuda.Stream.null.synchronize()
    except cupy.cuda.driver.CUDADriverError as e:
        raise DriverIncompatibleError(
            "the NVIDIA driver is incompatible with this CuPy/CUDA build "
            "(%s)." % e) from e
    except Exception as e:
        raise CudaUnavailableError(
            "GPU device 0 failed a basic kernel launch (%s)." % e) from e

    return cupy


def check_gpu_memory(required_gb: float, cupy_module) -> None:
    """Raise :class:`InsufficientMemoryError` if ``required_gb`` won't fit in
    the GPU's currently-free VRAM. Must run before any solver array is
    allocated on the device."""
    free_b, _total_b = cupy_module.cuda.Device(0).mem_info
    free_gb = free_b / 1e9
    if required_gb > free_gb:
        raise InsufficientMemoryError(
            "estimated field memory ~%.2f GB exceeds the ~%.2f GB currently "
            "free on the GPU. Reduce the grid resolution, use --precision "
            "float32, or run with --device cpu." % (required_gb, free_gb))


def _gpu_backend(required_gb: float | None) -> Backend:
    cupy = _probe_gpu()
    if required_gb is not None:
        check_gpu_memory(required_gb, cupy)
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as cspla
    return Backend(name="gpu", xp=cupy, sparse=csp, sparse_linalg=cspla)


def get_backend(device: str = "auto", *, required_gb: float | None = None,
                log: Callable[[str], None] = print) -> Backend:
    """Resolve the compute backend for a run.

    - ``device="cpu"``: always the CPU backend; no GPU probing at all.
    - ``device="gpu"``: returns a working GPU backend, or raises a
      categorized :class:`BackendError`/``NotImplementedError`` -- NEVER
      silently falls back to CPU.
    - ``device="auto"`` (default): tries GPU first; on ANY failure, logs the
      reason and returns the CPU backend instead.
    """
    device = (device or "auto").lower()
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError("device must be one of auto/cpu/gpu, got %r" % device)
    if device == "cpu":
        return _cpu_backend()
    if device == "gpu":
        return _gpu_backend(required_gb)
    # auto
    try:
        return _gpu_backend(required_gb)
    except (BackendError, NotImplementedError) as e:
        category = getattr(e, "category", "not_implemented")
        log("[backend] GPU unavailable (%s): %s -- falling back to CPU." % (category, e))
        backend = _cpu_backend()
        backend.fallback_reason = "%s: %s" % (category, e)
        return backend


__all__ = [
    "Backend", "BackendError", "MissingLibraryError", "DriverIncompatibleError",
    "CudaUnavailableError", "NoGpuDetectedError", "InsufficientMemoryError",
    "UnsupportedKernelError", "get_backend", "check_gpu_memory",
]
