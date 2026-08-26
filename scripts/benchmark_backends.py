"""Reproducible CPU/GPU benchmark for the meteorological_flow solver.

Runs the same short simulation at three problem sizes (the repo's own named
presets: ``fast``/``recommended``/``advanced`` -- see
``meteorological_flow.config.PRESETS``) on each requested device, with an
explicit warmup run (discarded) before timing to separate out CuPy's
first-call kernel-compile cost, then records timing/memory/speedup and writes
a JSON + Markdown report.

Usage::

    python scripts/benchmark_backends.py
    python scripts/benchmark_backends.py --sizes small,medium --device both
    python scripts/benchmark_backends.py --sizes large --device gpu --repeats 3
    python scripts/benchmark_backends.py --output outputs/benchmarks/my_run.json

Honesty note: CPU peak-memory reporting uses ``resource.getrusage`` (matches
``simulation.py``'s own ``_mem_kb`` helper) and is therefore ``None`` on
Windows, where that module doesn't exist. GPU peak memory is the CuPy default
memory pool's ``used_bytes()`` sampled immediately after the run -- a lower
bound on true peak (it does not track the running maximum via a memory hook),
not necessarily the maximum reached mid-run. Per-step transfer time is NOT
separately instrumented (would need to intercept every ``Backend.to_cpu``
call site) -- out of scope for this script; see docs/architecture.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from meteorological_flow import config as cfgmod  # noqa: E402
from meteorological_flow.backend import BackendError, get_backend  # noqa: E402
from meteorological_flow.simulation import Simulation, _mem_kb  # noqa: E402

SIZE_PRESETS = {"small": "fast", "medium": "recommended", "large": "advanced"}
_WARMUP_DURATION = 1.0     # s of simulated time; just enough to exercise every kernel once
_BENCH_DURATION = 5.0      # s of simulated time for the timed runs


def _build_cfg(preset: str, duration: float):
    cfg = cfgmod.SimulationConfig()
    cfg = cfgmod.apply_overrides(cfg, preset=preset, duration=duration,
                                 no_microphysics=True)   # isolate the core dynamical core
    cfg.output.format = []; cfg.output.figures = []; cfg.output.restart = False
    cfg.output.outdir = "outputs/_benchmark_scratch"
    return cfg


def _gpu_mem_used_bytes() -> float | None:
    try:
        import cupy
        return float(cupy.get_default_memory_pool().used_bytes())
    except Exception:
        return None


def _run_once(cfg, backend, duration: float) -> dict:
    cfg = cfgmod.apply_overrides(cfg, duration=duration)
    t0 = time.perf_counter()
    sim = Simulation(cfg, backend=backend)
    t1 = time.perf_counter()
    report = sim.run()
    t2 = time.perf_counter()
    if backend.name == "gpu":
        backend.synchronize()
    peak_kb = _mem_kb()
    return {
        "init_time_s": t1 - t0,
        "main_loop_time_s": t2 - t1,
        "total_time_s": t2 - t0,
        "n_steps": report["n_steps"],
        "per_step_time_s": (t2 - t1) / max(report["n_steps"], 1),
        "peak_host_memory_mb": (peak_kb / 1024.0) if peak_kb else None,
        "peak_gpu_memory_mb": (_gpu_mem_used_bytes() / 1e6) if backend.name == "gpu" else None,
    }


def _bench_one(size_name: str, preset: str, device: str, repeats: int,
               compute_threads: int | None, log) -> dict | None:
    cfg = _build_cfg(preset, _WARMUP_DURATION)
    mem_est = cfgmod.estimate_memory_gb(cfg)
    try:
        backend = get_backend(device, required_gb=mem_est, log=log)
    except BackendError as e:
        log(f"[benchmark] skipping {size_name}/{device}: {e}")
        return None
    if device != "auto" and backend.name != device:
        log(f"[benchmark] skipping {size_name}/{device}: resolved to {backend.name} instead")
        return None

    if compute_threads:
        from threadpoolctl import threadpool_limits
        ctx = threadpool_limits(limits=compute_threads)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        log(f"[benchmark] {size_name} ({preset}) on {backend.name}: warmup...")
        _run_once(cfg, backend, _WARMUP_DURATION)   # discarded: JIT/kernel-compile cost
        runs = []
        for r in range(repeats):
            log(f"[benchmark] {size_name} ({preset}) on {backend.name}: run {r + 1}/{repeats}...")
            runs.append(_run_once(cfg, backend, _BENCH_DURATION))

    geom = cfgmod.geometry(cfg)
    best = min(runs, key=lambda r: r["total_time_s"])
    return {
        "size": size_name, "preset": preset, "device": backend.name,
        "device_label": backend.device_info()["label"],
        "cpu_threads": compute_threads or os.cpu_count(),
        "grid": f"{geom['Nx']}x{geom['Ny']}x{geom['Nz']}", "n_cells": geom["n_cells"],
        "precision": cfg.physics.precision,
        "repeats": repeats,
        "best_run": best,
        "all_runs": runs,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", default="small,medium,large",
                   help="comma-separated subset of: small, medium, large")
    p.add_argument("--device", choices=("auto", "cpu", "gpu", "both"), default="both",
                   help="'both' runs cpu then gpu for each size (gpu rows skipped "
                        "gracefully if unavailable)")
    p.add_argument("--repeats", type=int, default=3, help="timed repeats per (size, device)")
    p.add_argument("--compute-threads", type=int, default=None, dest="compute_threads")
    p.add_argument("--output", default=None, help="output JSON path "
                   "(default: outputs/benchmarks/bench_<timestamp>.json)")
    args = p.parse_args(argv)

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    for s in sizes:
        if s not in SIZE_PRESETS:
            p.error(f"unknown size {s!r}; choices: {sorted(SIZE_PRESETS)}")
    devices = ("cpu", "gpu") if args.device == "both" else (args.device,)

    def log(msg):
        print(msg, flush=True)

    rows = []
    for size_name in sizes:
        for device in devices:
            row = _bench_one(size_name, SIZE_PRESETS[size_name], device, args.repeats,
                             args.compute_threads, log)
            if row is not None:
                rows.append(row)

    # post-hoc speedup: GPU best-run vs the CPU best-run at the same size
    by_size = {}
    for row in rows:
        by_size.setdefault(row["size"], {})[row["device"]] = row
    for size_name, by_dev in by_size.items():
        if "cpu" in by_dev and "gpu" in by_dev:
            cpu_t = by_dev["cpu"]["best_run"]["total_time_s"]
            gpu_t = by_dev["gpu"]["best_run"]["total_time_s"]
            by_dev["gpu"]["speedup_vs_cpu"] = cpu_t / gpu_t if gpu_t > 0 else None

    ts = time.strftime("%Y%m%dT%H%M%S")
    out_dir = "outputs/benchmarks"
    os.makedirs(out_dir, exist_ok=True)
    json_path = args.output or os.path.join(out_dir, f"bench_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    md_path = os.path.splitext(json_path)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(_to_markdown(rows))

    print(f"\nWrote {json_path}\nWrote {md_path}")
    print(_to_markdown(rows))
    return 0


def _to_markdown(rows: list) -> str:
    lines = ["| size | grid | device | threads | precision | init [s] | loop [s] | "
            "per-step [ms] | total [s] | peak host [MB] | peak GPU [MB] | speedup |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        b = row["best_run"]
        speedup = row.get("speedup_vs_cpu")
        lines.append("| %s | %s | %s | %s | %s | %.3f | %.3f | %.3f | %.3f | %s | %s | %s |" % (
            row["size"], row["grid"], row["device"], row["cpu_threads"], row["precision"],
            b["init_time_s"], b["main_loop_time_s"], b["per_step_time_s"] * 1000.0,
            b["total_time_s"],
            ("%.1f" % b["peak_host_memory_mb"]) if b["peak_host_memory_mb"] else "n/a",
            ("%.1f" % b["peak_gpu_memory_mb"]) if b["peak_gpu_memory_mb"] else "n/a",
            ("%.2fx" % speedup) if speedup else "-"))
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
