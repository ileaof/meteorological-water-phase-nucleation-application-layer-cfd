"""Post-run animation builder: turns plotting.plot_snapshot()'s per-timestep
PNG frames into per-field MP4s and a combined side-by-side panel (MP4 + GIF).

This is the library backing both the ``--animate`` CLI flag (see ``cli.py``)
and the standalone ``scripts/make_anim.py`` / ``scripts/make_panel.py`` --
the same logic either way, so a run made with ``--animate`` and a later
manual ``python scripts/make_anim.py <run>`` produce the same output.

ffmpeg is a runtime dependency of this module only (not the physics/CFD
code), and is deliberately not vendored/pip-installed: :func:`resolve_ffmpeg`
finds the most capable ffmpeg it can (PATH, then a few common extra install
locations, e.g. WinGet) and adapts the encoding pipeline to what it can
actually do -- some machines only have a very old ffmpeg on PATH (e.g. one
bundled with an older Tecplot 360 install: no PNG decoder, no libx264, no
palettegen/paletteuse GIF filters, older flag spellings only); this module
still produces a working (lower-quality) MP4 with that build.
"""
from __future__ import annotations

import functools
import glob
import os
import re
import shutil
import subprocess
import tempfile

from PIL import Image

_FRAME_RE = re.compile(r"_t(\d+)\.png$")

DEFAULT_PANEL_FIELDS = ("w", "S_w", "q_v")   # matches docs/media/storm_panel_w_Sw_qv.gif

# extra locations to scan if PATH's ffmpeg turns out to be a limited build
# (e.g. WinGet installs into a per-user, not-on-PATH-by-default location)
_EXTRA_FFMPEG_ROOTS = [
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"),
]


# ---------------------------------------------------------------------------
# ffmpeg discovery / capability probing
# ---------------------------------------------------------------------------
def _ffmpeg_candidates():
    seen = set()
    on_path = shutil.which("ffmpeg")
    if on_path:
        seen.add(os.path.normcase(on_path))
        yield on_path
    for root in _EXTRA_FFMPEG_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if "ffmpeg.exe" in files:
                p = os.path.join(dirpath, "ffmpeg.exe")
                if os.path.normcase(p) not in seen:
                    seen.add(os.path.normcase(p))
                    yield p


@functools.lru_cache(maxsize=None)
def probe_ffmpeg(ffmpeg_bin: str) -> dict:
    """{'h264': bool, 'palette': bool, 'png': bool} for one ffmpeg binary."""
    def _run(*args):
        try:
            return subprocess.run([ffmpeg_bin, *args], capture_output=True,
                                  text=True, timeout=15).stdout
        except Exception:
            return ""
    enc = _run("-encoders")
    dec = _run("-decoders")
    filt = _run("-filters")
    return {
        "h264": "libx264" in enc,
        "png": "png" in dec.lower(),
        "palette": "palettegen" in filt and "paletteuse" in filt,
    }


def resolve_ffmpeg(explicit: str | None = None) -> tuple[str, dict]:
    """Return (path, capabilities) for the best available ffmpeg.

    Prefers an explicit path if given (still probed for capabilities); else
    scans PATH plus a few common extra install locations and picks the most
    capable one found (h264 + palette support preferred over a limited/old
    build). Raises FileNotFoundError if none can be found at all.
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"ffmpeg path not found: {explicit}")
        return explicit, probe_ffmpeg(explicit)
    best = best_caps = None
    for cand in _ffmpeg_candidates():
        caps = probe_ffmpeg(cand)
        if best is None:
            best, best_caps = cand, caps
        if caps["h264"] and caps["palette"]:
            return cand, caps
    if best is None:
        raise FileNotFoundError(
            "no ffmpeg found on PATH or in common install locations "
            "(pass an explicit ffmpeg path)")
    return best, best_caps


def _encode_video(ffmpeg_bin: str, caps: dict, frame_pattern: str, out_path: str,
                  fps: float, qscale: int = 3) -> None:
    if caps["h264"]:
        cmd = [ffmpeg_bin, "-y", "-r", str(fps), "-i", frame_pattern,
              "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_path]
    else:
        cmd = [ffmpeg_bin, "-y", "-r", str(fps), "-i", frame_pattern,
              "-vcodec", "mpeg4", "-qscale", str(qscale), "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _encode_gif_ffmpeg(ffmpeg_bin: str, frame_pattern: str, out_path: str,
                       fps: float, width: int | None = None) -> None:
    scale = f"scale={width}:-1:flags=lanczos," if width else ""
    vf = f"{scale}fps={fps},split[a][b];[a]palettegen[p];[b][p]paletteuse"
    cmd = [ffmpeg_bin, "-y", "-i", frame_pattern, "-vf", vf, out_path]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _write_gif_pillow(frames: list[Image.Image], out_path: str, fps: float,
                      width: int | None) -> None:
    if width and frames[0].width != width:
        frames = [im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
                 for im in frames]
    duration_ms = int(1000 / fps)
    quantized = [im.convert("P", palette=Image.ADAPTIVE) for im in frames]
    quantized[0].save(out_path, save_all=True, append_images=quantized[1:],
                      duration=duration_ms, loop=0, optimize=True)


# ---------------------------------------------------------------------------
# frame discovery
# ---------------------------------------------------------------------------
def discover_fields(figures_dir: str) -> list[str]:
    """Every field with a time-sequence of frames (name_t<int>.png), sorted."""
    fields = set()
    for path in glob.glob(os.path.join(figures_dir, "*_t*.png")):
        name = os.path.basename(path)
        m = _FRAME_RE.search(name)
        if m:
            fields.add(name[: m.start()])
    return sorted(fields)


def frames_for_field(figures_dir: str, field: str) -> list[tuple[int, str]]:
    """[(time, path), ...] sorted by simulated time for one field."""
    out = []
    for path in glob.glob(os.path.join(figures_dir, f"{field}_t*.png")):
        m = _FRAME_RE.search(path)
        if m:
            out.append((int(m.group(1)), path))
    out.sort(key=lambda pair: pair[0])
    return out


def _common_canvas_size(paths: list[str]) -> tuple[int, int]:
    w = h = 0
    for p in paths:
        with Image.open(p) as im:
            w = max(w, im.width)
            h = max(h, im.height)
    # h264/yuv420p requires even width and height (4:2:0 chroma subsampling);
    # round up rather than down so no frame content gets cropped.
    return w + (w % 2), h + (h % 2)


def _write_frame_sequence(frames: list[tuple[int, str]], size: tuple[int, int],
                          tmp_dir: str, ext: str, bg=(255, 255, 255)) -> str:
    for i, (_t, path) in enumerate(frames):
        with Image.open(path) as im:
            canvas = Image.new("RGB", size, bg)
            canvas.paste(im.convert("RGB"), (0, 0))
            canvas.save(os.path.join(tmp_dir, f"frame_{i:04d}.{ext}"))
    return os.path.join(tmp_dir, f"frame_%04d.{ext}")


# ---------------------------------------------------------------------------
# per-field animation
# ---------------------------------------------------------------------------
def make_field_animation(run_dir: str, field: str, out_dir: str, fps: float,
                         ffmpeg_bin: str, caps: dict, qscale: int = 3) -> str:
    """Build <out_dir>/<field>_evolution.mp4; returns the output path."""
    figures_dir = os.path.join(run_dir, "figures")
    frames = frames_for_field(figures_dir, field)
    if not frames:
        raise ValueError(f"no frames found for field {field!r} in {figures_dir}")
    size = _common_canvas_size([p for _t, p in frames])
    out_path = os.path.join(out_dir, f"{field}_evolution.mp4")
    ext = "png" if caps["png"] else "bmp"
    with tempfile.TemporaryDirectory(prefix=f"animate_{field}_") as tmp_dir:
        pattern = _write_frame_sequence(frames, size, tmp_dir, ext)
        _encode_video(ffmpeg_bin, caps, pattern, out_path, fps, qscale)
    return out_path


# ---------------------------------------------------------------------------
# combined side-by-side panel
# ---------------------------------------------------------------------------
def _composite_panel(entries: list[tuple[str, tuple[int, int]]], height: int,
                     bg=(255, 255, 255)) -> Image.Image:
    """Horizontally concatenate one frame per field, each padded onto its own
    field's canvas width (not a shared global width) so a narrower field
    doesn't leave a blank gap before the next one; all slots share ``height``."""
    total_w = sum(size[0] for _p, size in entries)
    panel = Image.new("RGB", (total_w, height), bg)
    x = 0
    for p, size in entries:
        with Image.open(p) as im:
            canvas = Image.new("RGB", (size[0], height), bg)
            canvas.paste(im.convert("RGB"), (0, 0))
            panel.paste(canvas, (x, 0))
        x += size[0]
    return panel


def build_panel_frames(run_dir: str, fields: list[str]) -> list[Image.Image]:
    """One composited panel image per simulated time common to every field."""
    figures_dir = os.path.join(run_dir, "figures")
    per_field = {f: frames_for_field(figures_dir, f) for f in fields}
    missing = [f for f, fr in per_field.items() if not fr]
    if missing:
        raise ValueError(f"no frames found for field(s): {missing}")
    # each field gets its own canvas width (sized to that field's own frames
    # only) -- a field with narrower plots (e.g. fewer colorbar tick-label
    # digits) must not be padded out to match a wider field, or the padding
    # shows up as a visible gap between adjacent panels.
    sizes = {f: _common_canvas_size([p for _t, p in per_field[f]]) for f in fields}
    height = max(size[1] for size in sizes.values())
    common_times = {t for t, _p in per_field[fields[0]]}
    for f in fields[1:]:
        common_times &= {t for t, _p in per_field[f]}
    times = sorted(common_times)
    if not times:
        raise ValueError("no common timestamps across the requested fields")
    by_field_by_time = {f: dict(per_field[f]) for f in fields}
    return [
        _composite_panel([(by_field_by_time[f][t], sizes[f]) for f in fields], height)
        for t in times
    ]


def make_panel_animation(run_dir: str, fields: list[str], out_dir: str, fps: float,
                         ffmpeg_bin: str, caps: dict, mp4: bool = True, gif: bool = True,
                         gif_width: int = 820, qscale: int = 3, max_mp4_width: int = 1920,
                         mp4_path: str | None = None, gif_path: str | None = None) -> dict:
    """Build the combined side-by-side panel MP4 and/or GIF.

    Returns {'mp4': path_or_None, 'gif': path_or_None, 'n_frames': int}.
    """
    frames = build_panel_frames(run_dir, fields)
    tag = "_".join(fields)
    out_mp4 = mp4_path or os.path.join(out_dir, f"storm_panel_{tag}.mp4")
    out_gif = gif_path or os.path.join(out_dir, f"storm_panel_{tag}.gif")
    result = {"mp4": None, "gif": None, "n_frames": len(frames)}

    # ffmpeg-encoded frames (BMP/PNG on disk) are only needed for the MP4
    # path, or for the GIF path when ffmpeg's own palette filter is used;
    # a Pillow-only GIF (no MP4 requested, no palette support) needs neither.
    need_frame_files = mp4 or (gif and caps["palette"])
    tmp_ctx = tempfile.TemporaryDirectory(prefix="animate_panel_") if need_frame_files else None
    try:
        pattern = None
        if need_frame_files:
            tmp_dir = tmp_ctx.name
            ext = "png" if caps["png"] else "bmp"
            mp4_frames = frames
            if not caps["h264"] and frames[0].width > max_mp4_width:
                # old-build swscale width limit workaround (h264 has no such limit)
                scale = max_mp4_width / frames[0].width
                size = (max_mp4_width, round(frames[0].height * scale))
                mp4_frames = [im.resize(size, Image.LANCZOS) for im in frames]
            pattern = os.path.join(tmp_dir, f"frame_%04d.{ext}")
            for i, im in enumerate(mp4_frames):
                im.save(os.path.join(tmp_dir, f"frame_{i:04d}.{ext}"))

        if mp4:
            _encode_video(ffmpeg_bin, caps, pattern, out_mp4, fps, qscale)
            result["mp4"] = out_mp4

        if gif:
            if caps["palette"]:
                _encode_gif_ffmpeg(ffmpeg_bin, pattern, out_gif, fps, gif_width)
            else:
                _write_gif_pillow(frames, out_gif, fps, gif_width)
            result["gif"] = out_gif
    finally:
        if tmp_ctx:
            tmp_ctx.cleanup()
    return result


# ---------------------------------------------------------------------------
# high-level orchestration + manual fallback commands
# ---------------------------------------------------------------------------
def manual_commands(run_dir: str, fields: list[str] | None = None,
                    panel_fields=DEFAULT_PANEL_FIELDS, fps: float = 6.0) -> list[str]:
    """The exact ``python scripts/...`` commands equivalent to
    :func:`animate_run`, for the end-of-run fallback message."""
    field_arg = f" --fields {' '.join(fields)}" if fields else ""
    panel_arg = " ".join(panel_fields)
    return [
        f"python scripts/make_anim.py {run_dir}{field_arg} --fps {fps:g}",
        f"python scripts/make_panel.py {run_dir} --fields {panel_arg} --gif --fps {fps:g}",
    ]


def animate_run(run_dir: str, fields: list[str] | None = None,
                panel_fields=DEFAULT_PANEL_FIELDS, fps: float = 6.0,
                gif: bool = True, gif_width: int = 820,
                ffmpeg_bin: str | None = None, out_dir: str | None = None) -> dict:
    """Build one MP4 per field plus the combined panel (MP4 + GIF) for a run.

    Returns ``{'ffmpeg': path, 'caps': {...}, 'fields': {field: path_or_exc},
    'panel': dict_or_exc_or_None}``. A single field's (or the panel's)
    failure is caught and returned as the exception object rather than
    raised, so one bad field doesn't abort the rest.

    Raises ``FileNotFoundError`` if no ffmpeg can be found at all, or
    ``ValueError`` if the run has no ``figures/`` frames to animate --
    callers (e.g. ``cli.py``'s ``--animate``) should catch these two and fall
    back to printing :func:`manual_commands`.
    """
    figures_dir = os.path.join(run_dir, "figures")
    if not os.path.isdir(figures_dir):
        raise ValueError(f"{figures_dir} does not exist -- were figure snapshots "
                         f"(output.figures, e.g. 'slices') enabled for this run?")
    available_fields = discover_fields(figures_dir)
    if not available_fields:
        raise ValueError(f"no time-sequence frames found in {figures_dir}")

    ffmpeg_path, caps = resolve_ffmpeg(ffmpeg_bin)

    out_dir = out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    target_fields = fields or available_fields

    field_results: dict[str, str | Exception] = {}
    for field in target_fields:
        try:
            field_results[field] = make_field_animation(
                run_dir, field, out_dir, fps, ffmpeg_path, caps)
        except (ValueError, subprocess.CalledProcessError) as e:
            field_results[field] = e

    panel_result = None
    usable_panel_fields = [f for f in panel_fields if f in available_fields]
    if len(usable_panel_fields) >= 2:
        try:
            panel_result = make_panel_animation(
                run_dir, usable_panel_fields, out_dir, fps, ffmpeg_path, caps,
                gif=gif, gif_width=gif_width)
        except (ValueError, subprocess.CalledProcessError) as e:
            panel_result = e

    return {"ffmpeg": ffmpeg_path, "caps": caps, "fields": field_results, "panel": panel_result}


__all__ = [
    "DEFAULT_PANEL_FIELDS",
    "animate_run",
    "build_panel_frames",
    "discover_fields",
    "frames_for_field",
    "make_field_animation",
    "make_panel_animation",
    "manual_commands",
    "probe_ffmpeg",
    "resolve_ffmpeg",
]
