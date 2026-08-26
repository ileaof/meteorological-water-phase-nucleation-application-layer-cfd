"""Tests for meteorological_flow.animate (the --animate CLI flag's backing
module, shared with scripts/make_anim.py / scripts/make_panel.py) and its
CLI/config wiring.
"""
from __future__ import annotations

import os

import pytest

from meteorological_flow import animate as an
from meteorological_flow.cli import _maybe_animate, build_argparser
from meteorological_flow.config import SimulationConfig, apply_overrides


def _ffmpeg_available() -> bool:
    try:
        an.resolve_ffmpeg()
        return True
    except FileNotFoundError:
        return False


def test_cli_parses_animate_flag():
    args = build_argparser().parse_args(["--animate"])
    assert args.animate is True
    args2 = build_argparser().parse_args([])
    assert args2.animate is False


def test_apply_overrides_sets_animate():
    cfg = apply_overrides(SimulationConfig(), animate=True)
    assert cfg.output.animate is True
    cfg2 = apply_overrides(SimulationConfig())
    assert cfg2.output.animate is False


def test_discover_fields_strips_directory_prefix(tmp_path):
    # regression test: discover_fields used to search the frame-suffix regex
    # against the FULL PATH but slice the BASENAME with that match offset,
    # so it returned whole filenames (e.g. "S_i_t0.png") instead of clean
    # field names ("S_i") whenever figures_dir was non-trivial (i.e. always,
    # in practice) -- never caught earlier because every prior manual test
    # happened to pass an explicit --fields, bypassing discovery entirely.
    figures = tmp_path / "figures"
    figures.mkdir()
    for name in ("S_i_t0.png", "S_i_t1.png", "w_t0.png"):
        (figures / name).write_bytes(b"")
    assert an.discover_fields(str(figures)) == ["S_i", "w"]


def test_frames_for_field_sorted_by_time(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    for t in (30, 0, 15):
        (figures / f"w_t{t}.png").write_bytes(b"")
    frames = an.frames_for_field(str(figures), "w")
    assert [t for t, _p in frames] == [0, 15, 30]


def test_animate_run_raises_when_no_figures_dir(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        an.animate_run(str(tmp_path))


def test_animate_run_raises_when_figures_dir_empty(tmp_path):
    (tmp_path / "figures").mkdir()
    with pytest.raises(ValueError, match="no time-sequence frames"):
        an.animate_run(str(tmp_path))


def test_manual_commands_reference_the_scripts_and_run_dir():
    cmds = an.manual_commands("outputs/myrun")
    assert any("make_anim.py" in c and "outputs/myrun" in c for c in cmds)
    assert any("make_panel.py" in c and "outputs/myrun" in c for c in cmds)


def test_maybe_animate_falls_back_cleanly_without_ffmpeg(tmp_path, monkeypatch, capsys):
    # the core of the user's request: even with NO ffmpeg available anywhere,
    # --animate must never look like the simulation itself failed -- it
    # prints the exact manual commands instead.
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "w_t0.png").write_bytes(b"")
    monkeypatch.setattr(an, "resolve_ffmpeg",
                        lambda explicit=None: (_ for _ in ()).throw(FileNotFoundError("no ffmpeg")))
    cfg = SimulationConfig()
    cfg.output.outdir = str(tmp_path)
    cfg.output.animate = True
    _maybe_animate(cfg)   # must not raise
    out = capsys.readouterr().out
    assert "could not build animations automatically" in out
    assert "make_anim.py" in out and "make_panel.py" in out


def test_maybe_animate_falls_back_cleanly_with_no_figures(tmp_path, capsys):
    cfg = SimulationConfig()
    cfg.output.outdir = str(tmp_path)   # no figures/ subdir at all
    cfg.output.animate = True
    _maybe_animate(cfg)   # must not raise
    out = capsys.readouterr().out
    assert "could not build animations automatically" in out
    assert "make_anim.py" in out and "make_panel.py" in out


@pytest.mark.skipif(not _ffmpeg_available(), reason="no ffmpeg available in this environment")
def test_animate_run_end_to_end_with_real_frames(tmp_path):
    from PIL import Image
    figures = tmp_path / "figures"
    figures.mkdir()
    # ffmpeg's palettegen/paletteuse GIF filter chain needs a few real frames
    # to work with -- an overly-degenerate synthetic fixture (e.g. 3 frames
    # at 64x48) reproducibly fails ("Error writing trailer: Invalid
    # argument") even though every real run in this repo (hundreds of much
    # larger frames) encodes fine; 8 frames at 200x150 is enough margin.
    for field in ("w", "S_w", "q_v"):
        for t in range(8):
            Image.new("RGB", (200, 150), (t * 20, 50, 80)).save(figures / f"{field}_t{t}.png")
    result = an.animate_run(str(tmp_path), fps=2.0)
    assert not isinstance(result["fields"]["w"], Exception)
    assert os.path.isfile(result["fields"]["w"])
    assert result["panel"] and not isinstance(result["panel"], Exception)
    assert os.path.isfile(result["panel"]["mp4"])
    assert os.path.isfile(result["panel"]["gif"])
