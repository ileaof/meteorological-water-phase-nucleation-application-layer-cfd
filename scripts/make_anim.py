"""Build one MP4 animation per field from a run's plot_snapshot() PNG frames.

Thin CLI wrapper around ``meteorological_flow.animate`` -- the same logic
also backs the ``meteorological-flow --animate`` CLI flag, so a run made
with ``--animate`` and this script produce the same output. See that
module's docstring for the ffmpeg discovery/capability details.

Usage::

    python scripts/make_anim.py outputs/storm_stretched_fine_grid
    python scripts/make_anim.py outputs/storm_gpu --fields w S_w T --fps 8
    python scripts/make_anim.py outputs/storm_gpu --out-dir outputs/storm_gpu/figures
    python scripts/make_anim.py outputs/storm_gpu --ffmpeg "C:/path/to/modern/ffmpeg.exe"
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from meteorological_flow import animate as an  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="run output directory containing figures/ "
                   "(e.g. outputs/<run>)")
    p.add_argument("--fields", nargs="+", default=None,
                   help="fields to animate (default: every field with a "
                        "t-sequence in figures/, auto-discovered)")
    p.add_argument("--fps", type=float, default=6.0, help="output frame rate")
    p.add_argument("--qscale", type=int, default=3,
                   help="mpeg4 quality scale (only used with a limited/old ffmpeg "
                        "build, no h264 support), 1 (best/largest) .. 31 (worst/smallest)")
    p.add_argument("--out-dir", default=None,
                   help="where to write <field>_evolution.mp4 (default: run_dir itself)")
    p.add_argument("--ffmpeg", default=None,
                   help="path to a specific ffmpeg binary (default: auto-detect the "
                        "most capable one on PATH or in common install locations)")
    args = p.parse_args(argv)

    try:
        ffmpeg_bin, caps = an.resolve_ffmpeg(args.ffmpeg)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"using ffmpeg: {ffmpeg_bin} (h264={caps['h264']}, png={caps['png']})")

    figures_dir = os.path.join(args.run_dir, "figures")
    if not os.path.isdir(figures_dir):
        print(f"ERROR: {figures_dir} does not exist.", file=sys.stderr)
        return 1

    fields = args.fields or an.discover_fields(figures_dir)
    if not fields:
        print(f"ERROR: no time-sequence frames found in {figures_dir}.", file=sys.stderr)
        return 1

    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)

    rc = 0
    for field in fields:
        try:
            out_path = an.make_field_animation(args.run_dir, field, out_dir, args.fps,
                                               ffmpeg_bin, caps, args.qscale)
        except ValueError as e:
            print(f"  {field}: SKIPPED ({e})")
            continue
        except subprocess.CalledProcessError as e:
            print(f"  {field}: FAILED (ffmpeg exit {e.returncode})\n{e.stderr}")
            rc = 1
            continue
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  {field}: wrote {out_path} ({size_kb:.0f} KB)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
