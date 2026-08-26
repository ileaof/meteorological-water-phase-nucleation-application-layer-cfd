"""Build a combined side-by-side panel animation (MP4 and/or GIF) from a
run's plot_snapshot() PNG frames -- e.g. the w | S_w | q_v panel embedded in
the README (docs/media/storm_panel_w_Sw_qv.gif, itself downsampled from
outputs/storm_stretched_fine_grid/storm_panel_w_Sw_qv.mp4).

Thin CLI wrapper around ``meteorological_flow.animate`` -- the same logic
also backs the ``meteorological-flow --animate`` CLI flag, so a run made
with ``--animate`` and this script produce the same output. See that
module's docstring for the ffmpeg discovery/capability details.

Usage::

    python scripts/make_panel.py outputs/storm_stretched_fine_grid --gif
    python scripts/make_panel.py outputs/storm_stretched_fine_grid --fields w S_w q_v --gif
    python scripts/make_panel.py outputs/storm_gpu --fields w S_i --fps 8 --gif --gif-width 820
    python scripts/make_panel.py outputs/storm_gpu --no-mp4 --gif   # GIF only, no ffmpeg needed
    python scripts/make_panel.py outputs/storm_gpu --ffmpeg "C:/path/to/modern/ffmpeg.exe"
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
    p.add_argument("run_dir", help="run output directory containing figures/")
    p.add_argument("--fields", nargs="+", default=list(an.DEFAULT_PANEL_FIELDS),
                   help="fields to place side by side, in order (default: "
                        "w S_w q_v, matching docs/media/storm_panel_w_Sw_qv.gif)")
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--qscale", type=int, default=3,
                   help="mpeg4 quality scale (only with a limited/old ffmpeg build), "
                        "1 (best/largest) .. 31 (worst/smallest)")
    p.add_argument("--out", default=None,
                   help="MP4 output path (default: <run_dir>/storm_panel_<fields>.mp4)")
    p.add_argument("--gif", action="store_true", help="also write a GIF")
    p.add_argument("--gif-width", type=int, default=820,
                   help="downscale the GIF to this width (matches the README embed width)")
    p.add_argument("--no-mp4", action="store_true", help="skip MP4 (GIF only)")
    p.add_argument("--max-mp4-width", type=int, default=1920,
                   help="downscale the MP4 composite if wider than this AND only a "
                        "limited/old ffmpeg is available (works around its hardcoded "
                        "swscale max width, e.g. 2048px) -- not applied with a capable ffmpeg")
    p.add_argument("--ffmpeg", default=None,
                   help="path to a specific ffmpeg binary (default: auto-detect the "
                        "most capable one on PATH or in common install locations)")
    args = p.parse_args(argv)

    try:
        ffmpeg_bin, caps = an.resolve_ffmpeg(args.ffmpeg)
    except FileNotFoundError as e:
        print(f"ERROR: {e} (use --no-mp4 --gif for Pillow-only GIF output without ffmpeg)",
              file=sys.stderr)
        return 1
    print(f"using ffmpeg: {ffmpeg_bin} (h264={caps['h264']}, palette={caps['palette']})")

    gif_path = os.path.splitext(args.out)[0] + ".gif" if args.out else None
    try:
        result = an.make_panel_animation(
            args.run_dir, args.fields, args.run_dir, args.fps, ffmpeg_bin, caps,
            mp4=not args.no_mp4, gif=args.gif, gif_width=args.gif_width,
            qscale=args.qscale, max_mp4_width=args.max_mp4_width,
            mp4_path=args.out, gif_path=gif_path)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ffmpeg failed (exit {e.returncode}):\n{e.stderr}", file=sys.stderr)
        return 1

    print(f"composited {result['n_frames']} panel frames ({len(args.fields)} fields side by side)")
    if result["mp4"]:
        print(f"wrote {result['mp4']} ({os.path.getsize(result['mp4']) / 1024:.0f} KB)")
    if result["gif"]:
        print(f"wrote {result['gif']} ({os.path.getsize(result['gif']) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
