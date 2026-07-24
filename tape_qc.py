"""Tape-to-file conversion QC: locate REAL, deterministic defects.

Tape captures introduce a known catalogue of defects whose signatures are
structural (content-independent), so they detect with near-zero false
alarms -- unlike content-relative "spatial anomaly" scoring. This scans
for the highest-value, most reliable ones using ffmpeg's own broadcast-QC
filters in a single decode pass per file (dense: every frame, fast:
native, no per-frame Python):

  frozen / repeated frames  (freezedetect) -- the #1 tape-capture artifact
                             (dropouts, splices, capture stalls)
  black / dropped frames    (blackdetect)

Each hit is reported with file + timestamp + duration, and a representative
frame is saved as evidence.

Usage:
  python tape_qc.py --files A.mxf B.mxf [--out out_tapeqc]
        [--freeze-noise 0.003] [--freeze-dur 0.2] [--black-dur 0.05]
        [--limit-seconds N]   # scan only the first N s (testing)
"""

import argparse
import os
import re
import subprocess

FREEZE_RE = re.compile(r"freeze_(start|end|duration):\s*([-\d.]+)")
BLACK_RE = re.compile(r"black_start:([\d.]+)\s+black_end:([\d.]+)\s+"
                      r"black_duration:([\d.]+)")


def run_qc(path, freeze_noise, freeze_dur, black_dur, limit=None):
    vf = (f"freezedetect=n={freeze_noise}:d={freeze_dur},"
          f"blackdetect=d={black_dur}:pic_th=0.98")
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info"]
    if limit:
        cmd += ["-t", str(limit)]
    cmd += ["-i", path, "-map", "0:v:0", "-vf", vf, "-an", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace")
    log = proc.stderr
    events = []

    # black segments
    for m in BLACK_RE.finditer(log):
        start, end, dur = float(m[1]), float(m[2]), float(m[3])
        events.append(("black", start, end, dur))

    # freeze segments: freezedetect emits freeze_start then freeze_end;
    # pair them in order
    pend = None
    for m in re.finditer(r"freeze_(start|end|duration):\s*([-\d.]+)", log):
        kind, val = m[1], float(m[2])
        if kind == "start":
            pend = val
        elif kind == "end" and pend is not None:
            events.append(("frozen", pend, val, val - pend))
            pend = None
    events.sort(key=lambda e: e[1])
    return events


def save_frame(path, t, outpath):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-ss", f"{t:.3f}",
         "-i", path, "-map", "0:v:0", "-vf", "yadif", "-frames:v", "1",
         "-pix_fmt", "rgb24", "-y", outpath], capture_output=True)


def hms(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", default="out_tapeqc")
    ap.add_argument("--freeze-noise", type=float, default=0.003)
    ap.add_argument("--freeze-dur", type=float, default=0.2)
    ap.add_argument("--black-dur", type=float, default=0.05)
    ap.add_argument("--limit-seconds", type=int, default=None)
    ap.add_argument("--save", action="store_true",
                    help="save an evidence frame per event")
    args = ap.parse_args()

    if args.save:
        os.makedirs(args.out, exist_ok=True)
    grand = 0
    for path in args.files:
        name = os.path.basename(path)
        print(f"\n=== {name} ===", flush=True)
        events = run_qc(path, args.freeze_noise, args.freeze_dur,
                        args.black_dur, args.limit_seconds)
        if not events:
            print("  no frozen/black defects detected")
            continue
        for i, (kind, start, end, dur) in enumerate(events):
            print(f"  {kind:<7} {hms(start)} -> {hms(end)}  ({dur:.2f}s)",
                  flush=True)
            if args.save:
                save_frame(path, start + dur / 2,
                           os.path.join(args.out,
                                        f"{name}_{i:03d}_{kind}_"
                                        f"{int(start)}s.png"))
        grand += len(events)
        print(f"  -> {len(events)} defect segments")
    print(f"\nTOTAL: {grand} defect segments across {len(args.files)} files")


if __name__ == "__main__":
    main()
