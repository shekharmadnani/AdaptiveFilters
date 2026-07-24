"""Scan real video files for spatial-discontinuity errors.

Samples frame-bursts across each file (bursts give the consecutive
frames the temporal detector needs), calibrates the detector on the
content's OWN frames -- robust median/MAD tolerates the sparse errors we
are hunting -- then flags frames whose error-block fraction exceeds a
clean-calibrated threshold, reporting file + timestamp and saving
heatmaps for the worst offenders.

Handles broadcast masters: any ffmpeg-decodable input, deinterlaced on
the fly; frames are never fully copied -- only sampled bursts are pulled.

Usage:
  python scan_video.py --files A.mxf B.mxf [--weights models/wiener4_dvc.pt]
        [--seeks 15] [--burst 4] [--top 12] [--out out_scan]
"""

import argparse
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

from adaptive_filters.detect import SpatialErrorDetector, ROBUST_CHANNELS
from adaptive_filters.io import ffprobe_dims


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def extract_burst(path, t, burst, tmpdir, tag):
    """Deinterlace + decode `burst` consecutive frames starting at time t.
    Returns a list of (H, W, 3) uint8 luma-first RGB frames (or [])."""
    pat = os.path.join(tmpdir, f"{tag}_%03d.png")
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-ss", f"{t:.3f}",
           "-i", path, "-map", "0:v:0", "-vf", "yadif",
           "-frames:v", str(burst), "-pix_fmt", "rgb24", "-y", pat]
    subprocess.run(cmd, capture_output=True)
    frames = []
    for i in range(1, burst + 1):
        fp = os.path.join(tmpdir, f"{tag}_{i:03d}.png")
        if os.path.exists(fp):
            # keep uint8 in RAM (the detector converts per frame); float64
            # would be ~50 MB/frame and blow up memory over a long scan
            frames.append(np.asarray(Image.open(fp), dtype=np.uint8))
            os.remove(fp)
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--weights", default=os.path.join("models",
                                                      "wiener4_dvc.pt"))
    ap.add_argument("--seeks", type=int, default=15)
    ap.add_argument("--burst", type=int, default=4)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default="out_scan")
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--channels", default="robust",
                    choices=["robust", "all"],
                    help="robust = gen-4 residual/damage-map + seam only "
                         "(survives broadcast crowd texture + camera pans); "
                         "all = the full 8-channel set (film content)")
    args = ap.parse_args()

    chan = ROBUST_CHANNELS if args.channels == "robust" else None
    det = SpatialErrorDetector(args.weights, channels=chan)
    samples = []          # (file, t, cur, prev)
    tmp = tempfile.mkdtemp(prefix="scan_")
    try:
        for path in args.files:
            dur = duration(path)
            if dur <= 0:
                print(f"skip (no duration): {os.path.basename(path)}")
                continue
            name = os.path.basename(path)
            ts = np.linspace(0.05 * dur, 0.95 * dur, args.seeks)
            got = 0
            for si, t in enumerate(ts):
                burst = extract_burst(path, float(t), args.burst, tmp,
                                      f"s{si}")
                for k in range(1, len(burst)):
                    samples.append((name, float(t) + k / 29.97,
                                    burst[k], burst[k - 1]))
                    got += 1
            print(f"{name}: {got} test frames sampled across {dur / 60:.0f} min")

        if not samples:
            print("no frames extracted")
            return

        # calibrate on every 3rd sampled frame (robust stats tolerate the
        # sparse errors that may be present)
        cal = samples[::3]
        print(f"\nCalibrating on {len(cal)} frames "
              f"(detecting on {len(samples)})...")
        det.calibrate([s[2] for s in cal], prev_frames=[s[3] for s in cal])

        clean_like = np.median(
            [det.detect(s[2], prev=s[3], margin=args.margin, votes=args.votes)[
                "flagged_fraction"] for s in cal[: min(len(cal), 20)]])
        frame_thr = max(clean_like * 3.0, 0.01)
        print(f"anomaly threshold = {frame_thr * 100:.2f}% flagged blocks\n")

        results = []
        for name, t, cur, prev in samples:
            r = det.detect(cur, prev=prev, margin=args.margin, votes=args.votes)
            results.append((r["flagged_fraction"], name, t, cur, r))
        results.sort(reverse=True, key=lambda x: x[0])

        n_flag = sum(1 for r in results if r[0] > frame_thr)
        print(f"Flagged {n_flag} / {len(results)} sampled frames "
              f"above threshold.\n")
        print(f"{'flag%':>6}  {'time':>9}  file")
        os.makedirs(args.out, exist_ok=True)
        for rank, (frac, name, t, cur, r) in enumerate(results[: args.top]):
            mm, ss = int(t // 60), t % 60
            print(f"{frac * 100:>5.1f}%  {mm:>4d}:{ss:05.2f}  {name}")
            Image.fromarray(cur.astype(np.uint8)).save(
                os.path.join(args.out, f"{rank:02d}_{mm:02d}m{int(ss):02d}s_"
                             f"input.png"))
            votes = r["votes"]
            up = np.kron((votes / max(votes.max(), 1) * 255).astype(np.uint8),
                         np.ones((8, 8), np.uint8))
            Image.fromarray(up).save(
                os.path.join(args.out, f"{rank:02d}_{mm:02d}m{int(ss):02d}s_"
                             f"heat.png"))
        print(f"\nHeatmaps for the top {args.top} -> {args.out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
