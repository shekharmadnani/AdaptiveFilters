"""Rigorous stress test of the learned Wiener filter's residual features.

Runs the Wiener probe STANDALONE (no other filters) against severity
ladders in three groups, on real BVI content:

  REAL CODEC (x264 at 1080p, decoded error-resiliently):
    h264_deblk_on  : CRF 26..51 with the in-loop deblocking filter ON
                     (blockiness already smoothed by the codec -- the
                     hard detection case)
    h264_deblk_off : same CRFs, no-deblock (raw blockiness)
    pkt_loss_real  : CRF 28 deblocked stream, byte-flipped slice NALs at
                     rising fractions, decoded with concealment (-ec 1)

  SEVERE SIMULATED (beyond the standard suite's ranges):
    jpeg_severe  q = 15..3      blur_severe  box passes 2..10
    noise_severe sigma = 8..40

  STANDARD SUITE (all 9 artifacts incl. the new ringing, severities 1..5):
    compression, blur, noise, banding, ringing, pl_interp, pl_copy,
    block_fill, stale

For each ladder: features are averaged over several frames, SRCC is
computed per feature against severity (pristine included at level 0).
Gate: every condition must have >=1 feature with |SRCC| >= 0.9.

Usage:  python stress_test.py [--weights models/wiener.pt]
"""

import argparse
import os
import tempfile

import numpy as np

from adaptive_filters.artifacts import ARTIFACTS, apply_artifact
from adaptive_filters.bitstream import (
    encode_h264, corrupt_annexb, decode_yuv444_u8,
)
from adaptive_filters.features.stats import spearman
from adaptive_filters.io import ffprobe_dims
from adaptive_filters.synthetic import jpeg_like
from adaptive_filters.utils import box_filter

DEFAULT_BVI = r"F:\DVI\BVI-CC1\ORIG_MP4"
FRAME_IDS = [12, 24, 36]          # sampled frames (predecessors decoded too)
NFRAMES = 40
CROP = 512


def center_crop(f, size=CROP):
    h, w = f.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return f[y0 : y0 + size, x0 : x0 + size]


def per_channel(fn, frame, *a, **kw):
    out = np.empty_like(frame)
    for c in range(frame.shape[2]):
        out[:, :, c] = fn(frame[:, :, c], *a, **kw)
    return out


def mean_features(probe, frames):
    acc = {}
    for f in frames:
        for k, v in probe.run(f / 255.0).features.items():
            acc[k] = acc.get(k, 0.0) + v / len(frames)
    return acc


def ladder_srcc(probe, ladders):
    """ladders: list (per severity 0..n) of frame lists -> per-feature SRCC."""
    rows = {}
    for frames in ladders:
        for k, v in mean_features(probe, frames).items():
            rows.setdefault(k, []).append(v)
    sev = list(range(len(ladders)))
    return sorted(((abs(spearman(sev, v)), k) for k, v in rows.items()),
                  reverse=True)


def report(name, scored, thr):
    strong = sum(1 for s, _ in scored if s >= thr)
    top = ", ".join(f"{k}({s:+.2f})" for s, k in scored[:3])
    print(f"{name:<16} {strong:>7}  {top}")
    return strong >= 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=os.path.join("models", "wiener.pt"))
    ap.add_argument("--bvi", default=DEFAULT_BVI)
    ap.add_argument("--srcc", type=float, default=0.9)
    args = ap.parse_args()

    from adaptive_filters.probes.learned_wiener_probe import LearnedWienerProbe
    probe = LearnedWienerProbe(args.weights)

    masters = sorted(f for f in os.listdir(args.bvi)
                     if f.endswith(".mp4") and "1920x1080" in f)
    mpath = os.path.join(args.bvi, masters[-1])
    w, h = ffprobe_dims(mpath)
    print(f"Content: {os.path.basename(mpath)}  (frames {FRAME_IDS}, "
          f"{CROP}x{CROP} center crops)")

    refs_full = decode_yuv444_u8(mpath, w, h, max_frames=NFRAMES)
    pristine = [center_crop(refs_full[i]).astype(np.float64)
                for i in FRAME_IDS]
    prevs = [center_crop(refs_full[i - 1]).astype(np.float64)
             for i in FRAME_IDS]

    ok = True
    print(f"\n{'condition':<16} {'#strong':>7}  top features (SRCC)")

    # ---------------- real codec ladders
    with tempfile.TemporaryDirectory() as td:
        for name, crfs, deblock in (("h264_deblk_on", [26, 34, 42, 48, 51], True),
                                    ("h264_deblk_off", [26, 34, 42, 48, 51], False)):
            ladders = [pristine]
            for crf in crfs:
                enc = os.path.join(td, f"{name}_{crf}.264")
                encode_h264(mpath, enc, NFRAMES, crf, deblock=deblock)
                degs = decode_yuv444_u8(enc, w, h, max_frames=NFRAMES)
                ladders.append([center_crop(degs[i]).astype(np.float64)
                                for i in FRAME_IDS])
            ok &= report(name, ladder_srcc(probe, ladders), args.srcc)

        # real packet loss on a deblocked CRF-28 stream
        base = os.path.join(td, "pl_base.264")
        encode_h264(mpath, base, NFRAMES, 28, deblock=True)
        with open(base, "rb") as f:
            data = f.read()
        ladders = [pristine]
        for i, frac in enumerate([0.05, 0.15, 0.25, 0.35, 0.50]):
            cor = os.path.join(td, f"pl_{i}.264")
            with open(cor, "wb") as f:
                f.write(corrupt_annexb(data, frac, seed=40 + i))
            degs = decode_yuv444_u8(cor, w, h, max_frames=NFRAMES)
            m = len(degs)
            ladders.append([center_crop(degs[min(j, m - 1)]).astype(np.float64)
                            for j in FRAME_IDS])
        ok &= report("pkt_loss_real", ladder_srcc(probe, ladders), args.srcc)

    # ---------------- severe simulated ladders
    severe = {
        "jpeg_severe": (lambda f, p: jpeg_like(f, p), [15, 10, 7, 5, 3]),
        "blur_severe": (lambda f, p: _multi_box(f, p), [2, 4, 6, 8, 10]),
        "noise_severe": (lambda f, p, i=[0]: _noise(f, p), [8, 16, 24, 32, 40]),
    }
    for name, (fn, params) in severe.items():
        ladders = [pristine]
        for p in params:
            ladders.append([per_channel(fn, fr, p) for fr in pristine])
        ok &= report(name, ladder_srcc(probe, ladders), args.srcc)

    # ---------------- standard suite (incl. ringing), severities 1..5
    for aname in ARTIFACTS:
        ladders = [pristine]
        for s in range(1, 6):
            frames = []
            for fr, pv in zip(pristine, prevs):
                deg = np.empty_like(fr)
                for c in range(3):
                    deg[:, :, c] = apply_artifact(
                        aname, fr[:, :, c], s,
                        prev_frame=pv[:, :, c], seed=7)
                frames.append(deg)
            ladders.append(frames)
        ok &= report(aname, ladder_srcc(probe, ladders), args.srcc)

    print(f"\nSTRESS TEST {'PASSED' if ok else 'FAILED'} "
          f"(every condition needs >=1 feature with |SRCC|>={args.srcc})")
    raise SystemExit(0 if ok else 1)


def _multi_box(f, passes):
    out = f.copy()
    for _ in range(passes):
        out = box_filter(out, 1)
    return out


def _noise(f, sigma):
    rng = np.random.default_rng(int(sigma * 100))
    return np.clip(f + rng.normal(0.0, sigma, f.shape), 0, 255)


if __name__ == "__main__":
    main()
