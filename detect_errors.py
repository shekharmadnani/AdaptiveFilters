"""Validate the spatial-discontinuity detector on KNOWN error locations.

The artifact suite injects errors whose position we know exactly (block
fill, packet-loss bands), so the per-pixel difference (degraded vs
pristine) is a ground-truth error mask. We calibrate the detector on
clean frames the test never sees, then measure, per error type:

  recall     : fraction of true error blocks that were flagged
  precision  : fraction of flagged blocks that are truly erroneous
  clean FA   : flagged-area fraction on a CLEAN frame (false-alarm rate)

Usage:
  python detect_errors.py [--weights models/wiener4_dvc.pt]
                          [--bvi F:\\DVI\\BVI-CC1\\ORIG_MP4] [--save out_detect]
"""

import argparse
import os

import numpy as np

from adaptive_filters.artifacts import apply_artifact
from adaptive_filters.bitstream import decode_yuv444_u8
from adaptive_filters.detect import SpatialErrorDetector, _block_mean
from adaptive_filters.io import ffprobe_dims

DEFAULT_BVI = r"F:\DVI\BVI-CC1\ORIG_MP4"
CROP = 512


def center(f, size=CROP):
    h, w = f.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return f[y0:y0 + size, x0:x0 + size]


def gt_region(pris, deg, thr=3.0):
    """Blocks whose content was changed (mean |deg - pris| > thr)."""
    d = np.abs(pris.astype(np.float64) - deg.astype(np.float64))
    if d.ndim == 3:
        d = d[:, :, 0]
    return _block_mean(d) > thr


def gt_discontinuity(region):
    """The DISCONTINUITY ground truth: blocks at the boundary between a
    changed region and unchanged content (the seams/edges a spatial-
    discontinuity detector should find), rather than the smooth interior
    of a fill or interpolated band."""
    r = region
    # a block is a discontinuity block if it is changed and touches an
    # unchanged block, OR unchanged and touches a changed block
    changed_edge = r & _touches(~r)
    unchanged_edge = (~r) & _touches(r)
    return changed_edge | unchanged_edge


def _touches(mask):
    t = np.zeros_like(mask)
    t[1:, :] |= mask[:-1, :]; t[:-1, :] |= mask[1:, :]
    t[:, 1:] |= mask[:, :-1]; t[:, :-1] |= mask[:, 1:]
    return t


def score(pred, gt):
    tp = int((pred & gt).sum())
    recall = tp / max(int(gt.sum()), 1)
    precision = tp / max(int(pred.sum()), 1)
    return recall, precision


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=os.path.join("models",
                                                      "wiener4_dvc.pt"))
    ap.add_argument("--bvi", default=DEFAULT_BVI)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    masters = sorted(f for f in os.listdir(args.bvi)
                     if f.endswith(".mp4") and "1920x1080" in f)
    mpath = os.path.join(args.bvi, masters[-1])
    w, h = ffprobe_dims(mpath)
    frames = decode_yuv444_u8(mpath, w, h, max_frames=24)
    lum = [center(f).astype(np.float64) for f in frames]   # 0..255, HWC

    det = SpatialErrorDetector(args.weights)
    print("Calibrating on clean frames 2..9 ...")
    det.calibrate(lum[2:10], prev_frames=lum[1:9])

    pris, prev = lum[16], lum[15]
    conds = [
        ("clean", None, 0),
        ("block_fill", "block_fill", 3),
        ("block_fill_hi", "block_fill", 5),
        ("pl_interp", "pl_interp", 3),
        ("pl_copy", "pl_copy", 3),
    ]

    # frame-level detection threshold: calibrated from the clean frame's
    # flagged fraction plus a margin (so clean reads NO ERROR)
    clean_res = det.detect(pris, prev=prev)
    frame_thr = max(clean_res["flagged_fraction"] * 2.0, 0.004)

    print(f"\nframe-level error threshold = {frame_thr * 100:.2f}% "
          f"flagged blocks\n")
    print(f"{'condition':<14} {'GT seam':>7} {'recall':>7} {'prec':>6} "
          f"{'flag%':>6} {'ERROR?':>7}")
    if args.save:
        os.makedirs(args.save, exist_ok=True)
    for name, art, sev in conds:
        if art is None:
            deg = pris.copy()
        else:
            deg = np.empty_like(pris)
            for c in range(3):
                deg[:, :, c] = apply_artifact(art, pris[:, :, c], sev,
                                              prev_frame=prev[:, :, c],
                                              seed=7)
        res = det.detect(deg, prev=prev)
        pred = res["error_mask"]
        if art is None:
            gt = np.zeros_like(pred)
            rec, prec = float("nan"), float("nan")
        else:
            region = gt_region(pris, deg)
            gt = gt_discontinuity(region)
            gh = min(pred.shape[0], gt.shape[0])
            gw = min(pred.shape[1], gt.shape[1])
            pred, gt = pred[:gh, :gw], gt[:gh, :gw]
            rec, prec = score(pred, gt)
        flagged = res["flagged_fraction"]
        is_err = flagged > frame_thr
        print(f"{name:<14} {int(gt.sum()):>7} {rec:>7.2f} {prec:>6.2f} "
              f"{flagged * 100:>5.1f}% {str(is_err):>7}")
        if args.save:
            _save_maps(args.save, name, deg, pred, res["votes"][:pred.shape[0],
                                                                :pred.shape[1]])

    print("\nInterpretation: recall/prec are vs the DISCONTINUITY (seam) "
          "ground truth; ERROR? is the per-frame verdict (clean must be "
          "False, error types True).")


def _save_maps(outdir, name, deg, pred, votes):
    from PIL import Image
    Image.fromarray(np.clip(deg[:, :, 0], 0, 255).astype(np.uint8)).save(
        os.path.join(outdir, f"{name}_input.png"))
    up = np.kron(pred.astype(np.uint8) * 255, np.ones((8, 8), np.uint8))
    Image.fromarray(up).save(os.path.join(outdir, f"{name}_flagged.png"))
    vv = votes / max(votes.max(), 1)
    vv = np.kron((vv * 255).astype(np.uint8), np.ones((8, 8), np.uint8))
    Image.fromarray(vv).save(os.path.join(outdir, f"{name}_votes.png"))


if __name__ == "__main__":
    main()
