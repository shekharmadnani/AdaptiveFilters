"""Inspect the adaptive DCT filter: apply it to a frame and save images of
the reconstruction, the residual, and the K-map, at several degradation
levels.

Usage:
  python inspect_filter.py [--weights models/kmap.pt]
                           [--image path.png | uses a BVI master frame]
                           [--qualities 100,50,10] [--outdir out_inspect]

Outputs (per quality level q):
  <name>_q<q>_input.png      the (degraded) input frame
  <name>_q<q>_filtered.png   top-K DCT reconstruction
  <name>_q<q>_residual.png   |residual| x8 (the artifact-revealing signal)
  <name>_q<q>_kmap.png       predicted K per 8x8 block (bright = high K)
plus a stats table on stdout.
"""

import argparse
import os

import numpy as np
from PIL import Image

from adaptive_filters.learned.adaptive_dct import AdaptiveDctFilter
from adaptive_filters.synthetic import make_frame, jpeg_like

DEFAULT_BVI = r"F:\DVI\BVI-CC1\ORIG_MP4"


def save_gray(path, img01):
    Image.fromarray(
        np.clip(img01 * 255.0, 0, 255).astype(np.uint8)).save(path)


def get_frame(args):
    if args.image:
        from adaptive_filters.io import load_image_luma
        return os.path.splitext(os.path.basename(args.image))[0], \
            load_image_luma(args.image) * 255.0
    if os.path.isdir(args.bvi):
        from adaptive_filters.learned.patches import gather_frames
        f = gather_frames(args.bvi, frames_per_video=1, verbose=False)[-1]
        h, w = f.shape
        return "bvi_master", f[h // 2 - 256: h // 2 + 256,
                               w // 2 - 256: w // 2 + 256] * 255.0
    return "synthetic", make_frame(999, size=512)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=os.path.join("models", "kmap.pt"))
    ap.add_argument("--image", default=None)
    ap.add_argument("--bvi", default=DEFAULT_BVI)
    ap.add_argument("--qualities", default="100,50,10")
    ap.add_argument("--outdir", default="out_inspect")
    ap.add_argument("--gain", type=float, default=8.0,
                    help="residual display gain")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    name, frame = get_frame(args)
    filt = AdaptiveDctFilter(args.weights)
    qualities = [int(q) for q in args.qualities.split(",")]

    print(f"{'q':>4} {'rmse':>8} {'K_pred':>7} {'K_emp':>7} "
          f"{'res_energy':>11} {'res_p99':>8}")
    for q in qualities:
        deg = frame if q == 100 else jpeg_like(frame, q)
        r = filt.apply(deg / 255.0)

        rmse = float(np.sqrt(np.mean(r.residual ** 2)))
        p99 = float(np.percentile(np.abs(r.residual), 99))
        print(f"{q:>4} {rmse:>8.5f} {r.k_pred.mean():>7.2f} "
              f"{r.k_emp.mean():>7.2f} {np.mean(r.residual ** 2):>11.3e} "
              f"{p99:>8.4f}")

        pre = os.path.join(args.outdir, f"{name}_q{q}")
        crop = deg[: r.filtered.shape[0], : r.filtered.shape[1]] / 255.0
        save_gray(f"{pre}_input.png", crop)
        save_gray(f"{pre}_filtered.png", r.filtered)
        save_gray(f"{pre}_residual.png", np.abs(r.residual) * args.gain)
        kimg = np.repeat(np.repeat(r.k_pred / 63.0, 8, axis=0), 8, axis=1)
        save_gray(f"{pre}_kmap.png", kimg)

    print(f"\nimages -> {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
