"""Context ablation evaluation: baseline (A) vs dilated (B) vs U-Net (C).

For each checkpoint, measures on a real BVI frame:
  1. restoration PSNR per artifact x severity (luma, vs pristine) --
     context should matter most where synthesis-like behavior is needed
     (blur, heavy compression, concealment)
  2. JPEG severity-ladder feature monotonicity (count of |SRCC| >= 0.9)

Usage:
  python compare_context.py --models models/wiener.pt models/wiener_b.pt models/wiener_c.pt
"""

import argparse
import os

import numpy as np

from adaptive_filters.artifacts import apply_artifact
from adaptive_filters.features.stats import spearman
from adaptive_filters.learned.adaptive_dct import AdaptiveWienerFilter
from adaptive_filters.learned.patches import gather_frames
from adaptive_filters.synthetic import jpeg_like

DEFAULT_BVI = r"F:\DVI\BVI-CC1\ORIG_MP4"
ARTS = ["compression", "blur", "noise", "banding", "ringing",
        "pl_interp", "block_fill"]


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+",
                    default=[os.path.join("models", "wiener.pt"),
                             os.path.join("models", "wiener_b.pt"),
                             os.path.join("models", "wiener_c.pt")])
    ap.add_argument("--bvi", default=DEFAULT_BVI)
    args = ap.parse_args()

    f = gather_frames(args.bvi, frames_per_video=1, color=True,
                      verbose=False)[-1]
    h, w = f.shape[:2]
    pris = f[h // 2 - 256 : h // 2 + 256,
             w // 2 - 256 : w // 2 + 256].astype(np.float64)

    filters = {os.path.basename(p): AdaptiveWienerFilter(p)
               for p in args.models}
    tags = list(filters)

    # ---- restoration PSNR
    print(f"{'artifact':<12} {'sev':>3} {'input':>7} "
          + " ".join(f"{t:>18}" for t in tags))
    for a in ARTS:
        for sev in (2, 4):
            deg = np.empty_like(pris)
            for c in range(3):
                deg[:, :, c] = apply_artifact(a, pris[:, :, c], sev, seed=7)
            p_in = psnr(deg[:, :, 0] / 255, pris[:, :, 0] / 255)
            outs = []
            for t in tags:
                r = filters[t].apply(deg / 255.0)
                hh, ww = r.filtered.shape[:2]
                outs.append(psnr(r.filtered[:, :, 0],
                                 pris[:hh, :ww, 0] / 255))
            print(f"{a:<12} {sev:>3} {p_in:>7.2f} "
                  + " ".join(f"{o:>18.2f}" for o in outs))

    # ---- ladder monotonicity (probe features per model)
    from adaptive_filters.probes.learned_wiener_probe import LearnedWienerProbe
    print(f"\n{'model':<18} {'#monotone features (JPEG ladder)':>35}")
    qualities = [100, 90, 70, 50, 30, 10]
    sev_axis = [100 - q for q in qualities]
    for p in args.models:
        probe = LearnedWienerProbe(p)
        rows = {}
        for q in qualities:
            if q == 100:
                deg = pris
            else:
                deg = np.empty_like(pris)
                for c in range(3):
                    deg[:, :, c] = jpeg_like(pris[:, :, c], q)
            for k, v in probe.run(deg / 255.0).features.items():
                rows.setdefault(k, []).append(v)
        strong = sum(1 for v in rows.values()
                     if abs(spearman(sev_axis, v)) >= 0.9)
        print(f"{os.path.basename(p):<18} {strong:>35}")


if __name__ == "__main__":
    main()
