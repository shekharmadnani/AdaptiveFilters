"""Head-to-head against VMAF: which filter's residual features predict
quality best?

Single-probe ridge fusion per filter family, leave-one-content-out on the
x264 demo ladder (real VMAF labels):
  - lwn : learned DCT-domain Wiener (color model, gains <= gmax)
  - dct : blind content-adaptive DCT (per-block MAD threshold)
  - fdct: fixed-K DCT (K identical for every block -- non-adaptive baseline)

Usage:  python compare_vmaf.py  [--rebuild]
"""

import argparse
import math
import os

import numpy as np

from adaptive_filters.dataset import build_dataset, load_dataset
from adaptive_filters.pipeline import FeatureExtractor
from adaptive_filters.probes.dct_probe import DctKeepKProbe
from adaptive_filters.probes.fixed_dct_probe import FixedKDctProbe
from adaptive_filters.fusion import RidgeFusion
from adaptive_filters.features.stats import spearman, pearson

MANIFEST = os.path.join("examples", "demo_data", "manifest.json")
OUT = os.path.join("examples", "demo_data", "dataset_cmp.npz")

FAMILIES = {
    "learned Wiener (lwn)": "_lwn_",
    "adaptive DCT (dct)": "_dct_",
    "fixed-K DCT (fdct)": "_fdct_",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--wiener", default=os.path.join("models", "wiener.pt"))
    args = ap.parse_args()

    if args.rebuild or not os.path.exists(OUT):
        from adaptive_filters.probes.learned_wiener_probe import (
            LearnedWienerProbe,
        )
        extractor = FeatureExtractor(probes=[
            DctKeepKProbe(),
            FixedKDctProbe(k=6),
            LearnedWienerProbe(args.wiener),
        ])
        build_dataset(MANIFEST, OUT, frame_step=6, include_ref=False,
                      extractor=extractor,
                      cache_dir=os.path.join("examples", "demo_data",
                                             "cache_cmp"))

    d = load_dataset(OUT)
    x, y, names, gs = d["x"], d["y"], d["names"], d["groups"]
    contents = sorted(set(gs))

    print(f"\n{len(y)} frames, {len(contents)} contents, "
          f"VMAF range [{y.min():.1f}, {y.max():.1f}]")
    print(f"{'filter family':<24} {'#feat':>5} "
          + " ".join(f"{c:>12}" for c in contents)
          + f" {'SRCC_mean':>10} {'RMSE_mean':>10}")

    for fam, tag in FAMILIES.items():
        cols = [i for i, nm in enumerate(names) if tag in nm]
        srccs, rmses = [], []
        for held in contents:
            tr = [i for i, g in enumerate(gs) if g != held]
            te = [i for i, g in enumerate(gs) if g == held]
            model = RidgeFusion(alpha=5.0).fit(x[np.ix_(tr, cols)],
                                               np.asarray(y)[tr])
            pred = model.predict(x[np.ix_(te, cols)])
            yt = np.asarray(y)[te]
            srccs.append(spearman(yt, pred))
            rmses.append(math.sqrt(float(np.mean((yt - pred) ** 2))))
        row = " ".join(f"{s:>12.3f}" for s in srccs)
        print(f"{fam:<24} {len(cols):>5} {row} "
              f"{np.mean(srccs):>10.3f} {np.mean(rmses):>10.2f}")


if __name__ == "__main__":
    main()
