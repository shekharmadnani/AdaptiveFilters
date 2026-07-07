"""Artifact-response matrix: which filter features detect which artifact?

For every artifact type in the simulation suite, applies a 5-level severity
ladder to test frames (real BVI master + synthetic; with a true previous
frame so temporal artifacts and the temporal probe both work), extracts ALL
probe features, and reports per artifact:

  - how many features respond monotonically (|SRCC| >= threshold)
  - the top responding features and which probe family they belong to

Gate: every artifact type must have at least one strong responder on the
real-content frame. This is the coverage evidence for error conditions
(packet loss, concealment, compression, noise, banding, ...).

Usage:
  python validate_artifacts.py [--lkm models/kmap.pt] [--srcc 0.9]
"""

import argparse
import itertools
import os

import numpy as np

from adaptive_filters.artifacts import ARTIFACTS, apply_artifact
from adaptive_filters.features.stats import spearman
from adaptive_filters.io import iter_ffmpeg
from adaptive_filters.pipeline import FeatureExtractor, default_probes
from adaptive_filters.synthetic import make_frame

DEFAULT_BVI = r"F:\DVI\BVI-CC1\ORIG_MP4"


def real_frame_pair(bvi_dir, size=512):
    """Two consecutive frames (crop) from the last HD master."""
    files = sorted(f for f in os.listdir(bvi_dir)
                   if f.endswith(".mp4") and "1920x1080" in f)
    path = os.path.join(bvi_dir, files[-1])
    it = iter_ffmpeg(path)
    prev, cur = next(it), next(it)
    it.close()
    h, w = cur.shape
    y0, x0 = (h - size) // 2, (w - size) // 2
    return (cur[y0 : y0 + size, x0 : x0 + size] * 255.0,
            prev[y0 : y0 + size, x0 : x0 + size] * 255.0)


def synthetic_pair(size=512):
    cur = make_frame(999, size=size)
    prev = np.roll(cur, (2, 3), axis=(0, 1))  # small global motion
    return cur, prev


def probe_family(feature_name):
    return feature_name.split("_")[1]  # s0_dct_res_energy -> dct


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lkm", default=os.path.join("models", "kmap.pt"))
    ap.add_argument("--lwn", default=os.path.join("models", "wiener.pt"),
                    help="learned Wiener weights ('' to disable)")
    ap.add_argument("--bvi", default=DEFAULT_BVI)
    ap.add_argument("--srcc", type=float, default=0.9)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    probes = default_probes()
    if args.lkm and os.path.exists(args.lkm):
        from adaptive_filters.probes.learned_kmap_probe import LearnedKMapProbe
        probes.append(LearnedKMapProbe(args.lkm))
    if args.lwn and os.path.exists(args.lwn):
        from adaptive_filters.probes.learned_wiener_probe import LearnedWienerProbe
        probes.append(LearnedWienerProbe(args.lwn))
    extractor = FeatureExtractor(probes=probes)

    tests = []
    if os.path.isdir(args.bvi):
        tests.append(("bvi_master",) + real_frame_pair(args.bvi))
    tests.append(("synthetic",) + synthetic_pair())

    severities = [0, 1, 2, 3, 4, 5]
    ok = True
    for tname, frame, prev in tests:
        print(f"\n=== {tname} "
              f"({len(ARTIFACTS)} artifacts x {len(severities) - 1} levels) ===")
        print(f"{'artifact':<12} {'#strong':>7}  top features (SRCC)")
        for aname in ARTIFACTS:
            rows = {}
            for s in severities:
                deg = apply_artifact(aname, frame, s, prev_frame=prev, seed=7)
                feats = extractor.extract(deg / 255.0, prev / 255.0)
                for k, v in feats.items():
                    rows.setdefault(k, []).append(v)
            scored = sorted(((abs(spearman(severities, v)), k)
                             for k, v in rows.items()), reverse=True)
            strong = [k for sc, k in scored if sc >= args.srcc]
            fams = sorted({probe_family(k) for k in strong})
            top = ", ".join(f"{k}({sc:+.2f})" for sc, k in scored[: args.top])
            print(f"{aname:<12} {len(strong):>7}  {top}")
            if fams:
                print(f"{'':<12} {'':>7}  families: {', '.join(fams)}")
            if tname == "bvi_master" and not strong:
                ok = False
                print(f"{'':<12} {'':>7}  *** BLIND SPOT ***")
    print(f"\nMATRIX {'PASSED' if ok else 'FAILED'} "
          f"(every artifact needs >=1 feature with |SRCC|>={args.srcc} "
          f"on real content)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
