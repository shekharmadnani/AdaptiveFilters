"""Score a video with a trained model bundle (no reference needed).

Usage:
  python score_video.py video.mp4 --model model.json [--step 6]
                        [--width W --height H --pix-fmt yuv420p]  (raw .yuv)
                        [--csv per_frame.csv]

Outputs the distilled quality index (VMAF-like, higher = better) with
mean / p5 / min pooling, plus the naturalness anchor score and per-artifact
diagnostics on the worst sampled frame.
"""

import argparse
import csv
import json

import numpy as np

from adaptive_filters import FeatureExtractor, NaturalnessModel, RidgeFusion, to_vector
from adaptive_filters.io import iter_luma_frames, sample_frames


def load_bundle(path):
    with open(path, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    names = bundle["feature_names"]
    anchor = NaturalnessModel.from_dict(bundle["anchor"])
    if bundle["model_type"] == "gbt":
        import xgboost as xgb

        model = xgb.XGBRegressor()
        model.load_model(bundle["gbt_file"])
        predict = lambda v: float(model.predict(v[None, :])[0])
    else:
        ridge = RidgeFusion.from_dict(bundle["ridge"])
        predict = lambda v: float(ridge.predict(v[None, :])[0])
    return names, predict, anchor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--model", default="model.json")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--pix-fmt", default="yuv420p")
    ap.add_argument("--csv", help="optional per-frame output CSV")
    args = ap.parse_args()

    names, predict, anchor = load_bundle(args.model)
    extractor = FeatureExtractor()

    rows = []
    frames = iter_luma_frames(args.video, width=args.width,
                              height=args.height, pix_fmt=args.pix_fmt)
    for idx, frame, prev in sample_frames(frames, args.step):
        feats = extractor.extract(frame, prev)
        _, vec = to_vector(feats, names)
        rows.append((idx, predict(vec), anchor.score(vec), vec))
    if not rows:
        raise SystemExit("no frames sampled -- video too short for --step?")

    q = np.array([r[1] for r in rows])
    a = np.array([r[2] for r in rows])

    print(f"Sampled frames : {len(rows)} (every {args.step})")
    print(f"Quality index  : mean={q.mean():6.2f}  p5={np.percentile(q, 5):6.2f}  "
          f"min={q.min():6.2f}   (VMAF-distilled, higher = better)")
    print(f"Anchor score   : mean={a.mean():6.2f}  max={a.max():6.2f}   "
          f"(distance from pristine, lower = better)")

    worst = int(np.argmin(q))
    print(f"\nWorst sampled frame: #{rows[worst][0]} "
          f"(quality {rows[worst][1]:.2f}) -- top naturalness deviations:")
    for name, z in anchor.top_deviations(rows[worst][3], names, k=5):
        print(f"  {name:<28} z={z:.1f}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "quality_index", "anchor_score"])
            for idx, qi, ai, _ in rows:
                w.writerow([idx, f"{qi:.3f}", f"{ai:.3f}"])
        print(f"\nPer-frame scores -> {args.csv}")


if __name__ == "__main__":
    main()
