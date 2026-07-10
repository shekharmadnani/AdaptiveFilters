"""Model selection on held-out BinResults folders (never extracted for
training). For each held-out folder we have GT + 10 degraded bins with VIF
labels -- a natural per-content severity ladder in the SAME domain and
color space (RGB) as the training data.

For every candidate checkpoint:
  - features on a center crop of each bin image
  - per folder, per feature: SRCC against (1 - VIF)
  - score = number of features whose MEDIAN |SRCC| across folders >= 0.9
  - t diagnostics: damage-map mean on the mildest vs most severe bin

Usage:
  python validate_binpairs.py --models models/wiener4_bin_mu01.pt ...
        [--pairs datasets/binpairs] [--crop 512]
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

from adaptive_filters.binpairs import parse_folder
from adaptive_filters.features.stats import spearman

Image.MAX_IMAGE_PIXELS = None


def load_heldout_crops(pairs_dir, crop=512):
    """[(folder_id, [(vif, crop_rgb float01 (H,W,3)), ...sorted by vif])]"""
    with open(os.path.join(pairs_dir, "heldout.json"), encoding="utf-8") as f:
        folders = json.load(f)
    out = []
    for fld in folders:
        parsed = parse_folder(fld)
        if parsed is None:
            continue
        _, bins = parsed
        entries = []
        for b in sorted(bins):
            path, vif = bins[b]
            img = np.asarray(Image.open(path).convert("RGB"),
                             dtype=np.float64) / 255.0
            h, w = img.shape[:2]
            y0, x0 = (h - crop) // 2, (w - crop) // 2
            y0 -= y0 % 8
            x0 -= x0 % 8
            entries.append((vif, img[y0:y0 + crop, x0:x0 + crop]))
        if len(entries) >= 6:
            out.append((os.path.basename(fld), entries))
        print(f"  loaded {os.path.basename(fld)}: {len(entries)} bins")
    return out


def evaluate(model_path, heldout, thr=0.9):
    from adaptive_filters.probes.learned_wiener_probe import LearnedWienerProbe

    probe = LearnedWienerProbe(model_path)
    per_feature = {}
    t_mild, t_severe = [], []
    for _fid, entries in heldout:
        sev_axis = [1.0 - v for v, _ in entries]
        rows = {}
        for (vif, img) in entries:
            feats = probe.run(img)
            for k, v in feats.features.items():
                rows.setdefault(k, []).append(v)
        for k, vals in rows.items():
            per_feature.setdefault(k, []).append(
                abs(spearman(sev_axis, vals)))
        if "t_abs_mean" in rows:
            order = np.argsort([v for v, _ in entries])
            t_mild.append(rows["t_abs_mean"][int(order[-1])])   # best VIF
            t_severe.append(rows["t_abs_mean"][int(order[0])])  # worst VIF
    med = {k: float(np.median(v)) for k, v in per_feature.items()}
    strong = sorted(((s, k) for k, s in med.items() if s >= thr),
                    reverse=True)
    return {
        "n_strong": len(strong),
        "n_total": len(med),
        "top": strong[:5],
        "t_mild": float(np.mean(t_mild)) if t_mild else None,
        "t_severe": float(np.mean(t_severe)) if t_severe else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--pairs", default=r"datasets\binpairs")
    ap.add_argument("--crop", type=int, default=512)
    args = ap.parse_args()

    print("Loading held-out folders (network)...")
    heldout = load_heldout_crops(args.pairs, args.crop)
    print(f"{len(heldout)} held-out contents\n")

    print(f"{'model':<28} {'strong':>6} {'total':>6} "
          f"{'t(mild)':>9} {'t(severe)':>9}  top features")
    for mp in args.models:
        r = evaluate(mp, heldout)
        top = ", ".join(f"{k}({s:.2f})" for s, k in r["top"][:3])
        tm = f"{r['t_mild']:.5f}" if r["t_mild"] is not None else "-"
        ts = f"{r['t_severe']:.5f}" if r["t_severe"] is not None else "-"
        print(f"{os.path.basename(mp):<28} {r['n_strong']:>6} "
              f"{r['n_total']:>6} {tm:>9} {ts:>9}  {top}")


if __name__ == "__main__":
    main()
