"""Statistical anomaly scan: content-LOCAL statistical decision making.

The lesson from the earlier attempts: a global-baseline anomaly score
flags content outliers (graphics segments, crowd shots) because broadcast
content is heterogeneous. The fix is to make the statistical decision
LOCAL IN TIME -- score each frame against its OWN short temporal
neighborhood (same shot) and flag TRANSIENT deviations, which is the
signature of a real defect (dropout, glitch, digital hit, brief
corruption) as opposed to a globally-unusual-but-valid shot.

Per frame we build a statistical feature vector from the gen-4 filter's
residual and damage map (distribution SHAPE, not just magnitude -- shape
is content-invariant for natural content but shifts for real distortion):
  res_energy, res_ggd_alpha, res_kurtosis,
  damage-map mean / p90 / max, worst 64x64 residual tile.

Within each burst of consecutive frames we take the robust center
(median) and spread (MAD) per feature; a frame's anomaly score is its max
robust-z across features. A transient spike (one frame high, neighbors
low) is a defect candidate; a flat burst -- however complex the content --
scores low.

Usage:
  python scan_stat.py --files A.mxf B.mxf [--seeks 40] [--burst 12]
        [--z 6] [--top 20] [--out out_stat]
"""

import argparse
import os
import shutil
import tempfile

import numpy as np
from PIL import Image

from adaptive_filters.learned.adaptive_dct import AdaptiveWienerFilter
from adaptive_filters.features.stats import fit_ggd, kurtosis
from scan_video import extract_burst, duration

FEATS = ["res_energy", "ggd_alpha", "kurtosis", "t_mean", "t_p90",
         "t_max", "tile_max"]


def feature_vector(filt, frame):
    r = filt.apply(frame, light=True)
    resid = r.residual[:, :, 0] if r.residual.ndim == 3 else r.residual
    rs = resid[::2, ::2]
    a, _ = fit_ggd(rs - rs.mean())
    tm = r.t_map[0] if (r.t_map is not None and r.t_map.ndim == 3) else r.t_map
    if tm is None:
        tm = np.zeros((1, 1))
    h, w = resid.shape
    th, tw = h // 64, w // 64
    if th and tw:
        tile = float((resid[:th * 64, :tw * 64] ** 2)
                     .reshape(th, 64, tw, 64).mean(axis=(1, 3)).max())
    else:
        tile = float((resid ** 2).mean())
    return np.array([
        float((resid ** 2).mean()), a, float(kurtosis(rs)),
        float(tm.mean()), float(np.percentile(tm, 90)), float(tm.max()),
        tile,
    ], dtype=np.float64)


def burst_scores(vectors, scale):
    """Per-frame anomaly z: deviation from the BURST median (content-local
    reference) normalized by a GLOBAL per-feature scale `scale` (stable
    denominator). Using the within-burst MAD as the denominator explodes
    when a burst is legitimately stable (MAD -> 0), especially for the
    grid-snapped ggd_alpha; the global scale fixes that."""
    v = np.asarray(vectors)                       # (N, F)
    med = np.median(v, axis=0)
    z = np.abs(v - med) / scale                    # (N, F)
    return z.max(axis=1), z                        # per-frame score, full z


def typical_scale(bursts_vectors):
    """Per-feature scale = the TYPICAL within-shot frame-to-frame variation
    (median over bursts of each burst's within-burst MAD). This is the
    right noise floor: a defect deviates from its burst by far more than
    normal frame-to-frame variation, while cross-SHOT differences (which
    dominate a global MAD and would swamp the signal) are excluded because
    we only ever compare within a burst. Floored relative to the global
    feature magnitude so a near-constant feature cannot blow up."""
    mads = []
    for v in bursts_vectors:
        v = np.asarray(v)
        mads.append(np.median(np.abs(v - np.median(v, axis=0)), axis=0)
                    * 1.4826)
    scale = np.median(np.asarray(mads), axis=0)
    gv = np.concatenate([np.asarray(v) for v in bursts_vectors])
    floor = 1e-2 * (np.abs(np.median(gv, axis=0)) + 1e-6)
    return np.maximum(scale, floor)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--weights", default=os.path.join("models",
                                                      "wiener4_dvc.pt"))
    ap.add_argument("--seeks", type=int, default=40)
    ap.add_argument("--burst", type=int, default=12)
    ap.add_argument("--z", type=float, default=6.0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="out_stat")
    args = ap.parse_args()

    filt = AdaptiveWienerFilter(args.weights)
    bursts = []          # (file, t, [frames], [vectors])
    tmp = tempfile.mkdtemp(prefix="stat_")
    try:
        # pass 1: gather every burst's per-frame feature vectors
        for path in args.files:
            dur = duration(path)
            if dur <= 0:
                continue
            name = os.path.basename(path)
            ts = np.linspace(0.03 * dur, 0.97 * dur, args.seeks)
            n = 0
            for si, t in enumerate(ts):
                try:  # any per-burst failure (network glitch, decode,
                    #   corrupt frame) skips this sample, never the scan
                    frames = extract_burst(path, float(t), args.burst, tmp,
                                           f"s{si}")
                    if len(frames) < 5:
                        continue
                    vecs = [feature_vector(filt, f) for f in frames]
                    bursts.append((name, float(t), frames, vecs))
                    n += 1
                except Exception as e:
                    print(f"  skip seek {si} ({e.__class__.__name__})",
                          flush=True)
            print(f"{name}: {n} bursts sampled over {dur/60:.0f} min",
                  flush=True)

        if not bursts:
            print("no bursts sampled (check file paths / decode)")
            return
        # per-feature scale = typical within-shot frame-to-frame variation
        scale = typical_scale([vecs for _, _, _, vecs in bursts])

        # pass 2: content-local transient anomaly per burst
        hits = []           # (score, file, time, frame, feat_name)
        for name, t, frames, vecs in bursts:
            scores, z = burst_scores(vecs, scale)
            k = int(np.argmax(scores))
            rest = np.median(np.delete(scores, k))    # neighbours' level
            if scores[k] >= args.z and rest < args.z * 0.5:
                fname = FEATS[int(np.argmax(z[k]))]
                hits.append((float(scores[k]), name,
                             t + k / 29.97, frames[k], fname))

        hits.sort(reverse=True, key=lambda h: h[0])
        os.makedirs(args.out, exist_ok=True)
        print(f"\n{'z':>6}  {'time':>10}  {'feature':>10}  file")
        for rank, (sc, name, t, frame, fname) in enumerate(hits[: args.top]):
            hh, mm, ss = int(t // 3600), int(t % 3600 // 60), t % 60
            print(f"{sc:>6.1f}  {hh}:{mm:02d}:{ss:05.2f}  {fname:>10}  {name}",
                  flush=True)
            Image.fromarray(frame.astype(np.uint8)).save(
                os.path.join(args.out, f"{rank:02d}_{mm:02d}m{int(ss):02d}s_"
                             f"{fname}.png"))
        print(f"\n{len(hits)} transient anomalies total; "
              f"top {min(args.top, len(hits))} frames -> {args.out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
