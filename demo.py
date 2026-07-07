"""End-to-end demo and smoke test for the probe-bank NR-VQA stack.

Pipeline exercised (no external data, NumPy only):
  1. Generate synthetic "pristine" frames (gradients + sharp edges +
     oriented texture + fine grain).
  2. Degrade with JPEG-like 8x8 DCT quantization at decreasing quality
     (stand-in for an encode ladder).
  3. Extract probe-bank features.
  4. Layer B: fit the NIQE-style naturalness anchor on pristine TRAIN
     content only; check its score rises monotonically with distortion
     on held-out TEST content.
  5. Layer A: train RidgeFusion on TRAIN content with distortion level as
     a proxy label (stand-in for VMAF distillation); evaluate SRCC on
     held-out TEST content (content-grouped split -- no leakage).

Run:  python demo.py
"""

import numpy as np

from adaptive_filters import FeatureExtractor, NaturalnessModel, RidgeFusion, to_vector
from adaptive_filters.features.stats import spearman
from adaptive_filters.synthetic import make_frame, jpeg_like


# ------------------------------------------------------------------- driver

def main():
    rng_train = range(0, 10)   # train content ids
    rng_test = range(100, 104) # held-out content ids (content-grouped split)
    qualities = [100, 90, 70, 50, 30, 10]  # 100 = pristine

    extractor = FeatureExtractor()
    names = None

    def frame_vector(frame):
        nonlocal names
        feats = extractor.extract(frame)
        names, vec = to_vector(feats, names)
        return vec

    print("Extracting features (train: %d contents, test: %d contents, "
          "%d quality levels)..." % (len(rng_train), len(rng_test), len(qualities)))

    def build(content_ids):
        vecs, labels, groups = [], [], []
        for cid in content_ids:
            pristine = make_frame(cid)
            for q in qualities:
                frame = pristine if q == 100 else jpeg_like(pristine, q)
                vecs.append(frame_vector(frame))
                labels.append(100.0 - q)  # distortion level = proxy label
                groups.append(cid)
        return np.array(vecs), np.array(labels), groups

    x_train, y_train, _ = build(rng_train)
    x_test, y_test, g_test = build(rng_test)
    print("Feature vector length: %d\n" % x_train.shape[1])

    # ---- Layer B: naturalness anchor (fit on pristine train frames only)
    pristine_rows = x_train[y_train == 0.0]
    anchor = NaturalnessModel().fit(pristine_rows)

    print("Layer B -- naturalness anchor score vs quality (held-out content):")
    header = "  content | " + " | ".join("q=%3d" % q for q in qualities)
    print(header)
    anchor_srcc = []
    for cid in rng_test:
        rows = [i for i, g in enumerate(g_test) if g == cid]
        scores = [anchor.score(x_test[i]) for i in rows]
        levels = [y_test[i] for i in rows]
        anchor_srcc.append(spearman(levels, scores))
        print("  %7d | " % cid + " | ".join("%5.2f" % s for s in scores))
    print("  SRCC(distortion, anchor score) per content: "
          + ", ".join("%.3f" % s for s in anchor_srcc) + "\n")

    # ---- Layer A: ridge fusion, distortion level as proxy label
    fusion = RidgeFusion(alpha=5.0).fit(x_train, y_train)
    pred = fusion.predict(x_test)

    print("Layer A -- ridge fusion (proxy-label distillation, grouped split):")
    fusion_srcc = []
    for cid in rng_test:
        rows = [i for i, g in enumerate(g_test) if g == cid]
        fusion_srcc.append(spearman([y_test[i] for i in rows],
                                    [pred[i] for i in rows]))
    print("  SRCC per held-out content: "
          + ", ".join("%.3f" % s for s in fusion_srcc))
    print("  SRCC pooled over all held-out frames: %.3f\n"
          % spearman(y_test, pred))

    # ---- Diagnostics: which probes carry the signal
    print("Top fusion features by |weight|:")
    for name, wgt in fusion.feature_weights(names, k=8):
        print("  %-28s %+.3f" % (name, wgt))

    worst = int(np.argmax(y_test))
    print("\nTop naturalness deviations on a worst-quality test frame:")
    for name, z in anchor.top_deviations(x_test[worst], names, k=6):
        print("  %-28s z=%.1f" % (name, z))

    ok = min(anchor_srcc) > 0.8 and min(fusion_srcc) > 0.8
    print("\nSMOKE TEST %s (per-content SRCC threshold 0.8)"
          % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
