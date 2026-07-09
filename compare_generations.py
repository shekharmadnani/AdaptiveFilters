"""Compare ALL learned-filter generations on the same degraded images.

Generations (probe class -> checkpoint):
  gen1_kdct     LearnedDctProbe    models/kdct.pt         patch/threshold
  gen2_kmap     LearnedKMapProbe   models/kmap.pt         top-K, naturalness
  gen2_restore  LearnedKMapProbe   models/kmap_restore.pt top-K, paired
  gen3_wiener   LearnedWienerProbe models/wiener.pt       gains, color
  gen4_affine   LearnedWienerProbe models/wiener4_c.pt    gains + t, U-Net

Measured on one real BVI frame (512x512 center crop):
  1. restoration PSNR (luma, dB vs pristine) per artifact x severity
  2. number of severity-monotone features (JPEG ladder, |SRCC| >= 0.9)

Fairness note: generations differ in training data (gen1/2 pristine-only
luma; gen2_restore paired luma; gen3/4 paired color + real H.264), so PSNR
rows compare *designs as trained*, not architectures under equal data.

Usage:  python compare_generations.py
"""

import os

import numpy as np

from adaptive_filters.artifacts import apply_artifact
from adaptive_filters.features.stats import spearman
from adaptive_filters.learned.patches import gather_frames
from adaptive_filters.synthetic import jpeg_like

DEFAULT_BVI = r"F:\DVI\BVI-CC1\ORIG_MP4"
ARTS = ["compression", "blur", "noise", "banding", "ringing",
        "pl_interp", "block_fill"]
QUALITIES = [100, 90, 70, 50, 30, 10]


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def build_probes():
    from adaptive_filters.probes.learned_dct_probe import LearnedDctProbe
    from adaptive_filters.probes.learned_kmap_probe import LearnedKMapProbe
    from adaptive_filters.probes.learned_wiener_probe import LearnedWienerProbe

    m = os.path.join("models", "")
    return {
        "gen1_kdct": (LearnedDctProbe(m + "kdct.pt"), "gray"),
        "gen2_kmap": (LearnedKMapProbe(m + "kmap.pt"), "gray"),
        "gen2_restore": (LearnedKMapProbe(m + "kmap_restore.pt"), "gray"),
        "gen3_wiener": (LearnedWienerProbe(m + "wiener.pt"), "color"),
        "gen4_affine": (LearnedWienerProbe(m + "wiener4_c.pt"), "color"),
    }


def probe_input(frame_color, mode):
    """frame_color: (H, W, 3) in [0, 255] -> probe input in [0, 1]."""
    if mode == "gray":
        return frame_color[:, :, 0] / 255.0
    return frame_color / 255.0


def degrade(frame_color, artifact, severity):
    out = np.empty_like(frame_color)
    for c in range(3):
        out[:, :, c] = apply_artifact(artifact, frame_color[:, :, c],
                                      severity, seed=7)
    return out


def main():
    f = gather_frames(DEFAULT_BVI, frames_per_video=1, color=True,
                      verbose=False)[-1]
    h, w = f.shape[:2]
    pris = f[h // 2 - 256 : h // 2 + 256,
             w // 2 - 256 : w // 2 + 256].astype(np.float64)
    pris_y = pris[:, :, 0] / 255.0

    probes = build_probes()
    tags = list(probes)

    # ---------------- restoration PSNR
    print("Restoration PSNR (luma, dB; input column = the degraded frame)")
    print(f"{'artifact':<12} {'sev':>3} {'input':>7} "
          + " ".join(f"{t:>13}" for t in tags))
    for a in ARTS:
        for sev in (2, 4):
            deg = degrade(pris, a, sev)
            p_in = psnr(deg[:, :, 0] / 255.0, pris_y)
            outs = []
            for t in tags:
                probe, mode = probes[t]
                r = probe.run(probe_input(deg, mode))
                hh, ww = r.filtered.shape[:2]
                outs.append(psnr(r.filtered, pris_y[:hh, :ww]))
            print(f"{a:<12} {sev:>3} {p_in:>7.2f} "
                  + " ".join(f"{o:>13.2f}" for o in outs))

    # ---------------- ladder monotonicity
    print(f"\nJPEG severity ladder: monotone features (|SRCC| >= 0.9)")
    print(f"{'generation':<14} {'#monotone':>9} {'#total':>7}  top features")
    sev_axis = [100 - q for q in QUALITIES]
    for t in tags:
        probe, mode = probes[t]
        rows = {}
        for q in QUALITIES:
            if q == 100:
                deg = pris
            else:
                deg = np.empty_like(pris)
                for c in range(3):
                    deg[:, :, c] = jpeg_like(pris[:, :, c], q)
            for k, v in probe.run(probe_input(deg, mode)).features.items():
                rows.setdefault(k, []).append(v)
        scored = sorted(((abs(spearman(sev_axis, v)), k)
                         for k, v in rows.items()), reverse=True)
        strong = [k for s, k in scored if s >= 0.9]
        top = ", ".join(f"{k}({s:+.2f})" for s, k in scored[:3])
        print(f"{t:<14} {len(strong):>9} {len(rows):>7}  {top}")


if __name__ == "__main__":
    main()
