"""SAO-style edge-offset probe.

Naturalness model : a sample agrees with the mean of its two directional
                    neighbors (no systematic over/undershoot).
Blind theta       : per-category offset = mean pull toward the neighbor
                    average -- closed-form, reference-free by construction.
Residual exposes  : ringing and edge over/undershoot; the offset magnitudes
                    themselves are the primary features.

Reference: Fu et al., "Sample Adaptive Offset in the HEVC Standard",
IEEE TCSVT 2012.
"""

import numpy as np

from .base import Probe, ProbeResult, common_residual_features

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_CATS = [(-2, "valley"), (-1, "concave"), (1, "convex"), (2, "peak")]


class SaoProbe(Probe):
    name = "sao"

    def __init__(self, deadzone=1.0 / 255.0):
        self.deadzone = deadzone

    def run(self, frame, prev_frame=None):
        i = frame
        h, w = i.shape
        t = self.deadzone

        valid = np.zeros_like(i, dtype=bool)
        valid[1:-1, 1:-1] = True

        correction = np.zeros_like(i)
        offset_mag = {key: [] for _, key in _CATS}
        classified = []

        for dy, dx in _DIRS:
            a = np.roll(i, (dy, dx), axis=(0, 1))
            b = np.roll(i, (-dy, -dx), axis=(0, 1))
            d1 = i - a
            d2 = i - b
            s1 = np.where(d1 > t, 1, np.where(d1 < -t, -1, 0))
            s2 = np.where(d2 > t, 1, np.where(d2 < -t, -1, 0))
            cat = s1 + s2
            target = 0.5 * (a + b) - i  # pull toward the neighbor mean

            classified.append(float(np.mean((cat != 0) & valid)))
            for c, key in _CATS:
                m = (cat == c) & valid
                if m.sum() >= 16:
                    off = float(target[m].mean())
                    correction[m] += off
                    offset_mag[key].append(abs(off))

        correction /= len(_DIRS)
        filtered = i + correction
        residual = i - filtered  # = -correction

        feats = common_residual_features(filtered, residual)
        for _, key in _CATS:
            vals = offset_mag[key]
            feats[f"off_{key}"] = float(np.mean(vals)) if vals else 0.0
        feats["classified_frac"] = float(np.mean(classified))

        return ProbeResult(filtered=filtered, residual=residual, features=feats)
