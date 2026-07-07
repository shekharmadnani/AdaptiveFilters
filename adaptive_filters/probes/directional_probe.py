"""Directional (CDEF-style / structure-tensor) probe.

Naturalness model : content is locally 1-D along the dominant orientation.
Blind theta       : orientation + coherence from the smoothed structure
                    tensor; clamped 3-tap smoothing along the local tangent
                    (large differences are ignored, as in CDEF's constraint).
Residual exposes  : jaggies and ringing across edges.
Theta features    : coherence statistics (natural edges are coherent;
                    artifact edges are not).

References: Midtskogen & Valin, AV1 CDEF, ICASSP 2018;
Takeda, Farsiu, Milanfar, kernel regression (LARK), TIP 2007.
"""

import numpy as np

from ..features import stats
from ..utils import box_filter, stat_view
from .base import Probe, ProbeResult, common_residual_features

# tangent-direction offsets for quantized orientations 0/45/90/135 degrees
_OFFSETS = [(0, 1), (1, 1), (1, 0), (1, -1)]


class DirectionalProbe(Probe):
    name = "dir"

    def __init__(self, tensor_radius=2):
        self.tensor_radius = tensor_radius

    def run(self, frame, prev_frame=None):
        i = frame
        gy, gx = np.gradient(i)
        r = self.tensor_radius

        jxx = box_filter(gx * gx, r)
        jyy = box_filter(gy * gy, r)
        jxy = box_filter(gx * gy, r)

        trace = jxx + jyy
        diff = jxx - jyy
        mag = np.sqrt(diff * diff + 4.0 * jxy * jxy)
        coherence = mag / (trace + 1e-10)

        theta_normal = 0.5 * np.arctan2(2.0 * jxy, diff)
        theta_tangent = theta_normal + np.pi / 2.0
        bins = np.round(theta_tangent / (np.pi / 4.0)).astype(int) % 4

        sigma_n = stats.estimate_noise_sigma(i)
        thr = max(3.0 * sigma_n, 0.008)  # CDEF-like constraint

        filtered = i.copy()
        for b, (dy, dx) in enumerate(_OFFSETS):
            m = bins == b
            if not m.any():
                continue
            n1 = np.roll(i, (dy, dx), axis=(0, 1))
            n2 = np.roll(i, (-dy, -dx), axis=(0, 1))
            n1c = np.where(np.abs(n1 - i) < thr, n1, i)
            n2c = np.where(np.abs(n2 - i) < thr, n2, i)
            smoothed = (i + n1c + n2c) / 3.0
            filtered[m] = smoothed[m]

        residual = i - filtered
        feats = common_residual_features(filtered, residual)

        coh_s = stat_view(coherence)
        res_s = stat_view(residual)
        feats["coh_mean"] = float(coherence.mean())
        feats["coh_p95"] = float(np.percentile(coh_s, 95))

        # residual localization on coherent structure (edges) vs elsewhere
        hi = coh_s > np.percentile(coh_s, 75)
        e_hi = float(np.mean(res_s[hi] ** 2)) if hi.any() else 0.0
        e_lo = float(np.mean(res_s[~hi] ** 2)) if (~hi).any() else 0.0
        feats["edge_ratio"] = e_hi / (e_lo + 1e-12)

        return ProbeResult(filtered=filtered, residual=residual, features=feats)
