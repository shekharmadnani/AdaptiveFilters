"""Guided-filter / local-Wiener probe with adaptive epsilon.

Naturalness model : image is locally linear in itself: x ~ a*I + b per window.
Blind theta       : closed-form a = var / (var + eps); eps from a blind
                    global noise estimate (Immerkaer), so the filter adapts
                    per pixel through the local variance.
Residual exposes  : noise level, grain loss, banding steps in flat regions.

References: He, Sun, Tang, Guided Image Filtering, TPAMI 2013;
Lee local-statistics filter, TPAMI 1980.
"""

import numpy as np

from ..features import stats
from ..utils import box_filter, stat_view
from .base import Probe, ProbeResult, common_residual_features


class GuidedProbe(Probe):
    name = "gd"

    def __init__(self, radius=3):
        self.radius = radius

    def run(self, frame, prev_frame=None):
        i = frame
        r = self.radius

        sigma_n = stats.estimate_noise_sigma(i)
        eps = max(sigma_n ** 2, 1e-6)

        mu = box_filter(i, r)
        var = np.clip(box_filter(i * i, r) - mu * mu, 0.0, None)

        a = var / (var + eps)
        b = mu * (1.0 - a)
        filtered = box_filter(a, r) * i + box_filter(b, r)
        residual = i - filtered

        feats = common_residual_features(filtered, residual)

        # theta-map features
        feats["a_mean"] = float(a.mean())
        feats["a_low_frac"] = float(np.mean(stat_view(a) < 0.15))  # banding-prone
        feats["noise_sigma"] = sigma_n

        # flat-region behavior: banding shows as structured residual where
        # the content itself is flat (subsampled views for the order stats)
        var_s = stat_view(var)
        res_s = stat_view(residual)
        flat = var_s < np.quantile(var_s, 0.25)
        e_all = float(np.mean(res_s ** 2)) + 1e-12
        if flat.any():
            feats["flat_res_ratio"] = float(np.mean(res_s[flat] ** 2)) / e_all
            alpha_flat, _ = stats.fit_ggd(res_s[flat])
            feats["flat_res_alpha"] = alpha_flat
        else:
            feats["flat_res_ratio"] = 0.0
            feats["flat_res_alpha"] = 2.0

        return ProbeResult(filtered=filtered, residual=residual, features=feats)
