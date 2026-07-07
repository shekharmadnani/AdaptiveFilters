"""Temporal probe (frame-pair, no motion compensation in v1).

Naturalness model : brightness constancy between consecutive frames.
Blind theta       : none in v1 (the previous frame is the prediction);
                    v2 adds local gain/motion compensation.
Residual exposes  : flicker, quality pumping, frame repeats (zero residual).
"""

import numpy as np

from ..features import stats
from .base import Probe, ProbeResult, common_residual_features


class TemporalProbe(Probe):
    name = "tmp"

    def __init__(self, block=16):
        self.block = block

    def run(self, frame, prev_frame=None):
        if prev_frame is None:
            return None
        cur = frame
        prev = prev_frame
        h = min(cur.shape[0], prev.shape[0])
        w = min(cur.shape[1], prev.shape[1])
        cur, prev = cur[:h, :w], prev[:h, :w]

        residual = cur - prev
        feats = common_residual_features(prev, residual)

        # flicker: dispersion of blockwise mean temporal change
        n = self.block
        hb, wb = h // n, w // n
        if hb > 0 and wb > 0:
            bm = residual[: hb * n, : wb * n].reshape(hb, n, wb, n).mean(axis=(1, 3))
            feats["flicker"] = float(bm.std())
        else:
            feats["flicker"] = 0.0
        feats["gain_corr"] = stats.pearson(cur, prev)
        feats["repeat"] = float(np.mean(np.abs(residual)) < 1e-6)

        return ProbeResult(filtered=prev, residual=residual, features=feats)
