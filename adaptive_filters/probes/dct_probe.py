"""Adaptive DCT keep-K probe (the prototype of the whole family).

Naturalness model : each 8x8 block is sparse in the DCT basis.
Blind theta       : per-block noise floor sigma from MAD of high-frequency
                    coefficients; keep coefficient iff |X_k| > tau * sigma.
                    K (retained AC count) is the content-adaptive parameter.
Residual exposes  : ringing / mosquito noise; the K-map itself detects
                    compression (K collapses toward the codec's own count).

References: Yaroslavsky sliding-window DCT; Foi et al., SA-DCT, TIP 2007;
Donoho & Johnstone shrinkage, Biometrika 1994.
"""

import numpy as np

from ..features import stats
from ..utils import blockify, unblockify, gradient_magnitude, stat_view
from .base import Probe, ProbeResult, common_residual_features


def dct_matrix(n=8):
    k = np.arange(n)
    d = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    d[0, :] = np.sqrt(1.0 / n)
    return d


class DctKeepKProbe(Probe):
    name = "dct"

    def __init__(self, block=8, tau=2.5):
        self.block = block
        self.tau = tau
        self._d = dct_matrix(block)
        u, v = np.meshgrid(np.arange(block), np.arange(block), indexing="ij")
        # high-frequency set used for the blind noise-floor estimate
        self._hf_mask = (u + v) >= block

    def run(self, frame, prev_frame=None):
        n = self.block
        crop, blocks = blockify(frame, n)
        d = self._d

        x = np.matmul(np.matmul(d, blocks), d.T)  # X = D b D^T (batched BLAS)

        # blind per-block noise floor from high-frequency coefficients (MAD)
        hf = np.abs(x[..., self._hf_mask])
        sigma = np.median(hf, axis=-1) / 0.6745  # (Hb, Wb)

        thr = self.tau * sigma[..., None, None]
        keep = np.abs(x) > thr
        keep[..., 0, 0] = True  # always keep DC
        k_map = keep.sum(axis=(-1, -2)) - 1  # retained AC count per block

        xf = np.where(keep, x, 0.0)
        rec = np.matmul(np.matmul(d.T, xf), d)  # D^T X D
        filtered = unblockify(rec)
        residual = crop - filtered

        feats = common_residual_features(filtered, residual)

        # theta-map features: the estimated parameters are features themselves
        n_ac = n * n - 1
        feats["k_mean"] = float(k_map.mean())
        feats["k_std"] = float(k_map.std())
        feats["k_entropy"] = stats.histogram_entropy(
            k_map, bins=np.arange(0, n_ac + 5, 4)
        )
        feats["k_zero_frac"] = float(np.mean(k_map == 0))
        feats["sigma_mean"] = float(sigma.mean())

        # ringing localization: residual energy near strong edges vs elsewhere
        gm = gradient_magnitude(crop)
        hi = gm > np.percentile(stat_view(gm), 75)
        e_hi = float(np.mean(residual[hi] ** 2)) if hi.any() else 0.0
        e_lo = float(np.mean(residual[~hi] ** 2)) if (~hi).any() else 0.0
        feats["ring_ratio"] = e_hi / (e_lo + 1e-12)

        return ProbeResult(filtered=filtered, residual=residual, features=feats)
