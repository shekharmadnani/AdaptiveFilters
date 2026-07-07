"""Fixed-K DCT filter: the NON-adaptive baseline.

Keeps DC + a fixed number K of AC coefficients (largest magnitudes) in
every 8x8 block, regardless of content. This is the reference the
content-adaptive filters (blind-threshold DCT, learned Wiener) must beat:
its residual confounds content complexity with distortion, since K does
not adapt to either.
"""

import numpy as np

from ..utils import blockify, unblockify
from .base import Probe, ProbeResult, common_residual_features
from .dct_probe import dct_matrix


class FixedKDctProbe(Probe):
    name = "fdct"

    def __init__(self, k=6, block=8):
        self.k = k
        self.block = block
        self._d = dct_matrix(block)

    def run(self, frame, prev_frame=None):
        n = self.block
        crop, blocks = blockify(frame, n)
        d = self._d
        x = np.matmul(np.matmul(d, blocks), d.T)

        hb, wb = x.shape[:2]
        flat = x.reshape(hb, wb, n * n)
        ac = np.abs(flat[..., 1:])
        kth = np.partition(ac, -self.k, axis=-1)[..., -self.k]
        keep_ac = ac >= kth[..., None]
        mask = np.concatenate(
            [np.ones(flat.shape[:-1] + (1,), dtype=bool), keep_ac], axis=-1)

        xf = np.where(mask.reshape(x.shape), x, 0.0)
        rec = np.matmul(np.matmul(d.T, xf), d)
        filtered = unblockify(rec)
        residual = crop - filtered

        feats = common_residual_features(filtered, residual)
        energy = flat ** 2
        feats["kept_energy_frac"] = float(
            energy[mask].sum() / (energy.sum() + 1e-12))
        return ProbeResult(filtered=filtered, residual=residual,
                           features=feats)
