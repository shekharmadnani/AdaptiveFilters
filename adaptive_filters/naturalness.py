"""Opinion-unaware naturalness anchor (NIQE-style, layer B of the plan).

Fit the feature distribution on known-good (pristine/high-bitrate) content;
score = distance from that distribution. Zero labels required. Doubles as
the drift monitor for the supervised fusion stage.

Reference: Mittal, Soundararajan, Bovik, NIQE, IEEE SPL 2013.
"""

import numpy as np


class NaturalnessModel:
    """Diagonal-covariance Mahalanobis distance to the pristine corpus.

    Diagonal by default (robust with small corpora); switches to full
    covariance only when samples >> dimensions.
    """

    def __init__(self, full_cov_ratio=5.0, z_clip=30.0):
        self.full_cov_ratio = full_cov_ratio
        self.z_clip = z_clip  # cap per-feature deviation so one near-constant
        #                       pristine feature cannot dominate the distance
        self.mu = None
        self.sigma = None
        self._cov_inv = None

    def fit(self, vectors):
        x = np.asarray(vectors, dtype=np.float64)
        n, d = x.shape
        self.mu = x.mean(axis=0)
        sigma = x.std(axis=0)
        floor = 1e-2 * (np.abs(self.mu) + 1e-2)
        self.sigma = np.maximum(sigma, floor)

        self._cov_inv = None
        if n >= self.full_cov_ratio * d:
            cov = np.cov(x, rowvar=False) + np.diag(floor ** 2)
            self._cov_inv = np.linalg.inv(cov)
        return self

    def score(self, vector):
        """Larger = less natural. Normalized by sqrt(dims) for scale stability."""
        v = np.asarray(vector, dtype=np.float64)
        z = v - self.mu
        if self._cov_inv is not None:
            q = float(z @ self._cov_inv @ z)
            return float(np.sqrt(max(q, 0.0) / len(z)))
        zn = np.clip(z / self.sigma, -self.z_clip, self.z_clip)
        return float(np.sqrt(np.mean(zn * zn)))

    def to_dict(self):
        return {
            "z_clip": self.z_clip,
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "cov_inv": self._cov_inv.tolist() if self._cov_inv is not None else None,
        }

    @classmethod
    def from_dict(cls, d):
        m = cls(z_clip=d.get("z_clip", 30.0))
        m.mu = np.array(d["mu"], dtype=np.float64)
        m.sigma = np.array(d["sigma"], dtype=np.float64)
        ci = d.get("cov_inv")
        m._cov_inv = np.array(ci, dtype=np.float64) if ci is not None else None
        return m

    def top_deviations(self, vector, names, k=5):
        """Diagnostics: which features deviate most (drives per-artifact triage)."""
        v = np.asarray(vector, dtype=np.float64)
        z = np.abs((v - self.mu) / self.sigma)
        order = np.argsort(z)[::-1][:k]
        return [(names[j], float(z[j])) for j in order]
