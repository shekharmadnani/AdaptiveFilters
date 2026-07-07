"""Residual statistics shared by every probe.

Key references:
  - GGD moment-matching fit: Mittal et al., BRISQUE, IEEE TIP 2012.
  - MAD robust sigma: Donoho & Johnstone, Biometrika 1994.
  - Immerkaer fast noise estimation, CVIU 1996.
"""

import math

import numpy as np

# ---------------------------------------------------------------- GGD fit
# For a generalized Gaussian with shape alpha:
#   rho(alpha) = E[|x|]^2 / E[x^2] = Gamma(2/a)^2 / (Gamma(1/a) * Gamma(3/a))
# rho is monotonic in alpha, so we invert it with a precomputed grid.

_ALPHA_GRID = np.linspace(0.2, 10.0, 981)


def _ggd_ratio(alpha):
    return math.gamma(2.0 / alpha) ** 2 / (
        math.gamma(1.0 / alpha) * math.gamma(3.0 / alpha)
    )


_RATIO_GRID = np.array([_ggd_ratio(a) for a in _ALPHA_GRID])


def fit_ggd(x):
    """Moment-matching GGD fit. Returns (shape alpha, scale sigma).

    alpha == 2 is Gaussian; structured/artifact residuals push alpha well
    below the natural range, which is why alpha is a quality feature.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 16:
        return 2.0, float(np.std(x)) if x.size else 0.0
    v = float(np.mean(x * x))
    if v <= 1e-16:
        return 2.0, 0.0
    m = float(np.mean(np.abs(x)))
    rho = (m * m) / v
    alpha = float(_ALPHA_GRID[int(np.argmin(np.abs(_RATIO_GRID - rho)))])
    return alpha, math.sqrt(v)


# ------------------------------------------------------------ robust sigma
def mad_sigma(x):
    """Robust noise-scale estimate: median absolute deviation / 0.6745."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) / 0.6745)


def estimate_noise_sigma(img):
    """Immerkaer's fast blind noise-sigma estimator (Laplacian-difference kernel)."""
    i = np.asarray(img, dtype=np.float64)
    if i.shape[0] < 3 or i.shape[1] < 3:
        return 0.0
    k = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
    h, w = i.shape
    acc = np.zeros((h - 2, w - 2))
    for dy in range(3):
        for dx in range(3):
            acc += k[dy, dx] * i[dy : dy + h - 2, dx : dx + w - 2]
    return float(
        math.sqrt(math.pi / 2.0) * np.abs(acc).sum() / (6.0 * (w - 2) * (h - 2))
    )


# --------------------------------------------------------- misc statistics
def kurtosis(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 16:
        return 3.0
    x = x - x.mean()
    m2 = float(np.mean(x * x))
    if m2 <= 1e-16:
        return 3.0
    return float(np.mean(x ** 4) / (m2 * m2))


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(np.mean(a * a)) * float(np.mean(b * b))) + 1e-16
    return float(np.mean(a * b) / den)


def lag1_correlation(r):
    """Horizontal/vertical lag-1 autocorrelation of a residual map.

    A residual that is only removed noise is ~white (corr ~ 0); structure in
    the residual (destroyed content or coherent artifact) raises |corr|.
    """
    r = np.asarray(r, dtype=np.float64)
    r = r - r.mean()

    def _corr(a, b):
        den = math.sqrt(float(np.mean(a * a)) * float(np.mean(b * b))) + 1e-16
        return float(np.mean(a * b) / den)

    ch = _corr(r[:, :-1], r[:, 1:])
    cv = _corr(r[:-1, :], r[1:, :])
    return ch, cv


def histogram_entropy(values, bins):
    h, _ = np.histogram(np.asarray(values).ravel(), bins=bins)
    total = h.sum()
    if total == 0:
        return 0.0
    p = h[h > 0] / float(total)
    return float(-(p * np.log2(p)).sum())


def spearman(a, b):
    """Spearman rank correlation (used by the validation harness/demo)."""

    def _rank(x):
        order = np.argsort(np.asarray(x, dtype=np.float64))
        ranks = np.empty(len(order))
        ranks[order] = np.arange(len(order), dtype=np.float64)
        return ranks

    return pearson(_rank(a), _rank(b))
