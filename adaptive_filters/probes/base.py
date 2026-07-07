"""Probe template: naturalness model -> blind theta -> residual -> features."""

from dataclasses import dataclass, field

import numpy as np

from ..features import stats
from ..utils import stat_view


@dataclass
class ProbeResult:
    filtered: np.ndarray
    residual: np.ndarray
    features: dict = field(default_factory=dict)


class Probe:
    """Base class. Subclasses implement run(frame, prev_frame) -> ProbeResult.

    `frame` is float64 luma in [0, 1]. Probes that need temporal context
    return None when prev_frame is None (the pipeline then skips them).
    """

    name = "probe"

    def run(self, frame, prev_frame=None):
        raise NotImplementedError


def common_residual_features(filtered, residual):
    """Residual statistics every probe reports.

    - res_energy    : mean r^2 (raw artifact energy seen by this probe)
    - res_sigma_mad : robust residual scale
    - res_ggd_alpha : GGD shape (natural ~ Gaussian-ish, artifacts heavy-tailed)
    - res_kurtosis  : tail weight
    - res_lag1      : residual structuredness (white -> 0)
    - res_ortho     : |corr(filtered, residual)| -- MMSE orthogonality check,
                      ~0 when the filter split content/error cleanly
    """
    r = residual
    rs = stat_view(r)  # decimated view for the order-statistic-heavy fits
    alpha, _sigma = stats.fit_ggd(rs - rs.mean())
    ch, cv = stats.lag1_correlation(r)
    return {
        "res_energy": float(np.mean(r * r)),
        "res_tile_max": _tile_max_energy(r),  # worst 64x64 tile: localized
        #                                       corruption can't hide in means
        "res_sigma_mad": stats.mad_sigma(rs),
        "res_ggd_alpha": alpha,
        "res_kurtosis": stats.kurtosis(rs),
        "res_lag1": 0.5 * (abs(ch) + abs(cv)),
        "res_ortho": abs(stats.pearson(filtered, r)),
    }


def _tile_max_energy(r, tile=64):
    h, w = r.shape
    th, tw = h // tile, w // tile
    if th == 0 or tw == 0:
        return float(np.mean(r * r))
    v = r[: th * tile, : tw * tile].reshape(th, tile, tw, tile)
    return float((v ** 2).mean(axis=(1, 3)).max())
