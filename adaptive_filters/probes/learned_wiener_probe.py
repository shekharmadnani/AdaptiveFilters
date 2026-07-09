"""Feature-extraction wrapper around the learned DCT-domain Wiener filter.

Identical feature contract to LearnedKMapProbe, with K interpreted as the
effective AC count (sum of gains). The residual is this filter's
contribution to the companion residual bank.
"""

import numpy as np

from .base import Probe, ProbeResult, common_residual_features


class LearnedWienerProbe(Probe):
    name = "lwn"

    def __init__(self, weights, device=None):
        from ..learned.adaptive_dct import AdaptiveWienerFilter

        self.filter = AdaptiveWienerFilter(weights, device=device)

    def run(self, frame, prev_frame=None):
        r = self.filter.apply(frame)
        crop = frame[: r.filtered.shape[0], : r.filtered.shape[1]]

        # color results: the 2-D residual/edge statistics use the luma
        # plane; the K statistics cover all channels
        filtered2d, residual2d = r.filtered, r.residual
        if residual2d.ndim == 3:
            filtered2d = filtered2d[:, :, 0]
            residual2d = residual2d[:, :, 0]
            crop = crop[:, :, 0]

        feats = common_residual_features(filtered2d, residual2d)

        dk = (r.k_pred - r.k_emp).ravel()
        feats["dk_abs_mean"] = float(np.abs(dk).mean())
        feats["dk_abs_p90"] = float(np.percentile(np.abs(dk), 90))
        feats["dk_mean"] = float(dk.mean())
        feats["dk_std"] = float(dk.std())
        feats["k_pred_mean"] = float(r.k_pred.mean())
        feats["k_emp_mean"] = float(r.k_emp.mean())
        feats["k_tail_mean"] = float(r.k_tail.mean())

        if r.t_map is not None:  # gen-4: synthesis effort = damage map
            feats["t_abs_mean"] = float(r.t_map.mean())
            feats["t_abs_p90"] = float(np.percentile(r.t_map, 90))
            feats["t_tile_max"] = float(r.t_map.max())

        gx_o = np.abs(np.diff(crop, axis=1))
        gx_r = np.abs(np.diff(filtered2d, axis=1))
        gy_o = np.abs(np.diff(crop, axis=0))
        gy_r = np.abs(np.diff(filtered2d, axis=0))
        feats["new_edge1"] = float(np.maximum(gx_r - gx_o, 0).mean()
                                   + np.maximum(gy_r - gy_o, 0).mean())
        feats["lost_edge1"] = float(np.maximum(gx_o - gx_r, 0).mean()
                                    + np.maximum(gy_o - gy_r, 0).mean())

        return ProbeResult(filtered=filtered2d, residual=residual2d,
                           features=feats)
