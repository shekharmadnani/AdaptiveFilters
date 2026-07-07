"""Grid-aligned discontinuity (deblocking) probe.

Naturalness model : natural content is statistically identical on and off
                    the 8-px coding grid.
Blind theta       : adaptive clipping threshold beta from the off-grid
                    neighbor-difference statistics (H.264-deblocking style,
                    reference-free by construction).
Residual exposes  : blockiness -- residual energy is localized on the grid.
Theta features    : on/off-grid difference ratio per direction and grid-phase
                    peak ratio (detects the grid even at unknown alignment).

Reference: List et al., "Adaptive Deblocking Filter", IEEE TCSVT 2003.
"""

import numpy as np

from .base import Probe, ProbeResult, common_residual_features


class DeblockProbe(Probe):
    name = "dbk"

    def __init__(self, grid=8):
        self.grid = grid

    @staticmethod
    def _phase_stats(diffs, grid):
        """Mean |neighbor difference| grouped by grid phase.

        diffs[:, j] = |I[:, j+1] - I[:, j]|; the boundary between coding
        blocks (x-1, x) with x % grid == 0 lands at phase (j+1) % grid == 0.
        """
        n = diffs.shape[1]
        idx = (np.arange(n) + 1) % grid
        means = np.array(
            [diffs[:, idx == k].mean() if (idx == k).any() else 0.0
             for k in range(grid)]
        )
        on = means[0]
        off = np.median(means[1:])
        return on, off, means

    def run(self, frame, prev_frame=None):
        i = frame
        g = self.grid
        h, w = i.shape

        dv = np.abs(i[:, 1:] - i[:, :-1])
        dh = np.abs(i[1:, :] - i[:-1, :])

        on_v, off_v, means_v = self._phase_stats(dv, g)
        on_h, off_h, means_h = self._phase_stats(dh.T, g)

        # adaptive clipping threshold from off-grid (content) statistics;
        # decimate rows/cols (not the phase axis) to keep the median cheap
        off_all = np.concatenate(
            [dv[::2, (np.arange(dv.shape[1]) + 1) % g != 0].ravel()[::4],
             dh[(np.arange(dh.shape[0]) + 1) % g != 0, ::2].ravel()[::4]]
        )
        beta = 4.0 * float(np.median(off_all)) + 2e-3

        filtered = i.copy()
        active = []

        # weak deblocking across vertical boundaries (columns x = g, 2g, ...)
        for x in range(g, w - 1, g):
            p, q = i[:, x - 1], i[:, x]
            delta = q - p
            m = np.abs(delta) < beta
            active.append(float(m.mean()))
            filtered[:, x - 1][m] += delta[m] / 4.0
            filtered[:, x][m] -= delta[m] / 4.0
            if x - 2 >= 0:
                filtered[:, x - 2][m] += delta[m] / 8.0
            if x + 1 < w:
                filtered[:, x + 1][m] -= delta[m] / 8.0

        # horizontal boundaries (rows y = g, 2g, ...)
        for y in range(g, h - 1, g):
            p, q = i[y - 1, :], i[y, :]
            delta = q - p
            m = np.abs(delta) < beta
            active.append(float(m.mean()))
            filtered[y - 1, :][m] += delta[m] / 4.0
            filtered[y, :][m] -= delta[m] / 4.0
            if y - 2 >= 0:
                filtered[y - 2, :][m] += delta[m] / 8.0
            if y + 1 < h:
                filtered[y + 1, :][m] -= delta[m] / 8.0

        residual = i - filtered
        feats = common_residual_features(filtered, residual)

        feats["blockiness_v"] = on_v / (off_v + 1e-9)
        feats["blockiness_h"] = on_h / (off_h + 1e-9)
        # grid detectability at unknown phase: peak-to-median across phases
        feats["grid_peak_v"] = float(means_v.max() / (np.median(means_v) + 1e-9))
        feats["grid_peak_h"] = float(means_h.max() / (np.median(means_h) + 1e-9))
        feats["active_frac"] = float(np.mean(active)) if active else 0.0
        feats["beta"] = beta

        return ProbeResult(filtered=filtered, residual=residual, features=feats)
