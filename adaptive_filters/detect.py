"""Spatial-discontinuity error detection from adaptive-filter residuals.

Combines several LOCALIZED detectors, each sensitive to a different error
fingerprint, and flags a block only when independent detectors AGREE -- so
natural content (which fools at most one) survives, while real errors
(which trip a characteristic subset) are caught.

Detectors, per 8x8 block on the luma plane:
  wiener_res : residual energy of the gen-4 Wiener filter -- deviation
               from the learned natural-content model
  wiener_t   : the synthesis-effort / damage map (t-field)
  blocking   : shifted-DCT boundary energy -- a discontinuity running
               along the 8-px coding grid (blocking / tiling seams), which
               a within-block transform cannot see directly
  seam       : row/column projection saliency -- long axis-aligned bands
               (slice loss, concealment stripes)
  temporal   : motion-robust temporal residual -- structure no local
               motion can explain (packet-loss regions that do not track
               the scene). Needs a previous frame.

Decision: each detector is z-scored against a pristine baseline
(`calibrate`); a block is an error candidate when its z exceeds `margin`
for at least `votes` detectors. Per-frame flag = flagged-area fraction
above `area_thr`. The agreement rule is what suppresses the false alarms
any single detector suffers on busy natural texture.
"""

import numpy as np

from .probes.dct_probe import dct_matrix
from .learned.adaptive_dct import AdaptiveWienerFilter

_EPS = 1e-9


# ------------------------------------------------------------ block utils

def _block_mean(m, n=8):
    hb, wb = m.shape[0] // n, m.shape[1] // n
    return m[: hb * n, : wb * n].reshape(hb, n, wb, n).mean(axis=(1, 3))


def _blocks(gray, n=8, off=(0, 0)):
    oy, ox = off
    g = gray[oy:, ox:]
    hb, wb = g.shape[0] // n, g.shape[1] // n
    g = g[: hb * n, : wb * n]
    return g.reshape(hb, n, wb, n).transpose(0, 2, 1, 3)


def _ac_energy(gray, dmat, off):
    """sqrt of per-block AC energy on an 8x8 grid at pixel offset `off`."""
    b = _blocks(gray, 8, off)
    x = np.matmul(np.matmul(dmat, b), dmat.T)      # (Hb, Wb, 8, 8)
    e = (x ** 2).sum(axis=(-1, -2)) - x[..., 0, 0] ** 2
    return np.sqrt(np.clip(e, 0, None))


def _box3(m):
    return _boxN(m, 1)


def _boxN(m, rad):
    k = 2 * rad + 1
    p = np.pad(m, rad, mode="edge")
    acc = np.zeros_like(m)
    for dy in range(k):
        for dx in range(k):
            acc += p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return acc / (k * k)


def _mad(v):
    v = np.asarray(v).ravel()
    return float(np.median(np.abs(v - np.median(v))) * 1.4826)


# --------------------------------------------------------------- detector

class SpatialErrorDetector:
    def __init__(self, weights, device=None, mc_window=5):
        self.filter = AdaptiveWienerFilter(weights, device=device)
        self.dmat = dct_matrix(8)
        self.mc_window = mc_window
        self.baseline = {}          # name -> (median, mad)

    # ---- individual detector maps (each returns a (H/8, W/8) array)

    def _wiener_maps(self, gray_or_color, lum):
        r = self.filter.apply(gray_or_color, light=True)
        resid = r.residual
        if resid.ndim == 3:
            resid = resid[:, :, 0]
        # normalize residual energy by local activity so the channel
        # measures "unexplained relative to how busy the content is",
        # not raw texture energy (which the agreement rule must not
        # double-count with blocking/seam)
        gy, gx = np.gradient(lum[: resid.shape[0], : resid.shape[1]])
        act = _block_mean(gx ** 2 + gy ** 2)
        res_e = _block_mean(resid ** 2)
        gh = min(res_e.shape[0], act.shape[0])
        gw = min(res_e.shape[1], act.shape[1])
        res_norm = res_e[:gh, :gw] / (act[:gh, :gw] + 1e-4)
        t = r.t_map[0] if (r.t_map is not None and r.t_map.ndim == 3) \
            else r.t_map
        return res_norm, t

    def _blocking_map(self, gray):
        """Grid-phase blockiness: jump ACROSS the 8-px block boundary
        relative to jumps INSIDE the block. Natural texture has
        boundary ~ interior (ratio ~1); blocking spikes the boundary
        specifically at the coding-grid phase (ratio >> 1). Content-
        normalized (a ratio) and phase-specific -- so it does not fire
        on generic high-frequency content."""
        h, w = gray.shape

        def _axis(diff):                     # diff along the tested axis
            wd = (diff.shape[1] // 8) * 8
            g = diff[:, :wd].reshape(diff.shape[0], -1, 8)
            boundary = g[:, :, 7]            # jump at the block seam
            interior = g[:, :, :7].mean(axis=2) + _EPS
            ratio = boundary / interior
            hb = ratio.shape[0] // 8
            return ratio[:hb * 8].reshape(hb, 8, -1).mean(axis=1)

        bv = _axis(np.abs(gray[:, 1:] - gray[:, :-1]))          # vertical
        bh = _axis(np.abs(gray[1:, :] - gray[:-1, :]).T).T      # horizontal
        gh = min(bv.shape[0], bh.shape[0])
        gw = min(bv.shape[1], bh.shape[1])
        return np.maximum(bv[:gh, :gw], bh[:gh, :gw])

    def _seam_map(self, gray):
        # residual-free structural seam test: energy of row/col second
        # differences, salient rows/cols = axis-aligned bands
        d2r = np.abs(gray[2:, :] - 2 * gray[1:-1, :] + gray[:-2, :])
        d2c = np.abs(gray[:, 2:] - 2 * gray[:, 1:-1] + gray[:, :-2])
        row = d2r.mean(axis=1)
        col = d2c.mean(axis=0)
        rz = np.clip((row - np.median(row)) / (_mad(row) + _EPS), 0, None)
        cz = np.clip((col - np.median(col)) / (_mad(col) + _EPS), 0, None)
        sal = np.maximum(rz[1:-1, None], cz[None, 1:-1])
        return _block_mean(sal)

    def _smooth_map(self, gray):
        """Unnatural-smoothness anomaly: a block markedly flatter than its
        LOCAL neighborhood. Catches interpolated concealment bands (a
        smooth strip cutting through textured content) without firing on
        legitimately flat regions (whose neighbors are equally flat)."""
        gy, gx = np.gradient(gray)
        act = _block_mean(gx ** 2 + gy ** 2)
        neigh = _boxN(act, 3)
        return np.clip(neigh - act, 0, None) / (neigh + 1e-4)

    def _overshoot_map(self, gray):
        """SAO-style over/undershoot: pixels that exceed the range of
        their 4-neighbors (ringing / concealment halos). Natural
        monotonic edges sit BETWEEN their neighbors and score ~0."""
        up = np.roll(gray, 1, 0); dn = np.roll(gray, -1, 0)
        lf = np.roll(gray, 1, 1); rt = np.roll(gray, -1, 1)
        mx = np.maximum(np.maximum(up, dn), np.maximum(lf, rt))
        mn = np.minimum(np.minimum(up, dn), np.minimum(lf, rt))
        over = np.clip(gray - mx, 0, None) + np.clip(mn - gray, 0, None)
        return _block_mean(over)

    def _incoherence_map(self, gray):
        """Torn-edge test: gradient energy that is NOT locally coherent.
        A natural edge has one dominant orientation (high structure-tensor
        coherence); a concealment tear is strong but incoherent. Runs on
        the GPU (structure tensor + 8x8 block mean via pooling)."""
        torch = self.filter._torch
        import torch.nn.functional as F
        with torch.inference_mode():
            g = torch.from_numpy(np.ascontiguousarray(gray)).to(
                self.filter.device)[None, None]
            gp = F.pad(g, (1, 1, 1, 1), mode="replicate")
            gx = (gp[..., 1:-1, 2:] - gp[..., 1:-1, :-2]) * 0.5
            gy = (gp[..., 2:, 1:-1] - gp[..., :-2, 1:-1]) * 0.5

            def box(m):
                return F.avg_pool2d(m, 5, 1, 2, count_include_pad=False)

            jxx, jyy, jxy = box(gx * gx), box(gy * gy), box(gx * gy)
            tr = jxx + jyy
            coh = torch.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (tr + 1e-10)
            sig = tr * (1.0 - coh)
            bm = F.avg_pool2d(sig, 8, 8)[0, 0]
            return bm.cpu().numpy().astype(np.float64)

    def _temporal_map(self, gray, prev):
        """Motion-robust temporal residual: min over a search window of
        |cur - shifted prev|. Runs on the GPU (already loaded for the
        filter) -- 121 shifts on CPU cost ~1.8 s at 1080p, ~15 ms here."""
        torch = self.filter._torch
        dev = self.filter.device
        win = self.mc_window
        with torch.inference_mode():
            c = torch.from_numpy(np.ascontiguousarray(gray)).to(dev)
            p = torch.from_numpy(np.ascontiguousarray(prev)).to(dev)
            h, w = c.shape
            pp = torch.nn.functional.pad(
                p[None, None], (win, win, win, win), mode="replicate")[0, 0]
            best = None
            for dy in range(2 * win + 1):
                for dx in range(2 * win + 1):
                    diff = (c - pp[dy:dy + h, dx:dx + w]).abs_()
                    best = diff if best is None else torch.minimum(best, diff)
            bm = best[: h // 8 * 8, : w // 8 * 8].reshape(
                h // 8, 8, w // 8, 8).mean(dim=(1, 3))
            return bm.cpu().numpy().astype(np.float64)

    # ---- raw maps for a frame (dict name -> map)

    def _maps(self, frame, prev=None):
        # keep the filter input in its native dtype (uint8 uploads cheaply
        # and converts on the GPU); do the numpy channels in float32
        raw = np.asarray(frame)
        if raw.dtype == np.uint8:
            lum = (raw[:, :, 0] if raw.ndim == 3 else raw).astype(
                np.float32) / 255.0
            wiener_in = raw
        else:
            f = raw.astype(np.float32, copy=False)
            if f.max() > 2.0:
                f = f / 255.0
            lum = f[:, :, 0] if f.ndim == 3 else f
            wiener_in = f
        res_e, t = self._wiener_maps(wiener_in, lum)
        out = {"wiener_res": res_e, "blocking": self._blocking_map(lum),
               "seam": self._seam_map(lum), "smooth": self._smooth_map(lum),
               "overshoot": self._overshoot_map(lum),
               "incoherence": self._incoherence_map(lum)}
        if t is not None:
            out["wiener_t"] = t
        if prev is not None:
            pr = np.asarray(prev)
            p = (pr[:, :, 0] if pr.ndim == 3 else pr).astype(np.float32)
            if p.max() > 2.0:
                p = p / 255.0
            out["temporal"] = self._temporal_map(lum, p)
        # crop every map to the common grid size
        gh = min(m.shape[0] for m in out.values())
        gw = min(m.shape[1] for m in out.values())
        return {k: v[:gh, :gw] for k, v in out.items()}

    # ---- calibration on pristine content

    def calibrate(self, pristine_frames, prev_frames=None):
        acc = {}
        prevs = prev_frames or [None] * len(pristine_frames)
        for f, p in zip(pristine_frames, prevs):
            for k, v in self._maps(f, p).items():
                acc.setdefault(k, []).append(v.ravel())
        for k, vals in acc.items():
            allv = np.concatenate(vals)
            self.baseline[k] = (float(np.median(allv)), _mad(allv) + _EPS)
        return self

    # ---- detection

    def detect(self, frame, prev=None, margin=4.0, votes=3, area_thr=0.004,
               require=("blocking", "temporal")):
        """A block is flagged when >= `votes` detectors exceed `margin`
        AND at least one DISCRIMINATIVE detector (`require`) fires. The
        gate channels have false-alarm patterns independent of natural
        texture -- grid-phase blocking (coding seams), motion-robust
        temporal (regions not tracking motion), and the smoothness
        anomaly (interpolated bands). The remaining channels only
        CONTRIBUTE votes: they raise recall at genuine seams but cannot
        trip a flag alone, so texture co-flags stay suppressed."""
        maps = self._maps(frame, prev)
        gh, gw = next(iter(maps.values())).shape
        z = {}
        vote = np.zeros((gh, gw))
        disc = np.zeros((gh, gw), bool)
        for k, m in maps.items():
            med, mad = self.baseline.get(k, (float(np.median(m)), _mad(m)))
            zk = (m - med) / (mad + _EPS)
            z[k] = zk
            fired = zk > margin
            vote += fired.astype(float)
            if k in require:
                disc |= fired
        error_mask = (vote >= votes) & disc
        frac = float(error_mask.mean())
        return {
            "error_mask": error_mask,       # (H/8, W/8) bool
            "votes": vote,                  # agreement heatmap
            "z": z,                         # per-detector z maps
            "flagged_fraction": frac,
            "frame_flag": frac > area_thr,
            "detectors": list(maps),
        }
