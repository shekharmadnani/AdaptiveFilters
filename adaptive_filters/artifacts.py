"""Artifact simulation suite: controllable-severity degradations covering
compression, transmission (packet loss + concealment), and processing
artifacts. Doubles as the paired-training-data generator for the
restoration-trained adaptive DCT filter.

All functions take/return frames in [0, 255] float64 (same convention as
synthetic.jpeg_like). severity is an integer 1..5. Deterministic given
(seed, severity). Artifacts that need temporal context take prev_frame.
"""

import numpy as np

from .synthetic import jpeg_like
from .utils import box_filter


# --------------------------------------------------------------- helpers

def _rng(seed, severity):
    return np.random.default_rng(10_000 * seed + severity)


def _band_rows(h, frac, band=16, rng=None):
    """Random non-overlapping 16-row slice bands covering ~frac of height."""
    n = max(1, int(round(frac * h / band)))
    starts = rng.choice(np.arange(0, h - band, band), size=min(n, h // band - 1),
                        replace=False)
    return sorted(int(s) for s in starts)


# ------------------------------------------------------------- artifacts

def compression(frame, severity, prev_frame=None, seed=0):
    """JPEG-like DCT quantization (blockiness + ringing + detail loss)."""
    q = [90, 70, 50, 30, 10][severity - 1]
    return jpeg_like(frame, q)


def blur(frame, severity, prev_frame=None, seed=0):
    """Low-pass (repeated box filter): scaling/soft-focus/oversmoothing."""
    out = frame.copy()
    for _ in range(severity):
        out = box_filter(out, 1)
    return out


def noise(frame, severity, prev_frame=None, seed=0):
    """Additive Gaussian noise: sensor/transmission analog noise."""
    sigma = [2.0, 4.0, 8.0, 12.0, 16.0][severity - 1]
    r = _rng(seed, severity)
    return np.clip(frame + r.normal(0.0, sigma, frame.shape), 0, 255)


def banding(frame, severity, prev_frame=None, seed=0):
    """Bit-depth starvation / posterization: banding in gradients."""
    levels = [64, 32, 16, 8, 4][severity - 1]
    step = 256.0 / levels
    return np.clip(np.floor(frame / step) * step + step / 2.0, 0, 255)


def packet_loss_interp(frame, severity, prev_frame=None, seed=0):
    """Slice loss concealed by spatial interpolation (vertical smear across
    the lost band) -- decoder behavior without a reference frame."""
    frac = [0.03, 0.08, 0.15, 0.22, 0.30][severity - 1]
    r = _rng(seed, severity)
    out = frame.copy()
    h = frame.shape[0]
    band = 16
    for y0 in _band_rows(h, frac, band, r):
        top = out[max(y0 - 1, 0), :]
        bot = out[min(y0 + band, h - 1), :]
        t = (np.arange(band, dtype=np.float64) + 1.0) / (band + 1.0)
        out[y0 : y0 + band, :] = top[None, :] * (1 - t[:, None]) \
            + bot[None, :] * t[:, None]
    return out


def packet_loss_copy(frame, severity, prev_frame=None, seed=0):
    """Slice loss concealed by copy-from-previous-frame with a small motion
    offset: ghosting / shear at band borders. Falls back to interpolation
    concealment when no previous frame exists."""
    if prev_frame is None:
        return packet_loss_interp(frame, severity, seed=seed)
    frac = [0.03, 0.08, 0.15, 0.22, 0.30][severity - 1]
    r = _rng(seed, severity)
    out = frame.copy()
    h, w = frame.shape
    band = 16
    for y0 in _band_rows(h, frac, band, r):
        dy, dx = int(r.integers(-4, 5)), int(r.integers(-6, 7))
        src = np.roll(prev_frame, (dy, dx), axis=(0, 1))
        out[y0 : y0 + band, :] = src[y0 : y0 + band, :w]
    return out


def block_fill(frame, severity, prev_frame=None, seed=0):
    """Unconcealed macroblock loss: random 16x16 blocks filled mid-gray
    (severe transmission error / decoder giving up)."""
    frac = [0.005, 0.015, 0.03, 0.06, 0.10][severity - 1]
    r = _rng(seed, severity)
    out = frame.copy()
    h, w = frame.shape
    n = max(1, int(frac * (h // 16) * (w // 16)))
    for _ in range(n):
        y0 = int(r.integers(0, h - 16))
        x0 = int(r.integers(0, w - 16))
        out[y0 : y0 + 16, x0 : x0 + 16] = 128.0
    return out


def stale_regions(frame, severity, prev_frame=None, seed=0):
    """Concealment that 'succeeds' too well: regions frozen from the
    previous frame with no offset -- locally natural, temporally wrong.
    The spatial-blindness stress test (only temporal features can see it
    when motion is small)."""
    if prev_frame is None:
        return frame.copy()
    frac = [0.05, 0.12, 0.25, 0.40, 0.60][severity - 1]
    r = _rng(seed, severity)
    out = frame.copy()
    h, w = frame.shape
    gh, gw = h // 32, w // 32
    mask = r.random((gh, gw)) < frac
    big = np.kron(mask, np.ones((32, 32), dtype=bool))[:h, :w]
    out[big] = prev_frame[:h, :w][big]
    return out


ARTIFACTS = {
    "compression": compression,
    "blur": blur,
    "noise": noise,
    "banding": banding,
    "pl_interp": packet_loss_interp,
    "pl_copy": packet_loss_copy,
    "block_fill": block_fill,
    "stale": stale_regions,
}


def apply_artifact(name, frame, severity, prev_frame=None, seed=0):
    """severity 0 returns the frame unchanged; 1..5 applies the artifact."""
    if severity == 0:
        return frame.copy()
    return ARTIFACTS[name](frame, severity, prev_frame=prev_frame, seed=seed)
