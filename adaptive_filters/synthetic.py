"""Synthetic content and JPEG-like degradation (shared by demo, training,
and validation harnesses). Frames are float64 in [0, 255].
"""

import numpy as np

from .utils import blockify, unblockify
from .probes.dct_probe import dct_matrix


def make_frame(seed, size=256):
    """Synthetic pristine frame: gradient + rectangles + texture + grain."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / float(size)

    a, b = rng.uniform(-1, 1, 2)
    img = 110.0 + 70.0 * (a * xx + b * yy)

    for _ in range(6):  # sharp-edged rectangles
        y0, x0 = rng.integers(0, size - 40, 2)
        hgt, wid = rng.integers(20, 90, 2)
        img[y0 : y0 + hgt, x0 : x0 + wid] = rng.uniform(30, 225)

    for _ in range(2):  # oriented sinusoid texture patches
        y0, x0 = rng.integers(0, size - 64, 2)
        theta = rng.uniform(0, np.pi)
        freq = rng.uniform(4, 12)
        py, px = np.mgrid[0:64, 0:64] / 64.0
        wave = 18.0 * np.sin(
            2 * np.pi * freq * (px * np.cos(theta) + py * np.sin(theta))
        )
        img[y0 : y0 + 64, x0 : x0 + 64] += wave

    grain = rng.normal(0.0, 1.0, (size, size))  # fine natural grain
    grain = 0.25 * (grain + np.roll(grain, 1, 0) + np.roll(grain, 1, 1)
                    + np.roll(grain, (1, 1), (0, 1)))
    img += 5.0 * grain

    return np.clip(img, 0, 255)


_JPEG_Q = np.array([  # standard JPEG luminance quantization table
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)


def jpeg_like(frame, quality):
    """JPEG-like degradation: 8x8 DCT + luminance-table quantization.

    `frame` in [0, 255]; returns the same range.
    """
    s = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    table = np.clip(np.floor((_JPEG_Q * s + 50.0) / 100.0), 1, 255)
    d = dct_matrix(8)
    crop, blocks = blockify(frame - 128.0, 8)
    x = np.matmul(np.matmul(d, blocks), d.T)
    xq = np.round(x / table) * table
    rec = np.matmul(np.matmul(d.T, xq), d)
    out = frame.copy()
    out[: crop.shape[0], : crop.shape[1]] = unblockify(rec) + 128.0
    return np.clip(out, 0, 255)
