"""Shared low-level image utilities (NumPy only)."""

import numpy as np


def to_float(frame):
    """Convert a frame to float64 luma in [0, 1]."""
    f = np.asarray(frame, dtype=np.float64)
    if f.ndim == 3:  # naive luma if a color image sneaks in
        f = f.mean(axis=2)
    if f.max() > 2.0:
        f = f / 255.0
    return f


def downsample2(img):
    """2x2 average downsampling (crops to even size)."""
    h, w = img.shape
    h -= h % 2
    w -= w % 2
    c = img[:h, :w]
    return 0.25 * (c[0::2, 0::2] + c[1::2, 0::2] + c[0::2, 1::2] + c[1::2, 1::2])


def box_filter(img, radius):
    """Mean filter with square window (2r+1)^2 via integral image, edge-padded."""
    k = 2 * radius + 1
    p = np.pad(img, radius, mode="edge")
    c = np.cumsum(p, axis=0)
    c = np.vstack([np.zeros((1, p.shape[1])), c])
    s = c[k:, :] - c[:-k, :]
    c2 = np.cumsum(s, axis=1)
    c2 = np.hstack([np.zeros((s.shape[0], 1)), c2])
    return (c2[:, k:] - c2[:, :-k]) / float(k * k)


def blockify(img, n):
    """Crop to a multiple of n and return (crop, blocks[Hb, Wb, n, n])."""
    h, w = img.shape
    hb, wb = h // n, w // n
    crop = img[: hb * n, : wb * n]
    blocks = crop.reshape(hb, n, wb, n).transpose(0, 2, 1, 3)
    return crop, blocks


def unblockify(blocks):
    """Inverse of blockify: (Hb, Wb, n, n) -> (Hb*n, Wb*n)."""
    hb, wb, n, _ = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(hb * n, wb * n)


def gradient_magnitude(img):
    gy, gx = np.gradient(img)
    return np.sqrt(gx * gx + gy * gy)


def stat_view(x, limit=1_000_000):
    """Decimated view of a 2-D map for order statistics on large frames.

    Filters always run on full resolution; only medians/percentiles/moment
    fits use this stride-2 subsample (statistically equivalent, ~4x faster
    per level at 1080p+).
    """
    while x.ndim == 2 and x.size > limit:
        x = x[::2, ::2]
    return x
