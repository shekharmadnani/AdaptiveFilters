"""Pristine 64x64 patch sampling for training the K-predictor.

Sources, in order of preference:
  1. real masters (e.g. BVI-CC1 ORIG_MP4 HD files) decoded via ffmpeg
  2. synthetic frames (fallback, keeps the trainer self-contained)

Patches are 8-aligned so DCT blocks coincide with the natural block grid.
"""

import os

import numpy as np

from ..io import iter_ffmpeg
from ..synthetic import make_frame


def _bvi_hd_masters(bvi_dir):
    if not bvi_dir or not os.path.isdir(bvi_dir):
        return []
    return sorted(
        os.path.join(bvi_dir, f) for f in os.listdir(bvi_dir)
        if f.lower().endswith(".mp4") and "1920x1080" in f
    )


def gather_frames(bvi_dir=None, frames_per_video=5, frame_stride=60,
                  max_frame_index=290, verbose=True):
    """Decode a sparse set of frames from each master (float64 luma [0,1])."""
    frames = []
    for path in _bvi_hd_masters(bvi_dir):
        taken = 0
        for idx, f in enumerate(iter_ffmpeg(path)):
            if idx % frame_stride == 0:
                frames.append(f)
                taken += 1
            if taken >= frames_per_video or idx >= max_frame_index:
                break
        if verbose:
            print(f"  {os.path.basename(path)}: {taken} frames")
    if not frames:
        if verbose:
            print("  no masters found -- using synthetic frames")
        frames = [make_frame(s, size=512) / 255.0 for s in range(24)]
    return frames


def sample_patches(frames, n, size=64, seed=0):
    """Random 8-aligned crops across the frame pool -> (n, size, size) f32."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, size, size), dtype=np.float32)
    for i in range(n):
        f = frames[int(rng.integers(len(frames)))]
        h, w = f.shape
        y0 = int(rng.integers(0, (h - size) // 8 + 1)) * 8
        x0 = int(rng.integers(0, (w - size) // 8 + 1)) * 8
        out[i] = f[y0 : y0 + size, x0 : x0 + size]
    return out


# artifacts used to synthesize DEGRADED inputs for restoration training.
# pl_copy and stale are deliberately excluded (temporal; and held out to
# test generalization to artifact types never seen in training).
TRAIN_ARTIFACTS = ["compression", "blur", "noise", "banding",
                   "pl_interp", "block_fill"]


def gather_h264_crops(bvi_dir, n, size=128, seed=0, nframes=62,
                      verbose=True):
    """Aligned (pristine, degraded) crops from REAL H.264 encodes.

    Per master: one compression-only stream and one bitstream-corrupted
    stream (random CRF each; byte-flipped non-IDR slice NALs), both with
    in-loop deblocking disabled and concealment-deblock off. Crops are
    8-aligned and taken at identical positions in the pristine and decoded
    frames. Returns (pristine, degraded) float32 arrays in [0, 1].
    """
    import os
    import tempfile

    from ..io import ffprobe_dims
    from ..bitstream import encode_h264, corrupt_annexb, decode_gray_u8

    masters = _bvi_hd_masters(bvi_dir)
    if not masters or n <= 0:
        return None
    rng = np.random.default_rng(seed + 909)
    pris = np.empty((n, size, size), dtype=np.float32)
    degr = np.empty((n, size, size), dtype=np.float32)
    per_cfg = max(1, int(np.ceil(n / (2 * len(masters)))))
    idx = 0

    with tempfile.TemporaryDirectory() as td:
        for mi, mpath in enumerate(masters):
            if idx >= n:
                break
            w, h = ffprobe_dims(mpath)
            refs = decode_gray_u8(mpath, w, h, max_frames=nframes)
            configs = [
                (int(rng.choice([26, 32, 38, 44])), 0.0),
                (int(rng.choice([22, 28, 34, 40])),
                 float(rng.uniform(0.15, 0.4))),
            ]
            for ci, (crf, cfrac) in enumerate(configs):
                enc = os.path.join(td, f"m{mi}c{ci}.264")
                encode_h264(mpath, enc, nframes, crf)
                if cfrac > 0:
                    with open(enc, "rb") as f:
                        data = f.read()
                    with open(enc, "wb") as f:
                        f.write(corrupt_annexb(data, cfrac,
                                               seed + 100 * mi + ci))
                degs = decode_gray_u8(enc, w, h, max_frames=nframes)
                m = min(len(refs), len(degs))
                if m < 12:
                    continue
                fids = list(range(8, m, 10))
                took = 0
                while took < per_cfg and idx < n:
                    fi = fids[int(rng.integers(len(fids)))]
                    y0 = int(rng.integers(0, (h - size) // 8 + 1)) * 8
                    x0 = int(rng.integers(0, (w - size) // 8 + 1)) * 8
                    pris[idx] = refs[fi][y0:y0 + size, x0:x0 + size] \
                        .astype(np.float32) / 255.0
                    degr[idx] = degs[fi][y0:y0 + size, x0:x0 + size] \
                        .astype(np.float32) / 255.0
                    idx += 1
                    took += 1
                if verbose:
                    tag = f"corrupt {cfrac:.2f}" if cfrac > 0 else "clean"
                    print(f"  {os.path.basename(mpath)}: crf={crf} {tag} "
                          f"-> {took} crops ({len(degs)} frames decoded)")
            del refs
    return pris[:idx], degr[:idx]


def make_degraded(pristine, seed=0, clean_frac=0.15,
                  artifact_names=TRAIN_ARTIFACTS):
    """Degraded twin of each pristine crop: random artifact x severity.

    A `clean_frac` share stays pristine so the model learns to preserve
    clean content (identity on no-artifact input).
    """
    from ..artifacts import apply_artifact

    rng = np.random.default_rng(seed + 555)
    out = np.empty_like(pristine)
    for i in range(len(pristine)):
        if rng.random() < clean_frac:
            out[i] = pristine[i]
            continue
        name = artifact_names[int(rng.integers(len(artifact_names)))]
        sev = int(rng.integers(1, 6))
        out[i] = np.clip(
            apply_artifact(name, pristine[i].astype(np.float64) * 255.0,
                           sev, seed=seed + i) / 255.0, 0.0, 1.0)
    return out
