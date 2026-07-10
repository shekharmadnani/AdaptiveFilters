"""Training-pair generation: (degraded frame, pristine frame) patch pairs
from REAL codecs, at scale, stored on disk for reuse by every filter
generation.

Two families, as specified:

  compression : pristine stream -> H.264 / HEVC / MPEG-2 encoder with a
                randomized rate point, profile and tool set -> decode.
                Real quantization artifacts across three codec eras.
  loss        : the compressed stream is additionally subjected to packet
                loss (byte corruption inside slice NALs / MPEG-2 slices,
                parameter sets and IRAP/I pictures kept intact so the
                stream stays decodable and frame-aligned) -> decoded with
                the decoder's error resilience. Real concealment errors,
                real propagation -- the digital-tape / transmission case.

Patches are stored at 256x256 (the largest any generation needs); smaller
generations sub-crop at load time. Everything is uint8 YUV 4:4:4 at luma
resolution, 8-aligned so DCT blocks match the coding grid.

Output layout:
  out_dir/manifest.json           one record per shard
  out_dir/shard_NNNN.npz          pris (n,3,S,S) u8, deg (n,3,S,S) u8
"""

import json
import os
import re
import subprocess
import tempfile

import numpy as np

from .bitstream import decode_yuv444_u8

TARGET_W, TARGET_H = 1920, 1088   # encode resolution (divisible by 8)
PATCH = 256


# ------------------------------------------------------------- encoding

def sample_config(rng, codec):
    """Randomized rate point + profile + tool set per codec."""
    if codec == "h264":
        profile = str(rng.choice(["baseline", "main", "high"]))
        cfg = {
            "crf": int(rng.integers(20, 51)),
            "profile": profile,
            "deblock": bool(rng.random() < 0.5),
            "slices": int(rng.choice([1, 4])),
            "g": int(rng.choice([12, 30, 60])),
            "bf": 0 if profile == "baseline" else int(rng.choice([0, 3])),
        }
    elif codec == "hevc":
        cfg = {
            "crf": int(rng.integers(22, 46)),
            "sao": int(rng.random() < 0.5),
            "g": int(rng.choice([12, 30, 60])),
        }
    else:  # mpeg2
        cfg = {
            "q": int(rng.integers(4, 32)),
            "g": int(rng.choice([1, 12, 15, 30])),   # g=1: intra-only (tape)
            "bf": int(rng.choice([0, 2])),
        }
    return cfg


def encode_stream(src, out, nframes, codec, cfg, ffmpeg="ffmpeg"):
    """Encode `src` (scaled to TARGET res, 8-bit 4:2:0) to an elementary
    stream so packet-loss corruption can parse start codes directly."""
    base = [ffmpeg, "-y", "-v", "error", "-i", str(src),
            "-frames:v", str(nframes),
            "-vf", f"scale={TARGET_W}:{TARGET_H}:flags=bicubic",
            "-pix_fmt", "yuv420p"]
    if codec == "h264":
        params = ("deblock=1" if cfg["deblock"] else "no-deblock=1")
        cmd = base + ["-c:v", "libx264", "-preset", "veryfast",
                      "-crf", str(cfg["crf"]),
                      "-profile:v", cfg["profile"],
                      "-x264-params", params,
                      "-slices", str(cfg["slices"]),
                      "-g", str(cfg["g"]), "-bf", str(cfg["bf"]),
                      "-f", "h264", str(out)]
    elif codec == "hevc":
        cmd = base + ["-c:v", "libx265", "-preset", "veryfast",
                      "-crf", str(cfg["crf"]),
                      "-x265-params",
                      f"sao={cfg['sao']}:log-level=error",
                      "-g", str(cfg["g"]),
                      "-f", "hevc", str(out)]
    else:  # mpeg2 elementary stream
        cmd = base + ["-c:v", "mpeg2video", "-q:v", str(cfg["q"]),
                      "-g", str(cfg["g"]), "-bf", str(cfg["bf"]),
                      "-f", "mpeg2video", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)


# ------------------------------------------------------------ corruption

def corrupt_stream(data, codec, frac, seed, flips=6):
    """Byte-flip inside slice payloads; keep headers and random-access
    pictures intact so the stream decodes and stays frame-aligned."""
    rng = np.random.default_rng(seed)
    ba = bytearray(data)
    starts = [m.end() for m in re.finditer(b"\x00\x00\x01", data)]
    for i, p in enumerate(starts):
        end = starts[i + 1] - 3 if i + 1 < len(starts) else len(data)
        if end - p < 64:
            continue
        b0 = data[p]
        if codec == "h264":
            hit = (b0 & 0x1F) == 1                  # non-IDR slice
        elif codec == "hevc":
            hit = ((b0 >> 1) & 0x3F) <= 9           # non-IRAP slice segment
        else:  # mpeg2: slice start codes 0x01..0xAF
            hit = 0x01 <= b0 <= 0xAF
        if hit and rng.random() < frac:
            for _ in range(flips):
                off = int(rng.integers(p + 8, end - 1))
                ba[off] ^= 0xFF
    return bytes(ba)


# ---------------------------------------------------------------- driver

def _crop_pairs(pris_f, deg_f, k, rng, damage_biased, out_p, out_d, filled):
    """Extract k aligned 256x256 patch pairs from one frame pair."""
    h, w = pris_f.shape[:2]
    for _ in range(k):
        if filled >= len(out_p):
            break
        best = None
        for _try in range(8):
            y0 = int(rng.integers(0, (h - PATCH) // 8 + 1)) * 8
            x0 = int(rng.integers(0, (w - PATCH) // 8 + 1)) * 8
            p = pris_f[y0:y0 + PATCH, x0:x0 + PATCH]
            d = deg_f[y0:y0 + PATCH, x0:x0 + PATCH]
            if not damage_biased:
                best = (p, d)
                break
            mad = float(np.mean(np.abs(p.astype(np.int16)
                                       - d.astype(np.int16))))
            if best is None or mad > best[2]:
                best = (p, d, mad)
            if mad > 0.5:   # visibly damaged patch -- good enough
                break
        p, d = best[0], best[1]
        out_p[filled] = p.transpose(2, 0, 1)
        out_d[filled] = d.transpose(2, 0, 1)
        filled += 1
    return filled


def generate_pairs(sources, out_dir, clips=60, nframes=40,
                   frame_ids=(10, 22, 34), patches_per_frame=20,
                   loss_frac_range=(0.15, 0.45), seed=0, verbose=True):
    """Walk the source clips; per clip produce one compression shard and
    one loss shard (codec cycled h264 -> hevc -> mpeg2)."""
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for sdir in sources:
        files += sorted(os.path.join(sdir, f) for f in os.listdir(sdir)
                        if f.lower().endswith(".mp4"))
    if len(files) > clips:
        idx = np.linspace(0, len(files) - 1, clips).astype(int)
        files = [files[i] for i in idx]

    rng = np.random.default_rng(seed)
    codecs = ["h264", "hevc", "mpeg2"]
    manifest = []
    shard_id = 0

    for ci, src in enumerate(files):
        codec = codecs[ci % 3]
        try:
            with tempfile.TemporaryDirectory() as td:
                # pristine reference, decoded AND scaled to encode resolution
                pris = _decode_scaled(src, nframes)
                if len(pris) <= max(frame_ids):
                    continue

                for family in ("compression", "loss"):
                    cfg = sample_config(rng, codec)
                    enc = os.path.join(td, f"s{shard_id}.{codec}")
                    encode_stream(src, enc, nframes, codec, cfg)
                    if family == "loss":
                        with open(enc, "rb") as f:
                            data = f.read()
                        frac = float(rng.uniform(*loss_frac_range))
                        cfg["loss_frac"] = round(frac, 3)
                        with open(enc, "wb") as f:
                            f.write(corrupt_stream(data, codec, frac,
                                                   seed + shard_id))
                    degs = decode_yuv444_u8(enc, TARGET_W, TARGET_H,
                                            max_frames=nframes)
                    m = min(len(pris), len(degs))
                    fids = [i for i in frame_ids if i < m]
                    if not fids:
                        continue

                    n = len(fids) * patches_per_frame
                    out_p = np.empty((n, 3, PATCH, PATCH), np.uint8)
                    out_d = np.empty_like(out_p)
                    filled = 0
                    for fi in fids:
                        filled = _crop_pairs(pris[fi], degs[fi],
                                             patches_per_frame, rng,
                                             family == "loss",
                                             out_p, out_d, filled)
                    shard = f"shard_{shard_id:04d}.npz"
                    np.savez_compressed(os.path.join(out_dir, shard),
                                        pris=out_p[:filled],
                                        deg=out_d[:filled])
                    manifest.append({
                        "shard": shard, "family": family, "codec": codec,
                        "src": os.path.basename(src), "config": cfg,
                        "n": filled,
                    })
                    shard_id += 1
                    if verbose:
                        print(f"  [{ci + 1}/{len(files)}] {codec:<6} "
                              f"{family:<12} {os.path.basename(src)[:40]:<40}"
                              f" -> {filled} pairs  {cfg}")
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"  skip {os.path.basename(src)} ({codec}): "
                      f"{e.stderr.decode(errors='replace')[:120]}")
            continue
        # persist the manifest incrementally (resumable inspection)
        with open(os.path.join(out_dir, "manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)

    total = sum(e["n"] for e in manifest)
    if verbose:
        print(f"Done: {len(manifest)} shards, {total} pairs -> {out_dir}")
    return manifest


def _decode_scaled(src, nframes, ffmpeg="ffmpeg"):
    """Pristine frames decoded AND scaled to the encode resolution so the
    pairs align pixel-for-pixel with the codec output."""
    from .io import _read_exact

    cmd = [ffmpeg, "-v", "error", "-i", str(src), "-frames:v", str(nframes),
           "-vf", f"scale={TARGET_W}:{TARGET_H}:flags=bicubic",
           "-f", "rawvideo", "-pix_fmt", "yuv444p", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    frames = []
    try:
        while len(frames) < nframes:
            buf = _read_exact(proc.stdout, 3 * TARGET_W * TARGET_H)
            if buf is None:
                break
            frames.append(np.frombuffer(buf, np.uint8)
                          .reshape(3, TARGET_H, TARGET_W)
                          .transpose(1, 2, 0).copy())
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    return frames
