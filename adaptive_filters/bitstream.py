"""Real H.264 corruption for training data.

Pipeline: encode pristine video with x264 (in-loop deblocking DISABLED via
no-deblock; H.264 has no SAO), flip bytes inside non-IDR slice NALs to
emulate transmission corruption, decode with ffmpeg error resilience with
concealment-deblock OFF (-ec 1 = guess_mvs only). The decoded frames carry
real quantization artifacts, real corruption, real concealment and real
error propagation through the GOP -- with no deblocking filter anywhere in
the chain.

Frames are returned as uint8 gray (memory-friendly); convert at crop time.
"""

import re
import subprocess

import numpy as np

from .io import _read_exact


def encode_h264(src_path, out_path, nframes, crf, ffmpeg="ffmpeg"):
    """Annex-B elementary stream; loop deblocking off; 4 slices/frame so
    corruption stays localized; GOP 30 so errors propagate realistically."""
    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src_path),
           "-frames:v", str(nframes), "-c:v", "libx264",
           "-preset", "veryfast", "-crf", str(crf),
           "-x264-params", "no-deblock=1", "-slices", "4", "-g", "30",
           "-f", "h264", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def corrupt_annexb(data, frac, seed, flips=6):
    """Flip bytes inside a fraction of non-IDR slice NALs (type 1). SPS/PPS
    and IDR slices stay intact so the stream stays decodable and frame
    count is preserved (alignment with the pristine source)."""
    rng = np.random.default_rng(seed)
    ba = bytearray(data)
    starts = [m.end() for m in re.finditer(b"\x00\x00\x01", data)]
    for i, p in enumerate(starts):
        end = starts[i + 1] - 3 if i + 1 < len(starts) else len(data)
        if (data[p] & 0x1F) == 1 and end - p > 64 and rng.random() < frac:
            for _ in range(flips):
                off = int(rng.integers(p + 16, end - 1))
                ba[off] ^= 0xFF
    return bytes(ba)


def decode_gray_u8(path, width, height, max_frames=None, ffmpeg="ffmpeg"):
    """Decode any input to a list of uint8 gray frames. Error-resilient:
    ignore_err + concealment WITHOUT its deblock stage (-ec 1)."""
    return _decode_u8(path, width, height, "gray", 1, max_frames, ffmpeg)


def decode_yuv444_u8(path, width, height, max_frames=None, ffmpeg="ffmpeg"):
    """Same error-resilient decode, but color: uint8 (H, W, 3) YUV frames
    at luma resolution."""
    return _decode_u8(path, width, height, "yuv444p", 3, max_frames, ffmpeg)


def _decode_u8(path, width, height, pix_fmt, nplanes, max_frames, ffmpeg):
    cmd = [ffmpeg, "-v", "error", "-err_detect", "ignore_err", "-ec", "1",
           "-i", str(path), "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    frames = []
    try:
        while max_frames is None or len(frames) < max_frames:
            buf = _read_exact(proc.stdout, nplanes * width * height)
            if buf is None:
                break
            arr = np.frombuffer(buf, np.uint8)
            if nplanes == 1:
                frames.append(arr.reshape(height, width).copy())
            else:
                frames.append(arr.reshape(nplanes, height, width)
                              .transpose(1, 2, 0).copy())
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    return frames
