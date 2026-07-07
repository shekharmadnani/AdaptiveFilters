"""Frame ingestion: raw planar YUV, Y4M, still images, and anything ffmpeg
can decode (via a rawvideo pipe). All readers yield float64 luma in [0, 1].
"""

import json
import os
import subprocess

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_CHROMA_FACTOR = {"420": 1.5, "422": 2.0, "444": 3.0, "400": 1.0}


def _read_exact(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _luma_from_bytes(buf, width, height, bits):
    if bits > 8:
        y = np.frombuffer(buf, dtype="<u2").reshape(height, width)
    else:
        y = np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
    return y.astype(np.float64) / float(2 ** bits - 1)


def iter_yuv(path, width, height, pix_fmt="yuv420p"):
    """Headerless planar YUV (e.g. yuv420p, yuv420p10le, yuv444p)."""
    sub = next((k for k in _CHROMA_FACTOR if k in pix_fmt), "420")
    bits = 12 if "12" in pix_fmt else (10 if "10" in pix_fmt else 8)
    bps = 2 if bits > 8 else 1
    frame_bytes = int(width * height * _CHROMA_FACTOR[sub]) * bps
    luma_bytes = width * height * bps
    with open(path, "rb") as f:
        while True:
            buf = f.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield _luma_from_bytes(buf[:luma_bytes], width, height, bits)


def iter_y4m(path):
    """YUV4MPEG2 (.y4m) reader; 8/10-bit planar, luma only."""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"\n"):
            c = f.read(1)
            if not c:
                raise ValueError(f"{path}: truncated Y4M header")
            header += c
        tokens = header.decode("ascii", "replace").split()
        if not tokens or tokens[0] != "YUV4MPEG2":
            raise ValueError(f"{path}: not a Y4M file")
        width = height = None
        ctoken = "420"
        for t in tokens[1:]:
            if t.startswith("W"):
                width = int(t[1:])
            elif t.startswith("H"):
                height = int(t[1:])
            elif t.startswith("C"):
                ctoken = t[1:]
        if width is None or height is None:
            raise ValueError(f"{path}: Y4M header missing W/H")
        sub = next((k for k in _CHROMA_FACTOR if ctoken.startswith(k)), "420")
        bits = 10 if "p10" in ctoken else (12 if "p12" in ctoken else 8)
        bps = 2 if bits > 8 else 1
        frame_bytes = int(width * height * _CHROMA_FACTOR[sub]) * bps
        luma_bytes = width * height * bps
        while True:
            line = b""
            while not line.endswith(b"\n"):
                c = f.read(1)
                if not c:
                    return
                line += c
            if not line.startswith(b"FRAME"):
                raise ValueError(f"{path}: bad Y4M frame marker")
            buf = _read_exact(f, frame_bytes)
            if buf is None:
                return
            yield _luma_from_bytes(buf[:luma_bytes], width, height, bits)


def ffprobe_dims(path, ffprobe="ffprobe"):
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    s = json.loads(out.stdout)["streams"][0]
    return int(s["width"]), int(s["height"])


def iter_ffmpeg(path, ffmpeg="ffmpeg"):
    """Decode any container/codec to gray frames through an ffmpeg pipe."""
    w, h = ffprobe_dims(path)
    proc = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0",
         "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # early generator exit breaks the pipe;
    )                               # ffmpeg's complaint about it is noise
    try:
        while True:
            buf = _read_exact(proc.stdout, w * h)
            if buf is None:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w).astype(np.float64) / 255.0
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def load_image_luma(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def iter_luma_frames(path, width=None, height=None, pix_fmt="yuv420p"):
    """Unified entry point; dispatches on extension."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".yuv":
        if not width or not height:
            raise ValueError("raw .yuv needs explicit width/height")
        return iter_yuv(path, width, height, pix_fmt)
    if ext == ".y4m":
        return iter_y4m(path)
    if ext in IMAGE_EXTS:
        return iter((load_image_luma(path),))
    return iter_ffmpeg(path)


def sample_frames(frames, step):
    """Yield (frame_index, frame, previous_frame) every `step` frames.

    The immediate predecessor is kept so the temporal probe always has
    context; sampling starts at index `step` so vectors have fixed length.
    """
    prev = None
    for idx, f in enumerate(frames):
        if idx > 0 and idx % step == 0:
            yield idx, f, prev
        prev = f
