"""VMAF label generation and parsing (layer-A proxy labels).

Runs ffmpeg's libvmaf filter (distorted vs. reference must share resolution
and frame count) or parses a pre-computed VMAF JSON log.
"""

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

# bundled model (some ffmpeg builds default to a Unix-only model path)
DEFAULT_MODEL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models",
                 "vmaf_v0.6.1.json"))


def parse_vmaf_json(path):
    """Per-frame VMAF scores from a libvmaf JSON log (v1.x or v2.x layout)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scores = []
    for fr in data["frames"]:
        m = fr.get("metrics", fr)
        v = m.get("vmaf", m.get("VMAF_score"))
        if v is None:
            raise ValueError(f"{path}: no vmaf metric in frame record")
        scores.append(float(v))
    return np.array(scores, dtype=np.float64)


def compute_vmaf(reference, distorted, ffmpeg="ffmpeg", keep_log=None,
                 n_threads=4, model=None):
    """Run libvmaf and return per-frame scores.

    Log and model use relative paths inside a temp cwd to avoid Windows
    drive-colon escaping issues in lavfi filter arguments.
    """
    reference = os.path.abspath(reference)
    distorted = os.path.abspath(distorted)
    model = model or (DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else None)
    with tempfile.TemporaryDirectory() as td:
        log_name = "vmaf_log.json"
        opts = f"libvmaf=log_fmt=json:log_path={log_name}:n_threads={n_threads}"
        if model:
            shutil.copy(model, os.path.join(td, "vmaf_model.json"))
            opts += ":model_path=vmaf_model.json"
        cmd = [
            ffmpeg, "-hide_banner", "-v", "error",
            "-i", distorted,   # first input = main (distorted)
            "-i", reference,   # second input = reference
            "-lavfi", opts,
            "-f", "null", "-",
        ]
        proc = subprocess.run(cmd, cwd=td, capture_output=True, text=True)
        log_path = os.path.join(td, log_name)
        if proc.returncode != 0 or not os.path.exists(log_path):
            raise RuntimeError(
                f"libvmaf failed for {os.path.basename(distorted)}: {proc.stderr.strip()}"
            )
        scores = parse_vmaf_json(log_path)
        if keep_log:
            shutil.move(log_path, keep_log)  # works across drives
    return scores
