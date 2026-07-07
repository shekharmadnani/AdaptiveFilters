"""Dataset builder: manifest of (content, distorted, reference) -> features
+ per-frame VMAF labels, cached as a compressed .npz.

Manifest: JSON list of entries
  {
    "content":  "clip01",            # content id -- drives the grouped split
    "dist":     "encodes/clip01_crf28.mp4",
    "ref":      "masters/clip01.y4m",     # optional if vmaf_json given
    "vmaf_json": "logs/clip01_crf28.json",# optional pre-computed labels
    "width"/"height"/"pix_fmt":           # only for raw .yuv inputs
  }

Reference frames are also featurized once per unique ref (label 100,
is_ref=True) -- that is the pristine corpus the naturalness anchor fits on.
"""

import json
import os

import numpy as np

from .pipeline import FeatureExtractor, to_vector
from .io import iter_luma_frames, sample_frames
from .vmaf import compute_vmaf, parse_vmaf_json


def _extract_rows(extractor, path, entry, frame_step, names, cache_dir=None):
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"{os.path.basename(path)}.step{frame_step}.npz")
        if os.path.exists(cache_path):
            c = np.load(cache_path, allow_pickle=False)
            cached_names = [str(n) for n in c["names"]]
            if names is None or cached_names == names:
                return list(c["rows"]), list(c["idxs"]), cached_names

    rows, idxs = [], []
    frames = iter_luma_frames(
        path,
        width=entry.get("width"),
        height=entry.get("height"),
        pix_fmt=entry.get("pix_fmt", "yuv420p"),
    )
    for idx, frame, prev in sample_frames(frames, frame_step):
        feats = extractor.extract(frame, prev)
        names, vec = to_vector(feats, names)
        rows.append(vec)
        idxs.append(idx)

    if cache_path and rows:
        np.savez_compressed(cache_path, rows=np.array(rows),
                            idxs=np.array(idxs), names=np.array(names))
    return rows, idxs, names


def build_dataset(manifest_path, out_path, frame_step=6, include_ref=True,
                  num_scales=2, extractor=None, cache_dir=None, verbose=True):
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    base = os.path.dirname(os.path.abspath(manifest_path))

    def _resolve(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    if extractor is None:
        extractor = FeatureExtractor(num_scales=num_scales)
    names = None
    xs, ys, groups, is_ref, sources, frame_ids = [], [], [], [], [], []
    seen_refs = set()

    for e in entries:
        dist = _resolve(e["dist"])
        content = e["content"]

        vmaf_json = e.get("vmaf_json")
        if vmaf_json and os.path.exists(_resolve(vmaf_json)):
            vmaf = parse_vmaf_json(_resolve(vmaf_json))
        elif e.get("ref"):
            # compute, and persist to vmaf_json (if given) so reruns resume
            keep = _resolve(vmaf_json) if vmaf_json else None
            if keep:
                os.makedirs(os.path.dirname(keep) or ".", exist_ok=True)
            vmaf = compute_vmaf(_resolve(e["ref"]), dist, keep_log=keep)
        else:
            raise ValueError(f"{content}: entry needs 'ref' or 'vmaf_json'")

        rows, idxs, names = _extract_rows(extractor, dist, e, frame_step,
                                          names, cache_dir)
        kept = 0
        for vec, idx in zip(rows, idxs):
            if idx >= len(vmaf):
                break
            xs.append(vec)
            ys.append(float(vmaf[idx]))
            groups.append(content)
            is_ref.append(False)
            sources.append(os.path.basename(dist))
            frame_ids.append(idx)
            kept += 1
        if verbose:
            print(f"  {content:<16} {os.path.basename(dist):<28} "
                  f"frames={kept:3d}  vmaf_mean={vmaf.mean():6.2f}")

        ref = e.get("ref")
        if include_ref and ref and _resolve(ref) not in seen_refs:
            ref_path = _resolve(ref)
            seen_refs.add(ref_path)
            rrows, ridxs, names = _extract_rows(extractor, ref_path, e,
                                                frame_step, names, cache_dir)
            for vec, idx in zip(rrows, ridxs):
                xs.append(vec)
                ys.append(100.0)
                groups.append(content)
                is_ref.append(True)
                sources.append(os.path.basename(ref_path))
                frame_ids.append(idx)
            if verbose:
                print(f"  {content:<16} {os.path.basename(ref_path):<28} "
                      f"frames={len(rrows):3d}  (reference corpus)")

    np.savez_compressed(
        out_path,
        x=np.array(xs),
        y=np.array(ys),
        groups=np.array(groups),
        is_ref=np.array(is_ref),
        names=np.array(names),
        source=np.array(sources),
        frame=np.array(frame_ids),
    )
    if verbose:
        print(f"Saved {len(xs)} rows x {len(names)} features -> {out_path}")
    return out_path


def load_dataset(path):
    d = np.load(path, allow_pickle=False)
    return {
        "x": d["x"],
        "y": d["y"],
        "groups": [str(g) for g in d["groups"]],
        "is_ref": d["is_ref"].astype(bool),
        "names": [str(n) for n in d["names"]],
        "source": [str(s) for s in d["source"]],
        "frame": d["frame"],
    }
