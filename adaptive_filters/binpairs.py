"""Extract training-patch pairs from the BinResults share.

Layout of the source (network share):
  BinResults/Batch_N/TrainingImage_XXXXXXXX/
      GT.png                                   pristine ground truth
      <id>_bitrate_B_vif_V_binK.png            10 degraded versions,
                                               bins 0 (destroyed) .. 9 (near
                                               transparent), VIF label V

Extraction (configuration approved 2026-07-10):
  - N folders sampled across all batches (grouped split: a folder's
    patches never straddle train/held-out)
  - per folder, 3 bins: one severe (0-2), one mid (3-6), one mild (7-9),
    rotating deterministically so the dataset covers every bin
  - per (GT, degraded) pair: 10 random 8-aligned 256x256 crops, SAME
    coordinates in GT and degraded (pixel-aligned pairs)
  - RGB kept as-is (per configuration decision)
  - shards in the datasets/pairs format (+ per-pair bin and VIF labels)
  - RESUMABLE: folders already in the manifest are skipped on rerun
"""

import json
import os
import re

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None          # the GTs are ~16 MP, that's fine
PATCH = 256
DEG_RE = re.compile(r"_bitrate_(\d+)_vif_([0-9.]+)_bin(\d+)\.png$")
BIN_GROUPS = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]


def list_folders(root, seed=0):
    """All TrainingImage folders across batches, deterministically shuffled."""
    folders = []
    for b in sorted(os.listdir(root)):
        bdir = os.path.join(root, b)
        if not (b.startswith("Batch") and os.path.isdir(bdir)):
            continue
        folders += [os.path.join(bdir, f) for f in sorted(os.listdir(bdir))]
    rng = np.random.default_rng(seed + 4242)
    rng.shuffle(folders)
    return folders


def parse_folder(fld):
    """-> (gt_path, {bin: (path, vif)}) or None if malformed."""
    gt = os.path.join(fld, "GT.png")
    if not os.path.exists(gt):
        return None
    bins = {}
    for f in os.listdir(fld):
        m = DEG_RE.search(f)
        if m:
            bins[int(m.group(3))] = (os.path.join(fld, f),
                                     float(m.group(2)))
    return (gt, bins) if bins else None


def pick_bins(bins, folder_index):
    """One bin per severity group, rotating with the folder index."""
    chosen = []
    for gi, group in enumerate(BIN_GROUPS):
        want = group[(folder_index + gi) % len(group)]
        avail = [b for b in group if b in bins]
        if not avail:
            continue
        chosen.append(want if want in bins
                      else avail[folder_index % len(avail)])
    return chosen


def extract(root, out_dir, n_folders=1500, n_heldout=12,
            patches_per_pair=10, seed=0, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    man_path = os.path.join(out_dir, "manifest.json")
    manifest = []
    done = set()
    if os.path.exists(man_path):          # resume
        with open(man_path, encoding="utf-8") as f:
            manifest = json.load(f)
        done = {e["src"] for e in manifest}
        if verbose:
            print(f"resuming: {len(done)} folders already extracted")

    folders = list_folders(root, seed)
    heldout = folders[:n_heldout]
    with open(os.path.join(out_dir, "heldout.json"), "w",
              encoding="utf-8") as f:
        json.dump(heldout, f, indent=1)
    work = folders[n_heldout : n_heldout + n_folders]

    rng = np.random.default_rng(seed + 99)
    for fi, fld in enumerate(work):
        fid = os.path.basename(fld)
        if fid in done:
            continue
        try:
            parsed = parse_folder(fld)
            if parsed is None:
                continue
            gt_path, bins = parsed
            chosen = pick_bins(bins, fi)
            if not chosen:
                continue
            gt = np.asarray(Image.open(gt_path).convert("RGB"))
            h, w = gt.shape[:2]
            if h < PATCH or w < PATCH:
                continue

            n = len(chosen) * patches_per_pair
            pris = np.empty((n, 3, PATCH, PATCH), np.uint8)
            deg = np.empty_like(pris)
            blab = np.empty(n, np.int16)
            vlab = np.empty(n, np.float32)
            k = 0
            for b in chosen:
                dpath, vif = bins[b]
                dimg = np.asarray(Image.open(dpath).convert("RGB"))
                if dimg.shape != gt.shape:
                    continue
                for _ in range(patches_per_pair):
                    y0 = int(rng.integers(0, (h - PATCH) // 8 + 1)) * 8
                    x0 = int(rng.integers(0, (w - PATCH) // 8 + 1)) * 8
                    pris[k] = gt[y0:y0 + PATCH, x0:x0 + PATCH] \
                        .transpose(2, 0, 1)
                    deg[k] = dimg[y0:y0 + PATCH, x0:x0 + PATCH] \
                        .transpose(2, 0, 1)
                    blab[k], vlab[k] = b, vif
                    k += 1
            if k == 0:
                continue
            shard = f"{fid}.npz"
            np.savez_compressed(os.path.join(out_dir, shard),
                                pris=pris[:k], deg=deg[:k],
                                bin=blab[:k], vif=vlab[:k])
            manifest.append({"shard": shard, "family": "bins",
                             "codec": "photo", "src": fid,
                             "config": {"bins": chosen}, "n": int(k)})
            if (len(manifest) % 25) == 0 or fi == len(work) - 1:
                with open(man_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f)
                if verbose:
                    total = sum(e["n"] for e in manifest)
                    print(f"  [{fi + 1}/{len(work)}] {len(manifest)} folders,"
                          f" {total} pairs")
        except Exception as e:
            if verbose:
                print(f"  skip {fid}: {type(e).__name__}: {e}")
            continue

    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    total = sum(e["n"] for e in manifest)
    if verbose:
        print(f"Done: {len(manifest)} folders, {total} pairs -> {out_dir}")
    return manifest
