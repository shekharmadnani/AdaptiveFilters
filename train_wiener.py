"""Train the learned DCT-domain adaptive Wiener filter (gain-map model).

Usage:
  python train_wiener.py [--out models/wiener.pt] [--crops 6000]
                         [--epochs 12] [--naturalness] [--skip-train]

Default mode is PAIRED (restoration): degraded input, pristine target --
the gains learn Wiener shrinkage toward pristine content, attenuating what
they can. --naturalness switches to pristine-only self-reconstruction.
Held-out artifacts (never in training): pl_copy, stale.

Validation: per-feature SRCC harness on the JPEG severity ladder.
"""

import argparse
import os
import time

import numpy as np
import torch

from adaptive_filters.learned.wiener import WienerDctModel, wiener_loss, save_model
from adaptive_filters.learned.kdct import pick_device
from adaptive_filters.learned.patches import (
    gather_frames, sample_patches, make_degraded, gather_h264_crops,
)
from adaptive_filters.synthetic import make_frame, jpeg_like
from adaptive_filters.features.stats import spearman

DEFAULT_BVI = r"F:\DVI\BVI-CC1\ORIG_MP4"


def train(args):
    device = pick_device(args.device)
    print(f"Device: {device}")

    print("Gathering pristine frames (color)...")
    frames = gather_frames(args.bvi if os.path.isdir(args.bvi) else None,
                           color=True)
    crops = sample_patches(frames, args.crops, size=args.crop_size,
                           seed=args.seed)
    print(f"Training crops: {crops.shape}  (N, C, H, W)")

    model = WienerDctModel(lam_rate=args.lam_rate, in_channels=3,
                           gmax=args.gmax).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    if args.naturalness:
        data = torch.from_numpy(crops)
        targets = None
        print("Naturalness mode: self-reconstruction on pristine crops")
    else:
        print("Paired Wiener mode: degraded input -> pristine target "
              "(held out: pl_copy, stale)")
        pris_all, deg_all = crops, make_degraded(crops, seed=args.seed)
        n_h264 = int(args.h264_frac * args.crops)
        if n_h264 > 0 and os.path.isdir(args.bvi):
            print(f"Adding {n_h264} real-H.264 pairs "
                  "(no-deblock encodes; corrupted slice NALs; -ec 1)...")
            h264 = gather_h264_crops(args.bvi, n_h264, size=args.crop_size,
                                     seed=args.seed, color=True)
            if h264 is not None:
                ph, dh = h264
                pris_all = np.concatenate([pris_all, ph])
                deg_all = np.concatenate([deg_all, dh])
                print(f"Total pairs: {len(pris_all)} "
                      f"({len(ph)} from real H.264)")
        data = torch.from_numpy(deg_all)
        targets = torch.from_numpy(pris_all)

    rng = np.random.default_rng(args.seed)
    n_data = len(data)
    steps = max(1, n_data // args.batch)
    for epoch in range(args.epochs):
        order = rng.permutation(n_data)
        acc = {}
        t0 = time.time()
        for s in range(steps):
            idx = order[s * args.batch : (s + 1) * args.batch]
            batch = data[idx].to(device)
            tgt = targets[idx].to(device) if targets is not None else None
            loss, logs = wiener_loss(model, batch, target=tgt,
                                     w_e1=args.w_e1, w_e2=args.w_e2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for kk, v in logs.items():
                acc[kk] = acc.get(kk, 0.0) + v / steps
        print(f"epoch {epoch + 1:2d}/{args.epochs}  "
              f"loss={acc['loss']:.5f}  recon={acc['recon']:.5f}  "
              f"Kg={acc['k_mean']:5.1f}  g_mean={acc['g_mean']:.3f}  "
              f"g>1={acc['g_over1']:.3f}  tv1={acc['tv1']:.4f}  "
              f"({time.time() - t0:.1f}s)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_model(model, args.out, extra={"crops": len(crops),
                                       "paired": not args.naturalness})
    print(f"Saved -> {args.out}")


def _jpeg_like_color(frame_hw3, q):
    out = np.empty_like(frame_hw3)
    for c in range(frame_hw3.shape[2]):
        out[:, :, c] = jpeg_like(frame_hw3[:, :, c], q)
    return out


def validate(args):
    from adaptive_filters.learned.patches import synthetic_color_frame
    from adaptive_filters.probes.learned_wiener_probe import LearnedWienerProbe

    probe = LearnedWienerProbe(args.out, device=args.device)
    qualities = [100, 90, 70, 50, 30, 10]
    sev = [100 - q for q in qualities]

    tests = [("synthetic",
              synthetic_color_frame(999, size=512).astype(np.float64))]
    frames = gather_frames(args.bvi if os.path.isdir(args.bvi) else None,
                           frames_per_video=1, color=True, verbose=False)
    if frames:
        f = frames[-1].astype(np.float64)
        h, w = f.shape[:2]
        tests.append(("bvi_master",
                      f[h // 2 - 256 : h // 2 + 256,
                        w // 2 - 256 : w // 2 + 256, :]))

    print("\nValidation: per-feature SRCC vs JPEG-like severity (color)")
    ok = True
    for name, frame in tests:
        rows = {}
        for q in qualities:
            deg = frame if q == 100 else _jpeg_like_color(frame, q)
            for kk, v in probe.run(deg / 255.0).features.items():
                rows.setdefault(kk, []).append(v)
        scored = sorted(((abs(spearman(sev, v)), kk) for kk, v in rows.items()),
                        reverse=True)
        strong = [kk for s, kk in scored if s >= 0.9]
        ok = ok and len(strong) >= 3
        top = ", ".join(f"{kk}({s:+.2f})" for s, kk in scored[:4])
        print(f"  {name:<12} {len(strong)} features |SRCC|>=0.9 -> {top}")
        kp = "  ".join(f"{v:6.2f}" for v in rows["k_pred_mean"])
        re = "  ".join(f"{v:.1e}" for v in rows["res_energy"])
        print(f"  {'':<12} K_g       : {kp}")
        print(f"  {'':<12} res_energy: {re}")
    print(f"VALIDATION {'PASSED' if ok else 'FAILED'} "
          f"(>=3 monotone features with |SRCC|>=0.9 per test content)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("models", "wiener.pt"))
    ap.add_argument("--bvi", default=DEFAULT_BVI)
    ap.add_argument("--crops", type=int, default=6000)
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lam-rate", type=float, default=2.5e-3)
    ap.add_argument("--gmax", type=float, default=4.0,
                    help="upper bound of the coefficient gains (>1 allows "
                         "amplification)")
    ap.add_argument("--w-e1", type=float, default=0.05)
    ap.add_argument("--w-e2", type=float, default=0.05)
    ap.add_argument("--naturalness", action="store_true",
                    help="pristine-only self-reconstruction instead of paired")
    ap.add_argument("--h264-frac", type=float, default=0.5,
                    help="real-H.264 pairs as a fraction of --crops "
                         "(paired mode only; 0 disables)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    if not args.skip_train:
        train(args)
    raise SystemExit(0 if validate(args) else 1)


if __name__ == "__main__":
    main()
