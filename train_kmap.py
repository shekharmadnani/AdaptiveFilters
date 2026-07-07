"""Train the full-frame K-map network (rank-space top-K, asymmetric edges).

Usage:
  python train_kmap.py [--out models/kmap.pt] [--bvi F:\\DVI\\BVI-CC1\\ORIG_MP4]
                       [--crops 6000] [--crop-size 128] [--epochs 12]
                       [--batch 64] [--device cuda|cpu] [--skip-train]

Pristine-only training (BVI masters if available, synthetic fallback);
trained on crops, applied full-frame (fully convolutional). Validation is
the same harness as the other learned probe: >= 3 features per test content
must be monotone (|SRCC| >= 0.9) against a JPEG-like severity ladder.
"""

import argparse
import os
import time

import numpy as np
import torch

from adaptive_filters.learned.kmap import KMapModel, kmap_loss, save_model
from adaptive_filters.learned.kdct import pick_device
from adaptive_filters.learned.patches import (
    gather_frames, sample_patches, make_degraded,
)
from adaptive_filters.synthetic import make_frame, jpeg_like
from adaptive_filters.features.stats import spearman

DEFAULT_BVI = r"F:\DVI\BVI-CC1\ORIG_MP4"


def train(args):
    device = pick_device(args.device)
    print(f"Device: {device}")

    print("Gathering pristine frames...")
    frames = gather_frames(args.bvi if os.path.isdir(args.bvi) else None)
    crops = sample_patches(frames, args.crops, size=args.crop_size,
                           seed=args.seed)
    print(f"Training crops: {crops.shape}")

    model = KMapModel(lam_rate=args.lam_rate).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    if args.paired:
        print("Restoration mode: degrading inputs "
              "(targets stay pristine; held-out artifacts: pl_copy, stale)")
        degraded = make_degraded(crops, seed=args.seed)
        data = torch.from_numpy(degraded[:, None, :, :])
        targets = torch.from_numpy(crops[:, None, :, :])
    else:
        data = torch.from_numpy(crops[:, None, :, :])
        targets = None

    rng = np.random.default_rng(args.seed)
    steps = max(1, len(crops) // args.batch)
    for epoch in range(args.epochs):
        temperature = max(2.0 * (0.5 ** (epoch // 2)), 0.25)  # rank units
        order = rng.permutation(len(crops))
        acc = {}
        t0 = time.time()
        for s in range(steps):
            idx = order[s * args.batch : (s + 1) * args.batch]
            batch = data[idx].to(device)
            tgt = targets[idx].to(device) if targets is not None else None
            loss, logs = kmap_loss(model, batch, temperature,
                                   w_e1=args.w_e1, w_e2=args.w_e2,
                                   target=tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for kk, v in logs.items():
                acc[kk] = acc.get(kk, 0.0) + v / steps
        print(f"epoch {epoch + 1:2d}/{args.epochs}  T={temperature:.2f}  "
              f"loss={acc['loss']:.5f}  recon={acc['recon']:.5f}  "
              f"K={acc['k_mean']:5.1f}  newE1={acc['new_e1']:.5f}  "
              f"({time.time() - t0:.1f}s)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_model(model, args.out, extra={"crops": len(crops),
                                       "crop_size": args.crop_size})
    print(f"Saved -> {args.out}")


def validate(args):
    from adaptive_filters.probes.learned_kmap_probe import LearnedKMapProbe

    probe = LearnedKMapProbe(args.out, device=args.device)
    qualities = [100, 90, 70, 50, 30, 10]
    sev = [100 - q for q in qualities]

    tests = [("synthetic", make_frame(999, size=512))]
    frames = gather_frames(args.bvi if os.path.isdir(args.bvi) else None,
                           frames_per_video=1, verbose=False)
    if frames:
        f = frames[-1]
        h, w = f.shape
        tests.append(("bvi_master",
                      f[h // 2 - 256 : h // 2 + 256,
                        w // 2 - 256 : w // 2 + 256] * 255.0))

    print("\nValidation: per-feature SRCC vs JPEG-like severity")
    ok = True
    for name, frame in tests:
        rows = {}
        for q in qualities:
            deg = frame if q == 100 else jpeg_like(frame, q)
            for kk, v in probe.run(deg / 255.0).features.items():
                rows.setdefault(kk, []).append(v)
        scored = sorted(((abs(spearman(sev, v)), kk) for kk, v in rows.items()),
                        reverse=True)
        strong = [kk for s, kk in scored if s >= 0.9]
        ok = ok and len(strong) >= 3
        top = ", ".join(f"{kk}({s:+.2f})" for s, kk in scored[:4])
        print(f"  {name:<12} {len(strong)} features |SRCC|>=0.9 -> {top}")
        dk = "  ".join(f"{v:+6.2f}" for v in rows["dk_mean"])
        le = "  ".join(f"{v:6.4f}" for v in rows["lost_edge1"])
        print(f"  {'':<12} signed dk : {dk}")
        print(f"  {'':<12} lost_edge1: {le}")
    print(f"VALIDATION {'PASSED' if ok else 'FAILED'} "
          f"(>=3 monotone features with |SRCC|>=0.9 per test content)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("models", "kmap.pt"))
    ap.add_argument("--bvi", default=DEFAULT_BVI)
    ap.add_argument("--crops", type=int, default=6000)
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lam-rate", type=float, default=2.5e-3)
    ap.add_argument("--w-e1", type=float, default=0.05)
    ap.add_argument("--w-e2", type=float, default=0.05)
    ap.add_argument("--paired", action="store_true",
                    help="restoration mode: degraded input, pristine target")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    if not args.skip_train:
        train(args)
    raise SystemExit(0 if validate(args) else 1)


if __name__ == "__main__":
    main()
