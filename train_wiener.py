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

from adaptive_filters.learned.wiener import build_model, wiener_loss, save_model
from adaptive_filters.learned.kdct import pick_device
from adaptive_filters.learned.patches import (
    gather_frames, sample_patches, make_degraded, gather_h264_crops,
    gather_crops_dir, load_pairs_dir, make_masked,
)
from adaptive_filters.synthetic import make_frame, jpeg_like
from adaptive_filters.features.stats import spearman

DEFAULT_BVI = r"F:\DVI\BVI-CC1\ORIG_MP4"


def train(args):
    device = pick_device(args.device)
    print(f"Device: {device}")

    if args.pairs_dir:
        # pre-generated real-codec pairs (generate_pairs.py):
        # H.264/HEVC/MPEG-2 compression + packet-loss families
        print(f"Loading pre-generated pairs from {args.pairs_dir}...")
        pris_all, deg_all = load_pairs_dir(args.pairs_dir, args.crops,
                                           size=args.crop_size,
                                           seed=args.seed,
                                           family=args.pairs_family)
        model = build_model(arch=args.arch, lam_rate=args.lam_rate,
                            in_channels=3, gmax=args.gmax,
                            affine=args.affine, tmax=args.tmax).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        data = torch.from_numpy(deg_all)
        targets = torch.from_numpy(pris_all)
        _train_loop(args, model, opt, data, targets, device)
        return

    if args.pristine_dir and os.path.isdir(args.pristine_dir):
        print(f"Gathering pristine crops from {args.pristine_dir} "
              f"(streaming, up to {args.max_videos} videos)...")
        crops = gather_crops_dir(args.pristine_dir, args.crops,
                                 size=args.crop_size, seed=args.seed,
                                 max_videos=args.max_videos)
        print(f"Training crops: {crops.shape} uint8, "
              f"~{args.max_videos} source contents")
    else:
        print("Gathering pristine frames (color)...")
        frames = gather_frames(args.bvi if os.path.isdir(args.bvi) else None,
                               color=True)
        crops = sample_patches(frames, args.crops, size=args.crop_size,
                               seed=args.seed)
        print(f"Training crops: {crops.shape}  (N, C, H, W)")

    model = build_model(arch=args.arch, lam_rate=args.lam_rate,
                        in_channels=3, gmax=args.gmax,
                        affine=args.affine, tmax=args.tmax).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    if args.mae_epochs > 0 and not args.naturalness:
        # MAE-style pretraining phase: mask-and-predict on pristine crops
        # (unlimited clean data teaches the natural-content prior before
        # any degraded pair is seen)
        print(f"MAE pretraining: {args.mae_epochs} epochs, "
              f"mask_frac={args.mae_frac} on pristine crops...")
        masked = make_masked(crops, mask_frac=args.mae_frac, seed=args.seed)
        _train_loop(args, model, opt, torch.from_numpy(masked),
                    torch.from_numpy(crops), device,
                    epochs=args.mae_epochs, save=False)
        del masked
        print("MAE pretraining done; fine-tuning on degraded pairs...")

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
                if pris_all.dtype == np.uint8:  # match corpus storage
                    ph = np.clip(ph * 255.0 + 0.5, 0, 255).astype(np.uint8)
                    dh = np.clip(dh * 255.0 + 0.5, 0, 255).astype(np.uint8)
                pris_all = np.concatenate([pris_all, ph])
                deg_all = np.concatenate([deg_all, dh])
                print(f"Total pairs: {len(pris_all)} "
                      f"({len(ph)} from real H.264)")
        data = torch.from_numpy(deg_all)
        targets = torch.from_numpy(pris_all)

    _train_loop(args, model, opt, data, targets, device)


def _train_loop(args, model, opt, data, targets, device, epochs=None,
                save=True):
    rng = np.random.default_rng(args.seed)
    n_data = len(data)
    steps = max(1, n_data // args.batch)
    n_epochs = epochs if epochs is not None else args.epochs
    for epoch in range(n_epochs):
        order = rng.permutation(n_data)
        acc = {}
        t0 = time.time()
        for s in range(steps):
            idx = order[s * args.batch : (s + 1) * args.batch]
            batch = data[idx].to(device)
            if batch.dtype == torch.uint8:
                batch = batch.float().div_(255.0)
            tgt = targets[idx].to(device) if targets is not None else None
            if tgt is not None and tgt.dtype == torch.uint8:
                tgt = tgt.float().div_(255.0)
            loss, logs = wiener_loss(model, batch, target=tgt,
                                     w_e1=args.w_e1, w_e2=args.w_e2,
                                     mu=args.mu)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for kk, v in logs.items():
                acc[kk] = acc.get(kk, 0.0) + v / steps
        t_str = (f"  t_abs={acc['t_abs']:.4f}  t_act={acc['t_active']:.3f}"
                 if "t_abs" in acc else "")
        print(f"epoch {epoch + 1:2d}/{n_epochs}  "
              f"loss={acc['loss']:.5f}  recon={acc['recon']:.5f}  "
              f"Kg={acc['k_mean']:5.1f}  g_mean={acc['g_mean']:.3f}  "
              f"g>1={acc['g_over1']:.3f}  tv1={acc['tv1']:.4f}{t_str}  "
              f"({time.time() - t0:.1f}s)")

    if not save:
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_model(model, args.out, extra={"pairs": n_data,
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
    ap.add_argument("--arch", default="a", choices=["a", "b", "c", "d"],
                    help="a: baseline trunk (~40px view); b: + dilated "
                         "stack (~260px); c: + U-Net deep branch "
                         "(whole-crop view; pair with --crop-size 256); "
                         "d: c with global self-attention at the /32 "
                         "bottleneck (learned retrieval)")
    ap.add_argument("--mae-epochs", type=int, default=0,
                    help="MAE-style pretraining epochs on masked pristine "
                         "crops before the paired fine-tune (0 = off)")
    ap.add_argument("--mae-frac", type=float, default=0.4,
                    help="fraction of 32px tiles masked during pretraining")
    ap.add_argument("--affine", action="store_true",
                    help="gen-4: X_hat = g*X + t (adds the synthesis head)")
    ap.add_argument("--mu", type=float, default=0.02,
                    help="L1 price on t (synthesis); higher = sparser t")
    ap.add_argument("--tmax", type=float, default=1.0,
                    help="bound of the synthesis term |t| <= tmax")
    ap.add_argument("--w-e1", type=float, default=0.05)
    ap.add_argument("--w-e2", type=float, default=0.05)
    ap.add_argument("--naturalness", action="store_true",
                    help="pristine-only self-reconstruction instead of paired")
    ap.add_argument("--h264-frac", type=float, default=0.5,
                    help="real-H.264 pairs as a fraction of --crops "
                         "(paired mode only; 0 disables)")
    ap.add_argument("--pristine-dir", default=None,
                    help="directory of pristine videos for the crop corpus "
                         "(e.g. BVI-DVC Videos); streams crops in uint8, "
                         "overrides the BVI-CC1 frame pool")
    ap.add_argument("--max-videos", type=int, default=90,
                    help="cap on videos sampled from --pristine-dir")
    ap.add_argument("--pairs-dir", default=None,
                    help="pre-generated real-codec pair dataset from "
                         "generate_pairs.py (overrides all other data "
                         "sources)")
    ap.add_argument("--pairs-family", default=None,
                    choices=[None, "compression", "loss"],
                    help="restrict --pairs-dir to one family")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    if not args.skip_train:
        train(args)
    raise SystemExit(0 if validate(args) else 1)


if __name__ == "__main__":
    main()
