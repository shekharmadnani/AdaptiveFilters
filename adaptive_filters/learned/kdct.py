"""Learned per-block DCT sparsity (K) prediction.

The plan (as agreed):
  - input: 64x64 patch = 8x8 grid of 8x8 DCT blocks
  - network predicts, per block, a threshold tau (used for a differentiable
    keep-mask) and K_nat, the AC-coefficient count natural content with this
    appearance would need
  - loss = per-block reconstruction SSE
         + lam_rate * (number of kept coefficients)         [RD sparsity]
         + w_tv1 * |first derivatives of reconstruction|    [smoothness]
         + w_tv2 * |second derivatives of reconstruction|   [anti-staircase]
         + w_tvk * |gradient of the K-map|                  [coherent K field]
         + w_k   * MSE(K_nat, closed-form RD-optimal K)     [K head]

Training uses PRISTINE patches only, so the network learns the
natural-content K prior. At inference, deltaK = K_nat - K_emp (closed-form
RD count on the observed block) is a content-normalized compression-severity
signal: ~0 on pristine content, positive when coding has emptied blocks that
look like they should carry detail.

Closed-form benchmark (must be beaten to justify the network): without the
smoothness coupling the optimum is separable -- keep coefficient iff
X_k^2 > lam_rate. The network's value is the cross-block coupling, the
64x64 context, and the pristine prior.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..probes.dct_probe import dct_matrix


# ---------------------------------------------------------- block plumbing

def patch_to_blocks(x):
    """(B, 1, 64, 64) -> (B, 64, 8, 8), row-major block order."""
    b = x.shape[0]
    u = x.unfold(2, 8, 8).unfold(3, 8, 8)  # (B, 1, 8, 8, 8, 8)
    return u.reshape(b, 64, 8, 8)


def blocks_to_patch(blocks):
    """(B, 64, 8, 8) -> (B, 1, 64, 64); inverse of patch_to_blocks."""
    b = blocks.shape[0]
    v = blocks.reshape(b, 8, 8, 8, 8).permute(0, 1, 3, 2, 4)
    return v.reshape(b, 1, 64, 64)


def block_dct(dmat, blocks):
    return torch.matmul(torch.matmul(dmat, blocks), dmat.T)


def block_idct(dmat, coeffs):
    return torch.matmul(torch.matmul(dmat.T, coeffs), dmat)


# ------------------------------------------------------------------ model

class KDctModel(nn.Module):
    """Small CNN: 64x64 patch -> per-block (tau, K_nat) on the 8x8 block grid."""

    def __init__(self, lam_rate=2.5e-3, channels=32):
        super().__init__()
        self.lam_rate = lam_rate
        d = torch.tensor(dct_matrix(8), dtype=torch.float32)
        self.register_buffer("dmat", d)
        c = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # 32x32
            nn.Conv2d(c, 2 * c, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # 16x16
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # 8x8 = block grid
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.head_tau = nn.Conv2d(2 * c, 1, 1)
        self.head_k = nn.Conv2d(2 * c, 1, 1)

    def forward(self, x):
        h = self.features(x)
        tau = F.softplus(self.head_tau(h)) + 1e-4      # (B, 1, 8, 8) > 0
        k_nat = 63.0 * torch.sigmoid(self.head_k(h))   # (B, 1, 8, 8) in [0, 63]
        return tau, k_nat


def soft_mask(coeffs, tau, temperature):
    """Differentiable keep-mask; DC is always kept."""
    tau_b = tau.reshape(tau.shape[0], 64, 1, 1)
    m = torch.sigmoid((coeffs.abs() - tau_b) / temperature)
    dc = torch.zeros_like(m)
    dc[..., 0, 0] = 1.0
    return torch.maximum(m, dc)


def rd_optimal_k(coeffs, lam_rate):
    """Closed-form RD-optimal AC count: #{X_k^2 > lam} (DC excluded)."""
    keep = coeffs.pow(2) > lam_rate
    keep[..., 0, 0] = True
    return keep.float().sum(dim=(2, 3)) - 1.0


def kdct_loss(model, patch, temperature,
              w_tv1=0.01, w_tv2=0.01, w_tvk=1e-3, w_k=3e-4):
    """Returns (total_loss, logs). `patch` is (B, 1, 64, 64) in [0, 1]."""
    blocks = patch_to_blocks(patch)
    coeffs = block_dct(model.dmat, blocks)
    tau, k_nat = model(patch)

    m = soft_mask(coeffs, tau, temperature)
    rec_blocks = block_idct(model.dmat, m * coeffs)
    rec = blocks_to_patch(rec_blocks)

    # per-block SSE + lam * K : the RD objective (steps 4/7 of the plan)
    recon = (rec_blocks - blocks).pow(2).sum(dim=(2, 3)).mean()
    k_soft = m.sum(dim=(2, 3)) - 1.0                    # kept AC per block
    rate = k_soft.mean()

    # smoothness of the reconstruction (step 8): 1st + 2nd derivatives
    dx = rec[..., :, 1:] - rec[..., :, :-1]
    dy = rec[..., 1:, :] - rec[..., :-1, :]
    tv1 = dx.abs().mean() + dy.abs().mean()
    d2x = rec[..., :, 2:] - 2 * rec[..., :, 1:-1] + rec[..., :, :-2]
    d2y = rec[..., 2:, :] - 2 * rec[..., 1:-1, :] + rec[..., :-2, :]
    tv2 = d2x.abs().mean() + d2y.abs().mean()

    # natural K fields are spatially coherent
    kmap = k_soft.reshape(-1, 1, 8, 8)
    tvk = ((kmap[..., :, 1:] - kmap[..., :, :-1]).abs().mean()
           + (kmap[..., 1:, :] - kmap[..., :-1, :]).abs().mean())

    # K head learns the closed-form RD-optimal K of (pristine) content
    k_target = rd_optimal_k(coeffs, model.lam_rate).detach()
    k_loss = F.mse_loss(k_nat.reshape(-1, 64), k_target)

    total = (recon + model.lam_rate * rate
             + w_tv1 * tv1 + w_tv2 * tv2 + w_tvk * tvk + w_k * k_loss)
    logs = {
        "loss": float(total.detach()),
        "recon": float(recon.detach()),
        "k_mean": float(k_soft.detach().mean()),
        "k_rmse": float(k_loss.detach().sqrt()),
        "tv1": float(tv1.detach()),
    }
    return total, logs


# ------------------------------------------------------------ persistence

def save_model(model, path, extra=None):
    payload = {"state_dict": model.state_dict(),
               "lam_rate": model.lam_rate,
               "extra": extra or {}}
    torch.save(payload, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = KDctModel(lam_rate=ckpt["lam_rate"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        try:  # older GPUs may lack kernels in recent wheels
            (torch.zeros(2, device="cuda") + 1).sum().item()
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")
