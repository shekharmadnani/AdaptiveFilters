"""Full-frame K-map network: per-8x8-block count of retained DCT AC
coefficients, predicted by a fully convolutional CNN.

Design (as specified):
  - input frame (any W, H divisible by 8; per-plane for multi-channel)
  - CNN (own, not pretrained) -> (W/8) x (H/8) map of K in [0, 63]
  - reconstruction keeps the K largest AC magnitudes + DC, then IDCT
  - loss = fidelity + lam_rate * K                     [rate regularizer]
         + w_e1 * ReLU(|d1(rec)| - |d1(orig)|)         [no NEW 1st-order edges]
         + w_e2 * ReLU(|d2(rec)| - |d2(orig)|)         [no NEW 2nd-order structure]
    The edge terms are asymmetric hinges: zero wherever the reconstruction's
    derivatives do not exceed the original's -- they penalize only edges the
    filter CREATES (blocking, ringing, staircase), never existing content.

Differentiable top-K: AC magnitudes are sorted per block (gradients flow
through the values, not the permutation) and masked in rank space,
  m_r = sigmoid((K - r - 1/2) / T),  r = 0..62,
which is differentiable in the network output K and anneals to hard top-K
as T -> 0. K-form vs threshold-form duality: the K-th largest magnitude is
an implicit threshold, but K is bounded, contrast-invariant, and immune to
the quantization-rounding pathology observed with fixed thresholds.

Trained on PRISTINE content only (natural-content K prior).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..probes.dct_probe import dct_matrix


# ------------------------------------------------------- frame <-> blocks

def frame_to_blocks(x):
    """(B, 1, H, W) -> (B, N, 8, 8), N = (H/8)*(W/8), row-major grid order
    (matches a (B, 1, H/8, W/8) map flattened with .flatten(1))."""
    b, _, h, w = x.shape
    v = x.reshape(b, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4)
    return v.reshape(b, -1, 8, 8)


def blocks_to_frame(blocks, h, w):
    b = blocks.shape[0]
    v = blocks.reshape(b, h // 8, w // 8, 8, 8).permute(0, 1, 3, 2, 4)
    return v.reshape(b, 1, h, w)


# ------------------------------------------------------------------ model

class KMapModel(nn.Module):
    """Fully convolutional: (B, 1, H, W) -> K-map (B, 1, H/8, W/8) in [0, 63].

    Trained on crops, applied to full frames (FCN). Receptive field spans
    several neighboring blocks, so each K sees its spatial context.
    """

    def __init__(self, lam_rate=2.5e-3, channels=32):
        super().__init__()
        self.lam_rate = lam_rate
        d = torch.tensor(dct_matrix(8), dtype=torch.float32)
        self.register_buffer("dmat", d)
        c = channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # /2
            nn.Conv2d(c, 2 * c, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # /4
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # /8 = block grid
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, 1, 1),
        )

    def forward(self, x):
        return 63.0 * torch.sigmoid(self.net(x))   # (B, 1, H/8, W/8)


# ------------------------------------------------- rank-space soft top-K

def soft_topk_mask(coeffs, k, temperature):
    """coeffs (B, N, 8, 8), k (B, N) -> mask (B, N, 8, 8); DC always kept."""
    b, n = coeffs.shape[:2]
    flat = coeffs.reshape(b, n, 64)
    ac_mag = flat[..., 1:].abs()                     # index 0 is DC
    order = ac_mag.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(-1, order,
                   torch.arange(63, device=coeffs.device).expand_as(order))
    m_ac = torch.sigmoid((k.unsqueeze(-1) - ranks.float() - 0.5) / temperature)
    m = torch.cat([torch.ones_like(flat[..., :1]), m_ac], dim=-1)
    return m.reshape(b, n, 8, 8)


def hard_topk_mask(coeffs, k):
    """Inference version: exact top-round(K); DC always kept."""
    b, n = coeffs.shape[:2]
    flat = coeffs.reshape(b, n, 64)
    ac_mag = flat[..., 1:].abs()
    order = ac_mag.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(-1, order,
                   torch.arange(63, device=coeffs.device).expand_as(order))
    m_ac = (ranks.float() < k.round().unsqueeze(-1)).float()
    m = torch.cat([torch.ones_like(flat[..., :1]), m_ac], dim=-1)
    return m.reshape(b, n, 8, 8)


# ------------------------------------------------------------------- loss

def _d1(x):
    return (x[..., :, 1:] - x[..., :, :-1],
            x[..., 1:, :] - x[..., :-1, :])


def _d2(x):
    return (x[..., :, 2:] - 2 * x[..., :, 1:-1] + x[..., :, :-2],
            x[..., 2:, :] - 2 * x[..., 1:-1, :] + x[..., :-2, :])


def new_edge_penalty(rec, orig, deriv):
    """Asymmetric hinge: penalize derivative energy the reconstruction has
    but the original does not (edges the filter CREATED)."""
    total = 0.0
    for dr, do in zip(deriv(rec), deriv(orig)):
        total = total + F.relu(dr.abs() - do.abs()).mean()
    return total


def kmap_loss(model, frame, temperature, w_e1=0.05, w_e2=0.05, target=None):
    """frame (B, 1, H, W) in [0, 1], H and W divisible by 8.

    target=None  -> naturalness mode: reconstruct the input itself
                    (pristine-only training, self-supervised prior).
    target=x     -> restoration mode: `frame` is the DEGRADED input, the
                    reconstruction is scored against the PRISTINE target
                    (paired training; the K-map/DCT selection learns to
                    keep content coefficients and drop artifact energy).
    """
    if target is None:
        target = frame
    _, _, h, w = frame.shape
    kmap = model(frame)                              # (B, 1, H/8, W/8)
    k = kmap.flatten(1)                              # (B, N)

    blocks = frame_to_blocks(frame)                  # DCT of the OBSERVED
    coeffs = torch.matmul(torch.matmul(model.dmat, blocks), model.dmat.T)
    m = soft_topk_mask(coeffs, k, temperature)
    rec_blocks = torch.matmul(torch.matmul(model.dmat.T, m * coeffs),
                              model.dmat)
    rec = blocks_to_frame(rec_blocks, h, w)

    target_blocks = frame_to_blocks(target)
    recon = (rec_blocks - target_blocks).pow(2).sum(dim=(2, 3)).mean()
    rate = k.mean()
    e1 = new_edge_penalty(rec, target, _d1)          # no edges the PRISTINE
    e2 = new_edge_penalty(rec, target, _d2)          # frame doesn't have

    total = recon + model.lam_rate * rate + w_e1 * e1 + w_e2 * e2
    logs = {
        "loss": float(total.detach()),
        "recon": float(recon.detach()),
        "k_mean": float(k.detach().mean()),
        "new_e1": float(e1.detach()) if torch.is_tensor(e1) else float(e1),
        "new_e2": float(e2.detach()) if torch.is_tensor(e2) else float(e2),
    }
    return total, logs


# ------------------------------------------------------------ persistence

def save_model(model, path, extra=None):
    torch.save({"state_dict": model.state_dict(),
                "lam_rate": model.lam_rate,
                "extra": extra or {}}, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = KMapModel(lam_rate=ckpt["lam_rate"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model
