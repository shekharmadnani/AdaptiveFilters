"""Learned DCT-domain adaptive Wiener filter (gain-map model).

Generalizes the K-map: instead of a binary keep/drop of the top-K AC
coefficients, the CNN predicts a per-block, per-coefficient GAIN in [0, 1]:

    frame -> CNN -> gain map (64 channels on the (W/8)x(H/8) block grid)
    X_hat = g (.) X ;  filtered = IDCT(X_hat) ;  residual = frame - filtered

DC gain is fixed to 1. Gains only attenuate -- the filter removes what it
can and passes the rest; it never amplifies or synthesizes. With paired
training (degraded input, pristine target) the optimum gain approaches the
classical Wiener shrinkage E[X_pristine X]/E[X^2] per coefficient, learned
as a function of content context.

Loss (same structure as the K-map, no temperature needed -- gains are
already continuous):
    fidelity(rec, target) + lam_rate * sum(g)     [effective-K regularizer]
    + w_e1/w_e2 * asymmetric no-new-edges hinges vs the target

The effective coefficient count K_g = sum of AC gains per block is the
continuous analog of K and is exposed as a feature/theta-map.
"""

import torch
import torch.nn as nn

from ..probes.dct_probe import dct_matrix
from .kmap import frame_to_blocks, blocks_to_frame, _d1, _d2, new_edge_penalty


class WienerDctModel(nn.Module):
    """(B, 1, H, W) -> per-coefficient gains (B, N, 8, 8), N = (H/8)(W/8)."""

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
            nn.MaxPool2d(2),                                   # /8 block grid
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, 64, 1),        # one gain per DCT coefficient
        )

    def forward(self, x):
        b = x.shape[0]
        logits = self.net(x)                        # (B, 64, H/8, W/8)
        g = torch.sigmoid(logits)
        g = g.permute(0, 2, 3, 1).reshape(b, -1, 64)  # (B, N, 64) block order
        g = torch.cat([torch.ones_like(g[..., :1]), g[..., 1:]], dim=-1)
        return g.reshape(b, -1, 8, 8)               # DC forced to 1


def wiener_apply(model, frame_t):
    """frame_t (B, 1, H, W), H/W divisible by 8 -> (rec, gains, coeffs)."""
    _, _, h, w = frame_t.shape
    gains = model(frame_t)                          # (B, N, 8, 8)
    blocks = frame_to_blocks(frame_t)
    coeffs = torch.matmul(torch.matmul(model.dmat, blocks), model.dmat.T)
    rec_blocks = torch.matmul(torch.matmul(model.dmat.T, gains * coeffs),
                              model.dmat)
    return blocks_to_frame(rec_blocks, h, w), rec_blocks, gains, coeffs


def wiener_loss(model, frame, target=None, w_e1=0.05, w_e2=0.05):
    """target=None -> naturalness mode; else restoration (paired) mode."""
    if target is None:
        target = frame
    rec, rec_blocks, gains, _ = wiener_apply(model, frame)

    target_blocks = frame_to_blocks(target)
    recon = (rec_blocks - target_blocks).pow(2).sum(dim=(2, 3)).mean()
    k_g = gains.sum(dim=(2, 3)) - 1.0               # effective AC count
    rate = k_g.mean()
    e1 = new_edge_penalty(rec, target, _d1)
    e2 = new_edge_penalty(rec, target, _d2)

    total = recon + model.lam_rate * rate + w_e1 * e1 + w_e2 * e2
    logs = {
        "loss": float(total.detach()),
        "recon": float(recon.detach()),
        "k_mean": float(k_g.detach().mean()),
        "new_e1": float(e1.detach()) if torch.is_tensor(e1) else float(e1),
        "g_mean": float(gains.detach()[..., 1:].mean()),
    }
    return total, logs


def save_model(model, path, extra=None):
    torch.save({"state_dict": model.state_dict(),
                "lam_rate": model.lam_rate,
                "kind": "wiener",
                "extra": extra or {}}, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = WienerDctModel(lam_rate=ckpt["lam_rate"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model
