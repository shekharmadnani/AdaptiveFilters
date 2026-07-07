"""Learned DCT-domain adaptive Wiener filter (gain-map model), color-capable.

    frame (C channels) -> CNN -> per-channel gain map
                          (C x 64 gains on the (W/8)x(H/8) block grid)
    X_hat = g (.) X ;  filtered = IDCT(X_hat) ;  residual = frame - filtered

Design decisions (as specified):
  - gains g in [0, gmax] with gmax > 1: the filter may AMPLIFY coefficients
    (restoration of attenuated content, e.g. blur), not only attenuate.
    DC gain is fixed to 1.
  - smoothness regularization is on the OUTPUT ONLY: minimize the mean
    absolute first and second derivatives of the reconstruction. The input
    is degraded, so its edges are not a reference; fidelity to the pristine
    target counterbalances over-smoothing.
  - color: C input channels processed jointly by the trunk; the head
    predicts C*64 gains per block, i.e. a (W/8)x(H/8)xC K/gain field.

Loss:
    fidelity(rec, target) + lam_rate * sum(g_AC)   [K/rate regularizer]
    + w_e1 * mean|d1(rec)| + w_e2 * mean|d2(rec)|  [output smoothness]
"""

import torch
import torch.nn as nn

from ..probes.dct_probe import dct_matrix
from .kmap import frame_to_blocks, blocks_to_frame, _d1, _d2


# ------------------------------------------------ channel-aware plumbing

def frames_to_blocks_c(x):
    """(B, C, H, W) -> (B, C, N, 8, 8)."""
    b, c, h, w = x.shape
    blocks = frame_to_blocks(x.reshape(b * c, 1, h, w))
    return blocks.reshape(b, c, -1, 8, 8)


def blocks_to_frames_c(blocks, h, w):
    """(B, C, N, 8, 8) -> (B, C, H, W)."""
    b, c = blocks.shape[:2]
    frames = blocks_to_frame(blocks.reshape(b * c, -1, 8, 8), h, w)
    return frames.reshape(b, c, h, w)


# ------------------------------------------------------------------ model

class WienerDctModel(nn.Module):
    """(B, C, H, W) -> per-channel per-coefficient gains (B, C, N, 8, 8)."""

    def __init__(self, lam_rate=2.5e-3, channels=32, in_channels=3,
                 gmax=4.0):
        super().__init__()
        self.lam_rate = lam_rate
        self.in_channels = in_channels
        self.gmax = gmax
        d = torch.tensor(dct_matrix(8), dtype=torch.float32)
        self.register_buffer("dmat", d)
        c = channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # /2
            nn.Conv2d(c, 2 * c, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # /4
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # /8 block grid
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, in_channels * 64, 1),
        )
        # start gains just below 1 (sigmoid responsive region); with a
        # zero-init bias they start at gmax/2 and the rate/TV pressure
        # slams them into sigmoid saturation (dead, unrecoverable gains)
        nn.init.constant_(self.net[-1].bias, -1.6)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, x):
        b, c = x.shape[0], self.in_channels
        logits = self.net(x)                    # (B, C*64, H/8, W/8)
        g = self.gmax * torch.sigmoid(logits)   # gains in [0, gmax]
        g = g.reshape(b, c, 64, -1).permute(0, 1, 3, 2)   # (B, C, N, 64)
        g = torch.cat([torch.ones_like(g[..., :1]), g[..., 1:]], dim=-1)
        return g.reshape(b, c, -1, 8, 8)        # DC forced to 1


def wiener_apply(model, frames_t):
    """frames_t (B, C, H, W), H/W divisible by 8 -> (rec, rec_blocks,
    gains, coeffs)."""
    _, _, h, w = frames_t.shape
    gains = model(frames_t)                     # (B, C, N, 8, 8)
    blocks = frames_to_blocks_c(frames_t)
    coeffs = torch.matmul(torch.matmul(model.dmat, blocks), model.dmat.T)
    rec_blocks = torch.matmul(torch.matmul(model.dmat.T, gains * coeffs),
                              model.dmat)
    return blocks_to_frames_c(rec_blocks, h, w), rec_blocks, gains, coeffs


def wiener_loss(model, frame, target=None, w_e1=0.05, w_e2=0.05):
    """target=None -> naturalness mode; else restoration (paired) mode."""
    if target is None:
        target = frame
    rec, rec_blocks, gains, _ = wiener_apply(model, frame)

    target_blocks = frames_to_blocks_c(target)
    recon = (rec_blocks - target_blocks).pow(2).sum(dim=(-1, -2)).mean()
    k_g = gains.sum(dim=(-1, -2)) - 1.0         # effective AC count (B,C,N)
    rate = k_g.mean()

    # output-only smoothness: minimize derivatives of the reconstruction
    dx, dy = _d1(rec)
    e1 = dx.abs().mean() + dy.abs().mean()
    d2x, d2y = _d2(rec)
    e2 = d2x.abs().mean() + d2y.abs().mean()

    total = recon + model.lam_rate * rate + w_e1 * e1 + w_e2 * e2
    logs = {
        "loss": float(total.detach()),
        "recon": float(recon.detach()),
        "k_mean": float(k_g.detach().mean()),
        "g_mean": float(gains.detach()[..., 1:].mean()),
        "g_over1": float((gains.detach()[..., 1:] > 1.0).float().mean()),
        "tv1": float(e1.detach()),
    }
    return total, logs


# ------------------------------------------------------------ persistence

def save_model(model, path, extra=None):
    torch.save({"state_dict": model.state_dict(),
                "lam_rate": model.lam_rate,
                "in_channels": model.in_channels,
                "gmax": model.gmax,
                "kind": "wiener",
                "extra": extra or {}}, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = WienerDctModel(lam_rate=ckpt["lam_rate"],
                           in_channels=ckpt.get("in_channels", 1),
                           gmax=ckpt.get("gmax", 1.0))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model
