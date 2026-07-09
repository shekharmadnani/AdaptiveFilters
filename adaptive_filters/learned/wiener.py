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
                 gmax=4.0, affine=False, tmax=1.0):
        super().__init__()
        self.lam_rate = lam_rate
        self.in_channels = in_channels
        self.gmax = gmax
        self.affine = affine
        self.tmax = tmax
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
        if affine:  # gen-4 synthesis head: starts at exactly t = 0
            self.head_t = nn.Conv2d(2 * c, in_channels * 64, 1)
            nn.init.zeros_(self.head_t.bias)
            nn.init.normal_(self.head_t.weight, std=1e-3)

    def _shape_field(self, raw, b, c, dc_value):
        """(B, C*64, h8, w8) -> (B, C, N, 8, 8) with the DC entry pinned."""
        f = raw.reshape(b, c, 64, -1).permute(0, 1, 3, 2)     # (B, C, N, 64)
        dc = torch.full_like(f[..., :1], dc_value)
        f = torch.cat([dc, f[..., 1:]], dim=-1)
        return f.reshape(b, c, -1, 8, 8)

    def forward(self, x):
        b, c = x.shape[0], self.in_channels
        h = self.net[:-1](x)
        g = self._shape_field(
            self.gmax * torch.sigmoid(self.net[-1](h)), b, c, dc_value=1.0)
        t = None
        if self.affine:
            t = self._shape_field(
                self.tmax * torch.tanh(self.head_t(h)), b, c, dc_value=0.0)
        return g, t


class WienerDctModelV2(nn.Module):
    """Context-widened variants (the receptive-field ablation).

    arch="b": baseline trunk + dilated 3x3 stack (d=2,4,8) at the block
              grid. Each grid step is 8 px, so the dilations add
              (2+4+8)*2*8 = 224 px of receptive field for ~zero cost:
              view grows from ~40 px (+-2 blocks) to ~260 px (+-15 blocks).
    arch="c": arch b + a U-Net deep branch: pool to /16 and /32 (where one
              glance covers the whole training crop), then upsample back
              and concatenate with the fine features (skip connections --
              the analysis/synthesis pyramid with detail subbands carried
              around the bottleneck). Big-picture context for synthesis,
              fine detail preserved for gain decisions.

    Same forward contract as WienerDctModel: (B,C,H,W) -> gains
    (B,C,N,8,8) in [0,gmax], DC pinned to 1.
    """

    def __init__(self, lam_rate=2.5e-3, channels=32, in_channels=3,
                 gmax=4.0, arch="b", affine=False, tmax=1.0):
        super().__init__()
        assert arch in ("b", "c")
        self.lam_rate = lam_rate
        self.in_channels = in_channels
        self.gmax = gmax
        self.arch = arch
        self.affine = affine
        self.tmax = tmax
        d = torch.tensor(dct_matrix(8), dtype=torch.float32)
        self.register_buffer("dmat", d)
        c = channels
        r = nn.ReLU(inplace=True)
        self.stem = nn.Sequential(                      # -> (2c, H/8, W/8)
            nn.Conv2d(in_channels, c, 3, padding=1), r,
            nn.Conv2d(c, c, 3, padding=1), r,
            nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, 3, padding=1), r,
            nn.MaxPool2d(2),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), r,
            nn.MaxPool2d(2),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), r,
        )
        self.dilated = nn.Sequential(                   # spread-out vision
            nn.Conv2d(2 * c, 2 * c, 3, padding=2, dilation=2), r,
            nn.Conv2d(2 * c, 2 * c, 3, padding=4, dilation=4), r,
            nn.Conv2d(2 * c, 2 * c, 3, padding=8, dilation=8), r,
        )
        if arch == "c":
            self.down1 = nn.Sequential(                 # /16
                nn.MaxPool2d(2), nn.Conv2d(2 * c, 4 * c, 3, padding=1), r)
            self.down2 = nn.Sequential(                 # /32 (whole-crop view)
                nn.MaxPool2d(2), nn.Conv2d(4 * c, 4 * c, 3, padding=1), r)
            self.fuse16 = nn.Sequential(
                nn.Conv2d(8 * c, 2 * c, 3, padding=1), r)
            self.fuse8 = nn.Sequential(
                nn.Conv2d(4 * c, 2 * c, 3, padding=1), r)
        self.head = nn.Conv2d(2 * c, in_channels * 64, 1)
        nn.init.constant_(self.head.bias, -1.6)         # gains start ~0.7
        nn.init.normal_(self.head.weight, std=0.01)
        if affine:  # gen-4 synthesis head: starts at exactly t = 0
            self.head_t = nn.Conv2d(2 * c, in_channels * 64, 1)
            nn.init.zeros_(self.head_t.bias)
            nn.init.normal_(self.head_t.weight, std=1e-3)

    def forward(self, x):
        import torch.nn.functional as F

        b, c = x.shape[0], self.in_channels
        f = self.dilated(self.stem(x))                  # fine, at block grid
        if self.arch == "c":
            g16 = self.down1(f)
            g32 = self.down2(g16)
            u16 = F.interpolate(g32, size=g16.shape[-2:], mode="nearest")
            m16 = self.fuse16(torch.cat([u16, g16], dim=1))
            u8 = F.interpolate(m16, size=f.shape[-2:], mode="nearest")
            f = self.fuse8(torch.cat([u8, f], dim=1))   # big-picture + skip
        g = WienerDctModel._shape_field(
            self, self.gmax * torch.sigmoid(self.head(f)), b, c, dc_value=1.0)
        t = None
        if self.affine:
            t = WienerDctModel._shape_field(
                self, self.tmax * torch.tanh(self.head_t(f)), b, c,
                dc_value=0.0)
        return g, t


def wiener_apply(model, frames_t):
    """frames_t (B, C, H, W), H/W divisible by 8 -> (rec, rec_blocks,
    gains, t, coeffs). t is None for gain-only (gen-3) models; for affine
    (gen-4) models the reconstruction is IDCT(g*X + t)."""
    _, _, h, w = frames_t.shape
    gains, t = model(frames_t)                  # each (B, C, N, 8, 8)
    blocks = frames_to_blocks_c(frames_t)
    coeffs = torch.matmul(torch.matmul(model.dmat, blocks), model.dmat.T)
    xf = gains * coeffs
    if t is not None:
        xf = xf + t
    rec_blocks = torch.matmul(torch.matmul(model.dmat.T, xf), model.dmat)
    return blocks_to_frames_c(rec_blocks, h, w), rec_blocks, gains, t, coeffs


def wiener_loss(model, frame, target=None, w_e1=0.05, w_e2=0.05, mu=0.02):
    """target=None -> naturalness mode; else restoration (paired) mode.

    mu prices synthesis (L1 on t): reusing a coefficient costs lam_rate
    per unit gain, inventing one costs mu per unit magnitude. L1 (not L2)
    so t snaps to exactly zero wherever synthesis is not demanded --
    pristine input must yield t ~ 0 for the t-map to stay a clean
    damage indicator.
    """
    if target is None:
        target = frame
    rec, rec_blocks, gains, t, _ = wiener_apply(model, frame)

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
    if t is not None:
        t_pen = t.abs().sum(dim=(-1, -2)).mean()   # per-block L1
        total = total + mu * t_pen
        logs["loss"] = float(total.detach())
        logs["t_abs"] = float(t.detach().abs().mean())
        logs["t_active"] = float((t.detach().abs() > 0.01).float().mean())
    return total, logs


# ------------------------------------------------------------ persistence

def save_model(model, path, extra=None):
    torch.save({"state_dict": model.state_dict(),
                "lam_rate": model.lam_rate,
                "in_channels": model.in_channels,
                "gmax": model.gmax,
                "arch": getattr(model, "arch", "a"),
                "affine": getattr(model, "affine", False),
                "tmax": getattr(model, "tmax", 1.0),
                "kind": "wiener",
                "extra": extra or {}}, path)


def build_model(arch="a", **kw):
    if arch == "a":
        return WienerDctModel(**kw)
    return WienerDctModelV2(arch=arch, **kw)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(arch=ckpt.get("arch", "a"),
                        lam_rate=ckpt["lam_rate"],
                        in_channels=ckpt.get("in_channels", 1),
                        gmax=ckpt.get("gmax", 1.0),
                        affine=ckpt.get("affine", False),
                        tmax=ckpt.get("tmax", 1.0))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model
