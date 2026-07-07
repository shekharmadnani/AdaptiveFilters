"""Standalone adaptive DCT filter (the core object of study).

    frame -> CNN -> K-map -> keep top-K AC + DC -> IDCT -> filtered frame
    residual = frame - filtered

Weights come from train_kmap.py (RMSE + lam*K + asymmetric first/second
derivative regularizers, pristine-only training). The residual is the
artifact-revealing signal to be used alongside other adaptive-filter
residuals; the feature-extraction probe (LearnedKMapProbe) is a thin
wrapper around this class.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AdaptiveDctResult:
    filtered: np.ndarray   # (H8, W8) float64, top-K reconstruction
    residual: np.ndarray   # (H8, W8) float64, frame - filtered
    k_pred: np.ndarray     # (H8/8, W8/8) float64, CNN-predicted K per block
    k_emp: np.ndarray      # same grid: closed-form RD count #{X_ac^2 > lam}
    k_tail: np.ndarray     # same grid: smallest K holding 98% of AC energy


class AdaptiveDctFilter:
    """numpy in / numpy out; torch only inside. Frame is luma in [0, 1]
    (values in [0, 255] are rescaled automatically)."""

    def __init__(self, weights, device=None):
        import torch

        from .kdct import pick_device
        from .kmap import load_model

        self._torch = torch
        self.device = pick_device(device)
        self.model = load_model(weights, device=self.device)
        self.lam_rate = self.model.lam_rate

    def apply(self, frame):
        torch = self._torch
        from .kmap import frame_to_blocks, blocks_to_frame, hard_topk_mask

        frame = np.asarray(frame, dtype=np.float64)
        if frame.max() > 2.0:
            frame = frame / 255.0
        h8, w8 = (frame.shape[0] // 8) * 8, (frame.shape[1] // 8) * 8
        crop = frame[:h8, :w8]
        gh, gw = h8 // 8, w8 // 8

        with torch.no_grad():
            x = torch.from_numpy(crop.astype(np.float32))[None, None].to(
                self.device)
            k_pred = self.model(x).flatten(1)              # (1, N)

            blocks = frame_to_blocks(x)
            coeffs = torch.matmul(torch.matmul(self.model.dmat, blocks),
                                  self.model.dmat.T)
            m = hard_topk_mask(coeffs, k_pred)
            rec = blocks_to_frame(
                torch.matmul(torch.matmul(self.model.dmat.T, m * coeffs),
                             self.model.dmat), h8, w8)

            ac2 = coeffs.reshape(1, -1, 64)[..., 1:].pow(2)
            k_emp = (ac2 > self.lam_rate).float().sum(-1)
            srt = ac2.sort(dim=-1, descending=True).values
            cum = srt.cumsum(-1)
            total = cum[..., -1:].clamp(min=1e-12)
            k_tail = (cum < 0.98 * total).float().sum(-1) + 1.0
            k_tail = torch.where(total.squeeze(-1) < 1e-10,
                                 torch.zeros_like(k_tail), k_tail)

            filtered = rec.squeeze().cpu().numpy().astype(np.float64)
            k_pred_np = k_pred.reshape(gh, gw).cpu().numpy().astype(np.float64)
            k_emp_np = k_emp.reshape(gh, gw).cpu().numpy().astype(np.float64)
            k_tail_np = k_tail.reshape(gh, gw).cpu().numpy().astype(np.float64)

        return AdaptiveDctResult(
            filtered=filtered,
            residual=crop - filtered,
            k_pred=k_pred_np,
            k_emp=k_emp_np,
            k_tail=k_tail_np,
        )


class AdaptiveWienerFilter:
    """DCT-domain adaptive Wiener filter (gain-map model): numpy in/out.

    Same result contract as AdaptiveDctFilter, with k_pred = effective AC
    count (sum of predicted gains per block) -- the continuous analog of K.
    """

    def __init__(self, weights, device=None):
        import torch

        from .kdct import pick_device
        from .wiener import load_model

        self._torch = torch
        self.device = pick_device(device)
        self.model = load_model(weights, device=self.device)
        self.lam_rate = self.model.lam_rate

    def apply(self, frame):
        torch = self._torch
        from .wiener import wiener_apply

        frame = np.asarray(frame, dtype=np.float64)
        if frame.max() > 2.0:
            frame = frame / 255.0
        h8, w8 = (frame.shape[0] // 8) * 8, (frame.shape[1] // 8) * 8
        crop = frame[:h8, :w8]
        gh, gw = h8 // 8, w8 // 8

        with torch.no_grad():
            x = torch.from_numpy(crop.astype(np.float32))[None, None].to(
                self.device)
            rec, _, gains, coeffs = wiener_apply(self.model, x)

            k_g = gains.sum(dim=(2, 3)) - 1.0            # effective AC count
            ac2 = coeffs.reshape(1, -1, 64)[..., 1:].pow(2)
            k_emp = (ac2 > self.lam_rate).float().sum(-1)
            srt = ac2.sort(dim=-1, descending=True).values
            cum = srt.cumsum(-1)
            total = cum[..., -1:].clamp(min=1e-12)
            k_tail = (cum < 0.98 * total).float().sum(-1) + 1.0
            k_tail = torch.where(total.squeeze(-1) < 1e-10,
                                 torch.zeros_like(k_tail), k_tail)

            filtered = rec.squeeze().cpu().numpy().astype(np.float64)
            k_pred_np = k_g.reshape(gh, gw).cpu().numpy().astype(np.float64)
            k_emp_np = k_emp.reshape(gh, gw).cpu().numpy().astype(np.float64)
            k_tail_np = k_tail.reshape(gh, gw).cpu().numpy().astype(np.float64)

        return AdaptiveDctResult(
            filtered=filtered,
            residual=crop - filtered,
            k_pred=k_pred_np,
            k_emp=k_emp_np,
            k_tail=k_tail_np,
        )
