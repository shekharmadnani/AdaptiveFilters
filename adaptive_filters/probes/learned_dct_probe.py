"""Learned DCT-sparsity probe (requires PyTorch + trained weights).

Same template as every probe, with the theta estimator replaced by the
trained K-predictor network:

  theta        : per-block threshold tau + predicted natural K (K_nat)
  filter       : keep-K DCT reconstruction with the predicted tau
  residual     : frame - reconstruction
  key feature  : deltaK = K_nat - K_emp, where K_emp is the closed-form
                 RD-optimal count on the observed block. ~0 on pristine
                 content; positive where coding emptied blocks that look
                 like they should carry detail.

Not in the default probe bank -- add explicitly:
  FeatureExtractor(probes=default_probes() + [LearnedDctProbe("kdct.pt")])
"""

import numpy as np

from .base import Probe, ProbeResult, common_residual_features


class LearnedDctProbe(Probe):
    name = "ldct"

    def __init__(self, weights, device=None, batch_size=512):
        import torch  # local import keeps the core package numpy-only

        from ..learned.kdct import load_model, pick_device

        self._torch = torch
        self.device = pick_device(device)
        self.model = load_model(weights, device=self.device)
        self.lam_rate = self.model.lam_rate
        self.batch_size = batch_size

    def run(self, frame, prev_frame=None):
        torch = self._torch
        from ..learned.kdct import (
            patch_to_blocks, blocks_to_patch, block_dct, block_idct,
            rd_optimal_k,
        )

        h, w = frame.shape
        hp, wp = h // 64, w // 64
        if hp == 0 or wp == 0:
            raise ValueError("frame smaller than one 64x64 patch")
        crop = frame[: hp * 64, : wp * 64]

        patches = (crop.reshape(hp, 64, wp, 64).transpose(0, 2, 1, 3)
                   .reshape(-1, 1, 64, 64).astype(np.float32))

        rec_list, dk_list, knat_list, kemp_list, tau_list = [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                batch = torch.from_numpy(patches[i : i + self.batch_size]
                                         ).to(self.device)
                tau, k_nat = self.model(batch)

                blocks = patch_to_blocks(batch)
                coeffs = block_dct(self.model.dmat, blocks)

                # hard keep-mask with the predicted threshold (DC kept)
                tau_b = tau.reshape(-1, 64, 1, 1)
                keep = coeffs.abs() > tau_b
                keep[..., 0, 0] = True
                rec = blocks_to_patch(
                    block_idct(self.model.dmat, coeffs * keep))

                k_emp = rd_optimal_k(coeffs, self.lam_rate)     # (B, 64)
                dk = k_nat.reshape(-1, 64) - k_emp

                rec_list.append(rec.squeeze(1).cpu().numpy())
                dk_list.append(dk.cpu().numpy())
                knat_list.append(k_nat.reshape(-1, 64).cpu().numpy())
                kemp_list.append(k_emp.cpu().numpy())
                tau_list.append(tau.reshape(-1, 64).cpu().numpy())

        rec = np.concatenate(rec_list).reshape(hp, wp, 64, 64)
        filtered = rec.transpose(0, 2, 1, 3).reshape(hp * 64, wp * 64)
        filtered = filtered.astype(np.float64)
        residual = crop - filtered

        dk = np.concatenate(dk_list).ravel()
        k_nat = np.concatenate(knat_list).ravel()
        k_emp = np.concatenate(kemp_list).ravel()
        tau_all = np.concatenate(tau_list).ravel()

        feats = common_residual_features(filtered, residual)
        # |dk| = magnitude of the naturalness-prior breakdown (severity);
        # signed dk carries artifact type (negative ~ blur, positive ~ blocking)
        feats["dk_abs_mean"] = float(np.abs(dk).mean())
        feats["dk_abs_p90"] = float(np.percentile(np.abs(dk), 90))
        feats["dk_mean"] = float(dk.mean())
        feats["dk_std"] = float(dk.std())
        feats["dk_pos_frac"] = float(np.mean(dk > 2.0))
        feats["k_nat_mean"] = float(k_nat.mean())
        feats["k_emp_mean"] = float(k_emp.mean())
        feats["tau_mean"] = float(tau_all.mean())

        return ProbeResult(filtered=filtered, residual=residual, features=feats)
