# AdaptiveFilters — content-adaptive filter probe bank for no-reference VQA

> **Full technical report**: [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) —
> the complete design history with explanations, all experimental results,
> lessons learned, and an annotated bibliography (papers, tutorials,
> reference codebases) supporting every technique used here.

A NumPy-only prototype of the design discussed in the planning sessions:
a bank of **content-adaptive filters** whose parameters are blindly estimated
from the content itself. Each filter is a model of *natural* content, so its
residual `r = y − F_θ(y)` isolates unnatural discontinuities (coding
artifacts), and statistics of `(residual, θ-map)` form a content-normalized
feature vector for a no-reference quality index — no MOS anywhere.

Every probe follows one template:

```
1. Naturalness model   x ≈ M_θ   (sparse DCT / locally linear / directional / ...)
2. Blind θ estimation  from the degraded content (MAD, SURE-style, closed-form LS)
3. Probe residual      r = y − F_θ(y)
4. Features            stats of r, its spatial localization, and θ itself
```

## Components

| Component | File | Naturalness model | Residual exposes |
|---|---|---|---|
| Adaptive DCT keep-K probe | [dct_probe.py](adaptive_filters/probes/dct_probe.py) | block-sparse in DCT | ringing, mosquito noise; K-map detects compression |
| Guided / local-Wiener probe | [guided_probe.py](adaptive_filters/probes/guided_probe.py) | locally linear `x ≈ aI + b` | noise level, grain loss, banding in flat regions |
| Deblocking-grid probe | [deblock_probe.py](adaptive_filters/probes/deblock_probe.py) | on-grid ≡ off-grid statistics | blockiness (grid-phase peak ratio) |
| SAO edge-offset probe | [sao_probe.py](adaptive_filters/probes/sao_probe.py) | samples agree with directional neighbors | ringing, edge over/undershoot |
| Directional (CDEF-style) probe | [directional_probe.py](adaptive_filters/probes/directional_probe.py) | locally 1-D along structure-tensor orientation | jaggies, cross-edge ringing |
| Temporal probe | [temporal_probe.py](adaptive_filters/probes/temporal_probe.py) | brightness constancy | flicker, pumping, frame repeats |
| Shared residual statistics | [features/stats.py](adaptive_filters/features/stats.py) | GGD fit, MAD, Immerkær noise, lag-1 corr, orthogonality | — |
| Frame → feature pipeline | [pipeline.py](adaptive_filters/pipeline.py) | probe bank × 2 dyadic scales | ~110-dim vector |
| Naturalness anchor (layer B) | [naturalness.py](adaptive_filters/naturalness.py) | NIQE-style distance to pristine corpus | unsupervised score + drift monitor |
| Fusion (layer A) | [fusion.py](adaptive_filters/fusion.py) | RidgeFusion (baseline) / GbtFusion (optional XGBoost, monotone constraints) | supervised quality index |
| End-to-end demo / smoke test | [demo.py](demo.py) | synthetic frames + JPEG-like ladder | validation protocol in miniature |
| Frame ingestion | [io.py](adaptive_filters/io.py) | raw YUV, Y4M, images, ffmpeg pipe | float64 luma in [0,1] |
| VMAF labels | [vmaf.py](adaptive_filters/vmaf.py) | ffmpeg libvmaf runner + JSON parser | per-frame proxy labels |
| Dataset builder | [dataset.py](adaptive_filters/dataset.py) + [build_dataset.py](build_dataset.py) | manifest of (content, dist, ref) | features + labels .npz |
| Trainer / scorer CLIs | [train_fusion.py](train_fusion.py), [score_video.py](score_video.py) | grouped split, ridge-vs-GBT bake-off | versioned model bundle |
| Learned K-predictor v1 (patch/τ) | [learned/kdct.py](adaptive_filters/learned/kdct.py) + [train_kdct.py](train_kdct.py) | CNN predicts per-block DCT threshold + natural K from 64x64 context (PyTorch) | trained on pristine patches only |
| Learned DCT probe v1 | [learned_dct_probe.py](adaptive_filters/probes/learned_dct_probe.py) | keep-K with learned tau; deltaK = K_nat − K_emp | residual features monotone (SRCC 1.0); signed deltaK separates blur (−) from blocking (+) |
| **Learned K-map v2 (full-frame/top-K)** | [learned/kmap.py](adaptive_filters/learned/kmap.py) + [train_kmap.py](train_kmap.py) | FCN: frame → (W/8)×(H/8) K-map; rank-space differentiable top-K; asymmetric no-new-edges loss | K-form is contrast-invariant and immune to the fixed-τ quantization pathology |
| Learned K-map probe | [learned_kmap_probe.py](adaptive_filters/probes/learned_kmap_probe.py) | top-K reconstruction with predicted K | `lost_edge1` (frame edges the sparse model can't explain) perfectly monotone; `new_edge1` monotone on real content |
| Standalone adaptive DCT filter | [learned/adaptive_dct.py](adaptive_filters/learned/adaptive_dct.py) + [inspect_filter.py](inspect_filter.py) | frame → filtered/residual/K-map, numpy in/out | filter-level API for residual studies |
| Artifact simulation suite | [artifacts.py](adaptive_filters/artifacts.py) | 8 artifact types × 5 severities: compression, blur, noise, banding, packet-loss (interp/copy concealment), block fill, stale regions | doubles as paired-training data generator |
| Artifact-response matrix | [validate_artifacts.py](validate_artifacts.py) | every artifact × every feature, SRCC gate | all 8 artifacts covered; no blind spots on real content |
| Restoration training mode | `train_kmap.py --paired` → `models/kmap_restore.pt` | degraded input, pristine target; pl_copy + stale held out | K policy adapts; selection-only capacity bounds restoration (see notes) |
| **Learned DCT-domain Wiener filter** | [learned/wiener.py](adaptive_filters/learned/wiener.py) + [train_wiener.py](train_wiener.py) → `models/wiener.pt` | color (C-channel) CNN predicts per-channel per-coefficient gains in [0, gmax=4] (amplification allowed; DC pinned to 1); loss = RMSE + λ·Σg + output-only \|∇rec\| + \|∇²rec\|; paired training by default | primary residual generator; K_g = sum of gains is the continuous effective-K, one map per channel |
| Learned Wiener probe | [learned_wiener_probe.py](adaptive_filters/probes/learned_wiener_probe.py) | same feature contract as the K-map probe (`lwn` prefix) | standalone: responds to all 8 artifact classes on real content |
| Real H.264 corruption pipeline | [bitstream.py](adaptive_filters/bitstream.py) | x264 encode (in-loop deblocking OFF, no SAO in H.264) → byte-flip non-IDR slice NALs → decode with `-ec 1` (concealment-deblock OFF) | `train_wiener.py --h264-frac 0.5` mixes real pairs into paired training; loss unchanged (RMSE + λ·Σg + d1/d2 hinges) |

## Quickstart

```
pip install numpy
python demo.py
```

The demo generates synthetic pristine content, degrades it with JPEG-like
8×8 DCT quantization at 6 quality levels (a stand-in for an encode ladder),
and validates on **held-out content** (content-grouped split — no leakage):

- Layer B (naturalness anchor, zero labels): per-content SRCC = 1.000
- Layer A (ridge fusion, distortion-level proxy labels): per-content SRCC = 1.000

## Library use

```python
from adaptive_filters import FeatureExtractor, NaturalnessModel, RidgeFusion, to_vector

extractor = FeatureExtractor()                 # 6 probes x 2 scales
feats = extractor.extract(frame)               # {name: value}; luma, uint8 or [0,1]
names, vec = to_vector(feats)

anchor = NaturalnessModel().fit(pristine_vectors)   # layer B: no labels
score = anchor.score(vec)                           # larger = less natural
anchor.top_deviations(vec, names)                   # per-artifact triage

fusion = RidgeFusion().fit(X_train, y_train)        # layer A: proxy labels (e.g. VMAF)
```

## Design notes (agreed in planning)

- **The filter must not adapt to the distortion.** Defenses implemented:
  θ estimated with robust statistics (MAD, medians), θ read back as features
  (K-map, a-map, offset magnitudes), thresholds derived from off-artifact
  statistics (deblock β from off-grid diffs).
- **Content-grouped validation** is non-negotiable: frames of the same source
  never straddle train/test. The demo demonstrates the split.
- **Fusion is deliberately the most boring component.** Ridge is the floor;
  GBT (monotone-constrained) ships only if it beats ridge meaningfully on
  grouped held-out data.
- **Production labels**: replace the demo's distortion-level proxy with VMAF
  computed against masters on your own encode ladder (layer A), keep the
  naturalness anchor as the unsupervised sanity/drift layer (layer B).

## Learned K-predictor (v2 probe)

```
python train_kdct.py                 # trains on BVI-CC1 masters (F:\DVI) or synthetic fallback
python train_kdct.py --skip-train    # re-run validation on existing models/kdct.pt
```

Loss = per-block reconstruction SSE + λ·K (RD sparsity) + 1st/2nd-order
smoothness of the reconstruction + K-map coherence + K-head regression to
the closed-form RD-optimal count (keep iff X² > λ — the baseline the
network must beat). Empirical findings baked into the design:

- The τ-reconstruction **residual features are the primary severity
  signal** (perfectly monotone on real content).
- **Signed ΔK is an artifact-type signal**, not a severity signal: blur
  drags it negative, blocking pushes it positive (+3.3 at JPEG q=10).
  |ΔK| is noisy at mid severity (K-head RMSE ≈ 2.5 ACs).
- `k_emp` under a fixed λ is non-monotone in quality: quantization rounds
  surviving coefficients *up* past the threshold at mid qualities.

### K-map v2 (full-frame, rank-space top-K)

```
python train_kmap.py                 # trains on BVI-CC1 masters or synthetic fallback
python train_kmap.py --skip-train    # re-validate existing models/kmap.pt
```

Differences from v1: the network outputs K directly (bounded, contrast-
invariant, the dual of the τ form); selection is a differentiable top-K in
rank space, `m_r = σ((K − r − ½)/T)` applied through the per-block sort of
AC magnitudes; and the smoothness regularizers are **asymmetric hinges**
`ReLU(|∇rec| − |∇orig|)` (1st + 2nd order) that penalize only edges the
filter *creates*, never existing content. Two edge features fall out at
inference: `new_edge1` (created edges) and `lost_edge1` (frame edge energy
the sparse reconstruction cannot explain — a clean, monotone
detail-destruction signal).

## v1 → v2 roadmap

- Motion-compensated temporal probe (v1 is frame-difference only)
- Non-local / low-rank probe (grain and texture damage)
- Temporal pooling (mean + low-percentile + hysteresis) for segment scores
- Real validation harness: encode ladders, VMAF distillation, monotonicity
  and content-invariance regression tests, synthetic artifact confusion matrix
- Optional CNN feature branch (only if the feature-ceiling diagnostics
  — oracle fusion test, failure-set mining — show the probes have run out)
