# Content-Adaptive DCT Filtering for No-Reference Video Quality Assessment

**Technical report and design history**
Repository: https://github.com/shekharmadnani/AdaptiveFilters
Status: research prototype, all results reproducible from the repo CLIs.

---

## Table of contents

1. [Purpose and vision](#1-purpose-and-vision)
2. [The core framework: filters as naturalness models](#2-the-core-framework)
3. [The classical probe bank](#3-the-classical-probe-bank)
4. [The no-reference VQA pipeline](#4-the-no-reference-vqa-pipeline)
5. [Fusion-model choice: ridge, GBT, deep learning](#5-fusion-model-choice)
6. [The learned filter: four design generations](#6-the-learned-filter-four-generations)
7. [Training data: simulated artifacts and real H.264 corruption](#7-training-data)
8. [Experimental results](#8-experimental-results)
9. [Lessons learned (empirical)](#9-lessons-learned)
10. [Repository map and how to run](#10-repository-map)
11. [Appendix A: build-the-filter-yourself roadmap](#appendix-a-build-it-yourself-roadmap)
12. [Appendix B: annotated bibliography and tutorials](#appendix-b-references)

---

## 1. Purpose and vision

The ultimate deliverable is a **no-reference video quality (NR-VQA) index**
that is practical in an industrial environment. Three constraints shaped
every decision:

- **No MOS anywhere.** Subjective-score regression overfits legacy
  databases and does not transfer; labels must come from full-reference
  metrics (VMAF) computed on our own encode ladders, or from no-label
  naturalness statistics.
- **Interpretability and diagnostics.** A single opaque score does not
  survive contact with an operations team; the index must decompose into
  per-artifact signals (blockiness, ringing, banding, noise, temporal).
- **Deployability.** Deterministic, cheap, versionable components;
  monotonicity in encoder parameters; content-invariance (the score must
  measure degradation, not content complexity).

The mechanism chosen: a bank of **content-adaptive restoration filters**.
Each filter embodies a model of *natural* content; applied to a degraded
frame, whatever the model cannot explain lands in the **residual**
(frame − filtered). Statistics of the residual — and of the filter's own
estimated parameters — form the feature vector from which quality is
estimated. The restoration filters are instruments, not the product: the
residual characteristics are the product.

## 2. The core framework

Every filter (classical or learned) follows one template:

```
1. Naturalness model   x ≈ M_θ      (sparse in DCT, locally linear,
                                      directional, neighbor-consistent, ...)
2. Blind θ estimation  θ̂ from the degraded content itself
                       (MAD statistics, SURE, closed-form least squares,
                        or a trained CNN)
3. Probe residual      r = y − F_θ̂(y)
4. Features            statistics of r, its spatial localization,
                       and the θ̂-map itself
```

**Why content-adaptivity is the point, not a refinement.** A fixed filter's
residual confounds content complexity with distortion (grass looks
"distorted" forever). When θ adapts to content, the content is *explained
by the model* and subtracted out; what remains correlates with degradation.
This was later demonstrated quantitatively: a fixed-K DCT filter collapses
to SRCC 0.78 on grainy content while the adaptive versions hold ≥ 0.93
(Section 8.4).

**The one design trap** (recurring theme): the filter must adapt to the
*content* but not to the *distortion*. If the parameter estimator is too
faithful to the observed block, it models the artifact as content,
reproduces it, and the residual goes blind exactly when quality is worst.
Defenses used: robust statistics for θ (medians/MAD), reading θ̂ back as
features (deviation from the natural prior is itself a signal), estimating
θ at a larger spatial scale than the filtering, and — in the learned
generation — training on pristine data only or on paired
(degraded → pristine) data, never on degraded self-reconstruction.

**Two operating regimes.** Encoder-side (original available): θ by least
squares, distortion-driven — this is how codec in-loop filters (ALF, SAO,
AV1 loop restoration) work. Decoder-side/blind (our case): θ from the
degraded signal via SURE, robust statistics, or a learned prior.

## 3. The classical probe bank

Six hand-designed probes (all NumPy, `adaptive_filters/probes/`), each an
instance of the Section-2 template:

| Probe | Naturalness model | Blind θ | Residual exposes |
|---|---|---|---|
| `dct` — adaptive keep-K DCT | 8×8 blocks sparse in DCT | per-block noise floor σ̂ = MAD(high-freq coeffs)/0.6745; keep iff \|X\| > τσ̂ | ringing, mosquito noise; K-map detects compression |
| `gd` — guided/local Wiener | locally linear x ≈ aI + b | closed-form a = var/(var+ε), ε from Immerkær blind noise estimate | noise level, grain loss, banding (a-map plateaus) |
| `dbk` — deblocking grid | on-grid ≡ off-grid statistics | clipping threshold β from off-grid neighbor differences | blockiness (grid-phase energy ratio) |
| `sao` — edge offsets | samples agree with directional neighbors | per-category mean pull toward neighbor average (HEVC-SAO classification) | ringing, edge over/undershoot |
| `dir` — directional | locally 1-D along structure-tensor orientation | orientation/coherence from smoothed structure tensor; CDEF-style clamped smoothing | jaggies, cross-edge ringing |
| `tmp` — temporal | brightness constancy | none in v1 (prev frame = prediction) | flicker, pumping, frame repeats |

Key shared mathematics (`features/stats.py`):

- **GGD moment-matching fit** (BRISQUE machinery): for a generalized
  Gaussian, ρ(α) = E[\|x\|]²/E[x²] = Γ(2/α)²/(Γ(1/α)Γ(3/α)) is monotone in
  the shape α; natural residuals sit near Gaussian (α ≈ 2), artifact
  residuals push α down.
- **MAD robust sigma** σ̂ = median\|x − median\|/0.6745 (Donoho–Johnstone).
- **Orthogonality statistic** ρ(x̂, r): for an MMSE-optimal linear filter,
  E[x̂·r] = 0 (the orthogonality principle) — deviation from zero is a
  principled mis-tuning signal, not a heuristic.
- **Lag-1 residual autocorrelation**: a residual that removed only noise is
  ~white; structure in the residual means destroyed content or a coherent
  artifact.
- **Worst-tile pooling** `res_tile_max`: max over 64×64 tiles of mean r² —
  added so localized corruption (slice loss, block fill) cannot hide in
  frame means.

The **oracle keep rule** underlying the DCT probe (and later the learned
loss): with an orthonormal DCT, Parseval gives per-coefficient additivity of
distortion, so keeping coefficient X_k reduces block SSE by exactly X_k².
Under a rate penalty λ per kept coefficient, the optimum is separable:
**keep iff X_k² > λ**. This closed form is both the blind estimator's
threshold rule and the baseline any learned K predictor must beat.

## 4. The no-reference VQA pipeline

```
frame → probe bank (2 dyadic scales) → ~190 named features
      → layer A: RidgeFusion distilled against VMAF   (supervised proxy)
      → layer B: NIQE-style naturalness distance       (no labels; drift alarm)
      → temporal pooling (mean + low percentile) → segment score + sub-scores
```

**Label strategy without MOS** (the key industrial decision):

- **A. Full-reference proxy distillation** (primary): encode own masters
  across a ladder, compute per-frame VMAF against the master (free,
  unlimited, in-domain labels), train a small regressor features → VMAF.
  Precedented by NR-VMAF (IEEE TBC 2024) with a deep CNN; ours distills
  into interpretable features instead.
- **B. Opinion-unaware anchor**: fit mean/covariance of the feature vector
  on pristine frames only; score = clipped Mahalanobis distance. Catches
  out-of-distribution degradations layer A never saw; doubles as a drift
  monitor. (NIQE, IEEE SPL 2013.)
- **C. Learning-to-rank on encode ladders** (QP32 < QP27 < QP22 for the
  same content) — ordinal supervision for free; held in reserve.

**Validation protocol** (replaces MOS correlation):

1. Monotonicity: per-content SRCC(score, bitrate) across the ladder.
2. FR agreement: SRCC/PLCC against VMAF on held-out encodes.
3. Content invariance: pristine-score spread across contents must be small
   vs. the spread across QPs.
4. Synthetic artifact confusion matrix: each artifact at controlled
   severity must move at least one feature monotonically.
5. **Content-grouped splits, always**: frames of one source never straddle
   train/test. This single rule matters more than any hyperparameter —
   frame-level splits let the model memorize content signatures and produce
   fictional validation numbers (the reason many published results don't
   deploy).

## 5. Fusion-model choice

**Ridge regression** (chosen): standardized closed-form
w = (XᵀX + λI)⁻¹Xᵀy — the same normal-equations-with-diagonal-loading
mathematics as a regularized Wiener solve. Deterministic, inspectable,
extrapolates linearly (degrades gracefully on unfamiliar content).

**GBT (XGBoost)** (parked by decision): handles interactions and supports
monotonicity constraints, but trees cannot extrapolate — predictions
flatline outside the training feature hull. Observed live: on the grain
content held out from a 2-content training set, GBT scored SRCC −0.17 vs
ridge 0.94. Rule adopted: GBT ships only if it beats ridge by > 0.02
held-out SRCC on content-grouped splits ("ship ridge on ties").

**Deep learning / transformers** (assessed, deferred): on a ~100-dim
engineered feature vector, tree ensembles ≈ small MLPs (Grinsztajn et al.,
NeurIPS 2022) — no reason to switch the fusion stage. The legitimate
insertion point is *learned features* (a CNN branch), triggered only by
evidence: the oracle-fusion test (deliberately overfit model approximates
the information ceiling of the features) separating "bad features" from
"bad model", and failure-set mining naming a missing probe. In the
classical stack, **the features are the model** — feature design carries
~80% of the effort and sets the accuracy ceiling; the fusion model only
determines how close to that ceiling you get.

## 6. The learned filter: four generations

The DCT probe's parameter estimator was progressively replaced by a CNN.
Each generation kept the same probe interface, so every downstream
component (features, fusion, validation) measured the change objectively.

### Generation 1 — `kdct`: patch-based threshold predictor

64×64 patch → small CNN → per-8×8-block threshold τ (soft-gated,
temperature-annealed) + a K head regressing the closed-form RD-optimal
count. Pristine-only training (naturalness prior). Findings:

- The τ-reconstruction **residual features were perfectly monotone** with
  JPEG severity on real content (SRCC 1.0).
- **Signed ΔK = K_nat − K_emp is an artifact-type signal, not severity**:
  blur drags it negative (appearance co-degrades), blocking pushes it
  positive (+3.3 at q=10). The V-shape defeats any single-feature gate.
- `k_emp` under a fixed threshold is **non-monotone in quality**:
  mid-quality JPEG quantization rounds surviving coefficients *up* past
  the threshold before the count collapses.

### Generation 2 — `kmap`: full-frame K-map with differentiable top-K

Per the revised specification: fully convolutional CNN, frame →
(W/8)×(H/8) map of K (retained AC count), reconstruction keeps the K
largest AC magnitudes + DC. Two mathematical devices:

- **Rank-space soft top-K.** Sort AC magnitudes per block (gradients flow
  through values, not the permutation), mask by rank:
  m_r = σ((K − r − ½)/T). Differentiable in K, anneals to hard top-K as
  T → 0. K-form vs τ-form are duals (the K-th largest magnitude is an
  implicit threshold), but K is bounded, contrast-invariant, and immune to
  the k_emp rounding pathology.
- **Asymmetric no-new-edges hinges** (later superseded):
  ReLU(\|∇rec\| − \|∇orig\|) at first and second order — penalize only
  edges the filter *creates*. Second-order needed because first-order TV
  alone promotes staircase (banding) artifacts.

Findings: `lost_edge1` (frame edge energy the sparse natural model cannot
explain) emerged as a near-perfect monotone severity feature; a paired
"restoration mode" (degraded in, pristine target) confirmed the **capacity
limit of coefficient selection** — zeroing coefficients cannot restore
information that quantization removed, so the reconstruction floor rises
(per-block SSE 0.010 → 0.070) and the residual cannot flip polarity.

### Generation 3 — `wiener`: the current filter (gain-map)

Per the final specification, three changes over generation 2:

1. **Per-coefficient gains instead of binary selection**:
   X̂ = g ⊙ X with g ∈ [0, gmax], gmax = 4 — the filter may amplify
   (deblur-type restoration), not only attenuate; DC pinned to 1. The
   binary K-map is the special case g ∈ {0,1}; with paired training the
   MSE-optimal gain approaches classical Wiener shrinkage
   E[X_pristine·X]/E[X²] per coefficient, learned as a function of spatial
   context. K_g = Σg (effective AC count) replaces K in all features.
2. **Output-only smoothness**: the regularizers are plain
   mean\|∇rec\| + mean\|∇²rec\| — no reference to input or target
   derivatives (the input is degraded; its edges are not a reference).
   Fidelity to the pristine target counterbalances over-smoothing.
3. **Color**: 3-channel (YUV 4:4:4 at luma resolution) input processed
   jointly; the head outputs C×64 gains per block — a (W/8)×(H/8)×C
   effective-K field, exactly the original specification.

Loss (verbatim):

```
L = Σ_blocks ‖IDCT(g⊙X) − pristine_block‖²   (fidelity, per-block SSE)
  + λ · Σ g_AC                                (K/rate regularizer, λ = 2.5e-3)
  + w₁ · mean|∇rec| + w₂ · mean|∇²rec|        (output smoothness)
```

**The saturation-collapse lesson** (hard-won): with gains parameterized
gmax·σ(logits) and zero-initialized head, training starts at g = gmax/2 = 2;
the rate + smoothness pressure slams logits deep negative in one epoch,
σ′ ≈ 0, and the network freezes at a degenerate DC-only solution (K stuck
at 1.3, identical output for every input). Fix: initialize the head bias
so gains start ≈ 0.7 — inside the sigmoid's responsive region, just below
1. Collapse is visible in logs as constant K/g_mean across epochs; the
trainer logs `g_mean`, `K_g`, and `frac(g>1)` precisely to catch it.

### Generation 4 — the affine filter: X̂ = g·X + t

The final form, proposed by the project owner: each coefficient gets a
multiplier AND an additive term. In plain words: **g recycles what
survived, t invents what died.** A gain cannot revive a coefficient that
arrived as zero (g·0 = 0); only an additive term, predicted from the
surrounding content, can. This completes the filter to the textbook
affine form of the classical estimator (the Lee filter is exactly
a·input + (1−a)·local-estimate; t is that second half with a CNN
supplying the estimate).

Four guardrails keep the invented content honest:

1. **L1 price on t** (μ·Σ|t|, μ = 0.02): reusing a coefficient costs λ
   per unit gain, inventing one costs μ per unit magnitude — the filter
   invents only where the frame truly cannot supply the information.
2. **L1, not L2**, so t snaps to exactly zero where unneeded — clean
   frames keep a hard-zero baseline.
3. **t-head initialized at zero**: training starts as the proven
   gain-only filter and must earn its synthesis.
4. **Bounded and DC-excluded**: |t| ≤ 1 via tanh; no brightness synthesis.

**The context ablation — the project's most instructive experiment.**
The baseline CNN sees only ~40×40 px (±2 blocks). Theory predicted: g
("how much survived here?") is a local question; t ("what should be
here?") needs wide context. Both halves were measured:

- *Gen-3 negative control*: widening the view (dilated convolutions to
  ~260 px; a U-Net zoom-out/zoom-in branch to whole-crop view) did NOT
  improve the gain-only filter — the small baseline won every restoration
  row. **g is local. Confirmed.**
- *Gen-4 real test*: with the narrow view the t-head **died** (taxed to
  the noise floor, unable to predict anything useful from ±2 blocks).
  With the U-Net view, t came alive and its upper-percentile statistic
  `t_abs_p90` became the top-ranked severity feature (SRCC +1.00).
  **t needs context. Confirmed.**

**The damage map.** t doubles as a per-block map of "how much the model
had to invent" — near zero on clean frames, lighting up where damage
lives. Using this synthesis-effort map as a no-reference quality signal
appears to be novel (no match found in the free-energy, restoration-IQA,
or learned-shrinkage literatures).

### Robust training (gen4_robust — the production checkpoint)

Identical model, loss and guardrails, retrained on a 10× broader pristine
corpus: ~90 BVI-DVC source contents (streamed uint8 crops; 11,200 pairs =
8,000 simulator + 3,200 real H.264). Validated on BVI-CC1 content the
model never saw:

- **13 of 19 features severity-monotone — the best of all six
  checkpoints** — with the entire damage-map channel (t_tile_max,
  t_abs_p90, t_abs_mean) at SRCC +1.00. Corpus diversity matured the t
  signal: on 9 scenes only its upper percentiles tracked severity; on 90
  scenes even its plain mean does.
- Restoration PSNR ~0.5–2 dB below the 9-scene model on a CC1 test frame
  — but the in-domain models carried a home-field advantage (trained on
  the test content family). The robust model is measured honestly
  out-of-domain and still holds the severe-damage wins. Less in-domain
  polish, more generalization: the right trade for a probe.

## 7. Training data

Pristine source: BVI-CC1 HD masters (9 scenes, 1080p60, 10-bit). All
training crops are 8-aligned (DCT blocks coincide with the natural grid).

**Simulated artifact suite** (`artifacts.py`) — 9 types × 5 severities,
deterministic: compression (JPEG-like DCT quantization), blur, noise,
banding (posterization), **ringing** (brick-wall Fourier low-pass → Gibbs
oscillations; test-side only), packet loss with interpolation concealment,
packet loss with copy-from-previous concealment (ghost/shear), unconcealed
block fill, stale regions (concealment that "succeeds": locally natural,
temporally wrong). Training uses six of these (`TRAIN_ARTIFACTS`); pl_copy
and stale are **held out** to test generalization to unseen artifact types.

**Real H.264 corruption pipeline** (`bitstream.py`):
x264 Annex-B encode with **in-loop deblocking disabled** (`no-deblock=1`;
H.264 has no SAO), 4 slices/frame (localized damage), GOP 30 (real error
propagation) → **byte-flips inside non-IDR slice NALs** (SPS/PPS/IDR intact
so frame count and alignment are preserved) → error-resilient decode with
`-err_detect ignore_err -ec 1` (concealment **without** its deblock stage).
No deblocking anywhere in the training chain, per the design constraint
that the Wiener filter stands alone with no deblock/SAO admixture.
Training mixes 6,000 simulator pairs with 3,000 aligned real-H.264 pairs
(per master: one clean-compression stream CRF 22–44 and one corrupted
stream, 15–40% of slices damaged).

For **stress testing only**, `encode_h264(deblock=True)` produces normal
deblocked streams — blockiness the codec has already smoothed, the harder
detection case.

**Three-codec pair generator** (`pairgen.py` + `generate_pairs.py`):
the scaled-up successor to the H.264-only pipeline. Two families, stored
as reusable on-disk shards (256×256 uint8, one dataset serving every
generation via sub-cropping):

- *compression*: clean clips through real encoders with randomized
  settings — H.264 (CRF 20–50, three profiles, deblock on/off, slices,
  GOP, B-frames), HEVC (CRF 22–45, SAO on/off), MPEG-2 (qscale 4–31,
  including intra-only — the digital-tape profile);
- *loss*: the compressed stream additionally gets codec-aware packet
  loss (byte flips inside slice payloads, with the correct NAL parsing
  per codec; headers and random-access pictures kept intact so streams
  stay decodable and frame-aligned), decoded with error resilience.

Generated: 7,200 pairs, perfectly balanced across the six codec×family
cells, every encoder configuration recorded in the manifest.

**External dataset integration** (`binpairs.py` + `extract_binpairs.py`):
a resumable extractor for GT-plus-degraded image collections (built for
the BinResults share: 44,209 folders, each GT.png + 10 bitrate-ladder
degradations with VIF labels). Extracted 40,820 patch pairs with
per-patch bin and VIF labels; 12 folders reserved as held-out judges
(`validate_binpairs.py` scores any checkpoint on their own VIF ladders).

**Resource lesson (machine-hang fix)**: the original pair loader
concatenated shard arrays, briefly needing ~2× the dataset in RAM
(≈28 GB on the 32 GB machine) — Windows paged to disk and the machine
appeared hung. The loader is now memory-flat (preallocate + fill; peak =
dataset + one shard), in-RAM pairs are capped at ~24k at 256px, and long
trainings run at BelowNormal process priority. Watch the loading step's
*peak* memory, not the dataset's size.

## 8. Experimental results

### 8.1 Severity-ladder validation (JPEG ladder, real + synthetic content)

Every generation passed the gate (≥ 3 features with \|SRCC\| ≥ 0.9 per
content). Final color Wiener model: 6–12 monotone features per content;
K_g is content- and severity-responsive (1.65 → 1.03 on real content);
residual energy rises at severe degradation (artifact-suppression
polarity).

### 8.2 Artifact-response matrix (all probes × all artifacts)

All 8 artifact classes have strong monotone responders on real content
(75–150 features each); every probe family, including the learned filter,
responds to every class. Wiener-standalone: 3–14 monotone features per
artifact, **including the held-out types it never saw in training**.

### 8.3 Restoration accuracy (luma PSNR vs pristine, real content)

The λ-imposed fidelity ceiling sits at ≈ 34–35 dB. Where the input is worse
than the ceiling the filter genuinely restores (**+7.2 dB on severe
noise**, +1.9 dB severe banding); where the input is better (mild
compression at 42 dB) the filter's own sparsification costs more than the
artifact. Blur/concealment sit near break-even — per-coefficient gains
cannot recreate absent information (the known capacity limit). λ is the
fidelity-vs-sparsity knob. Restoration PSNR is *not* the objective in the
probe role; residual informativeness is.

### 8.4 Head-to-head against VMAF (single-probe ridge fusion, LOO by content)

x264 ladder, real per-frame VMAF labels (range 0–99):

| Filter | SRCC mean | RMSE mean | Note |
|---|---|---|---|
| **Learned Wiener (lwn)** | 0.929 | **7.41** | out-of-domain here (trained on BVI) |
| Blind adaptive DCT (dct) | 0.938 | 24.36 | ranks well, calibrates poorly |
| Fixed-K DCT, K=6 (fdct) | 0.890 | 10.06 | collapses to 0.78 on grain content |

Conclusions: (a) the learned Wiener **beats the fixed-K baseline on both
metrics**, and the fixed-K grain failure is the textbook demonstration of
why content-adaptivity is the mechanism, not a nicety; (b) vs the classical
adaptive filter, ranking is tied but the learned filter's **VMAF
calibration is 3× better** — the decisive property when absolute score
thresholds drive decisions.

### 8.5 Cross-generation comparison (identical degraded images)

All six checkpoints (gen1 kdct, gen2 kmap, gen2 restore-trained, gen3
wiener, gen4 affine, gen4 robust) on the same degraded real frame,
7 artifacts × 2 severities:

- **Division of labor, no dominant generation.** Selection filters
  (gens 1–2) are gentle preservers — best fidelity on mild damage
  (38+ dB where gain filters give ~34–35). Gain filters (gens 3–4) are
  aggressive restorers — best on severe damage (+6–7 dB on heavy noise,
  best severe banding). Crossover around input ≈ 32–35 dB.
- **Gen-4's t term helps in 10 of 14 rows over gen-3** (+0.3–1.1 dB);
  loses only on blur (the known information-theoretic limit).
- **Feature quality**: gen1 6, gen2 9, gen2-restore 10, gen3 7, gen4 8,
  **gen4-robust 13** monotone ladder features — and gen4 is the only
  family where a parameter-field channel (the damage map) outranks all
  residual statistics.
- Implication: the generations are complementary instruments; the
  strongest probe configuration pairs gen4-robust (primary) with
  gen2-restore (best selection-family probe) — the companion-residuals
  ensemble.

### 8.6 The champion's three defenses (final model selection)

After gen4_robust was crowned, three challengers trained on
better-looking data tried to take the title. All three lost:

1. **gen4_pairs** (7,200 real three-codec pairs incl. packet loss): the
   best *restorer* of any model — best severe-noise (34.1 dB) and best
   gain-family compression repair — but 12 monotone features to the
   champion's 13, plus a diet-induced quirk (rewrites unfixable regions),
   and its evaluation carried partial home-field advantage.
2. **BinResults μ sweep** (in-domain, 16k pairs each, μ ∈ {.01,.02,.04}):
   judged on 11 held-out folders' own VIF ladders — champion 17/19
   strong features vs 16 (μ=.04), 13 (μ=.01), 11 (μ=.02). The champion
   had never seen a BinResults image and ran in a different color space.
3. **Full-budget in-domain run** (24k pairs, 36 epochs): got *worse*
   (13/19) — more in-domain training specialized the model on that
   dataset's damage at the expense of judging naturalness.

Conclusion, now measured across content type, damage source and color
space: **broad pristine-content diversity beats damage specialization
for probe quality** — the filter's power is knowing what natural looks
like, exactly as the framework in Section 2 predicts. `wiener4_dvc.pt`
is the confirmed production probe. Side finding: training-log intuition
misjudged the sweep (μ=.04 looked over-taxed in the logs but won the
sweep on held-out data) — only held-out ladders decide.

### 8.7 Stress campaign (15 severity ladders, Wiener standalone)

All 15 conditions pass (≥ 1 feature \|SRCC\| ≥ 0.9):

- **H.264 with deblocking ON** (CRF 26→51): 9 strong features — detected
  as well as raw blockiness (8). Detection rides on detail-loss and
  reconstruction-mismatch channels (`new_edge1` +1.00), not on visible
  block edges — the within-block-DCT limitation is immaterial in practice.
- Real packet loss (5→50% corrupted slices, concealed decode): passes with
  the thinnest margin (2 strong; `k_pred_mean` +0.94 — the K-field itself
  reacts to concealment damage). A temporal companion residual is the
  structural fix if more margin is ever needed.
- Severe JPEG (q→3), severe blur, severe noise (σ→40), isolated ringing,
  and the full simulated suite: 5–16 strong features each.

## 9. Lessons learned

1. **Content-grouped validation splits are non-negotiable** — the single
   largest cause of non-reproducible published results.
2. **The features are the model**: fusion (ridge vs GBT vs MLP) changes
   little; feature information sets the ceiling. Measure the ceiling
   (oracle-fusion test) instead of guessing.
3. **Trees cannot extrapolate** — GBT fails catastrophically outside its
   training hull; linear models degrade gracefully. Ship the simple model
   on ties.
4. **The filter must not adapt to the distortion**: pristine-only or
   paired training; never degraded self-reconstruction.
5. **Signed ΔK encodes artifact type, \|ΔK\|/residual encode severity** —
   one probe, two orthogonal signals.
6. **Fixed thresholds interact pathologically with quantization**
   (coefficients round up past τ) — the K/rank formulation removes this.
7. **Selection (binary keep-K) cannot restore; gains ≤ 1 can only
   attenuate; gains > 1 permit restoration** — but MSE-optimal gains rarely
   exceed 1, so amplification capability matters less than expected.
8. **Sigmoid-parameterized gains collapse under rate pressure** unless the
   head starts in the responsive region — log g_mean/K per epoch to catch
   frozen training within one epoch.
9. **First-order TV alone promotes staircase/banding** — always pair with
   a second-order term.
10. **Deblocked (loop-filtered) blockiness is still detectable** through
    detail-loss channels — detection does not require visible block edges.
11. **Order statistics dominate runtime at 1080p** — decimated views for
    medians/percentiles (filters stay exact) and batched matmul DCT
    (9× over einsum) made full-frame extraction practical.
12. **Gain estimation is local; synthesis needs context** — measured with
    a negative control (widening the view did nothing for gen-3) before
    trusting the positive result (the gen-4 t-head dies at ±2 blocks and
    becomes the top feature with a U-Net view).
13. **Corpus diversity matures learned signals** — the damage map went
    from percentile-only usefulness (9 training scenes) to fully
    monotone in its plain mean (90 scenes), measured on content the
    model never saw.
14. **Beware the home-field advantage in evaluation** — models tested on
    the content family they trained on report flattered restoration
    numbers; only out-of-training-content results count as robustness
    evidence.
15. **Preservers and restorers are complementary, not competing** —
    selection filters win on mild damage, gain filters on severe;
    residuals from both families carry different information about the
    same frame.
16. **Knowing "natural" beats knowing the damage** — in-domain training
    on the target dataset's own degradations lost to the broadly-educated
    champion three times, at two budgets; more in-domain epochs made it
    worse. Invest in pristine diversity, not damage specialization.
17. **Watch peak memory, not dataset size** — a concatenate-based loader
    briefly doubles memory and can page a 32 GB machine into a hang;
    preallocate-and-fill keeps loading memory-flat. Cap in-RAM pairs and
    run long jobs at reduced process priority.
18. **Only held-out ladders decide** — training-log intuition misread the
    μ sweep (the "over-taxed" run won); model selection must come from
    held-out, same-domain severity ladders, never from loss curves.

## 10. Repository map

```
adaptive_filters/
  probes/           six classical probes + fixed-K baseline + learned wrappers
  learned/          kdct (gen1), kmap (gen2), wiener (gen3, CURRENT),
                    adaptive_dct (filter-level API), patches (training data)
  features/stats.py GGD, MAD, Immerkær, lag-1, Spearman
  artifacts.py      9-artifact simulation suite
  bitstream.py      real H.264 encode/corrupt/decode (no-deblock, -ec 1)
  io.py / vmaf.py / dataset.py / pipeline.py / naturalness.py / fusion.py
models/
  wiener4_dvc.pt    PRODUCTION probe: gen-4 affine + U-Net, ~90 BVI-DVC
                    contents -- confirmed champion after three challenges
  wiener4_pairs.pt  gen-4 on real 3-codec pairs (best restorer)
  wiener4_bin_*.pt  BinResults in-domain sweep + full-budget run
                    (recorded evidence that specialization lost)
  wiener4_c.pt / wiener4_a.pt   gen-4 on 9 scenes / keyhole negative control
  wiener.pt         gen-3 gain-only baseline (fastest)
  wiener_b/c.pt     gen-3 context-ablation variants
  kmap.pt / kmap_restore.pt / kdct.pt   generations 2 and 1
train_wiener.py     trainer (+ validation); --arch a|b|c, --affine,
                    --pristine-dir for large corpora
train_kmap.py / train_kdct.py   earlier-generation trainers
inspect_filter.py   visual inspection (input/filtered/residual/K-map PNGs)
validate_artifacts.py  probe-bank artifact matrix (regression gate)
stress_test.py      15-ladder Wiener-standalone stress campaign
compare_vmaf.py     single-probe VMAF head-to-head (lwn vs dct vs fdct)
compare_context.py  receptive-field ablation (baseline/dilated/U-Net)
compare_generations.py  all checkpoints on identical degraded images
generate_pairs.py   real 3-codec + packet-loss pair dataset generator
extract_binpairs.py / validate_binpairs.py  external GT-dataset extractor
                    and held-out VIF-ladder model selection
demo.py             end-to-end synthetic smoke test
build_dataset.py / train_fusion.py / score_video.py   VQA pipeline CLIs
```

Interpreter: `E:\Python_Workspace\envs\PythonGPU\Scripts\python.exe`.
Standing regression gates: `demo.py`, `validate_artifacts.py`,
`stress_test.py` — all print PASSED/FAILED and exit accordingly.

Core usage:

```python
from adaptive_filters.learned.adaptive_dct import AdaptiveWienerFilter
r = AdaptiveWienerFilter("models/wiener.pt").apply(frame)  # (H,W) or (H,W,3)
r.filtered   # IDCT(g ⊙ X)
r.residual   # the artifact-revealing signal
r.k_pred     # effective-K field (C, H/8, W/8)
```

## Appendix A: build-it-yourself roadmap

A 7-stage path for reimplementing the generation-3 filter from scratch
(for learning; the reference implementation is the answer key):

1. **DCT matrix** — orthonormal 8×8 DCT-II. Checks: D·Dᵀ = I;
   constant block → DC = 8v, AC = 0.
2. **Frame ↔ blocks** — (B,C,H,W) ↔ (B,C,N,8,8), row-major block order.
   Checks: exact round trip; single-nonzero-block ordering probe (catches
   the universal first transpose bug).
3. **Batched DCT** — D @ blocks @ Dᵀ via broadcasting matmul.
   Checks: Parseval; IDCT∘DCT = identity.
4. **CNN** — three ÷2 stages to the block grid; head → C·64 logits;
   g = gmax·σ; DC pinned; head bias so g starts ≈ 0.7 (reason through the
   saturation collapse before coding).
5. **Gains × coefficients** — the ordering trap: wrong permutes still
   *run*, silently multiplying wrong gains onto wrong blocks. Check with a
   one-block gain-pattern probe.
6. **Loss** — derive the keep-iff-X² > λ duality from Parseval (one line);
   decide per-block-SSE vs per-pixel-MSE deliberately (it sets λ's scale).
7. **Training loop** — reuse the data machinery; log g_mean, K_g,
   frac(g>1) per epoch; validate by pointing the existing probe harness at
   your checkpoint.

## Appendix B: references

### B.1 Theory of adaptive filtering (the unifying view)

- P. Milanfar, "A Tour of Modern Image Filtering," *IEEE Signal Processing
  Magazine*, 2013. **Read first** — unifies bilateral/NLM/LARK/kernel
  regression as data-adaptive filters.
- J.-S. Lee, "Digital image enhancement and noise filtering by use of local
  statistics," *IEEE TPAMI*, 1980 (the local Wiener / Lee filter).
- D. Kuan et al., "Adaptive noise smoothing filter…," *IEEE TPAMI*, 1985.
- A. Hillery, R. Chin, "Iterative Wiener filters for image restoration,"
  *IEEE TSP*, 1991 (the pilot-estimate feedback loop).
- C. Tomasi, R. Manduchi, "Bilateral filtering," *ICCV*, 1998.
- P. Perona, J. Malik, "Scale-space and edge detection using anisotropic
  diffusion," *IEEE TPAMI*, 1990.
- L. Rudin, S. Osher, E. Fatemi, "Nonlinear total variation based noise
  removal," *Physica D*, 1992 (TV; why 2nd-order terms fight staircase).

### B.2 DCT-domain and transform-domain filtering

- A. Foi, V. Katkovnik, K. Egiazarian, "Pointwise Shape-Adaptive DCT for
  High-Quality Denoising and Deblocking," *IEEE TIP*, 2007.
- D. Donoho, I. Johnstone, "Ideal spatial adaptation by wavelet shrinkage,"
  *Biometrika*, 1994 (oracle keep-iff-\|X\|>σ; threshold theory).
- T. Blu, F. Luisier, "The SURE-LET approach to image denoising,"
  *IEEE TIP*, 2007; S. Ramani, T. Blu, M. Unser, "Monte-Carlo SURE,"
  *IEEE TIP*, 2008 (blind risk-driven parameter selection).
- K. Dabov et al., "Image denoising by sparse 3-D transform-domain
  collaborative filtering" (BM3D), *IEEE TIP*, 2007 (stage 2 = empirical
  Wiener in transform domain — the classical ancestor of our gain map).
- M. Elad, M. Aharon, "Image denoising via sparse and redundant
  representations over learned dictionaries," *IEEE TIP*, 2006 (K-SVD;
  error-constrained sparsity = self-selecting K).
- A. B. Watson, "DCTune: perceptual optimization of quantization matrices,"
  *SID*, 1993 (content-dependent coefficient budgets).

### B.3 Codec in-loop filters (the industrial instances of the framework)

- C.-Y. Tsai et al., "Adaptive Loop Filtering for Video Coding,"
  *IEEE J-STSP*, 2013; M. Karczewicz et al., "VVC In-Loop Filters,"
  *IEEE TCSVT*, 2021 (ALF: classify by gradients → per-class Wiener via
  normal equations — the cleanest real-world version of our pipeline).
- C.-M. Fu et al., "Sample Adaptive Offset in the HEVC Standard,"
  *IEEE TCSVT*, 2012.
- P. List et al., "Adaptive Deblocking Filter," *IEEE TCSVT*, 2003.
- S. Midtskogen, J.-M. Valin, "The AV1 Constrained Directional Enhancement
  Filter," *ICASSP*, 2018; D. Mukherjee et al., "A switchable
  loop-restoration with side-information framework…," *ICIP*, 2017.
- Reference code worth reading: VTM `EncAdaptiveLoopFilter`, libaom
  `cdef.c` / `restoration.c` / `pickrst.c`.

### B.4 No-reference quality assessment

- A. Mittal, A. K. Moorthy, A. C. Bovik, "No-Reference Image Quality
  Assessment in the Spatial Domain" (BRISQUE), *IEEE TIP*, 2012 (the
  GGD/MSCN feature machinery).
- A. Mittal, R. Soundararajan, A. C. Bovik, "Making a 'Completely Blind'
  Image Quality Analyzer" (NIQE), *IEEE SPL*, 2013 (our layer B).
- Free-energy school — the closest theoretical umbrella for
  "filter-residual = quality": G. Zhai et al., "A Psychovisual Quality
  Metric in Free-Energy Principle," *IEEE TIP*, 2012; K. Gu et al.,
  "Using Free Energy Principle for Blind Image Quality Assessment" (NFERM),
  *IEEE TMM*, 2015; J. Wu et al., "Perceptual Quality Metric With Internal
  Generative Mechanism," *IEEE TIP*, 2013; overview: *Digital Signal
  Processing*, 2019 — https://doi.org/10.1016/j.dsp.2019.02.017
- Pseudo-reference (the dual: add distortion instead of removing it):
  X. Min et al., "Blind Image Quality Estimation via Distortion
  Aggravation," *IEEE Trans. Broadcasting*, 2018 —
  https://ieeexplore.ieee.org/document/8326697/ ; F. Crete et al., "The
  blur effect…" *SPIE HVEI*, 2007 (blur-of-blur).
- Restoration-network NR-IQA (learned free-energy): VCRNet, *IEEE TIP*,
  2022 — https://dl.acm.org/doi/10.1109/TIP.2022.3144892 ; "Joint
  Distortion Restoration and Quality Feature Learning," *ACM TOMM*, 2024 —
  https://dl.acm.org/doi/10.1145/3649899
- VMAF distillation: "No-Reference VMAF: A Deep Neural Network-Based
  Approach to Blind Video Quality Assessment," *IEEE Trans. Broadcasting*,
  2024 — https://ieeexplore.ieee.org/document/10564175/ (our layer A with
  a CNN; the baseline to differentiate against).
- FR-models-as-annotators: DUBMA, *IJCAI*, 2025 —
  https://www.ijcai.org/proceedings/2025/0227.pdf ; rank supervision:
  K. Ma et al., dipIQ, *IEEE TIP*, 2017; X. Liu et al., RankIQA, *ICCV*,
  2017.
- Artifact-specific deployed metrics: Z. Wang, H. Sheikh, A. Bovik,
  "No-reference perceptual quality assessment of JPEG compressed images,"
  *ICIP*, 2002 (blockiness); H. Liu et al., ringing, *IEEE TCSVT*, 2010;
  Z. Tu et al., BBAND, *ICASSP*, 2020; P. Tandon et al., CAMBI (Netflix),
  *PCS*, 2021 — the best template for an industrial NR artifact metric.
- Video: M. Saad et al., V-BLIINDS, *IEEE TIP*, 2014 (DCT statistics of
  frame differences); J. Korhonen, TLVQM, *IEEE TIP*, 2019
  (compute-conscious feature design); Z. Tu et al., RAPIQUE, *IEEE OJSP*,
  2021.
- Standards: ITU-T P.1203 / P.1204.3 (deployed bitstream/hybrid models —
  fuse bitstream features with pixel features when available).
- VMAF itself: Netflix, https://github.com/Netflix/vmaf (model, code, and
  the fusion pattern — few interpretable features + small regressor — that
  our design imitates).

### B.5 Learned transform-domain filtering (nearest neighbors of the final model)

- Z. Wang et al., "D3: Deep Dual-Domain Based Fast Restoration of
  JPEG-Compressed Images," *CVPR*, 2016 — first DCT-domain priors in a
  deep JPEG-artifact remover.
- J. Guo, H. Chao, "Building Dual-Domain Representations for Compression
  Artifacts Reduction" (DDCN), *ECCV*, 2016 —
  https://link.springer.com/chapter/10.1007/978-3-319-46448-0_38
- M. Ehrlich et al., "Quantization Guided JPEG Artifact Correction"
  (QGAC), *ECCV*, 2020 (uses quant tables to guide DCT-domain correction).
- DCTNet, *Signal, Image and Video Processing*, 2023 —
  https://link.springer.com/article/10.1007/s11760-023-02593-0 (deep
  shrinkage via DCT filterbanks — closest published relative of the
  gain-map model, for denoising).
- S. Herbreteau, C. Kervrann, "DCT2net: an interpretable shallow CNN for
  image denoising," 2021 — https://arxiv.org/abs/2107.14803
- K. Gregor, Y. LeCun, "Learning Fast Approximations of Sparse Coding"
  (LISTA), *ICML*, 2010 (learned thresholds in transform domain).
- M. Scetbon, M. Elad, P. Milanfar, "Deep K-SVD Denoising," *IEEE TIP*,
  2021.
- J. Dong et al., "Deep Wiener Deconvolution," *NeurIPS*, 2020
  (feature-space Wiener — the "deep Wiener" idea for deblurring).
- K. Xu et al., "Learning in the Frequency Domain," *CVPR*, 2020
  (per-DCT-channel gating, for recognition — same mechanism, different
  purpose).
- Y. Romano, J. Isidoro, P. Milanfar, "RAISR," *IEEE TCI*, 2017
  (hash content features → per-bucket learned filters; the offline-trained
  cousin of ALF and of our learned θ estimator).
- **Our differentiators against all of the above**: the residual (not the
  restored image) is the product; the gain/K/t fields are read back as
  feature channels — in particular the **synthesis-effort (damage) map**,
  which appears unpublished as a no-reference quality signal; training
  includes real corrupted H.264 with concealment; and the target is a
  quality index, not restoration quality.

### B.6 Supporting results

- L. Grinsztajn et al., "Why do tree-based models still outperform deep
  learning on tabular data?" *NeurIPS*, 2022 (fusion-stage choice).
- T. Chen, C. Guestrin, "XGBoost," *KDD*, 2016 (incl. monotone constraints).
- J. Immerkær, "Fast Noise Variance Estimation," *CVIU*, 1996 (the blind
  noise estimator used by the guided/directional probes).
- FAST-VQA (*ECCV* 2022), DOVER (*ICCV* 2023), MUSIQ (*ICCV* 2021),
  MANIQA (*CVPRW* 2022) — the end-to-end deep VQA state of the art;
  assessed and deferred (MOS dependence, compute, no diagnostics).

### B.7 Tutorials and practical resources

- PyTorch tutorials: https://pytorch.org/tutorials/ (autograd, `nn.Module`,
  and the custom-training-loop pattern used by all three trainers).
- Netflix TechBlog, "Toward a Practical Perceptual Video Quality Metric"
  (the VMAF design rationale) and the VMAF GitHub wiki.
- FFmpeg documentation: https://ffmpeg.org/documentation.html (rawvideo
  piping, libvmaf filter, error-resilience flags `-err_detect`, `-ec`).
- x264 parameter reference (deblocking, slices, GOP control):
  https://www.videolan.org/developers/x264.html
- M. Elad, *Sparse and Redundant Representations*, Springer, 2010 (book —
  the sparsity/K-selection theory in depth).
- Z. Wang, A. C. Bovik, *Modern Image Quality Assessment*, Morgan &
  Claypool, 2006 (book — IQA foundations).
- BVI datasets (University of Bristol): BVI-CC, BVI-DVC —
  https://fan-aaron-zhang.github.io/BVI-DVC/ (the pristine corpus used
  here).
