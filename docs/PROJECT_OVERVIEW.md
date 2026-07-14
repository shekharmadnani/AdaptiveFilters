# AdaptiveFilters — The Whole Project, Simply Explained

*A one-document overview for anyone joining the project. Deeper reading:
[THE_FOUR_FILTERS.md](THE_FOUR_FILTERS.md) (the filters in detail),
[CONVERSATION_LOG.md](CONVERSATION_LOG.md) (the full journey),
[PROJECT_REPORT.md](PROJECT_REPORT.md) (technical depth + references).*

---

## 1. What is this project trying to do?

**Measure the quality of a video without having the original to compare
against** (a "no-reference" quality score) — in a way that actually works
in industry: no human rating panels, cheap to run, explainable, and
robust across content types and damage types.

## 2. The core idea (if you remember one thing, remember this)

> **Train a filter to be an expert on what CLEAN video looks like.
> Show it a possibly damaged frame. Whatever the expert cannot explain
> IS the damage.**

The filter tries to rebuild each frame using its knowledge of clean
content. The difference between the frame and the rebuilt version — the
**residual** — is small and boring on clean video, and lights up where
damage lives. Statistics of that residual (how big, how spiky, where)
become the measurements a quality score is built from.

One rule governs everything: **the filter must adapt to the content,
never to the damage.** A filter that learns to reproduce blockiness
"explains" the blockiness — and goes blind exactly when the video is
worst. So training always aims at clean content.

## 3. How the filter works (the mechanics)

Every 8×8 block of pixels can be rewritten as a recipe of 64 ingredients
(DCT coefficients): one for average brightness (DC), 63 for patterns of
increasing fineness (AC). Clean image blocks need only a few strong
ingredients — that sparsity is why JPEG and every video codec use the
DCT.

Our filter, in its final form, gives every ingredient two controls:

```
output ingredient  =  g × (input ingredient)  +  t
                       ↑ recycle                ↑ invent
```

- **g (gain, 0 to 4)** — a dimmer switch: how much of what *arrived* to
  keep (or boost). Decided per ingredient, per block.
- **t (synthesis, −1 to +1)** — what to *write in* that arrived as zero
  (deleted by compression). Predicted from the surrounding context.
- A small neural network (under a million parameters, built from
  scratch) looks at the frame and outputs g and t for every block. It
  has a **U-Net** shape — zoom out to understand the region, zoom back
  in to act precisely — so t gets the wide view it needs.

Training balances a budget: rebuild the clean target faithfully (RMSE),
pay a price λ for every unit of gain (economy — keep only ingredients
that earn their keep), pay a price μ for every unit of invention
(honesty — invent only where the frame truly cannot supply the
information), and keep the output smooth (first + second derivatives).

**The bonus discovery:** t doubles as a **damage map** — a per-block
chart of "how much did I have to invent, and where." Near zero on clean
frames, lighting up in damaged regions, perfectly ordered with damage
severity. Using this map as a quality signal appears to be novel.

## 4. The journey in five steps

1. **Classical probes first.** Six hand-made adaptive filters (DCT
   keep-K, local Wiener, deblocking-grid, SAO-style, directional,
   temporal), each an instance of the same template. Proven working on
   day one: perfect ordering of a compression ladder.
2. **Generation 1–2 (learned selection).** A CNN learns how many
   ingredients each block needs (first per-patch, then as a full-frame
   K-map with a differentiable "keep the top K"). Lesson: selection can
   only *delete* — you cannot restore missing detail by dropping
   coefficients.
3. **Generation 3 (gains).** Dimmer switches instead of on/off — the
   learned DCT-domain Wiener filter, in color, trained on damaged→clean
   pairs including *real* corrupted H.264 streams. Beat the fixed-K
   baseline soundly (the fixed filter collapsed on grainy content —
   proof that content-adaptivity is the mechanism, not a luxury).
4. **Generation 4 (gains + invention).** The affine form `g·X + t` — the
   textbook estimator shape, with four honesty guardrails on t. The
   **context experiment** proved both halves of the theory: gains are a
   *local* question (wider view didn't help), synthesis is a *context*
   question (with a keyhole view t died; with the U-Net view it became
   the best feature in the model).
5. **The championship.** The robust model — same design, educated on
   ~90 different clean videos — defended its title **four times**:
   against a model trained on real three-codec damage, against models
   trained on the owner's own 44,000-image dataset (twice, at two
   budgets — more in-domain training made them *worse*), and against a
   transformer with masked pretraining. Every challenger lost.

## 5. The key results, at a glance

| Test | Result |
|---|---|
| Quality prediction vs real VMAF (unseen content) | rank correlation 0.93; score error 7.4 VMAF points — 3× better calibrated than the classical adaptive DCT filter |
| Fixed-K (non-adaptive) baseline | collapses on grain content (0.78) — adaptivity is the mechanism |
| 15-ladder stress campaign (severe compression, deblocked H.264, real packet loss, ringing, …) | all 15 pass; deblocked blockiness detected as well as raw |
| Champion on its own held-out judges (owner's dataset, VIF ladders) | 17 of 19 measurements perfectly ordered — best of all 12+ checkpoints, on data it never saw, in a color space it wasn't trained in |
| Damage map (t-field) | perfectly severity-ordered on two independent data worlds |

## 6. The laws this project learned (the short list)

1. **The features are the model** — the final regression stage barely
   matters; the information in the measurements sets the ceiling.
2. **Adapt to content, never to damage** — train toward clean targets.
3. **Knowing "natural" beats knowing the damage** — breadth of clean
   education beat damage specialization and architectural
   sophistication, four times.
4. **Gains are local; synthesis needs context** — measured with a
   negative control before trusting the positive result.
5. **Only held-out data decides** — content-grouped splits always;
   training-log intuition was wrong twice.
6. **Death by saturation, not by zero** — a gate driven into its flat
   tail stops learning (the gain collapse); a gate started at zero
   *output* has maximal gradients (the t init). Log the averages every
   epoch; a frozen number is a dead network.
7. **Watch peak memory, not data size** — a loader that briefly doubles
   memory can page a 32 GB machine into a hang.

## 7. What exists in the repository

- **The production filter**: `models/wiener4_dvc.pt` (four title
  defenses) — plus every previous generation and challenger kept as the
  ablation record.
- **Six evaluation harnesses**, all print PASS/FAIL: end-to-end demo,
  artifact-response matrix, 15-ladder stress test, VMAF head-to-head,
  context ablation, all-generations comparison.
- **Three data generators**: an artifact simulator (9 types × 5
  severities), a real three-codec + packet-loss pair generator
  (H.264/HEVC/MPEG-2, every encoder setting recorded), and a resumable
  extractor for external GT-image collections.
- **The VQA pipeline** (ready for the next phase): feature extraction →
  ridge fusion trained against VMAF → naturalness anchor → scoring CLI.
- **Three documentation tiers** (see the links at the top).

## 8. What happens next

- **Generation 5 (in progress): the CNN+MLP hybrid.** Today the g/t
  decisions see only the *context* (the CNN's view); the new design adds
  a small per-block MLP that looks directly at the block's own 64
  coefficients — "what actually arrived, and does it look natural or
  not?" That judgment stays *internal*: the MLP's features feed only the
  estimation of g and t — no extra outputs, no extra supervision. Prior
  (context) + evidence (block) fusion is exactly the structure of
  classical Wiener estimation. It must beat 17/19 on the held-out judges
  to take production.
- **The quality-index stage** (the original destination): fuse the
  champion's residuals, the damage map, the health map and the companion
  classical probes into the final no-reference score — with 40k
  VIF-labeled pairs already extracted for exactly this purpose.
