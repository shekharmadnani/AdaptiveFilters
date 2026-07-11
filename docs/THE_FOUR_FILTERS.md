# The Four Filter Generations — A Plain-Language Guide

This document explains, in simple language, the four learned adaptive DCT
filters built in this project: what each one does, why it was designed
that way, what we learned from it, which experiments proved what, and
where each idea comes from in the literature.

The companion technical report is [PROJECT_REPORT.md](PROJECT_REPORT.md).
This guide sacrifices precision for clarity; the report does the opposite.

---

## Table of contents

1. [The big idea](#1-the-big-idea)
2. [Foundations everyone shares](#2-foundations)
3. [Generation 1 — the patch expert (kdct)](#3-generation-1)
4. [Generation 2 — the K-map (kmap)](#4-generation-2)
5. [Generation 3 — the dimmer switches (wiener)](#5-generation-3)
6. [Generation 4 — recycle and invent (affine)](#6-generation-4)
7. [All four side by side](#7-side-by-side)
8. [The experiments, explained simply](#8-the-experiments)
9. [Where the ideas come from (literature)](#9-literature)
10. [Glossary](#10-glossary)

---

## 1. The big idea

We want to judge the quality of a video **without seeing the original**
(no-reference quality assessment). The trick used throughout this project:

> **Train a filter to be an expert on what CLEAN video looks like.
> Show it a possibly-damaged frame. Whatever the expert cannot explain
> is, by definition, the damage.**

The filter tries to rebuild the frame using its knowledge of clean
content. The difference between the frame and the rebuilt version is
called the **residual**. On a clean frame the residual is small and
boring; on a damaged frame it lights up. Statistics of the residual —
how big, how spiky, where concentrated — become the measurements from
which a quality score is computed.

One rule governs everything (learned the hard way, enforced everywhere):
**the filter must adapt to the content, never to the damage.** A filter
that learns to reproduce blockiness "explains" the blockiness, the
residual goes quiet exactly when the video is worst, and the whole
instrument goes blind. That is why training always uses clean video
(alone, or as the target of a damaged input) and never asks the filter to
reproduce a damaged frame.

## 2. Foundations

These pieces are shared by all four generations.

**The DCT — a recipe for image blocks.** Every 8×8 block of pixels can be
rewritten as a recipe of 64 ingredients (DCT coefficients): one for the
average brightness (called DC) and 63 for patterns of increasing fineness
(called AC). Natural image blocks need only a few strong ingredients —
this "sparsity" is the whole reason JPEG and every video codec use the
DCT. Our filters all work by deciding, per block, **what to do with each
of the 64 ingredients**.

**K — the ingredient count.** The original idea of this project: the
number of AC ingredients a block genuinely needs (K) depends on its
content — a flat sky needs almost none, a crowd scene needs many. If a
machine can estimate the *natural* K from the surrounding content, then
comparing it against what the block *actually contains* reveals
compression damage.

**The CNN.** A small convolutional network (built from scratch, not
pretrained, under a million parameters) looks at the frame and outputs
its decisions on the 8×8 block grid — one decision set per block. It is
fully convolutional: trained on small crops, applied to whole frames.

**The price system (the loss function).** Training balances a budget:

- *Fidelity*: the rebuilt frame should match the clean target (RMSE).
- *Economy*: every ingredient kept costs a price λ — so the filter keeps
  only ingredients that earn their keep. There is a beautiful closed-form
  consequence: an ingredient is worth keeping exactly when its energy
  exceeds λ. This one-line rule is both the classical baseline and the
  standard our learned models must beat.
- *Tidiness*: the output should not contain invented edges — measured by
  the first and second derivatives of the output (using both matters:
  the first alone actually *encourages* staircase-like banding).

**Training data.** Clean masters from the BVI datasets (real 1080p
video), damaged copies made two ways: an artifact simulator (compression,
blur, noise, banding, ringing, packet loss with several concealment
styles, block loss, frozen regions — each at 5 severities), and **real
corrupted H.264 streams** (encoded with x264, bytes flipped inside the
bitstream, decoded with the decoder's error concealment — real blockiness,
real error propagation). Two artifact types are deliberately *excluded*
from training so we can test whether the filter generalizes to damage it
has never seen. It does.

**How we test everything.** We damage a real frame at increasing
severities (a "ladder") and ask: do the filter's measurements move
*steadily in one direction* as damage grows? Steadiness is measured by
rank correlation (SRCC; 1.0 = perfectly ordered). A measurement that
tracks severity monotonically is useful for a quality score; one that
jumps around is not. Additional harnesses: an artifact-response matrix
(every artifact × every measurement), restoration accuracy (PSNR against
the clean original), a stress campaign of 15 ladders including deblocked
H.264 and real packet loss, and a head-to-head against real VMAF scores.

---

## 3. Generation 1 — the patch expert (kdct)

**Idea.** Feed the CNN a 64×64 patch; it outputs, for each of the 64
blocks inside, a *threshold*: keep ingredients louder than the threshold,
drop the rest. A second output guesses the natural K directly. Trained on
clean patches only — the network learns "what K does content that looks
like this naturally need?"

**What we learned.**

- The residual measurements were perfectly ordered with severity
  (SRCC 1.0) — proof the whole concept works.
- The gap ΔK between "K the content should need" and "K actually present"
  turned out to encode the *type* of damage, not the amount: blur pulls
  it negative, blockiness pushes it positive (+3.3 at the worst JPEG
  setting). A V-shape — useless alone as a severity meter, valuable as a
  damage classifier.
- A subtle pathology: with a *fixed* threshold, the measured K first
  **rises** with compression before collapsing, because quantization
  rounds surviving ingredients *upward* past the threshold. This bug in
  nature, not in code, motivated generation 2.

**Status.** Superseded; kept as the historical baseline (`models/kdct.pt`).

## 4. Generation 2 — the K-map (kmap)

**Idea** (a redesign specified by the project owner): drop thresholds,
predict **K itself** — a full map, one K per block over the whole frame —
and keep exactly the K loudest ingredients plus DC. Two technical devices
made it work:

- **Differentiable "keep the top K".** Sorting is not something a neural
  network can learn through directly. The trick: sort the ingredients
  once, then apply a soft yes/no to each *rank* — "rank below K → keep" —
  with a softness that anneals away during training. Gradients flow into
  K itself.
- **"No new edges" penalty.** Instead of penalizing all output edges, we
  penalized only edges the output has *but the target does not* — a
  one-sided (hinged) penalty at first and second order. This was a course
  correction adopted during the project: penalize creation, never
  existence.

**What we learned.**

- K is bounded, contrast-independent, and immune to the fixed-threshold
  pathology — the K-form is simply the better parameterization.
- A new star measurement appeared: **lost_edge1**, the edge energy in the
  frame that the sparse reconstruction *cannot explain*. It fell from
  0.009 to 0.000 in perfect order as compression destroyed fine detail —
  one of the cleanest severity signals in the project.
- A hard limit was proven: selection can only *delete* ingredients. A
  paired experiment (damaged input, clean target) showed the
  reconstruction error floor rising 7× — you cannot restore information
  by deleting, which set up generation 3.

**Status.** Superseded (`models/kmap.pt`, `models/kmap_restore.pt`).

## 5. Generation 3 — the dimmer switches (wiener)

**Idea.** Replace the on/off decision per ingredient with a **dimmer
switch**: each of the 64 ingredients gets its own multiplier ("gain")
between 0 and 4, predicted per block. Keep = 1, mute = 0, boost = above 1.
This is the DCT-domain version of the classical **Wiener filter** —
statistics-driven per-ingredient volume control — with the statistics
supplied by a CNN instead of a formula. Three deliberate design choices
(all specified by the project owner):

1. **Gains may exceed 1** (up to 4) — restoration sometimes needs to
   amplify weakened ingredients, not just attenuate noise.
2. **Tidiness is judged on the output alone** — the damaged input's edges
   are not a reference for anything; minimize the output's own first and
   second derivatives, and let fidelity to the clean target prevent
   over-smoothing.
3. **Color**: all three channels (Y, U, V) go in together; the network
   outputs a gain map per channel — a (W/8)×(H/8)×C field of effective-K
   values, matching the original project specification.

**A hard lesson: the collapse.** The first training run died silently.
Gains start at the midpoint (2.0); the economy and tidiness pressures
slam them toward zero in one epoch; the sigmoid that produces them
saturates; gradients vanish; the network freezes forever at "delete
everything". The fix is one line — initialize so gains start just below
1, inside the responsive region — but the diagnosis method matters more:
**log the average gain and effective K every epoch; a frozen number is a
dead network.**

**What the experiments said.**

- Against real VMAF scores (single-filter test, unseen content):
  SRCC 0.93 and score error 7.4 VMAF points — **3× better calibrated**
  than the classical blind adaptive DCT filter (24.4) and better on both
  counts than a fixed-K filter (0.89 / 10.1), which collapsed to 0.78 on
  grainy content. That collapse is the textbook demonstration of why
  content-adaptivity is the mechanism, not a luxury.
- The 15-ladder stress campaign passed on every condition — including
  H.264 whose blockiness the codec's own deblocking filter had already
  smoothed (detected through detail-loss, 9 strong measurements) and real
  packet loss (passed with the thinnest margin, 2 strong measurements).
- Restoration accuracy has a ceiling (~34 dB) set by the economy price λ:
  the filter genuinely repairs severe noise (+7 dB) and banding, breaks
  even on blur (nothing to amplify), and "over-cleans" mildly damaged
  input. For our purpose this is fine — the residual is the product, not
  the picture.

**Status.** Production gain-only filter (`models/wiener.pt`).

## 6. Generation 4 — recycle and invent (affine)

**Idea** (proposed by the project owner): the dimmer switch cannot fix an
ingredient that arrived as zero — no multiplier revives K·0. So give each
ingredient a second control: `output = K·x + t`, where **K recycles what
survived and t writes in what died**, with t predicted from the
surrounding context. This completes the filter to the textbook form of
the classical estimator — the Lee filter is exactly
`a·input + (1−a)·(local estimate)`; t is that second half, with a neural
network supplying the estimate instead of a local average.

**The four guardrails** (t is invented content — it must be kept honest):

1. **Tax the invention**: an L1 price μ on t, parallel to λ on gains —
   recycling costs λ per unit, inventing costs μ per unit. The filter
   invents only where the frame truly cannot supply the information.
2. **L1, not L2**: the absolute-value tax snaps t to *exactly zero* where
   unneeded, so clean frames stay clean and t keeps a hard zero baseline.
3. **Start at zero**: the t-head is initialized so training begins as the
   proven gain-only filter and must earn its synthesis (the collapse
   lesson, applied preemptively).
4. **Bounded and DC-excluded**: |t| ≤ 1 via tanh, and t never touches
   average brightness.

**The context experiment — the project's most instructive result.**
Before building gen-4, we measured the network's field of view: about
40×40 pixels (±2 blocks) — a keyhole. Theory said: K ("how much survived
*here*?") is a local question and doesn't need more; t ("what *should*
be here?") is a context question and does. Both halves were then measured:

- *Negative control (gen-3)*: widening the view (dilated convolutions;
  then a U-Net zoom-out/zoom-in branch) did **not** improve the gain-only
  filter — baseline A won every row. K is local. Confirmed.
- *The real test (gen-4)*: with the keyhole view, **the t-head died** —
  taxed into silence because it could not predict anything useful from
  ±2 blocks. With the U-Net's wide view, **t came alive and its
  upper-percentile statistic became the single best severity measurement
  in the model** (perfectly ordered, SRCC +1.00, on both test contents).

**The damage map.** t has a second life: it is literally a map of "how
much the network had to invent, and where" — near zero on clean frames,
lighting up in damaged regions. Localized damage shows in the map's upper
percentiles (most blocks need nothing; the hurt ones shout), exactly like
our worst-tile residual statistics. To our knowledge, using this
synthesis-effort map as a no-reference quality signal is novel.

**The robust upgrade (gen4_robust).** The first gen-4 model learned
"what natural video looks like" from only 9 scenes — a narrow education.
We retrained the *identical* model (same architecture, same prices, same
guardrails) on crops streamed from ~90 different source videos (the
BVI-DVC collection: sports, faces, water, textures, streets), 11,200
training pairs in total. The effect was exactly what a broader education
should give:

- **The damage map matured.** With 9 scenes, only the map's loudest spots
  tracked damage severity; with 90 scenes, even its plain *average* is
  perfectly ordered with severity. The model's sense of "natural" became
  broad enough to be trusted everywhere.
- **The numbers became honest.** The earlier models were tested on the
  same content family they trained on — a home-field advantage. The
  robust model was tested on content it had *never seen* and still
  produced the best measurements of any checkpoint in the project
  (13 of 19 perfectly ordered). It gave up ~1 dB of in-domain picture
  polish in exchange — the classic signature of less memorizing and more
  understanding, and the right trade for a measuring instrument.

**Status.** `models/wiener4_dvc.pt` (robust) is the production probe.
`wiener4_c.pt` is the 9-scene version, `wiener4_a.pt` the keyhole
negative control — both kept as evidence.

### The champion's three title defenses

After the robust model was crowned, it was challenged three times by
models trained on data that should, on paper, have beaten it. It won all
three — and the pattern of those wins is the project's most important
finding.

**Challenge 1 — real codec damage.** We built a generator that takes
clean clips and runs them through *real* encoders — H.264, HEVC, and
MPEG-2, with randomized bitrates, profiles and encoder tools — and also
through *real packet loss* (bytes corrupted inside the compressed stream,
decoded with the decoder's own error concealment). 7,200 damage-realistic
training pairs. The model trained on them (`gen4_pairs`) became the best
*repairman* — best severe-noise and compression repair of any model — but
its measurements were slightly less orderly than the champion's, and its
harsh diet made it trigger-happy: it rewrites regions it distrusts even
when they cannot be fixed.

**Challenge 2 — the owner's own dataset.** A 44,000-folder image
collection (each folder: a ground-truth photo plus 10 degraded versions
spanning a bitrate ladder, each labeled with its VIF quality score). We
extracted 40,820 patch pairs, held 12 folders out as untouched judges,
and trained three in-domain models varying the invention tax μ. Judged on
the held-out folders' own quality ladders: **the champion — which had
never seen a single image from this dataset, and was even trained in a
different color space — beat all three** (17 orderly measurements vs 16,
13 and 11).

**Challenge 3 — maybe they were just under-trained?** We reran the best
in-domain recipe with 1.5× the data and 1.5× the epochs. It got *worse*
(13 vs 16). More in-domain training made the model a better specialist in
that dataset's damage and a worse judge of what natural content looks
like — and judging naturalness is the whole job.

**The lesson, in one sentence:** *the filter's power comes from knowing
what NATURAL looks like, not from knowing the damage* — a broad education
on clean content beats any amount of studying the disease. This is
exactly what the project's theory said from the start; now it is measured
three ways (across content type, damage source, and color space).

### A hardware lesson worth keeping

One training run froze the machine ("hung under heavy disk access"). The
real cause was not the disk: the data loader briefly needed **twice** the
dataset's memory while assembling it (28 GB on a 32 GB machine), Windows
started swapping memory to disk, and everything stalled. Fixes now built
in: the loader fills a pre-allocated block piece by piece (memory never
spikes), big-data trainings cap at ~24k patches in RAM, and long runs
execute at low CPU priority so the desktop always stays responsive. Rule
of thumb: watch the *peak* memory of the loading step, not the size of
the data.

## 7. Side by side

| | Gen 1 (kdct) | Gen 2 (kmap) | Gen 3 (wiener) | Gen 4 (affine) |
|---|---|---|---|---|
| Per-ingredient action | keep if louder than a learned threshold | keep the K loudest | dimmer switch, 0–4 | dimmer + invent (K·x + t) |
| Input | 64×64 patch, luma | full frame, luma | full frame, **color** | full frame, color |
| Training | clean only | clean only (+ paired variant) | paired (simulator + real H.264) | paired (same) |
| Edge regularizer | soft (via loss on patch) | "no NEW edges" hinge | **output-only** smoothness | output-only smoothness |
| Can restore? | no (deletes only) | no (deletes only) | partially (reweighs) | yes, within its tax budget |
| Signature finding | ΔK sign = damage type | lost_edge1; selection can't restore | 3× better VMAF calibration than classical | t needs context; damage map |
| Fatal trap avoided | — | fixed-threshold pathology | sigmoid collapse | t dying / t hallucinating |
| Monotone features (same ladder, same frame) | 6 | 9–10 | 7 | 8 (9 scenes) → **13 (robust, 90 scenes)** |

**How the generations divide the work** (measured on identical degraded
images): the selection filters (gens 1–2) are *gentle preservers* — most
faithful when damage is mild, because keep/drop cannot hurt what it
keeps. The gain filters (gens 3–4) are *aggressive restorers* — clearly
best when damage is severe (+6–7 dB on heavy noise), at the cost of
over-cleaning nearly-clean input. No generation dominates; they are
complementary instruments, and the strongest measurement setup pairs the
robust gen-4 with the best selection filter (gen-2 restore-trained) so
their different residuals are read side by side.

## 8. The experiments, explained simply

Each harness answers one question and can be re-run anytime:

- **`demo.py`** — *does the whole quality pipeline work end to end?*
  Synthetic content, compression ladder, both scoring layers; PASS/FAIL.
- **`train_*.py` built-in validation** — *do this filter's measurements
  move steadily with damage?* JPEG ladder on real + synthetic frames;
  requires at least 3 perfectly-ordered measurements per content.
- **`validate_artifacts.py`** — *which measurements see which artifact?*
  All 9 artifact types × all measurements; every artifact must have at
  least one strong responder. (It does — including the two artifact
  types held out of training.)
- **`stress_test.py`** — *does the filter survive severe, realistic
  abuse?* 15 ladders: real H.264 up to CRF 51 with and without the
  codec's deblocking, real packet loss at up to 50% slice corruption,
  severe blur/noise/JPEG, isolated ringing, and the standard suite.
- **`compare_vmaf.py`** — *do the residual measurements actually predict
  quality?* Each filter's features alone → ridge regression → compared
  against real VMAF on content never seen in training.
- **`compare_context.py`** — *does a wider view help?* Same filter,
  three fields of view, judged on restoration accuracy and measurement
  orderliness. (Answer: not for gains; decisively yes for synthesis.)
- **`compare_generations.py`** — *how do all the generations behave on
  the very same damaged images?* Six checkpoints, seven artifact types,
  restoration accuracy plus measurement orderliness in one table. This
  is where the preserver-vs-restorer division of labor was measured.
- **`generate_pairs.py`** — *build real training data*: clean clips
  through real H.264/HEVC/MPEG-2 encoders (randomized settings) and real
  packet loss, saved as reusable patch-pair shards.
- **`extract_binpairs.py` / `validate_binpairs.py`** — *use an external
  GT-plus-degraded image collection*: a resumable network extractor, and
  a judge that scores any checkpoint on held-out folders' own quality
  ladders (this is the harness the champion defended its title on).

Two testing principles used everywhere, both worth stealing:
**(1) content-grouped splits** — frames from one source video never
appear in both training and testing, otherwise the model memorizes
content and the numbers are fiction; **(2) negative controls** — we
tested context on gen-3 *expecting* no effect, got none, and only then
trusted the gen-4 effect.

## 9. Literature

Plain-language anchors — one line on why each matters here. (Full
citations in [PROJECT_REPORT.md](PROJECT_REPORT.md), Appendix B.)

**The filtering tradition we build on.**
- *Lee (1980), Kuan (1985)*: the classical local filter is already
  "a·input + (1−a)·estimate" — gen-4's exact shape, 45 years early.
- *Milanfar, "A Tour of Modern Image Filtering" (2013)*: the survey that
  unifies all content-adaptive filters; the project's conceptual map.
- *Donoho & Johnstone (1994)*: keep an ingredient if it's louder than
  the noise — the mathematical root of every K decision here.
- *Foi's SA-DCT (2007), BM3D (2007)*: the best classical DCT-domain
  filters; BM3D's second stage is a hand-made version of our gain map.
- *Codec in-loop filters — ALF (2013), SAO (2012), CDEF (2018)*:
  industrial proof that "classify content, choose filter parameters"
  works at scale; our classical probe bank mirrors them.

**The quality-assessment tradition.**
- *NIQE (2013), BRISQUE (2012)*: quality from "naturalness statistics";
  our anchor layer and several residual measurements descend from them.
- *Free-energy school — Zhai (2012), Gu/NFERM (2015), Wu (2013)*: the
  theory that perception itself works by "predict, then measure the
  surprise" — our residual idea has this as its formal umbrella.
- *Min, distortion aggravation (2018)*: the mirror image of our idea —
  they add damage and watch; we remove it and watch.
- *VMAF (Netflix)*: the accepted full-reference score we distill from;
  *NR-VMAF (2024)* did the same distillation with an opaque deep network
  — our version keeps the features interpretable.
- *CAMBI (Netflix, 2021)*: the best example of a deployed, honest,
  single-artifact quality meter; a design role model.

**The learned-transform tradition (closest neighbors of gens 3–4).**
- *LISTA (2010)*: networks can learn thresholding schedules.
- *D3 / DDCN (2016)*: deep JPEG repair with a DCT-domain branch —
  our g,t model with all structure removed; ours stays interpretable.
- *DCTNet (2023), DCT2net (2021)*: learned shrinkage in DCT filterbanks —
  the published relatives of the gain map, aimed at denoising.
- *Deep Wiener Deconvolution (2020)*: the "learned Wiener" idea in
  feature space; ours lives in the classical DCT domain instead.
- *U-Net (2015)*: the zoom-out/zoom-in architecture with skip
  connections — in DSP terms, a learned analysis/synthesis pyramid that
  carries the detail subbands around the bottleneck.
- **Our differentiators**: the residual (not the restored image) is the
  product; the K/gain/t fields are read back as measurements; training
  includes real corrupted bitstreams; and the synthesis-effort map as a
  quality signal appears to be new.

## 10. Glossary

- **DCT / coefficient / DC / AC** — the recipe transform; one ingredient;
  the average-brightness ingredient; the 63 pattern ingredients.
- **K** — how many AC ingredients a block keeps (or effectively keeps,
  when gains are fractional: K = sum of gains).
- **Gain (g)** — the dimmer setting multiplying one ingredient, 0–4.
- **t** — the invented addition to one ingredient (gen-4 only).
- **Residual** — frame minus filtered frame; where damage shows up.
- **λ, μ** — the prices: λ per unit of gain (economy), μ per unit of
  invention (honesty).
- **Receptive field** — how much of the image one output decision can
  actually see; ours grew from ~40 px (keyhole) to the whole crop (U-Net).
- **SRCC** — rank correlation; 1.0 means a measurement orders the
  severity ladder perfectly.
- **VMAF** — Netflix's full-reference quality score, 0–100; our training
  teacher.
- **Ridge regression** — least squares with diagonal loading (the same
  stabilization as a regularized Wiener solve); our score fusion.
- **Content-grouped split** — the testing rule: no source video appears
  on both sides of train/test.
- **Monotone / monotonicity** — moves steadily in one direction as damage
  grows; the property every useful measurement must have.
