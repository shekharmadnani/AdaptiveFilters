# The Complete Project Conversation — in Plain Language

This is the full record of the conversation that built this project, from
the first planning question to the final architecture experiments. Every
question asked, every decision made, every result found — retold simply,
with nothing important left out. (The two companion documents go deeper:
[THE_FOUR_FILTERS.md](THE_FOUR_FILTERS.md) for the filters,
[PROJECT_REPORT.md](PROJECT_REPORT.md) for full technical detail and
references.)

---

## 1. The opening question: content-adaptive filters

**Asked:** Plan (no code yet) a set of filters whose parameters are
learned from the surrounding content. Applied to an image block, such a
filter should remove unnatural discontinuities; the residual (original
minus filtered) then exposes the errors. Example: an adaptive DCT filter
where the number of kept AC coefficients is decided from the content.

**Answered:** A full catalog of such filters with their mathematics and
literature — the adaptive DCT keep-K filter (with the classic rule: keep
a coefficient if its energy exceeds a noise threshold), the local Wiener
/ guided filter, the codec in-loop filters (ALF, SAO, deblocking, CDEF)
which are the industrial versions of exactly this idea, steering kernels,
sparse coding, and trained filter lookups (RAISR). One template unifies
them all: *model natural content → estimate the model's parameters from
the content itself → filter → the residual holds what the model of
"natural" could not explain.*

## 2. The real goal: a no-reference video quality score

**Asked:** The end goal is a no-reference video quality index built from
the residuals of several adaptive filters. Human opinion scores (MOS) are
impractical industrially, and published methods don't deploy well. A
practical method is needed.

**Answered:** The plan that shaped everything after: use the filters as
*probes* (each one an expert on clean content whose residual exposes one
damage family); turn residual statistics into a feature vector; train a
small regression against **VMAF** (Netflix's accepted full-reference
score, computed on our own encodes — free unlimited labels, no humans
needed); keep a second, label-free "naturalness distance" layer as a
safety net; and validate with mechanical tests (scores must rise steadily
with bitrate; scores must not depend on content) instead of MOS
correlation. Key rule adopted: **frames from one source video must never
appear in both training and testing** — breaking this rule is the main
reason published results fail to reproduce.

## 3–5. Choosing the score-fusion machine

**Asked:** Why GBT (gradient-boosted trees)? Will it overfit? Could deep
learning or transformers replace it? And if the features are weak, can
any fusion model fix that?

**Answered:** Ridge regression (least squares with diagonal loading —
the same stabilization as a regularized Wiener solve) was chosen as the
default, GBT as a challenger that must *earn* its extra complexity, and
deep models deferred until evidence demands them. The deciding insight,
confirmed later by experiment: **the features are the model** — the
fusion stage only gets you close to whatever ceiling the features set;
no fusion model can recover information the features already lost.
(GBT later failed spectacularly in a live test — trees cannot
extrapolate outside their training range.)

## 6. Back to the DCT filter as the core instrument

**Asked:** Can the adaptive DCT design be generalized into a family of
filters whose content-adaptive residuals feed the quality score?

**Answered:** Yes — the keep-K DCT filter is the prototype of the whole
family. Every probe repeats: naturalness model → blindly estimated
parameter → residual → features. Crucial design trap identified early:
*the filter must adapt to the content, never to the damage* — otherwise
it "explains" the artifacts and the residual goes blind exactly when
quality is worst. Defenses: robust statistics, reading the estimated
parameters back as features, training only toward clean content.

## 7. First code: the classical probe bank

**Asked:** Get the code for each component.

**Built:** A NumPy package with six classical probes (adaptive DCT,
guided/local-Wiener, deblocking-grid, SAO-style offsets, directional,
temporal), shared statistics (GGD shape fit, robust sigma, residual
whiteness, an orthogonality check), the naturalness anchor, ridge fusion,
and a self-contained demo. The demo passed immediately: on held-out
synthetic content, both scoring layers ordered a compression ladder
perfectly (rank correlation 1.0).

## 8–10. Real machinery: environment, real video, VMAF labels

The right Python environment was pinned down
(`E:\Python_Workspace\envs\PythonGPU`). Frame readers were built for raw
YUV, Y4M, images and anything ffmpeg decodes; a VMAF runner (with the
model file bundled, since the local ffmpeg lacked it); a dataset builder
with resumable caches; training and scoring command-line tools. Speed
work made 1080p feature extraction practical (batched matrix DCT — 9×
faster than the naive version; decimated views for medians). The user
pointed to the BVI datasets on the F: drive — real 4K/1080p masters and
a codec-comparison set — which became the project's data backbone.

## 11–12. Generation 1: the first learned filter

**Proposed (by the owner):** Feed 64×64 patches to a CNN that estimates
the optimal number of kept DCT coefficients per 8×8 block, trained with
reconstruction fidelity plus regularizers on the coefficient count and on
image smoothness (first and second derivatives).

**Built and found:** It worked — residual features tracked damage
perfectly — and taught three lessons: the gap between "K the content
should need" and "K actually present" encodes the *type* of damage
(blur pulls it negative, blockiness pushes it positive), not the amount;
fixed thresholds interact badly with quantization (counts rise before
they fall); and the network must train on clean content only.

## 13–14. Generation 2: the full-frame K-map

**Proposed (by the owner):** A new design — the CNN takes the whole
frame and outputs a (W/8)×(H/8) map of K; keep the K largest
coefficients plus DC; regularize the count and the smoothness of the
output so no new edges are invented.

**Built and found:** A differentiable "keep the top K" (sort once, apply
a soft yes/no by rank) made K directly learnable. The "no NEW edges"
one-sided penalty replaced plain smoothness. A star measurement emerged:
`lost_edge1` — edge energy the sparse model cannot explain — fell in
perfect order as compression destroyed detail. And a hard limit was
proven: *selection can only delete*; a paired experiment showed you
cannot restore missing information by dropping coefficients — which
motivated generation 3.

## 15–17. Robustness demanded, and generation 3 born

**Asked:** Is the CNN trained for artifacts like packet loss,
concealment, compression? It must be tested on most error conditions.
Then: input should be the degraded patch, output the pristine one. Then:
no restoration heroics needed — a **DCT-domain Wiener filter** (gains,
"filter to whatever extent it can") whose output generates the residual.

**Built and found:** An artifact simulation suite (compression, blur,
noise, banding, two packet-loss concealments, block loss, frozen
regions — five severities each) doubling as training-data generator; an
artifact-response matrix proving every damage type moves some
measurement; worst-tile pooling so local damage can't hide in averages;
paired training (damaged in → clean target). Generation 3 replaced the
binary keep/drop with per-coefficient **dimmer switches (gains)** —
the learned DCT-domain Wiener filter. Against real VMAF it beat a
fixed-K baseline soundly (the fixed filter collapsed on grainy content —
the textbook proof that content-adaptivity is the mechanism) and was 3×
better calibrated than the classical blind DCT filter.

## 18. Real corruption in training

**Asked:** Use H.264-corrupted data for training too — with strict
conditions: the loss stays exactly RMSE + coefficient-count regularizer +
derivative terms, and no deblocking or SAO sneaks in anywhere.

**Built:** x264 encoding with the loop filter disabled, bytes flipped
inside slice NALs (headers kept intact so streams stay decodable and
aligned), decoding with concealment-deblock off. Retrained on the mix,
the filter alone detected all 8 artifact classes — including two types
held out of training entirely.

## 19–21. Repo, explanations, related work

The project went to GitHub (the owner made the first commit; every later
round was committed on request with detailed messages). The .py/.pt/.y4m
files were explained. A literature check found the nearest published
relatives (DCTNet, DCT2net, dual-domain JPEG networks, LISTA) — none of
which use the residual as the product or read the parameter fields as
quality signals.

## 22–25. Generation 3 refined, then stress-tested

**Asked (three changes):** Gains may exceed 1 (cap 4); smoothness judged
on the *output only* (the damaged input's edges are no reference); and
full **color** — a (W/8)×(H/8)×C gain field.

**Built and found:** The first training collapsed silently — gains
started at the midpoint, got slammed into the flat tail of the sigmoid,
gradients died. Fix: start gains just below 1, in the responsive region;
lesson: log the average gain every epoch — a frozen number is a dead
network. The retrained color model passed everything, and the 15-ladder
stress campaign (severe compression, **H.264 with deblocking on** — the
codec-smoothed blockiness case — real packet loss, ringing, severe
blur/noise) passed on every condition.

## 26–28. Learning mode and plain teaching

The owner asked to learn to build the filter personally; a 7-stage
self-checkable roadmap was written (and kept in the report's appendix).
Two documents were created — the full technical report with ~60
annotated references, and later a plain-language guide. Along the way,
teaching answers explained VMAF distillation (train a cheap
no-reference student to imitate the reference-needing VMAF teacher) and
ridge regression ("ridge" = the raised diagonal added to the normal
equations — the same diagonal loading as a regularized Wiener solve).

## 29–33. Generation 4 designed, and the context question

**Proposed (by the owner):** `output = K·x + t` — keep the multiplier
(now up to 4) and add an **additive term t** that can revive zeroed-out
coefficients, predicted from context. Feedback given: this completes the
filter to the classical affine estimator (the Lee filter's exact shape);
it unlocks the two known limits (revive deleted detail, fix inter-block
seams); it needs four guardrails (tax t with an absolute-value penalty so
it stays exactly zero when unneeded; keep it bounded; start it at zero;
keep DC out). Bonus: **t doubles as a damage map** — "how much had to be
invented, and where" — and using that as a quality signal appears novel.

**Then asked:** t depends on context — should we keep the CNN, enlarge
its view, or switch to a transformer? Measured answer: today's view was
a 40-pixel keyhole; the plan chose to renovate the CNN (spread-out
dilated vision, a zoom-out/zoom-in U-Net branch with skip connections —
explained in DSP terms as a learned analysis/synthesis pyramid — and
bigger training crops), with the transformer held as an evidence-
triggered escalation.

## 34–35. The two-part context experiment (the project's cleanest result)

Run first on generation 3 as a deliberate negative control: widening the
view did **not** help the gain-only filter — K is a local question,
exactly as theory predicted. Then generation 4 was built on both the
keyhole and the U-Net: **with the keyhole view the synthesis term
died** (taxed into silence, unable to predict anything useful); **with
the U-Net view it came alive and its damage-map statistic became the
single best severity measurement in the model.** Both halves of the
theory measured, not assumed.

## 36–39. All generations compared; the robust champion

A cross-generation test on identical damaged images revealed a clean
division of labor: the selection filters (gens 1–2) are gentle
preservers, best on mild damage; the gain filters (gens 3–4) are
aggressive restorers, best on severe damage; no one dominates —
they're complementary instruments. Then the question "can gen-4 be a
novelty, and can it be made robust with more data?" led to retraining on
~90 different source videos (streamed crops from BVI-DVC):
**gen4_robust** scored 13 of 19 perfectly-ordered measurements *on
content it never saw* — the best of all checkpoints — with the entire
damage-map channel perfectly ordered. Corpus diversity matured the
signal. Novelty assessment: the damage-map-as-quality-signal is the
strong claim; the explicit g/t decomposition with separate taxes and the
"K is local / t needs context" finding support it.

## 40–41. Real three-codec training pairs

**Asked:** Generate training pairs from real encoders — H.264, HEVC,
MPEG-2 across bitrates, profiles and tools — plus a packet-loss family
(for transmission and digital-tape conversion errors).

**Built:** A generator producing 7,200 perfectly balanced pairs (both
families × three codecs, every encoder setting recorded), with
codec-aware corruption (correct bitstream parsing per codec, headers
protected so frames stay aligned) and damage-biased patch selection.
The model trained on it (**gen4_pairs**) became the best *restorer* —
best severe-noise repair of all models — but the measuring crown stayed
with gen4_robust. When confusion arose about "which is best," the
answer was laid out: for this project the filter is a measuring
instrument, so feature quality on honestly held-out data decides — and
that's gen4_robust; gen4_pairs is the real-codec repair specialist;
they hold one good ingredient each (content diversity vs damage
realism).

## 42–43. The owner's own dataset — and a machine crash

**Provided:** A network share with 44,209 folders, each holding a
ground-truth photo and 10 degraded versions spanning a bitrate ladder,
each labeled with its VIF quality score. After a formal configuration
approval (scale, bin coverage, hyperparameter sweep, color handling), a
resumable extractor pulled 40,820 patch pairs; 12 folders were reserved
as untouched judges. A μ sweep trained three in-domain models — and
**the champion beat all three on their own held-out data** (17/19 vs
16/13/11), despite never seeing a single image from this dataset and
running in a different color space.

Then a full-budget rerun (more data, more epochs) **hung the machine**.
Diagnosis: not the disk itself — the data loader briefly needed twice
the dataset's memory (~28 GB on a 32 GB machine) and Windows paged to
disk. Fixes now permanent: a memory-flat loader (pre-allocate and fill),
a ~24k-pair cap, and low process priority for long runs. The rerun
completed safely — and **lost even harder** (13/19): more in-domain
training made the model a better damage specialist and a worse judge of
naturalness. Third title defense. The lesson crystallized: *the
filter's power is knowing what natural looks like, not knowing the
damage.*

## 44–45. The transformer challenge (fourth defense)

**Asked:** Can a ViT — attention, dynamic context, masking — produce g
and t? Plan first, introspect.

**Planned honestly:** attention adds one genuinely new ability —
*retrieval* (find the matching undamaged texture elsewhere in the frame)
— and masking enables MAE-style pretraining (cover parts of clean
images, predict them — exactly t's job, trainable on unlimited clean
data). Predictions were registered in advance: the transformer with the
same data loses; the MAE-pretrained one is the real contest.

**Built and found:** A hybrid (the champion's exact architecture with
attention replacing only the bottleneck convolution — isolating the
retrieval question) plus the masking-practice pretraining. Results on
the held-out judges: champion 17, transformer+MAE 12, plain transformer
11. Both predictions held; MAE clearly helped (doubling the plain
transformer's feature count) but not remotely enough; and the *bigger*
model losing dissolved the capacity caveat. Architecture question
closed at this scale, with one honest asterisk: our MAE dose was small
compared to the literature's — "doesn't pay here" is proven, "never"
is not.

## 46–48. The learning path begins

The owner started the build-it-yourself journey, and two foundational
questions were answered in teaching mode:

- **Is a training sample the whole image or a patch?** A patch, always
  — memory forces it, statistics favor it (32 crops from 32 different
  contents per batch beat correlated whole-frames), and convolution's
  translation-equivariance makes "train on crops, infer on full frames"
  legitimate. Crop size limits only the *context* the network can learn
  to use. For transformers, "patch" means two things (the training crop
  and the token tile), and transformers are *not* naturally
  size-flexible — position embeddings and sequence lengths differ
  between crop training and full-frame inference.
- **If t starts at 0, aren't its gradients 0?** No — zero *output* is
  not zero *gradient*. `tanh(0) = 0` but its slope there is the maximum
  of the whole curve; the loss is unhappy whenever the frame is damaged;
  and the head's inputs are nonzero — so the weights move on the very
  first step. What kills training is *saturation* (the flat tails —
  exactly the gain-collapse accident), the opposite end of the curve
  from where t starts. The classic zero-init danger is a *symmetry*
  problem in hidden layers, which doesn't apply to a final head.
  Precedent: ControlNet's zero-convolutions start at exact zero by
  design, for the same reason as our guardrail: a new branch must start
  silent and earn its contribution.
- **Why is t limited to ±1?** Because t is the only component allowed to
  *invent*, its budget is capped at texture-level energy. In our units
  (pixels 0–1, orthonormal DCT), ordinary detail coefficients live well
  below 1 while strong structural edges reach 1–2 — so a bounded t can
  wrongly smudge but cannot fabricate structure. It also matches t's
  mission: quantization deletes the *small* coefficients (the big ones
  survive and are g's job), so everything t legitimately restores fits
  inside ±1. In practice trained |t| runs around 0.001–0.005 — two
  orders of magnitude under the ceiling — so the tanh operates in its
  linear zone and the bound is pure insurance, invisible until something
  goes wrong. It's an exposed knob (`--tmax`), raisable on evidence
  (watch for t pressing the rail), and the same logic taken to its
  extreme is why DC gets t = 0 exactly: brightness synthesis is all
  hazard, no mission.

---

## The state of the project at the end of this conversation

- **Production filter:** `models/wiener4_dvc.pt` — the generation-4
  affine DCT filter (gains up to 4 + taxed synthesis term, U-Net
  context, color), educated on ~90 clean video scenes, **four title
  defenses** against challengers with better-looking data, more
  training, and a fancier architecture.
- **Its central law, proven four ways:** breadth of natural-content
  education beats damage specialization and architectural
  sophistication.
- **The novel instrument:** the damage map (the t-field) — a per-block
  chart of "how much the filter had to invent" — perfectly ordered with
  damage severity on data from two independent worlds.
- **Everything is repeatable:** six evaluation harnesses, three data
  generators (simulator, three-codec pairs, external GT extractor), all
  checkpoints kept as the ablation record, machine-safety limits
  institutionalized.
- **Open frontiers:** the quality-index (fusion) stage — where the
  champion's residuals, the damage map, the companion selection filter,
  and 40k VIF-labeled pairs are waiting — and the owner's from-scratch
  reimplementation, now underway.
