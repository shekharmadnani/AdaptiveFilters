# The Complete Discussion Trail — Every Step, Simply Told

This document walks through the entire project discussion, step by step,
in the easiest language possible. Each step tells you three things:
**what was discussed**, **what was done**, and **what was learned**.
Nothing important is skipped.

*(Shorter companion reads: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for
the big picture, [THE_FOUR_FILTERS.md](THE_FOUR_FILTERS.md) for the
filters, [PROJECT_REPORT.md](PROJECT_REPORT.md) for full technical depth.)*

---

# Part I — Setting the Direction

## Step 1. The first idea: filters that adapt to content

**Discussed:** Could we build filters whose settings are not fixed, but
chosen by looking at the surrounding picture content? Applied to an image
block, such a filter should clean away unnatural roughness. Then the
difference between the original and the cleaned version — the
**residual** — would show us where the errors are. The first concrete
example: a DCT filter where the *number of frequency components to keep*
is decided from the content.

**Done:** A full survey (planning only, no code) of every filter family
that works this way, with the math behind each and the research papers
where they come from — including the filters inside real video codecs
(which are industrial versions of exactly this idea).

**Learned:** All of these filters follow ONE recipe: build a model of
what *natural* content looks like → estimate the model's settings from
the content itself → filter → whatever the model could not explain lands
in the residual. That recipe became the backbone of the whole project.

## Step 2. The real destination: a quality score without the original

**Discussed:** The true goal is a **no-reference video quality score** —
judging quality without having the original video to compare against.
Human rating panels (MOS) are impractical in industry, and published
methods do not survive real deployment.

**Done:** A practical plan: use the adaptive filters as *probes*; turn
their residual statistics into measurements; train a small regression to
imitate **VMAF** (Netflix's trusted quality score, which we can compute
ourselves on our own encodes — free labels, no humans); add a
label-free "naturalness distance" as a safety net; and replace human
correlation with mechanical tests (does the score rise steadily with
bitrate? is it independent of content?).

**Learned:** The single most important testing rule: **frames from one
source video must never appear in both training and testing.** Breaking
this rule lets a model memorize content and produce beautiful but
fictional numbers — the main reason published results fail in practice.

## Steps 3–5. Choosing the final scoring machine

**Discussed:** Why gradient-boosted trees (GBT)? Would they overfit?
Could deep learning or transformers do better? And if the measurements
themselves are weak, can any scoring machine save us?

**Done/Learned:** Ridge regression (ordinary least squares made stable
by adding a small constant along the matrix diagonal — the same
"diagonal loading" trick used in classical Wiener filtering) was chosen
as the default. GBT stays as a challenger that must *prove* itself. Deep
models wait for evidence they are needed. The deciding insight —
**the measurements ARE the model**: the final regression only gets you
close to whatever ceiling the measurements set; nothing downstream can
recover information the measurements already lost. (Later, a live test
proved the caution right: GBT scored *worse than random* on unfamiliar
grainy content, because trees cannot guess outside what they saw in
training.)

## Step 6. The DCT filter becomes the core instrument

**Discussed:** Can the adaptive-DCT idea be generalized into the family
of residual-producing filters for the quality score?

**Learned:** Yes — and one danger was named early and never forgotten:
**the filter must adapt to the CONTENT, never to the DAMAGE.** A filter
that learns to reproduce blockiness "explains" the blockiness, and its
residual goes quiet exactly when the video is worst. Every design
decision afterwards respected this rule.

---

# Part II — Building the Foundation

## Step 7. The first working code

**Done:** A Python package with six classical probes (adaptive DCT,
local Wiener, deblocking-grid, edge-offset, directional, temporal),
shared statistics, a naturalness anchor, ridge fusion, and a
self-contained demo.

**Learned:** The concept works on day one — on unseen synthetic content,
the quality layers ordered a compression ladder *perfectly*.

## Steps 8–10. Real machinery

**Done:** The right Python environment pinned down; readers for real
video files; a VMAF label generator; a dataset builder that survives
interruptions; training and scoring command-line tools; speed work that
made full-HD feature extraction practical (a 9× faster DCT among other
things). The BVI video datasets on the local drive became the data
backbone: real masters, real encodes.

---

# Part III — The Learned Filter Grows Up

## Steps 11–12. Generation 1: the first neural filter

**Proposed by the owner:** A small neural network (CNN) looks at a
64×64 patch and estimates, for each 8×8 block inside, the ideal number
of DCT components to keep — trained to rebuild the patch faithfully,
with penalties on the component count and on rough output.

**Done and learned:** It worked, and taught three things: (1) the *gap*
between "how many components content like this should need" and "how
many are actually present" tells you the **type** of damage (blur pulls
it one way, blockiness the other) but not the amount; (2) fixed
thresholds behave weirdly under compression (counts rise before they
collapse, because quantization rounds surviving values upward); (3) the
network must study clean content only.

## Steps 13–14. Generation 2: the full-frame K-map

**Proposed by the owner:** Redesign — the network sees the whole frame
and outputs a map of K (components to keep) for every block; keep the K
loudest components; penalize the count and any *newly invented* edges.

**Done and learned:** A trick made "keep the top K" learnable (sort
once, then apply a soft yes/no by rank). A star measurement appeared:
the frame's edge energy that the sparse rebuild *cannot explain* — it
fell in perfect order as compression destroyed detail. And a hard truth
was proven: **selection can only delete** — you cannot bring back
missing detail by choosing what to drop. That truth demanded the next
generation.

## Steps 15–17. Robustness demanded; Generation 3 born

**Asked by the owner:** Is this tested against packet loss, error
concealment, and the rest? Make it robust against most error conditions.
Then: train with damaged input and clean target. Then the key
simplification: no restoration heroics — build a **DCT-domain Wiener
filter** (per-component volume controls, "filter to whatever extent it
can") purely as a residual generator.

**Done:** An artifact simulator (nine damage types, five severities
each), a coverage matrix proving every damage type moves some
measurement, worst-region statistics so local damage can't hide in
averages, and generation 3: **per-component gains** — the learned
Wiener filter. Against real VMAF it clearly beat a fixed-K filter
(which collapsed on grainy content — the cleanest proof that
adaptivity is the mechanism) and was three times better calibrated than
the classical adaptive filter.

## Step 18. Real corruption in the training diet

**Asked by the owner:** Also train on genuinely corrupted H.264 —
with strict conditions: the loss function stays exactly as specified,
and **no deblocking or SAO** sneaks in anywhere.

**Done:** Real x264 encodes (loop filter off), bytes flipped inside the
compressed stream (headers protected so frames stay aligned), decoded
with the decoder's own error concealment. Retrained on this diet, the
filter alone detected **all** damage classes — including two types
deliberately held out of training.

---

# Part IV — Becoming a Real Project

## Steps 19–21. Repository, explanations, related research

The project went to GitHub, with every round committed under detailed
messages. The file types were explained (.py = code, .pt = trained
network weights, .y4m = uncompressed video masters). A literature check
found the nearest published relatives — none of which use the residual
as the product or read the filter's own settings as quality signals.

## Steps 22–25. Generation 3 refined, then stress-tested

**Asked by the owner (three changes):** Gains may exceed 1 (up to 4);
smoothness judged on the *output only* (a damaged input's edges are no
reference for anything); and full **color** processing.

**Done and learned:** The first training died silently — gains started
at the midpoint of their range, got slammed to zero by the penalties,
and the squashing function's flat tail killed all learning. The fix was
one line (start gains just below 1, on the steep part of the curve), but
the lesson was bigger: **log the average settings every epoch — a frozen
number is a dead network.** The repaired color model passed a 15-ladder
stress campaign: severe compression, H.264 whose blockiness the codec
itself had already smoothed (detected anyway!), real packet loss,
ringing, severe blur and noise.

## Steps 26–28. Teaching mode begins

The owner asked to learn to build the filter personally; a 7-stage
self-checkable roadmap was written. Two documents captured the project.
Plain-language teaching covered **VMAF distillation** (train a cheap
student that needs no reference to imitate a teacher that does) and
**ridge regression** (why the "ridge" is a raised diagonal in a matrix).

---

# Part V — Generation 4 and the Context Saga

## Step 29. The owner proposes generation 4

**Proposed:** `output = K·x + t` — keep the multiplier, and add an
**additive term t** that can revive components that arrived as zero,
predicted from the surrounding context.

**Feedback given:** This is the *textbook* form of the classical
estimator (the venerable Lee filter has exactly this shape) and it
unlocks the two known limits: reviving deleted detail, and fixing seams
*between* blocks. But t invents content, so it needs four guardrails:
a tax so it stays exactly zero when unneeded; a hard bound; birth at
zero; and hands off the brightness component. Bonus insight: **t is
also a damage map** — "how much had to be invented, and where."

## Steps 30–33. Understanding before building

Plain-language answers covered: what g is in the economy term (the sum
of gains = the effective number of kept components); the whole gen-4
design retold simply ("K recycles, t invents; tax both; the tax bill on
t doubles as a damage report"); the context question (the network's
field of view was measured — a 40-pixel keyhole — and a renovation was
planned: spread-out vision, a zoom-out/zoom-in U-Net branch explained as
a learned analysis/synthesis pyramid, bigger training crops); and the
U-Net itself, explained in detail with the wavelet analogy.

## Steps 34–35. The two-part context experiment

**Done:** First on generation 3 as a deliberate *negative control*:
widening the view did NOT help the gain-only filter — proving K is a
local question, exactly as predicted. Then generation 4 was built on
both a keyhole and a U-Net: **with the keyhole, the invention term
died** (taxed into silence — it couldn't predict anything useful from
±2 blocks); **with the U-Net's wide view it came alive**, and its
damage-map statistic became the single best severity measurement in the
model.

**Learned:** Gains are local; synthesis needs context. Both halves
measured, not assumed.

---

# Part VI — The Championship Era

## Steps 36–37. All generations on the same damaged images

**Done:** Every checkpoint tested identically. **Learned:** a clean
division of labor — the selection filters (gens 1–2) are *gentle
preservers*, best on mild damage; the gain filters (gens 3–4) are
*aggressive restorers*, best on severe damage. Nobody dominates; they
are complementary instruments.

## Steps 38–39. The robust champion is crowned

**Asked:** Can gen-4 be a novelty? Can it be made robust with more data?

**Done:** Retrained on ~90 different clean videos (streamed crops so
memory stays flat). The result — **gen4_robust** — scored 13 of 19
perfectly-ordered measurements *on content it never saw*: the best yet,
with the entire damage-map channel perfectly ordered. Novelty
assessment: the damage-map-as-quality-signal is the strong claim.

## Steps 40–41. Real three-codec training pairs

**Asked:** Generate pairs from real encoders — H.264, HEVC, MPEG-2,
wide settings — plus a packet-loss family (transmission and
digital-tape errors).

**Done:** 7,200 perfectly balanced pairs, every encoder setting
recorded, codec-aware corruption. The model trained on them became the
best *repairman* — but the measuring crown stayed with gen4_robust.
When "which is best?" caused confusion, the answer was laid out: the
filter is a **measuring instrument**, so honestly held-out measurement
quality decides — and that's gen4_robust.

## Steps 42–43. The owner's own dataset — and a crash

**Provided:** A network share: 44,209 folders, each a clean photo plus
10 degraded versions with quality labels. After a formal configuration
approval, 40,820 patch pairs were extracted (12 folders reserved as
untouched judges). Three in-domain models were trained — and **the
champion beat all three on their own held-out data** (17/19), never
having seen a single image from it.

Then a bigger rerun **froze the machine**. Real cause: not the disk —
the data loader briefly needed *twice* the dataset's memory (28 GB on a
32 GB machine) and Windows started swapping. Permanent fixes: a
memory-flat loader, a size cap, low process priority. The safe rerun
then **lost even harder** (13/19): more in-domain training made the
model a damage specialist and a worse judge of naturalness. Third title
defense. The central law crystallized: **the filter's power is knowing
what NATURAL looks like, not knowing the damage.**

## Steps 44–45. The transformer challenge (fourth defense)

**Asked:** Could a ViT (attention, dynamic context, masking) produce g
and t? Plan first, introspect.

**Planned with predictions registered in advance:** attention's one new
ability is *retrieval* (find matching undamaged texture elsewhere);
masking enables practice-on-clean-images pretraining (cover parts,
predict them — exactly t's job, unlimited data). Prediction: the plain
transformer loses; the pretrained one is the real contest.

**Result:** Champion 17, transformer+pretraining 12, plain transformer
11. Both predictions held. The bigger model losing also dissolved the
"maybe it just had more capacity" worry. Architecture question closed
at this scale — with the honest asterisk that our pretraining dose was
small by literature standards.

---

# Part VII — The Learning Path and Generation 5

## Step 46. Batch samples: whole image or patch?

**Asked:** During training, is one sample a whole image or a patch? And
for transformers?

**Answered:** A patch, always — memory forces it, statistics favor it
(32 crops from 32 different videos per batch beat correlated whole
frames), and convolution's slide-everywhere nature makes "train on
crops, run on full frames" legitimate. Crop size limits only the
*context* the network can learn to use. For transformers, "patch" means
two different things (the training crop AND the small tiles attention
compares), and transformers are *not* naturally size-flexible — their
learned positions and sequence lengths differ between crop training and
full-frame use.

## Step 47. If t starts at zero, how can it learn?

**Asked:** Zero initialization should mean zero gradients — how does
training happen?

**Answered:** Zero *output* is not zero *gradient*. The squashing
function tanh gives 0 at input 0 — but its *slope* there is the maximum
of the entire curve; the loss is unhappy wherever the frame is damaged;
and the head's inputs are real features. So the weights move on the
very first step. What kills learning is *saturation* (the flat tails —
exactly the earlier gain collapse), the opposite end of the curve. The
classic zero-init danger is a symmetry problem in hidden layers, which
doesn't apply to a final output head. Famous precedent: ControlNet's
"zero convolutions."

## Step 48. Why is t limited to ±1?

**Answered:** t is the only part allowed to *invent*, so its budget is
capped at texture-level energy. In our units, ordinary detail lives well
below 1 and strong structural edges reach 1–2 — so a bounded t can
wrongly smudge but cannot fabricate structure. It also matches t's
mission: compression deletes the *small* components (the big ones
survive and belong to g). In practice trained t sits a hundred times
below the ceiling — the bound is pure insurance. Same logic at its
extreme: the brightness component gets t = 0 exactly.

## Steps 49–50. The overview document, and generation 5 proposed

**Asked:** First a simple whole-project document for group reading —
then proceed with the new idea: a **hybrid CNN+MLP**. The CNN looks at
context as before; a small MLP examines the current 8×8 block itself —
is it healthy and natural, or not? Both together estimate g and t.

**Done:** [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) written (the whole
project in one simple read). The hybrid was analyzed: it completes the
classical estimator structure — the CNN is the *prior* ("what should be
here"), the MLP is the *evidence* ("what actually arrived") — the very
shape of Wiener estimation. First design included a supervised "health"
output; **the owner then refined the design: no health output, no third
parameter — the MLP's healthy/unnatural judgment stays internal and
feeds only g and t.** The health head was stripped accordingly.

## Steps 51–52. Generation 5 trained and judged (fifth defense)

**Done:** Built at fair size (0.72M parameters vs the champion's
0.68M), trained on the champion's exact recipe, judged on the held-out
photo dataset.

**Result:** Champion 17/19 — **fifth title defense.** The hybrid scored
a respectable 14/19 (better than the transformers and the in-domain
full run), but the way it lost was the project's oldest trap in a new
costume: direct sight of the block's own coefficients made it *trust
the observed evidence too much* — its effective K collapsed toward 1,
and the damage map's polarity inverted (higher on mild than severe
content) because the evidence branch silently absorbed part of the
invention job. The residual measurements stayed strong; the instrument
character weakened. The law sharpened: **every attempt to make the
filter cleverer about the observed evidence has traded away measuring
quality.**

## Step 53. Branch and pull-request housekeeping

The gen-5 work was committed on a side branch (`gen5-hybrid-cnn-mlp`)
when a create-pull-request command arrived; the GitHub CLI tool isn't
installed on this machine, so the PR itself needs one click on GitHub's
pre-staged page. The owner chose to handle the merge/PR separately.
`main` holds everything through the fourth defense; the branch holds
generation 5.

---

# Part VIII — Where Everything Stands

- **Production filter:** `models/wiener4_dvc.pt` — generation 4
  (recycle + invent, U-Net context, color), educated on ~90 clean video
  scenes. **Five title defenses**: real three-codec data, the owner's
  dataset at two training budgets, a transformer with masked
  pretraining, and the evidence hybrid.
- **The central law, proven five ways:** breadth of clean-content
  education beats damage specialization, architectural sophistication,
  and evidence-cleverness — because the filter is a measuring
  instrument, and instruments need a stable sense of "normal."
- **The novel signal:** the damage map (the t-field) — a per-block chart
  of how much the filter had to invent — perfectly ordered with damage
  severity across two independent data worlds.
- **Everything is repeatable:** six PASS/FAIL evaluation harnesses,
  three data generators, every challenger checkpoint kept as evidence,
  machine-safety limits built into code and memory.
- **Next frontiers:** the quality-index (fusion) stage — where the
  champion's residuals, the damage map, the companion classical probes,
  and 40k quality-labeled pairs converge into the final score — and the
  owner's own from-scratch rebuild, guided by the staged roadmap.
