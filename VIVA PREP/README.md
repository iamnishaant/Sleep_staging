# Viva prep: slide-by-slide guide for `FYP_final.pptx`

Built from the actual content of the 34 slides. For each slide: what it says, **why** it's there /
what to infer from any figure on it, and a quick-check question. After the slide walkthrough there are three
focused sections: **architecture & signal processing** (deep dive, as you asked), the **four questions you
pasted**, and a larger **general viva question bank** organised by topic, plus a **"gotcha" section** — the
kind of sharp follow-up a teacher asks when something looks inconsistent at first glance.

**Naming note, say this early if asked:** release **v1 = student "N4kd"** (139,606 params, EEG only).
Release **v2 = student "N6adv"** (175,599 params, EEG + EOG). "N6adv" is the architecture's internal name; the
PPT calls it **v2** throughout. If your teacher says "v2 N4kd" she likely means "v2 compared with N4kd" —
clarify once, politely, and move on.

---

## Part 1 — Slide by slide

### Slide 1 — Title
Title, team, guide. Nothing to defend here except pronouncing "Cohen's kappa" correctly later.

### Slide 2 — Outline
A map of the 17 sections. Mention in passing that the structure follows the standard thesis order
(Intro → Problem → Objectives → Contribution → Literature → Gap → Dataset → Method → Design → Algorithms →
Metrics → Results → Limitations → Conclusion → Future work → Publication → References) — this is the
expected institutional format, not something you chose arbitrarily.

### Slide 3 — Introduction
**What it says:** PSG background, AASM 5-stage scoring, why manual scoring is slow (1–2 h/night) and
inconsistent (experts agree only ~80% overall, 25–45% on N1), and the three trends (model compression,
local LLMs, offline edge deployment) that motivate the project.
**Why it matters:** this sets up *every* later design choice — compression → small student model; local LLM
risk → the verifier; offline → the whole edge chapter.
**Quick check:** *"Why 30-second epochs?"* → It's the AASM clinical scoring standard; using it keeps our
hypnogram directly comparable to clinically scored ones.

### Slide 4 — Problem Definition
**What it says:** five concrete gaps — subject leakage (κ 0.6663 reported vs 0.5066 true), naive KD hurting
accuracy, bare hypnograms with no trust signal, unverified SLM hallucination, and a fragile unattended agent.
**Why it matters:** each gap maps 1:1 to a later chapter — this is literally the "problem → solution" spine
of the whole report. If asked "what's your contribution," you can answer by just naming which gap each
contribution (N1–N6, Slide 6) closes.

### Slide 5 — Objectives
**What it says:** O1–O5, each with a measurable target and the achieved number (e.g. O2: <200K params, κ ≥
teacher → got 175,599 params, κ 0.756 vs teacher's 0.732).
**Why it matters:** these were **pre-registered** — written down *before* the final test run, so the
"achieved" column isn't cherry-picked after seeing results. This is important to say if asked about rigor.
**Quick check:** *"What does 'pre-registered' mean and why does it matter?"* → The target and the exact test
to run were fixed in writing before the locked test set was touched, so the result can't be adjusted after
the fact to look better — it either met the bar or it didn't.

### Slide 6 — Research Contribution
**What it says:** N1–N6, the six novelties. N1 pre-release verifier, N2 uncertainty enforced in wording, N3
the ensemble-distillation mechanism, N4 zero-init gated fusion, N5 the crash-safe agent, N6 the label-shift
discovery.
**Why it matters:** this is your "elevator pitch" slide — if your teacher asks "what's actually new here,"
point here. N1 and N3 are the two strongest, most defensible claims (verifier blocks before release; the
teacher-must-exceed-student rule is a real, provable mechanism, not just an empirical tweak).

### Slide 7 — Literature Review Summary (table)
**What it says:** comparison against 9 published systems on params, accuracy/F1/κ, whether they write a
report, and how (if at all) the output is checked.
**Inference to draw out loud:** every single row in the "Output Check" column for the non-LLM staging models
says "None" (they're pure classifiers with no language-generation risk), and every LLM-reporting row says
"Physicians after" or "Post-hoc audit" — **none check before release**. That's the one-sentence gap this
whole project exploits: nobody blocks a flawed report *before* a human reads it except us.

### Slide 8 — Comparison with Existing Work
**What it says / figure:** the scorecard image breaks the comparison into 7 "ahead," 3 "level," 8 "behind"
categories (from Table 4.11 in the report).
**Inference:** don't let "8 behind > 7 ahead" sound alarming — read the *categories*. "Behind" items are
mostly things that are **out of scope by design** (external cohort, clinician rating, FDA clearance, measured
Pi hardware) rather than failures of the method itself. The "ahead" items are all about the verification
mechanism, which is the actual contribution. Be ready to say this explicitly: *"we are behind on validation
breadth, not on the core claim."*
**Quick check:** *"Isn't 83.0% below MultiChannelSleepNet's 85.0%?"* → Yes, by ~2 points, but that model is
13M parameters — 74× larger. We state the honest comparison and never claim state-of-the-art, only
"competitive at a fraction of the size."

### Slide 9 — Research Gap
**What it says:** the same five gaps from Slide 4, now paired with the specific Phase I fix for each (G1
subject-ID split + relabel, G2 teacher-ceiling rule + ensemble, G3 Schema 1.3 packet, G4 grammar + verifier,
G5 hash-pinned agent).
**Note the last line:** it explicitly says Docker/ARM64 simulation **was** run in Phase I (3 Nov) — so this
slide was updated *after* your friend's test came back. Good to know in case she asks "when was this done."

### Slide 10 — Dataset Description
**What it says:** Sleep-EDFx, 197 nights / 100 subjects, SC (healthy ageing) + ST (mild sleep-onset
difficulty) cohorts, class imbalance (N2 37.4% down to N3 8.2%), the five-way split table, and the label fix.
**Why the splits table matters:** note that **5-fold CV (168 nights)** and **B1 (153 nights, SC only)** are
*different* evaluation protocols from the main 69/16/15 split — they reuse train+val nights in new
combinations for a specific purpose (ablation, standard-benchmark comparison) but never touch the locked test
set. This is a common confusion point — be ready to explain why there are five different "splits."

### Slide 11 — Dataset Split & Leakage Audit
**Figures:** subject-level split diagram + the exposure-gradient bar chart.
**Inference from the exposure gradient:** κ falls from 0.806 (same recording, i.e. the training data itself)
→ 0.680 (same person, different night — "leaked" because the person was still seen in training) → 0.5066
(genuinely unseen person). **The 0.68→0.51 drop is the real finding** — it proves the *original* 0.6663 number
was inflated by subject leakage, not just recording-level leakage. N3 F1 hitting exactly 0.000 on the unseen
subject is the most damning single number on this slide — the old model had *never actually learned* N3, it
had just memorized who gets which labels.

### Slide 12 — Methodology
**What it says:** the five pipeline stages (staging → packet → SLM → verifier → delivery) plus the agent
wrapping all of them, and the four terminal outcomes.
**Key phrase to repeat if asked "what does the agent actually decide":** *"the agent decides workflow, never
clinical content."* It can send a night to review or retry a step, but it can never alter a number or write a
sentence — all content comes from the model + the frozen verifier.

### Slide 13 — System Overview
**Figure:** the five-stage + agent overview diagram (Fig. 3.1).
**Inference:** this is the figure to point to for literally any "walk me through the pipeline" question —
narrate it left to right: raw signal → student model → evidence packet (facts) → SLM (words) → verifier
(check) → one page. Emphasize the facts/words separation: the SLM never touches raw data, only the packet's
already-computed numbers.

### Slide 14 — Design / Architecture
**What it says:** the full parameter table for N6adv (Table 3.2) and the Schema 1.3 / verifier summary
(Table 3.3).
**This is the table to memorize** for any "how many parameters does X contribute" question. See the
dedicated architecture section below for the full reasoning behind each row.

### Slide 15 — Receptive Field & Model Size
**Figures:** receptive-field comparison (0.25 s vs 8.75 s) and κ-vs-parameters scatter.
**Inference from receptive field:** 0.25 s is far shorter than a K-complex (~0.5–1.5 s) or a slow wave
(~0.5–2 s period) — the *old* encoder was architecturally incapable of seeing the waveforms that actually
define N3 and spindle-related stages, regardless of how much it was trained. Widening it to 8.75 s is a
structural fix, not a tuning fix — that's why it moved raw EEG's prediction contribution from ~0% to 71%.
**Inference from κ-vs-parameters:** v2 sits above and to the left of its own teacher and of v1 on the same
plot — smaller and more accurate, which is the headline efficiency claim.

### Slide 16 — Algorithms
**What it says:** Algorithm 1 (ensemble KD loss), Algorithm 2 (verified report generation state machine),
Algorithm 3 (Path B two-pass staging + edge bandwidth model).
**One number worth remembering:** the Pi 5 bandwidth ceiling, **7.43 tokens/s**, comes from
`tokens/s ≈ effective_bandwidth / bytes_read_per_token` — i.e. decoding a quantized LLM on CPU is a memory
problem, not a compute problem. Be ready to explain *why* that formula is true: every generated token has to
read the entire model's weights from memory once (no reuse across tokens), so speed is capped by how fast RAM
can feed the CPU, not by how many FLOPs the CPU can do.

### Slide 17 — Why Single-Teacher Distillation Failed
**Figures:** KL-divergence-vs-training-progress, and penalty-vs-teacher-quality.
**Inference:** figure 4.4 (penalty vs teacher quality) is the key one — it shows the KD penalty (how much
worse the distilled student is than the hard-label-only control) *shrinking* as teacher quality rises, and
crossing from "distillation hurts" to "distillation helps" once the teacher passes the student's own
undistilled performance. That crossing point *is* the "teacher must exceed student" rule stated as data, not
assertion.
**Quick check:** *"Why can't you just lower α or T to fix the single-teacher case?"* → Because the problem
isn't the *strength* of the soft-target signal, it's that the signal itself points in a wrong direction (the
teacher's own errors). No amount of down-weighting a bad direction removes the contamination; you need a
teacher that doesn't have the problem in the first place — which is why the fix is "get a better teacher"
(ensemble), not "tune α/T."

### Slide 18 — Evaluation Metrics
**What it says:** staging metrics (κ primary), calibration metrics (ECE, night-confidence correlation),
SLM safety metrics (fidelity, flawed-release rate + Clopper–Pearson bound), agent/edge metrics (fault matrix,
byte-identical reruns, latency/memory).
**This slide is the natural home for "why kappa not accuracy" — see the dedicated Q&A below**, but also note
the **Clopper–Pearson lower bound** here: when you only have 29 test nights and 28 pass, the honest statement
isn't "96.5% success," it's "we can only be 95% confident the *true* success rate is at least 82.2%." Knowing
this formula exists and why it's needed (small-sample binomial proportions are noisy) is a good rigor point.

### Slide 19 — Result Analysis (table)
**What it says:** the master results table across every stage — leaked baseline → single-teacher KD penalty →
held-out test (teacher, v1, v2) → B1. Read top to bottom as the project's *narrative arc* in one table.
**Inference:** the single most important row-to-row comparison is "ENSmc Teacher 0.732" vs "v2 N6adv 0.756" —
**the student beats its own teacher**. This is unusual and worth flagging explicitly if not asked, because
it's counter-intuitive (students are supposed to approach, not exceed, the teacher) — the explanation is that
v2 has information (EOG, TF, local context) the teacher never had, on top of what it learned from the teacher.

### Slide 20 — Results: Staging on Held-Out Test
**Figures:** confusion matrix, per-class F1 bar chart (v1 vs v2).
**Inference from the confusion matrix:** look at where the off-diagonal mass concentrates — almost certainly
N1↔Wake and N1↔N2 confusion (N1 is a short, ambiguous transitional stage; even human experts only agree
25–45% of the time on it). This is *expected* and *literature-consistent*, not a flaw unique to our model —
say so if asked why N1 looks weak.
**Inference from per-class F1:** v2's bars should be at or above v1 on **every** class — that uniform
improvement (not just average improvement) is stronger evidence than the single headline κ number, because it
rules out "v2 only improved by sacrificing one class to boost another."

### Slide 21 — Results: Ablation & Robustness
**Figures:** CV ablation bar chart (base → +TF/local → +EOG), missing-channel robustness chart.
**Inference:** this is the slide that answers "how much did EOG specifically contribute" cleanly — the CV
ablation isolates it as **+0.0179 κ** on a controlled, like-for-like basis (same labels, same teacher, same
everything except the architecture arm), separate from the held-out test's combined +0.038 (which also
includes label-correction and recipe effects). **Always use the CV ablation number, not the held-out-test
delta, when asked specifically "what does EOG add."**
**Inference from missing-channel chart:** κ only drops 0.012 (0.756→0.744) when EOG is entirely removed at
inference, and it's still *above* v1's 0.718 — proving the gate mechanism (Slide 14/15) actually works: the
model doesn't collapse without a modality it was trained to expect.

### Slide 22 — Results: Standard Benchmark & YASA Baseline
**Figures:** B1 per-fold κ, v2-vs-YASA comparison.
**Inference from B1 per-fold:** if the bars are roughly level across all 10 folds (not one fold wildly
higher), that's evidence the result isn't a lucky split — it generalizes across different subject groupings.
**Inference from YASA:** we beat an established *open-source offline tool* on N1 F1 by a wide margin (54.5 vs
30.2) — but flag the caveat yourself: our model is *in-domain* (trained on data very similar to this test set)
while YASA is a generic pretrained tool not tuned to this exact cohort, so this isn't a fully fair head-to-head
— it shows we're competitive with a real deployed tool, not that we're unconditionally better.

### Slide 23 — Results: Evidence Packet
**Figure:** the robust/fragile/unreliable metric-reliability chart.
**Inference:** only **3 of 24** derived metrics (likely total sleep time, sleep efficiency, and one other) are
safe to state as plain facts; 13 are "fragile" (state with a hedge, e.g. "approximately"), 8 are "unreliable"
(never state as fact). This 3/13/8 split is itself a result — it shows *most* commonly reported sleep metrics
(e.g. time to fall asleep, counts of awakenings) are measured from data too noisy to assert plainly, which is
a finding worth citing on its own, separate from the staging accuracy numbers.

### Slide 24 — Results: Report Model Selection
**Figure:** verified-reports-by-model bar chart with confidence intervals.
**Inference:** note Qwen-1.5B only reaches 10/31 — it's not that the *idea* doesn't work, it's that 1.5B
parameters isn't enough *capacity* to hold the whole reporting contract reliably. This tells you model size
has a floor for this task, which justifies picking the 3B model over cheaper options despite the edge-latency
cost.

### Slide 25 — Results: Verifier Ablation & the Blocked Night
**Figures:** ablation bar chart (gate on/off/off+grammar) + a screenshot of the actual fallback page shown to
the blocked night's reader.
**Inference, the single best story on the whole deck:** for the untrained base model, the gate takes flawed
releases from **31/31 → 0/31**. For the *fine-tuned* v2 model, the gate makes little difference in the
ablation table (already near-zero flaws) but **it still caught the one real mistake on the locked test set**
(SC4611E0, an invented −1.0 value). That's the cleanest demonstration you have: *fine-tuning makes the model
usually right; the verifier is what makes the system always safe, even when fine-tuning isn't perfect.*

### Slide 26 — Results: Agentic Workflow on Real Recordings
**Figures:** Path B scorecard (v1 vs v2), line-noise/quality chart.
**Inference from the line-noise chart:** κ collapsing from 0.775 to 0.340 under 49 Hz noise is a strong
argument for *why the Q03 signal-quality gate exists at all* — the verifier checks claims against the packet,
but can't tell if the **packet itself** is wrong because the input signal was bad. That's a layered-defense
argument: staging quality checks catch what the verifier structurally cannot.

### Slide 27 — Results: Edge Simulation
**What it says:** L0–L5 ladder with the 3 Nov Docker/ARM64 results (this is the part covered in detail
earlier in this conversation — see the "Edge Simulation" section below for the plain-language explainer you
can reuse almost verbatim).
**One correction to have ready:** the headline numbers (151 s, 5.16 GB) are **x86 container timings**, not Pi
timings — if asked "so it takes 151 seconds on the Pi?" the answer is **no**, that's explicitly flagged as an
x86 desktop CPU number; the Pi estimate (6–9 min) is a separate, analytical calculation, not yet measured.

### Slide 28 — Demo: Somnia Console
Screenshots of the actual running UI. If you can do a live demo, this is where — show the Overview page
(headline numbers read live from result files, not hard-coded) and a Recording detail page (reliability tags
on every metric). If asked "is this live or a screenshot," be honest: it's a local console
(`python -m ui`), not a hosted web app — it runs entirely offline on the machine it's demoed on.

### Slide 29 — Limitations
**What it says:** dataset scope, clinical scope, edge-hardware scope — three honest buckets. Already covered
in detail in the "is the project fully ready" discussion. Read it exactly as written; don't soften it further
— an examiner respects a limitations slide that's specific (named numbers, named gaps) far more than a vague
one.

### Slide 30 — Conclusion
**What it says:** the six-stage summary plus the one "headline sentence" the whole project boils down to:
*"a 0.18M-parameter staging model competitive with models 5–74× larger, joined to a local 3B report model
whose output is released only after a deterministic check — with no flawed report released on held-out
data."* **Memorize this exact sentence.** It's calibrated precisely (not overclaiming "best," not
underselling "just a prototype") and answers "what did you actually build" in one breath.

### Slide 31 — Phase II Plan
Seven prioritized milestones (P1–P7), already covered above in "is it fully ready." P1 (real Pi hardware) is
clearly the single most important remaining item — lead with that if asked "what's next."

### Slide 32 — Publication Plan
Two submission tracks: a full journal paper (JBHI/TBME) and a shorter conference paper (EMBC/ICASSP) focused
specifically on the distillation-mechanism finding. If asked "why two papers instead of one," the honest
answer: the full system (6 stages) is a journal-length contribution; the teacher-ceiling mechanism alone
(Slide 17) is a focused, self-contained enough finding to stand as its own shorter paper.

### Slides 33–34 — References
Standard APA bibliography. One honest caveat is printed directly on slide 34 — several entries were taken
from the project's own related-work survey and still need checking against the original papers before
real submission. If asked, say exactly that; don't pretend every citation has been hand-verified.

---

## Part 2 — Architecture & signal processing deep dive

This is the section to actually study if she probes "how are signals extracted and processed." Walk it in
this order: **raw signal → features → encoders → fusion → sequence model → output.**

### Step 0 — What comes in
Each night is split into **30-second epochs** (the AASM clinical standard). Per epoch:
- **EEG (Fpz-Cz lead), 100 Hz → 3,000 raw samples**
- **EOG (horizontal), 100 Hz → 3,000 raw samples** (v2 only; v1 used EEG alone)

### Step 1 — Two parallel views of the EEG are built
1. **Raw waveform** (3,000 samples) is fed straight into a learned CNN encoder.
2. **34 hand-engineered spectral features** are computed separately and fed into a simple linear projection.

**Why build both?** Hand-crafted features are compact, interpretable, stable summaries (e.g. "how much power
is in the delta band") but they *throw away* shape information. The raw-waveform CNN can learn to recognise
the actual *morphology* of a K-complex or spindle, which no summary statistic captures. They're fused later so
the model gets both views.

**The 34 features, precisely:**
| Count | Source | What |
|---|---|---|
| 24 | Discrete Wavelet Transform (Daubechies-4, 5 decomposition levels, 6 sub-bands) | energy, log-energy, Shannon entropy, variance — 4 stats × 6 bands |
| 10 | Short-Time Fourier Transform (2 s Hann window, 1 s hop) | 5 relative band powers (delta/theta/alpha/beta/gamma), spectral entropy, 95% spectral-edge frequency, and 3 band ratios (delta/beta, theta/alpha, (delta+theta)/(alpha+beta)) |

**Why both DWT and STFT, not just one?** DWT is good at **localizing transient events in time** (a spindle is
a short burst — wavelets are built for exactly this). STFT/band-power is good at summarizing **steady
frequency content** (which band dominates overall — useful since N3 is delta-dominant, wake is often
alpha-dominant). They're complementary resolutions of the same signal, which is a standard combination in EEG
feature engineering, not something invented for this project.

### Step 2 — The raw-EEG CNN encoder (multi-scale, "v1 backbone")
Two parallel convolutional branches on the *same* 3,000-sample window:
- **Fine branch:** small kernel (k=50), small stride (s=6) — short-duration, higher-frequency detail.
- **Coarse branch:** large kernel (k=200), large stride (s=25) — longer-duration, lower-frequency structure
  (e.g. slow waves).

Each branch is pooled down to a small number of time "tokens," concatenated, and combined through an
**attention-pooling** step (the model learns which of the 4 pooled tokens matter most for this epoch) into one
64-dimensional vector per epoch.

**Why two branches instead of one?** A single kernel size is a trade-off: wide enough to see slow waves means
too blurry for fast transients, and vice-versa. Two branches at two scales sidestep the trade-off entirely —
this is the fix that raised the network's **receptive field from 0.25 s to 8.75 s**, wide enough to contain a
full spindle, K-complex, or slow wave.

### Step 3 — EOG encoder (v2 only)
Two more parallel branches, same idea but tuned to eye-movement physiology:
- **Saccade branch** (k=50, s=6): catches **rapid eye movements** — the single strongest physiological REM cue.
- **Slow-eye branch** (k=400, s=50, a much wider kernel): catches **slow rolling eye movements** typical of
  drowsy wake / sleep onset.

Plus **8 hand-crafted EOG features**: four band log-powers (0.3–1, 1–2, 2–4, 4–8 Hz — slowest to fastest
eye-movement frequency content), log-variance, a direct **saccade-rate** count (fraction of samples where the
signal's derivative exceeds a threshold — literally counting rapid eye flicks), mean absolute derivative, and
the **EOG–EEG correlation** (catches cases where eye-movement artifact has bled into the EEG channel — a
signal-quality cue, not just a sleep-stage cue).

### Step 4 — The time-frequency (TF) branch
A full 2D spectrogram is computed from the raw EEG (2 s window, 1 s hop, 0.5–30 Hz in 0.5 Hz bins → 60
frequency bins × ~29 time frames), normalized per frequency, and passed through two 1D convolutions over time
(frequency treated as channels), then mean+max pooled into a 32-d vector.

**How is this different from the 10 STFT scalar features above?** Those 10 are single numbers per epoch
(already reduced/summarized). The TF branch instead keeps the **full 2D time-frequency image** and lets a CNN
learn arbitrary patterns across both axes — e.g. "power rising in the alpha band specifically in the second
half of the epoch" — a pattern no single scalar feature could represent. One is engineered and compact; the
other is learned and expressive.

### Step 5 — Gated fusion (the key architectural idea, N4)
EOG and TF branches don't just get concatenated in — they're added through a **gate**:

```
h ← h + sigmoid(gate(h, branch_output)) × (zero_init_projection(branch_output))
```

Two deliberate choices:
- The **projection weight starts at exactly zero.** So at the very first training step, regardless of what the
  gate says, the added term is mathematically zero — **the untrained v2 model computes exactly v1's function.**
- The **gate's bias starts at −4** (sigmoid(−4) ≈ 1.8%), so even once the projection weight starts moving
  during training, the new branch's influence is deliberately small at first and only grows if it actually
  helps reduce the loss.

**Why not just bias = −∞ (fully closed)?** Because a sigmoid at exactly 0 has zero gradient — the gate could
never learn to open. −4 leaves a small, trainable crack.

**Why this matters for the viva:** it's a *safe, incremental* architecture search — you're never risking the
new branch destabilizing a model that already works, and you get proof (the ablation table) that each addition
is net-positive before you'd ever ship it.

After EOG/TF are added, **spectral features are fused in** (concat → Linear(128→64) → LayerNorm → GELU), then
a **local-context depthwise convolution** (kernel 7, so ±3 neighbouring epochs) refines each epoch's
representation using its immediate neighbours — also zero-initialized, same "start as identity" logic.

### Step 6 — The sequence model
All 256 epochs in a window (~128 minutes of sleep) get **sinusoidal positional encoding** and go through a
**2-layer, pre-norm Transformer encoder** (64-dim, 2 attention heads, feed-forward 256). A final linear layer
(64→5) outputs one of {Wake, N1, N2, N3, REM} per epoch.

**Why look at the whole 256-epoch window, not just one epoch at a time?** Sleep has structure across time —
cycles are roughly 90 minutes, and certain stage transitions are much more likely than others. Letting the
Transformer attend across ~2 hours lets it implicitly use that context (e.g. "we just had a long N3 block, a
transition to REM or lighter sleep is more likely next") without hand-coding any sleep-cycle rule.

### Step 7 — The auxiliary head
A small extra head (Linear 64→1) is trained, during training only, to predict whether a **stage transition**
occurs between consecutive epochs, with a small loss weight (0.1). This is multi-task learning: it forces the
shared representation to be sensitive to *boundaries*, which are exactly where staging errors concentrate (a
drowsy-wake→N1 transition is genuinely ambiguous even for humans).

### The full parameter ledger (memorize this table)
| Component | Params |
|---|---:|
| v1 backbone (multi-scale CNN + spectral proj + fusion + Transformer + head) | 139,606 |
| EOG encoder | 6,768 |
| EOG gated projection | 8,320 |
| Time-frequency encoder | 7,848 |
| TF gated projection | 8,320 |
| Local context + auxiliary head | 4,737 |
| **Total (N6adv / v2)** | **175,599** |

---

## Part 3 — Your four pasted questions, answered in full

### Q1. Why do we use kappa (κ) instead of accuracy?

**Short answer:** because the classes are heavily imbalanced, and accuracy rewards a model for simply
predicting the majority class, while κ corrects for chance-level agreement.

**Give the number:** N2 alone is 37.4% of all epochs, Wake 29.5%, down to N3 at only 8.2%. A model that
**always predicts N2** would score ~37% accuracy while being clinically useless — it never identifies N3, N1,
or REM at all. κ accounts for exactly this: it measures agreement **beyond what chance/majority-class
guessing would already give you**, so a trivial always-N2 model scores κ ≈ 0, not 37%.

**Second reason — comparability:** κ is the standard metric the sleep-staging literature already uses (every
paper in our Table 2.1 reports κ), and it's also how **inter-human-scorer agreement is reported** (κ ≈
0.75–0.79 between two expert scorers, 25–45% specifically on N1). Reporting κ lets us directly compare our
model (a) to published models on the same scale and (b) to the human expert ceiling, which accuracy alone
can't do.

**If pushed further — "don't you also report accuracy and F1?"** → Yes, both, but as **secondary** metrics.
κ is primary because it's the one metric that's both chance-corrected and literature-standard; macro-F1
(unweighted average across the 5 classes) is reported alongside it because it penalizes doing badly on a rare
class (like N3) just as much as doing badly on a common one, which is also useful given the imbalance.

### Q2. Why didn't we use EOG in the distillation, even though the ensemble teacher had EOG?

**Correct the premise first, politely:** actually, the **ensemble teacher does *not* have EOG** — it's an
**EEG-only** model (see Slide 19: "ENSmc Teacher (EEG) | 412,311"). EOG is added **only in the v2 student**,
on top of a backbone that was distilled from the EEG-only teacher. So the real question to answer is: *"why
build the student's EOG branch on top of an EEG-only teacher, instead of first building a new EOG-aware
teacher?"*

**The honest, defensible reasons:**

1. **It's a deliberate, controlled ablation design.** The teacher ensemble (3 students, 412,311 params) was
   already built and validated during Stage 1 (the distillation-mechanism work) *before* EOG was introduced in
   Stage 2. If we'd built a brand-new EOG-aware teacher at the same time as adding EOG to the student, we'd no
   longer be able to cleanly answer "does EOG, as an architectural addition to the student, help?" — the
   teacher's own EOG-derived soft targets would already bake that signal in, confounding the measurement. By
   keeping the teacher EEG-only and adding EOG only on the student side, the CV ablation (base → +TF/local →
   +EOG) isolates EOG's contribution precisely: **+0.0179 κ, like-for-like.**
2. **The student isn't just copying the teacher.** Training combines two signals: the **KD loss** (match the
   EEG-only teacher's softened output) and the **hard-label cross-entropy loss** (match the true expert
   label). The EOG branch is driven mainly by the second term — it has to earn its place by actually reducing
   error against the *true* labels, not by mimicking an EEG-only teacher's blind spot. The zero-init gate
   (Part 2, Step 5) means the branch starts fully closed and only opens during training if doing so reduces
   the total loss — so by construction, EOG's contribution in the final model is "whatever genuinely helped,"
   not "whatever the teacher happened to know."
3. **Cost.** Training a *second* three-model EOG-aware ensemble teacher (3 more full training runs) to then
   distill from, just to add one new input modality to the student, is a significant extra compute and
   engineering cost for what's explicitly flagged as future work, not something skipped carelessly.

**If she asks "so would an EOG-aware teacher do even better?"** → Possibly, yes — that's a reasonable next
step (not currently planned in the Phase II list, but a fair thing to float as a natural extension: "a
richer teacher that also sees EOG might push the student further, though at the cost of losing the clean
ablation we currently have").

### Q3. How many parameters does our model have?
Already on the slides (175,599 for the student; 2.02 GB / ~3B for the report SLM) — you said you can skip
this one, but for completeness the full breakdown is in Part 2's parameter ledger table above if she asks for
the component-by-component split rather than just the total.

### Q4. What improvements did you make in v2 (N6adv) to increase κ above v1 (N4kd)?

Give this as **layered, measured increments**, not a vague list — this is the strongest way to answer it,
because every number is independently measured (CV ablation, Table 4.3):

| Step | What changed | κ (5-fold CV, same labels) | Δ |
|---|---|---:|---:|
| `base` | v1 architecture exactly | 0.7584 | — |
| `full_eeg` | + time-frequency branch + local context + the new training recipe | 0.7653 | +0.0069 |
| `full` (= v2) | + gated EOG branch | 0.7763 | +0.0179 further |

So, concretely, four architectural additions and one training-recipe upgrade:

1. **Time-frequency (STFT-based) learned branch** — a second, flexible spectral view beyond the 34 fixed
   hand-crafted features (Part 2, Step 4).
2. **Local temporal context** — a small convolution over ±3 neighbouring epochs, capturing that adjacent
   epochs' stages are correlated (sleep doesn't change state instantly).
3. **Gated EOG branch** — the single largest individual gain (+0.0179 κ alone), adding eye-movement
   information that's especially informative for REM and N1 (both partly *defined* by eye-movement behaviour).
4. **Auxiliary transition-prediction head** — a secondary training signal (weight 0.1) that sharpens the
   model's sensitivity near stage-transition boundaries.
5. **Training-recipe upgrades** applied alongside the architecture: warm-start from the v1 backbone (frozen
   for the first 10 epochs, then fine-tuned at 0.1× the learning rate, instead of training from scratch),
   boundary-aware label smoothing, a quality mask that excludes flat/stuck epochs from the loss, exponential
   moving-average (EMA) weight averaging, and **modality dropout** (EOG and TF each randomly switched off for
   25% of training sequences) — this last one is *why* the model still works at κ 0.744 when EOG is missing at
   inference (Slide 21), because it was explicitly trained to cope without it.

**One more factor, separate from the architecture, worth mentioning if she asks "is that the whole gap?":**
the **held-out test** delta (v1 0.718 → v2 0.756, +0.038) is *larger* than the CV ablation's architectural
delta (+0.0248 total) because the test comparison also includes the **corrected-label fix** (worth ~3 accuracy
points project-wide) as a confound. **Always clarify which number you're quoting**: the CV ablation isolates
*just the architecture*; the held-out test number is architecture + labels + recipe combined.

---

## Part 4 — General viva question bank, by topic

### Dataset & evaluation design
- *Why split by subject and not by recording?* — Nearly every subject has two consecutive nights; splitting
  by recording would put the same person's second night in test after their first night was in training,
  letting the model "recognise the person" rather than generalise — exactly the leakage the inherited model had.
- *How is the test set "locked," and why does that matter?* — It's read only for pre-registered, logged
  evaluations (currently read 7 times across the whole model line, all logged in `TEST_ACCESS_LOG.md`), never
  for tuning or model selection. Any future architecture needs a *new* test set (Phase II, P7) because this one
  has now been "seen" too many times to trust for ranking new models.
- *Why does B1 (0.769–0.775) differ from the held-out test (0.756)?* — Different protocol: B1 is SC-cohort
  only (no harder ST data), 10-fold CV (so ~90% of data trains each fold vs. the fixed split's ~70%), and
  trained **from scratch** (no distillation/warm-start) to match the standard published protocol exactly, so
  it's directly comparable to other papers' numbers — it is not meant to replace the main held-out result.
- *What's the difference between the 5-fold CV, B1, and the held-out test — isn't that three test sets?* —
  Three *different purposes*: the held-out test is the one locked, final number used for the pre-registered
  objectives; the 5-fold CV is specifically for *architecture ablation* (which part adds how much); B1 exists
  purely to let us compare against the literature on their exact protocol. None of them touch each other's data
  in a way that leaks.

### Distillation & training
- *What is Cohen's kappa, formally?* — κ = (observed agreement − expected chance agreement) / (1 − expected
  chance agreement). It ranges −1 to 1; 0 means no better than chance, 1 is perfect agreement.
- *Why T² in the KD loss?* — Dividing logits by temperature T before softmax flattens the distribution, which
  shrinks the gradient magnitude of the KL term by roughly 1/T²; multiplying back by T² keeps the soft-target
  gradient on a comparable scale to the hard-label gradient so one term doesn't get drowned out. (We
  unit-test this in code specifically because it's easy to silently get wrong.)
- *What does "warm start" mean and why do it?* — v2 initialises its shared backbone from the already-trained
  v1 weights (frozen for the first 10 epochs, then fine-tuned slowly) instead of training from random
  initialisation — it's faster to converge and less likely to destabilise a part of the network that already
  works.
- *What's EMA (exponential moving average) and why use it for evaluation?* — A slowly-updated running average
  of the model's weights (decay 0.999) kept alongside the actively-trained weights; evaluating with the EMA
  weights instead of the raw weights tends to generalise slightly better and reduces the effect of noisy
  late-training updates.

### Calibration & the evidence packet
- *What is temperature scaling, and what does Tc = 1.0021 mean?* — A single learned scalar that divides the
  logits before the final softmax, fit on validation data to make confidence match real accuracy (e.g. "80%
  confident" predictions should be right ~80% of the time). Tc very close to 1.0 means the model was **already
  well-calibrated** before any correction — a side-effect of training with EMA and label smoothing.
- *What is ECE?* — Expected Calibration Error: bin predictions by confidence (15 bins), compare each bin's
  average confidence to its actual accuracy, and average the gap, weighted by bin size.
- *What does "night confidence" mean and how is it computed without ground truth?* — The mean prediction
  entropy across all epochs in a night (higher entropy = more uncertain predictions on average). It needs no
  ground truth because entropy is computed purely from the model's own output probabilities. It's validated
  by checking it *correlates* with real expert agreement on the test set (r = −0.637) — high entropy nights
  really do score worse.
- *How is a metric classed robust/fragile/unreliable?* — By measuring, on validation data, the error between
  the metric computed from predicted stages vs. from the true expert stages, and thresholding that error.

### SLM, grammar, and the verifier
- *What's the difference between the grammar and the verifier?* — The GBNF grammar constrains *decoding* —
  it guarantees the model can only output syntactically valid JSON in the expected schema (it cannot produce
  malformed output), but it says nothing about whether the **values** are true. The verifier runs *after*
  generation and checks the content against the evidence packet.
- *Why two verifier layers instead of one?* — Layer 1 (structure) is packet-independent — closed-world JSON
  checks like valid claim types and no duplicates; it doesn't need to know anything about this specific night.
  Layer 2 (policy) reads the packet and does the semantic checks (does this cited number actually match, is
  a fragile value hedged). Separating them means a structurally broken claim never gets passed into the more
  expensive/complex policy checks, and the two kinds of bugs are easier to debug independently.
- *Why temperature 0, seed 0?* — Determinism — the same input always produces the same output, which is
  required for the byte-identical-rerun guarantee and for reproducibility of every evaluated number.
- *Why not just use a bigger/cloud model to avoid all this?* — Defeats the offline/privacy goal (Slide 3's
  third trend), and a cloud model still needs the same verification — hallucination isn't a *small-model*
  problem, it's an LLM problem; the verifier is required regardless of model size.

### Agent & edge
- *What does "the agent decides workflow, never clinical content" actually mean in code terms?* — The agent's
  state-machine functions only ever return a `Decision` naming the *next step*; none of them can write to a
  report's text or numeric fields — those come only from the staging model, the packet builder, and the SLM
  (itself gated by the verifier).
- *How does crash-safe resume work?* — Every step writes a completion marker to its own directory before the
  next step starts; on restart, finished steps are replayed from their markers (not recomputed), and the one
  interrupted step is wiped and re-run from scratch — so a crash mid-step never leaves a half-written result.
- *What's Path A vs Path B?* — Path A starts from an already-built evidence packet (used for controlled
  evaluation). Path B starts from a raw, uncropped EDF file with no expert hypnogram — the model has to find
  the sleep period itself (two-pass staging: stage the whole file once to estimate where sleep starts/ends,
  then re-stage just that window).
- *(Edge/Docker questions are already covered in detail in the simulation explainer from earlier in this
  session — see the slide-27 notes above for the condensed version.)*

### Comparative claims
- *Are you claiming to beat all existing sleep staging models?* — No — explicitly "competitive with," not
  "better than," models 5–74× larger; we are behind the single best published model (MultiChannelSleepNet,
  85.0%/κ0.79) on raw accuracy, ahead on parameter efficiency and uniquely ahead on pre-release verification.
- *Is 28/29 or 31/31 "good enough" statistically?* — With that few trials, raw percentage is misleading; the
  Clopper–Pearson 95% lower bound (82.2% for 28/29) is the honest, defensible number — report that, not "97%."

---

## Part 5 — "Gotcha" questions (things that look inconsistent until explained)

These are the sharpest kind of question — where two slides look like they contradict each other. Knowing the
resolution in advance stops you from being caught off guard.

1. **"Your student beats its own teacher — doesn't that mean distillation did nothing, or even hurt?"**
   No — it means the student had *extra information the teacher never had* (EOG, TF branch, local context),
   on top of what it gained from distillation. The controlled ablation shows the gain from each piece
   separately; it's addition of capability, not disproof of KD. (Also tie back to Q2 above if this follow-up
   comes right after it.)

2. **"You say 'no flawed report released,' but 1 report was blocked and the reader got nothing — isn't a
   fallback also a kind of failure?"**
   It's a *designed, safe* outcome, not a silent failure — the reader sees an honest "packet numbers only"
   summary instead of a wrong claim. The system's promise is never "every report is a full report," it's
   "every report shown is true." A fallback keeps that promise; a flawed release would break it.

3. **"The Pi 5 numbers are just estimates — so none of the edge chapter is real?"**
   Push back gently: the *analytical* budget is an estimate, but the Docker/ARM64 run (3 Nov) is a **real
   measurement** — offline execution confirmed, 31/31 byte-identical reports, real memory numbers, and real
   (if emulated) ARM64 staging parity on 8 nights. What's *not yet real* is specifically speed/heat/power on
   physical Pi silicon — be precise about which claim is measured vs. estimated.

4. **"Test set read 7 times — isn't that basically tuning on the test set, just spread out?"**
   No, because nothing was *changed* in response to a test result and then re-measured on that same test —
   each read corresponds to a different model generation's **single, final, pre-registered** evaluation, and
   each read is logged specifically so later analysis can check this. The risk you're describing (iterative
   tuning against test feedback) is exactly what logging + pre-registration prevents.

5. **"Only 8 ARM64 nights, and no Sleep Telemetry nights — isn't that too small a sample to claim ARM
   parity?"**
   Fair, and the slide says "partly done" for exactly this reason — it's stated as "met on a subset," not
   "met." The honest framing is: the architecture-level code path is proven portable; full-cohort and
   report-model ARM parity are explicitly still open (see Phase II, P1).

6. **"If the report model is 2.02 GB and the staging model is 0.70 MB, isn't calling this an 'edge/compact'
   system misleading — the SLM is 99.9% of the footprint?"**
   The compactness claim is specifically about the **staging model** (the actual research contribution on
   accuracy-per-parameter); the SLM's size is disclosed plainly (Slide 29) as the dominant footprint and
   explicitly flagged as a Phase II target (P6: train a 1–1.5B SLM to roughly halve this).

---

## How to use this during the viva

- Lead every answer with the **number**, then the **reason**. Examiners respond well to "28 of 29, 95% lower
  bound 82.2%, because..." rather than starting with the reasoning and arriving at the number.
- If a question targets something explicitly marked a **limitation** on Slide 29, don't defend it — agree,
  state the number, state the Phase II plan item that addresses it. Trying to argue a disclosed limitation
  away reads worse than owning it.
- For anything about the edge/Docker/ARM64 results, keep repeating the one caveat: **x86 timing, not Pi
  timing** — that single sentence pre-empts most follow-ups in that area.
