# Presentation Script: Sleep-Staging Pipeline

**For explaining the project to the guide, using the "Sleep-Staging Pipeline Flow" page.**
**About 12 minutes, plus questions.** Each part says what to show, then what to say.

---

## 1. Opening (1 min)

**Show:** the top of the page, the coloured hypnogram.

> This is one real night from our data, recording SC4081E0. Every coloured block is the sleep stage our
> model predicted: Wake, REM, N1, N2 and N3. It covers 1,134 epochs of 30 seconds, about nine and a half
> hours.
>
> Our project does two things. First, it labels every 30 seconds of a night with a sleep stage. Second,
> it writes a report a clinician can read, and it guarantees that every number in that report is
> correct. I'll walk through the pipeline phase by phase: what goes in, what we do to it, and what comes
> out.

---

## 2. The whole flow (1 min)

**Show:** the flow diagram.

> The top row turns the signal into numbers. A raw night goes into preprocessing, then a teacher
> model, then a smaller student model, then calibration.
>
> The bottom row turns numbers into a report. We put everything about the night into an evidence packet.
> A small language model reads the packet and writes claims. A verifier checks every claim. If all pass,
> we produce the report; if anything fails, the night goes to human review.
>
> The dashed box is the agent, which runs and supervises this second half.

---

## 3. The data (30 s)

**Show:** the split bar.

> We use Sleep-EDF Expanded: 197 nights, from healthy subjects and from a temazepam study. We split by
> person, so nobody appears in two splits. 137 nights are for training. 31 are for development, and
> every threshold and table is fitted on those. 29 are the test set. We locked it and read it once, at
> the very end. Its checksum hasn't changed since.

---

## 4. Phase 0: Preprocessing (1 min)

**Show:** Phase 0 panel.

> **Input:** the raw EDF recording and the expert's hypnogram file.
>
> **Process:** we take one EEG channel, Fpz-Cz, at 100 Hz. We map the expert's labels to the five
> modern stages, merging the old stages 3 and 4 into N3. We trim to 30 minutes of wake on either side of
> sleep and cut the signal into 30-second epochs of 3,000 samples. For each epoch we also compute 34
> features: wavelet energies, band powers for delta, theta, alpha, beta and gamma, and three band ratios.
>
> **Output:** for each night, a tensor of N epochs by 3,000 samples and a feature matrix of N by 34.
>
> We also found a bug here ourselves. The trimming shifted 5 to 7 percent of the labels by one epoch.
> We've corrected it, and the model actually scores higher on the corrected labels.

---

## 5. Phase 1a and 1b: Teacher and student (2 min)

**Show:** Phase 1 panel and the architecture diagram.

> **Input:** the tensors, the features and the labels.
>
> **Process:** first we build a **teacher**, an ensemble of three models whose predictions we average.
> It's accurate but large, about 412,000 parameters.
>
> Then we train a **student** by knowledge distillation. The student learns from two things at once:
> the expert's labels, and the teacher's probabilities. Half the loss is each.
>
> The student reads an epoch through two convolution branches. The fine one looks at half-second
> patterns like spindles and K-complexes. The coarse one looks at two-second patterns like slow waves.
> These are fused with the 34 features. Then a small Transformer looks across 256 epochs, about two
> hours, because the stage depends on what comes before and after.
>
> **Output:** a model of 139,606 parameters, 4.65 times smaller than the teacher. It gives five
> probabilities for every epoch.
>
> **Result:** on the test set the teacher has κ 0.714 and the student κ 0.699, so the student keeps
> almost all of the teacher's accuracy at a fraction of the size.

---

## 6. Phase 1c: Calibration and reliability (1 min)

**Show:** the table of five tables.

> A probability alone isn't enough. We need to know how far to trust it. So on the development nights we
> fit five tables:
>
> - a **temperature**, so that 80 percent confidence really means right 80 percent of the time;
> - a **decoder** that removes single stray REM epochs;
> - an **N1 flag**, because N1 is our weakest stage;
> - a **night confidence tier**: is this whole night easy or hard?
> - and **metric reliability**: for each clinical number, how big the error was on validation.
>
> This last one decides what the report may state plainly and what it must hedge.

---

## 7. Phase 1d: The evidence packet (1.5 min)

**Show:** the 19 coloured cards.

> **Input:** the night's probabilities and the five tables.
>
> **Output:** one JSON file per night, the evidence packet. It lists 19 evidence items, and these are
> the **only** things a report may talk about.
>
> The green ones are the 5 **mandatory** facts. Our model is reliable enough to state them plainly:
> total sleep time, here 410 minutes; time in bed; sleep efficiency, 76.3 percent; the night's confidence
> tier; and a warning that N1 is unreliable.
>
> The amber ones are the 14 **hedged** items. These must always carry their error and a caveat. REM
> latency, for example, is 227.5 minutes, but it depends on a single epoch and its average error is 48
> minutes. So it can never be stated as a bare fact.

---

## 8. Phase 2: The report language model (2 min)

**Show:** Phase 2 panel and the bar chart.

> **Input:** the packet, a fixed prompt and a grammar.
>
> **Process:** a small language model reads the packet. It doesn't write sentences. It writes small
> structured claims, each one pointing at the evidence item it's based on.
>
> **How we chose the model:** this chart. We first tried five small models as they come, without
> training. All of them failed. They failed on the rules, not the numbers: in 155 reports there was only
> one wrong number. What they got wrong was which items to hedge and which evidence to cite.
>
> So the question was whether the task is impossible or the models are just too weak. We ran a large
> hosted model, Gemini, on the same packets, prompt and verifier. It passed every night it answered, 30
> of 30, so the task is possible.
>
> We had written the rule down **before** seeing any results: if the large model passes at least 25 of
> 31, we train a small one. It did, so we trained. We used Gemini's correct outputs on 123 training
> nights as examples and fine-tuned two small models with LoRA on a Kaggle T4.
>
> **Result:** the 3-billion-parameter Llama student passed 29 of 29 test nights with zero violations.
> The 1.5-billion Qwen student passed only 12 of 29: it never lied, but it missed facts on harder
> nights. So we deployed Llama.

---

## 9. Phase 1e: The verifier gate (1 min)

**Show:** the three boxes: claim, verifier, sentence.

> This is how one claim travels.
>
> **Left:** the model writes a claim: REM latency, 227.5 minutes, as a hedged value, citing the REM
> latency item.
>
> **Middle:** the verifier checks it with 30 rules in two layers. Is the JSON valid? Does the cited item
> exist? Does 227.5 match the packet exactly? The item is unsafe, so is it hedged? Are all five
> mandatory items present? The verifier never guesses. If the packet doesn't say it, the claim fails.
>
> **Right:** only then does a fixed template write the English sentence. The model never writes the
> final wording, so it cannot slip in a wrong number.

---

## 10. Phase 3: Deployment (45 s)

**Show:** Phase 3 panel.

> `pipeline.py` runs all of this for one new night. You give it an EDF file. It preprocesses, stages,
> builds the packet, runs the model and verifies. A pass gives an HTML report; a fail goes to human
> review. Every run goes into an audit log.
>
> It runs on a laptop CPU in about two minutes per report and 4.7 GB of memory, with no internet. That's
> what makes our Raspberry Pi target realistic.

---

## 11. Phase 4: Adding EOG (45 s)

**Show:** the κ dot chart.

> We then improved the staging model by adding the eye-movement channel, EOG. N1 is partly defined by
> slow eye movements, so this helps exactly where we were weakest.
>
> The new model reaches 81.96 percent accuracy and κ 0.756. It beats the teacher with 43 percent of its
> size. And because it produces the same packet format, the report model works with it unchanged, with
> no retraining: 30 of 31 on development and 27 of 29 on test. That's slightly below the 29 of 29 with the
> old packets. One test night was blocked by the verifier, so nothing wrong reached a report. Retraining
> the report model on the new packets is the planned fix.

---

## 12. Phase 5: The agent (45 s)

**Show:** the four coloured outcomes.

> Finally we built an agent around the pipeline. It decides what to do: check quality, retry, fall back,
> deliver. It never decides what is true. Every night ends in exactly one of four outcomes:
>
> - a **verified report**;
> - a **fallback summary** built only from packet values;
> - **review only**;
> - or **rejected**, for an unreadable file.
>
> It is also crash-safe: if it's killed midway, re-running it gives exactly the same result.

---

## 13. Close (30 s)

**Show:** the five rules at the bottom.

> To sum up, three things make our results trustworthy:
>
> - the test set was locked and used once;
> - every threshold came from development data only;
> - and every decision rule was written down before we saw results.
>
> The outcome is a small model that runs offline. It labels sleep stages with about 80 percent accuracy
> and writes reports whose every number is checked against the evidence.

---

## Likely questions

**Why a teacher and a student, not just one model?**
The large ensemble is accurate but three times the size. Distillation gives a small model that keeps
almost all of the accuracy and can run on cheap hardware.

**Why is N1 so weak?**
N1 is defined partly by slow eye movements, and the deployed model sees EEG only. Even human experts
disagree most on N1. Adding EOG raised N1 F1 by about 5 points.

**Why not let the language model just write the report?**
Language models sometimes invent numbers, and a fluent wrong number looks exactly like a right one.
So the model only picks claims; the verifier checks them, and templates write the text.

**Isn't 29/29 suspicious?**
The test nights come from people never seen in training. We read them once, after the development gate
had already passed, and their checksum hasn't changed.

**What happens if the model fails on a night?**
The verifier blocks the report, and the night goes to human review or gets a fallback summary. Nothing
unverified is ever shown.

**What isn't finished?**
- The deployed server path scores lower than the evaluation path (20–23 vs 30 of 31). It needs to be
  aligned before deployment to the Pi.
- The staging models still need retraining on the corrected labels.
- Some result files exist only on this machine, not in git.
