# RUNLOG — End-of-Turn Detection (24BT10010)

Metric: mean response delay (ms) @ ≤5% interrupted turns, official `score.py`.
All model scores below are **out-of-fold** (GroupKFold by turn; 5-fold unless noted) — the model
scoring a pause has never seen that turn, so these estimate unseen-turn performance, not train fit.
(AUC = eot-vs-hold ranking quality, the scorer's own diagnostic.)

| # | run | english | hindi | what changed / what we learned |
|---|-----|---------|-------|--------------------------------|
| 0 | silence-only baseline (`baseline.py`) | **1600 ms** (AUC 0.514) | **850 ms** (AUC 0.501) | Reference to beat. Insight: Hindi holds are much shorter (p95 0.80s vs 1.80s English), so a plain silence timer is already "good" for Hindi at 850 ms — the model must earn a LOW-delay operating point to beat it, i.e., high recall while long holds stay quiet. |
| 1 | v1: 32 scalar prosody features (windows measured back from `pause_start`) + GBM/LR ensemble | 1235 ms (AUC 0.719) | 850 ms (AUC 0.685) | Beat EN baseline; Hindi unchanged — the sweep ignored our scores (threshold 0.05 = fire always). Feature AUC audit showed F0 cues weirdly weak. |
| 2 | v2: anchored windows at DETECTED speech offset after measuring the annotation gap (last 100 ms before `pause_start` is already ~44 dB below speech: consistent ~110 ms VAD hangover in the labels) | 1366 ms (AUC 0.652) | 857 ms (AUC 0.707) | EN got worse, HI slightly better. Anchoring was right (windows now measure speech, not silence) but discarding everything after the offset threw away the offset SHAPE. Plotted turns: **holds cut off sharply; eots decay gradually with trailing breath/low-energy wisps**. That's the cue we weren't measuring. |
| 3 | v3: + offset-shape family (energy fall-time, offset slope, trailing-tail level/centroid/voicing) + loudness-gated F0 (the tracker was reporting noise-floor "pitch" at 300–400 Hz which polluted every F0 statistic) + creak fraction | 1250 ms (AUC 0.689) | 857 ms (AUC 0.746) | HI ranking improved (0.707→0.746), EN flat. Scalar features appear to plateau ~0.70–0.75: two very different views (structure vs prosody scalars) hit the same ceiling. |
| 4 | CNN v1: 48-mel log-spectrogram (1.5 s ending at `pause_start`) + turn-structure scalars, ~70k params, jitter/gain/mask augmentation, 5-fold; prob-averaged with GBM (w=0.3) | 1210 ms (AUC 0.732) | 850 ms (AUC 0.758) | CNN alone ≈ GBM alone (~0.70); ensemble helps a little. Same ceiling from a third model family ⇒ the *inputs*, not the models, are the bottleneck. Decision: give the model (a) more labeled examples of "speech stopped but turn continues", (b) explicit pitch. |
| 5 | side test: GBM/LR trained WITH the 882 mined micro-gap negatives (weight 0.5) | 1281 ms (AUC 0.678) | 857 ms (AUC 0.728) | **Worse than GBM without them** (0.689/0.746). Mined gaps sit at random positions, so they distort the turn-structure prior (elapsed, pause_index…) the tabular model leans on. Conclusion: mined negatives belong in the acoustic model only; tabular stays annotated-only. |
| 6 | CNN v2: mined micro-gap hard negatives (unannotated <100 ms intra-speech silences = guaranteed continuations, ~8/turn, weight 0.5, train-side only), all windows re-anchored at detected speech offset, +F0-semitone & voicing input channels, 2.0 s context, 2 seeds × 5 folds = 10 snapshots; ensemble w=0.6 with GBM | **1080 ms** (AUC 0.745) | **784 ms** (AUC 0.804) | The bet paid: CNN alone 0.708/0.771, ensemble AUC +0.05 on Hindi. **Hindi now beats its 850 ms baseline; English is 33% under its 1600 ms.** Mined negatives + explicit F0 channels were the difference vs run 4. |
| 7 | Metric-aligned experiment: (a) duration-weighted hold cost in BOTH models (firing on a hold only causes a cutoff if it outlasts the action delay), (b) CNN snapshot selection by eot-vs-LONG-hold pAUC, (c) test-time end-shift averaging | 1160 ms (AUC 0.726) | 784 ms (AUC 0.792) | Mixed: cost-weighted **GBM clearly better** (1222/808 vs 1250/857 alone — same AUC, errors migrate to harmless short holds, exactly the mechanism). But the CNN regressed: only ~25 long holds per val fold makes the pAUC stopping criterion too noisy for snapshot selection. Lesson: align the *loss* with the metric where data allows; don't align the *validation signal* with a region too small to estimate. |
| 8 | Merge attempt: run-6 CNN config + run-7 cost-weighted GBM, TTA kept | 1125 ms (AUC 0.727) | 799 ms (AUC 0.797) | Better than run 7, still short of run 6 (1080/784). Reading runs 6–8 together: CNN fold-to-fold variance (val AUC 0.61–0.87 across folds/seeds) swamps the deltas from TTA and loss tweaks. Conclusion: stop tuning what the noise floor hides. |
| 9 | **FINAL**: run-6 CNN recipe (rerun; torch CPU threading makes it near- but not bit-deterministic) + cost-weighted tabular models, ensemble w=0.6 | **1094 ms** (AUC 0.739, op: thr 0.60 / 450 ms) | **755 ms** (AUC 0.811, op: thr 0.45 / 650 ms) | Confirmed with the official `score.py` on the OOF csvs. Best Hindi of all runs (−11% vs its 850 ms baseline, which a silence timer cannot beat); English −32% vs its 1600 ms baseline. Artifacts `model.pkl` + `model_cnn.pt` are from this run; `predictions_*.csv` written by `predict.py`, the exact shipped path, which was also smoke-tested on a fresh folder copy with `label`/`pause_end` columns removed (248/248 rows). |

| 10 | Variance-reduction retrain: 10-fold × 2 seeds = 20 snapshots, each fold training on 90% of turns | 1116 ms (AUC **0.765**) | 824 ms (AUC **0.820**) | **Rejected despite the best AUCs of the night** — global ranking improved but recall at the ≤5%-cutoff operating region got worse (Hindi +69 ms). The metric prices the top of the ranking against long holds, not the whole curve. Also rejected: merging both snapshot pools (30 models) — a mixed pool has no honest OOF, and we don't ship numbers we can't defend. |
| 11 | **FINAL (adopted)**: run-9 snapshots + post-hoc blend refinement on OOF — weight grid at 0.05 steps, prob- vs logit-space blending, objective 0.3·EN + 0.7·HI to match the announced mostly-Hindi hidden set → w=0.55, prob space | **1105 ms** (AUC 0.738) | **745 ms** (AUC 0.810) | Trades +11 ms English for −10 ms Hindi — the right trade for the hidden distribution, chosen without touching the models. Verified with the official `score.py` on the shipped `oof_*.csv`. |

**Final numbers (out-of-fold, official scorer):** english **1600 → 1105 ms**, hindi **850 → 745 ms**
at ≤5% interrupted turns (AUC 0.738 / 0.810). Inference: 248 pause decisions in 17 s on a laptop
CPU including model load (~55–70 ms per decision). Note `predictions_*.csv` for the provided
folders come from the final model which trained on all provided turns — score.py on those files
shows train-fit (~0.99 AUC), not generalization; the honest unseen-turn estimate is the OOF table.

## Listening notes (human)
Worst OOF errors were exported as clips (4 s before the pause + 1.5 s in) and listened to by me:

- `en__049 p3` (hold scored 0.95): caller gives a date, says it WRONG, stops, then corrects —
  "…July thir— … 9th, 9th". A self-repair: the stop before the restart sounds exactly like a
  finished statement. Only the words reveal it isn't over.
- `hi__097 p4` (hold 0.73): fast casual Hindi ("…puri raat nahi so paya, sari raat meri neend
  kharab rahi yaar, tu mujhe—") stopping mid-sentence on a dangling pronoun. Acoustically clean
  stop; syntactically obviously incomplete.
- `en__010 p1` (eot 0.05, missed): the turn ends by dictating an email address
  ("…dot white at yahoo dot com"). Spell-out/list prosody is flat — no terminal fall — so the
  model hears "more coming".
- `hi__033 p2` (eot 0.05, missed): the turn ends on a QUESTION ("mujhe advance payment karni
  hogi? …toh bataiye uske baare mein") with a rising/level final pitch.

Taxonomy this gives us: **false alarms = semantic incompleteness that sounds final (self-repairs,
trail-offs); misses = real endings without a terminal fall (dictation prosody, question rises).**
The first class needs lexical content (banned: no ASR/pretrained), which is why similarity of the
two classes' acoustics caps the ranking. The second class is partially learnable — rising-final
eots exist in training — but with 100 eots/language the rise-final subclass is rare.
