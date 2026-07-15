# End-of-Turn Detection — Plivo ML Assignment (roll 24BT10010)

## Results at a glance

| | mean response delay @ ≤5% interrupted turns | AUC | silence baseline |
|---|---|---|---|
| **English** | **1105 ms** | 0.738 | 1600 ms (−31%) |
| **Hindi** | **745 ms** | 0.810 | 850 ms (−12%) |

Runs at ~55–70 ms per pause decision on a laptop CPU (248 decisions in 17 s incl. model load).

**Every number above is out-of-fold** (5-fold GroupKFold by turn): the model scoring a pause
never trained on that turn — these estimate unseen-turn performance, not train fit.

## The five ideas that matter

1. **Anchor at the true speech offset** — the labels' `pause_start` trails the acoustic offset
   by a ~110 ms VAD hangover (we measured it); all windows end at the detected offset.
2. **Read *how* speech stops, not just *that* it stops** — holds cut off sharply mid-phrase;
   true ends decay gradually with trailing breath and a pitch fall. A small CNN over
   log-mel + explicit F0/voicing channels learns this shape.
3. **Mine free hard negatives** — every unannotated <100 ms silence inside fluent speech is a
   *guaranteed continuation*: 882 extra "stopped but not done" examples (+2.8× negatives).
4. **Cost-align training with deployment** — a hold only causes a cutoff if it outlasts the
   action delay, so long holds carry the training cost; errors migrate to harmless short pauses.
5. **Trust only honest validation** — OOF everywhere, cross-language stress tests, and the
   worst errors *listened to by a human* (see RUNLOG: false alarms are semantic — self-repairs,
   mid-sentence stops; misses are endings without a pitch fall — dictation, question rises).

## Verify the claims in 30 seconds

```bash
# our claimed numbers, reproduced with YOUR scorer on out-of-fold predictions:
python score.py --data_dir <eot_data>/english --pred oof_english.csv    # → 1105 ms, AUC 0.738
python score.py --data_dir <eot_data>/hindi   --pred oof_hindi.csv     # → 745 ms,  AUC 0.810

# run the shipped model on any same-schema folder (unseen folders work):
python predict.py --data_dir <folder> --out predictions.csv
```

(`predictions_english.csv` / `predictions_hindi.csv` in this repo come from the final model,
which trained on all provided turns — scoring those shows train fit, ~0.99 AUC. The honest
generalization numbers are the OOF ones above.)

## Reproduce from scratch

```bash
python train_model.py --data_root <root with english/ hindi/>   # tabular models → model.pkl
python train_cnn.py   --data_root <root with english/ hindi/>   # CNN + ensemble → model_cnn.pt
```

## Files

| file | role |
|---|---|
| `predict.py` | shipped inference CLI (loads `model.pkl` + `model_cnn.pt`) |
| `features_eot.py` | causal prosody/structure features (42 scalars) + contour extractors |
| `train_model.py` | GBM + logistic models, GroupKFold OOF + cross-language eval |
| `train_cnn.py` | log-mel+F0 CNN, micro-gap hard-negative mining, snapshot ensemble |
| `model.pkl`, `model_cnn.pt` | trained artifacts (from the provided data only) |
| `oof_english.csv`, `oof_hindi.csv` | out-of-fold predictions backing the claimed numbers |
| `predictions_english.csv`, `predictions_hindi.csv` | required predictions for the provided folders |
| `RUNLOG.md` | every scoring run: what changed, score, what we learned (graded) |
| `NOTES.md` | 10-sentence summary |
| `SUMMARY.html` | full report: method, figures, human-vs-agent breakdown |
| `score.py` | the official scorer (copied from the starter kit, unmodified) |
| `make_figures.py` | report figures only — NOT part of the model (uses matplotlib) |

## Causality

For a pause at `pause_start`, features use only audio frames that END at or
before `pause_start`, plus label-file fields from strictly earlier pauses of
the same turn. The current pause's `pause_end`/duration and `label` are never
read at feature time. Grep points: `features_eot.py` docstring +
`pause_features`, `train_cnn.py` `make_window`/`speech_end_frame`,
`predict.py` main loop.

## Library compliance

Model pipeline imports only: numpy, scipy, scikit-learn, pandas, librosa,
PyTorch, Python stdlib. CPU only, no pretrained weights, no external data.
(`make_figures.py` additionally uses matplotlib but produces report images
only and is not imported by any model code.)
