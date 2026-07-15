# End-of-Turn Detection — Plivo ML Assignment (roll 24BT10010)

Predicts, at every silence pause of a caller's turn, the probability that the
turn is over (`p_eot`), so a voice agent can respond fast without talking over
people. Scored by `score.py`: mean response delay at ≤5% interrupted turns.

## Results (out-of-fold = every score comes from folds that never saw that turn)

See `RUNLOG.md` for the full iteration history and `SUMMARY.html` for the
readable report with figures. Final numbers are in both.

## Run

```bash
# predictions for a data folder (unseen folders with the same schema work)
python predict.py --data_dir <folder> --out predictions.csv

# score them (official scorer from the starter kit)
python score.py --data_dir <folder> --pred predictions.csv

# full reproduction of training (writes model.pkl, model_cnn.pt, oof_*.csv)
python train_model.py --data_root <root with english/ hindi/>   # tabular models
python train_cnn.py   --data_root <root with english/ hindi/>   # CNN + ensemble
```

## Files

| file | role |
|---|---|
| `predict.py` | shipped inference CLI (loads `model.pkl` + `model_cnn.pt`) |
| `features_eot.py` | causal prosody/structure features (42 scalars) + contour extractors |
| `train_model.py` | GBM + logistic models, GroupKFold OOF + cross-language eval |
| `train_cnn.py` | log-mel+F0 CNN, micro-gap hard-negative mining, snapshot ensemble |
| `model.pkl`, `model_cnn.pt` | trained artifacts (from the provided data only) |
| `predictions_english.csv`, `predictions_hindi.csv` | predictions for the provided folders |
| `RUNLOG.md` | every scoring run: what changed, score, what we learned (graded) |
| `NOTES.md` | 10-sentence summary |
| `SUMMARY.html` | full report: method, figures, human-vs-agent breakdown |
| `make_figures.py` | report figures only — NOT part of the model (uses matplotlib) |

## Causality

For a pause at `pause_start`, features use only audio frames that END at or
before `pause_start`, plus label-file fields from strictly earlier pauses of
the same turn. The current pause's `pause_end`/duration and `label` are never
read at feature time (`pause_end` is only ever used as the causal history of
*later* pauses, and by the scorer itself). Grep points: `features_eot.py`
docstring + `pause_features`, `train_cnn.py` `make_window`/`speech_end_frame`,
`predict.py` main loop.

## Library compliance

Model pipeline imports only: numpy, scipy, scikit-learn, pandas, librosa,
PyTorch, Python stdlib. CPU only, no pretrained weights, no external data.
(`make_figures.py` additionally uses matplotlib but produces report images
only and is not imported by any model code.)
