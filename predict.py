"""End-of-turn prediction CLI (the shipped model).

    python predict.py --data_dir <folder> --out predictions.csv

<folder> must contain labels.csv and the audio/ files it references, same
schema as the handout. Works on folders it has never seen.

CAUSALITY: for each pause only these fields are read: turn_id, audio_file,
pause_index, pause_start — plus pause_start/pause_end of EARLIER pauses of
the same turn (they ended before this pause started: causal past). The
current row's pause_end and label (if present) are NEVER read. All audio
features use frames that end at or before pause_start (see features_eot.py
and train_cnn.py docstrings).

Model = probability average of
  - 10 CNN snapshots (log-mel + F0 + voicing window ending at the detected
    speech offset; 2 seeds x 5 folds), weight ens_w
  - GBM + logistic regression on 42 causal prosody/structure scalars

Allowed-library compliance: numpy, scipy, scikit-learn, pandas, librosa,
PyTorch + stdlib only (audio I/O via librosa).
"""
import argparse
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from features_eot import FEATURE_NAMES, pause_features, wav_contours
from train_cnn import (EotCNN, SCALARS, TTA_SHIFTS, make_window,
                       speech_end_frame, wav_tensors)

FALLBACK_P = 0.35     # ~ class prior; used only if a row cannot be processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "model.pkl"), "rb") as fh:
        tab = pickle.load(fh)
    cnn_pack = torch.load(os.path.join(here, "model_cnn.pt"),
                          map_location="cpu", weights_only=False)
    models = []
    for st in cnn_pack["states"]:
        m = EotCNN(len(SCALARS))
        m.load_state_dict(st)
        m.eval()
        models.append(m)
    ens_w = float(cnn_pack["ens_w"])
    s_mu, s_sd = cnn_pack["scalar_mu"], cnn_pack["scalar_sd"]
    sidx = [FEATURE_NAMES.index(s) for s in SCALARS]

    df = pd.read_csv(os.path.join(args.data_dir, "labels.csv"))
    df = df.sort_values(["turn_id", "pause_index"]).reset_index(drop=True)
    wavs = sorted({os.path.join(args.data_dir, a) for a in df.audio_file})

    def safe_pair(w):
        try:
            return w, (wav_contours(w), wav_tensors(w))
        except Exception:
            return w, None
    with ThreadPoolExecutor(max_workers=8) as ex:
        loaded = dict(ex.map(safe_pair, wavs))

    # tabular features per row (prior pauses of the same turn = causal past)
    X = np.zeros((len(df), len(FEATURE_NAMES)), dtype=np.float32)
    windows = {sh: np.zeros((len(df), 50, cnn_pack["win"]), dtype=np.float32)
               for sh in TTA_SHIFTS}
    ok = np.zeros(len(df), dtype=bool)
    prior_map = {}
    has_end = "pause_end" in df.columns
    for i, r in enumerate(df.itertuples()):
        wav = os.path.join(args.data_dir, r.audio_file)
        prior = prior_map.setdefault(r.turn_id, [])
        pair = loaded.get(wav)
        if pair is not None:
            try:
                c, tz = pair
                ps = float(r.pause_start)
                X[i] = pause_features(c, ps, int(r.pause_index), prior)
                endf = speech_end_frame(tz["e"], ps)
                for sh in TTA_SHIFTS:
                    windows[sh][i] = make_window(tz, endf, sh)
                ok[i] = True
            except Exception:
                ok[i] = False
        # append AFTER computing features: this pause becomes history for the
        # next pause of the turn. Uses this row's end only for FUTURE rows.
        end = float(getattr(r, "pause_end")) if has_end else float(r.pause_start)
        prior.append((float(r.pause_start), end))

    p = np.full(len(df), FALLBACK_P, dtype=np.float64)
    if ok.any():
        p_gbm = tab["gbm"].predict_proba(X[ok])[:, 1]
        p_lr = tab["lr"].predict_proba(X[ok])[:, 1]
        p_tab = 0.5 * p_gbm + 0.5 * p_lr
        sc = ((X[ok][:, sidx] - s_mu) / s_sd).astype(np.float32)
        with torch.no_grad():
            sc_t = torch.from_numpy(sc)
            ps_all = []
            for sh in TTA_SHIFTS:                 # test-time end-shift averaging
                mel_t = torch.from_numpy(windows[sh][ok])
                ps_all += [torch.sigmoid(m(mel_t, sc_t)).numpy() for m in models]
            p_cnn = np.mean(ps_all, axis=0)
        p[ok] = ens_w * p_cnn + (1 - ens_w) * p_tab

    out = df[["turn_id", "pause_index"]].copy()
    out["p_eot"] = np.round(p, 4)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} predictions -> {args.out} "
          f"({int(ok.sum())}/{len(df)} rows fully processed)")


if __name__ == "__main__":
    main()
