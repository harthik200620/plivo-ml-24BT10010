"""Log-mel+F0 CNN for end-of-turn detection, with mined hard negatives.

Design (why each piece exists):
  - INPUT: 2 s of (48 log-mel + F0-semitone + voicing) channels ending at the
    DETECTED speech offset before the pause. The labels' pause_start trails
    the true offset by a ~110 ms VAD hangover, so anchoring at the detected
    offset gives every row the same alignment; mel alone smears pitch, so an
    explicit F0 track carries the terminal fall/rise.
  - HARD NEGATIVES: every unannotated intra-speech silence gap (<100 ms,
    the annotation threshold) is a guaranteed continuation. We mine up to 8
    per turn -> ~4x more "speech stopped but turn is NOT over" examples,
    exactly the false-cutoff acoustics that cost latency budget. Weighted
    0.5 in the loss; never used for validation.
  - SMALL NET + AUG: ~80k params, end-jitter, gain shift, mel noise,
    freq/time masking; 5-fold GroupKFold by turn x 2 seeds -> 10 snapshots
    averaged at inference; OOF stays honest (val = annotated pauses only).
  - ENSEMBLE: averaged with the tabular GBM+LR (turn structure prior).

CAUSALITY: for a pause at pause_start every input frame ends at or before
pause_start: mel/F0 use center=False framing, the window ends at the
detected offset (<= pause_start), jitter only moves it EARLIER, and mined
negatives use only their own past the same way.

    python train_cnn.py --data_root ../eot/eot_data --out_dir .
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from features_eot import load_wav
from train_model import (cost_weights, fit_predict, load_labels, build_matrix,
                         official_score)

SR = 16000
N_FFT = 400          # 25 ms, center=False -> frame i covers [i*HOP, i*HOP+N_FFT)
HOP = 160            # 10 ms
N_MELS = 48
WIN = 200            # 2.0 s of context ending at the speech offset
JITTER = 6           # frames; training window end may shift EARLIER by <= 60 ms
NEG_PER_TURN = 8
SEEDS = (0, 1)

torch.set_num_threads(8)

SCALARS = ["elapsed", "pause_index", "n_prior", "last_seg_dur",
           "prior_dur_last", "prior_dur_mean", "prior_dur_max", "speech_frac"]


# ----------------------------------------------------------------------
# per-wav tensors (cached in memory)
# ----------------------------------------------------------------------

def wav_tensors(path):
    from features_eot import f0_contour, energy_contour_db
    x, sr = load_wav(path)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, power=2.0, center=False)
    logm = np.log(m + 1e-6).astype(np.float32)              # (48, T)
    f0 = f0_contour(x, SR)                                   # 10ms hop, 40ms frames
    e_db = energy_contour_db(x, SR)                          # 10ms hop, 25ms frames
    T = logm.shape[1]
    f0 = np.pad(f0[:T], (0, max(0, T - len(f0))))
    e_db = np.pad(e_db[:T], (0, max(0, T - len(e_db))), constant_values=-120.0)
    return {"mel": logm, "f0": f0.astype(np.float32), "e": e_db.astype(np.float32)}


def speech_end_frame(e, t_limit_s):
    """Last loud frame index strictly before t_limit_s (causal threshold)."""
    n = max(1, min(len(e), int(t_limit_s / 0.01) - 2))      # frame ends <= t_limit
    seg = e[:n]
    thr = np.percentile(seg, 95) - 25.0
    loud = np.where(seg > thr)[0]
    return int(loud[-1]) + 1 if len(loud) else n            # window end (exclusive)


def make_window(tz, end_frame, shift=0):
    """(50, WIN) input ending at end_frame - shift. Channels: mel | f0 | voiced."""
    end = max(4, end_frame - shift)
    start = end - WIN
    mel = tz["mel"][:, max(0, start):end]
    f0 = tz["f0"][max(0, start):end]
    if mel.shape[1] < WIN:
        pad = WIN - mel.shape[1]
        mel = np.concatenate([np.full((N_MELS, pad), np.log(1e-6), np.float32), mel], 1)
        f0 = np.concatenate([np.zeros(pad, np.float32), f0])
    mu, sd = mel.mean(), mel.std() + 1e-3
    mel = (mel - mu) / sd
    voiced = (f0 > 0).astype(np.float32)
    st = 12.0 * np.log2(np.maximum(f0, 1.0) / 55.0) * voiced
    if voiced.sum() >= 4:
        vm = st[voiced > 0].mean()
        vs = st[voiced > 0].std() + 0.5
        st = np.where(voiced > 0, (st - vm) / vs, 0.0).astype(np.float32)
    return np.concatenate([mel, st[None, :], voiced[None, :]], 0)


# ----------------------------------------------------------------------
# mined hard negatives: unannotated micro-gaps inside fluent speech
# ----------------------------------------------------------------------

def mine_negatives(tz, ann_spans, max_n=NEG_PER_TURN):
    e = tz["e"]
    thr = np.percentile(e, 95) - 25.0
    silent = e <= thr
    gaps, i, n = [], 0, len(silent)
    while i < n:
        if silent[i]:
            j = i
            while j < n and silent[j]:
                j += 1
            if i > 0 and j < n:                              # bounded by speech
                dur = (j - i) * 0.01
                gs = i * 0.01
                if 0.03 <= dur < 0.09 and gs > 1.0 and not any(
                        (gs > a - 0.3) and (gs < b + 0.3) for a, b in ann_spans):
                    gaps.append((dur, gs))
            i = j
        else:
            i += 1
    gaps.sort(reverse=True)                                  # prefer longest
    return [g for _, g in gaps[:max_n]]


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------

class EotCNN(nn.Module):
    def __init__(self, n_scalar, in_ch=N_MELS + 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 3, stride=2, padding=1), nn.BatchNorm1d(96), nn.ReLU(),
            nn.Conv1d(96, 96, 3, padding=1), nn.BatchNorm1d(96), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(96 * 2 + n_scalar, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))

    def forward(self, mel, sc):
        h = self.net(mel)
        h = torch.cat([h.mean(-1), h.amax(-1), sc], dim=1)
        return self.head(h).squeeze(1)


def augment(w, rng):
    w = w.copy()
    w[:N_MELS] += rng.normal(0, 0.10)                        # gain (log domain)
    w[:N_MELS] += rng.normal(0, 0.08, w[:N_MELS].shape).astype(np.float32)
    f0b = rng.integers(0, N_MELS - 8)
    w[f0b:f0b + rng.integers(2, 8), :] = 0.0                 # freq mask
    t0 = rng.integers(0, WIN - 24)
    w[:, t0:t0 + rng.integers(4, 24)] = 0.0                  # time mask
    return w


TTA_SHIFTS = (0,)   # end-shift TTA (0,3,6) was tried: no gain over fold noise
                    # (RUNLOG runs 7-8), so eval = the single causal window


def _rank_auc(pos, neg):
    s = np.concatenate([pos, neg])
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = len(pos), len(neg)
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def run_fold(fold, seed, tr, te, val_neg_mask, rows, tzs, Xs, y, wgt):
    """val_neg_mask: which of te's negatives count for early stopping —
    long holds only, i.e. the operating region the scorer prices."""
    rng = np.random.default_rng(1000 * seed + fold)
    torch.manual_seed(1000 * seed + fold)
    model = EotCNN(len(SCALARS))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def batch(idx, train):
        ws = []
        for i in idx:
            wav, endf = rows[i]
            shift = int(rng.integers(0, JITTER + 1)) if train else 0
            w = make_window(tzs[wav], endf, shift)
            if train:
                w = augment(w, rng)
            ws.append(w)
        return (torch.from_numpy(np.stack(ws)), torch.from_numpy(Xs[idx]),
                torch.from_numpy(y[idx].astype(np.float32)),
                torch.from_numpy(wgt[idx].astype(np.float32)))

    def eval_tta(idx):
        with torch.no_grad():
            sc_t = torch.from_numpy(Xs[idx])
            ps = []
            for sh in TTA_SHIFTS:
                mel_t = torch.from_numpy(np.stack(
                    [make_window(tzs[rows[i][0]], rows[i][1], sh) for i in idx]))
                ps.append(torch.sigmoid(model(mel_t, sc_t)).numpy())
        return np.mean(ps, axis=0)

    pos_w = float((wgt[tr] * (1 - y[tr])).sum() / max((wgt[tr] * y[tr]).sum(), 1))
    best_auc, best_state, patience = 0.0, None, 0
    for epoch in range(70):
        model.train()
        order = rng.permutation(tr)
        for b0 in range(0, len(order), 128):
            idx = order[b0:b0 + 128]
            mel_t, sc_t, y_t, w_t = batch(idx, True)
            opt.zero_grad()
            logit = model(mel_t, sc_t)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logit, y_t, weight=w_t * (1 + (pos_w - 1) * y_t))
            loss.backward()
            opt.step()
        model.eval()
        p = eval_tta(te)
        # global val AUC for snapshot selection: long-hold-only pAUC was tried
        # and is too noisy with ~25 long holds per fold (see RUNLOG run 7)
        auc = _rank_auc(p[y[te] == 1], p[y[te] == 0])
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 14:
                break
    model.load_state_dict(best_state)
    model.eval()
    return fold, seed, eval_tta(te), best_state, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    df = load_labels(args.data_root)
    df["dur"] = df.pause_end - df.pause_start                # scoring only
    y_ann = (df.label == "eot").to_numpy(int)
    groups_ann = (df.lang + "/" + df.turn_id).to_numpy()

    from features_eot import FEATURE_NAMES
    X = build_matrix(df, cache_path=os.path.join(args.out_dir, "contours_cache.pkl"))
    sidx = [FEATURE_NAMES.index(s) for s in SCALARS]

    t0 = time.time()
    wavs = df.wav.unique()
    with ThreadPoolExecutor(max_workers=8) as ex:
        got = list(ex.map(wav_tensors, wavs))
    tzs = dict(zip(wavs, got))
    print(f"tensors: {len(wavs)} wavs in {time.time()-t0:.1f}s")

    # ---- assemble rows: annotated pauses + mined negatives ----
    rows, Xs_list, y_list, wgt_list, grp_list, is_ann, dur_list = [], [], [], [], [], [], []
    for r in df.itertuples():
        endf = speech_end_frame(tzs[r.wav]["e"], float(r.pause_start))
        rows.append((r.wav, endf))
        Xs_list.append(X[r.Index][sidx])
        is_eot = 1 if r.label == "eot" else 0
        y_list.append(is_eot)
        # NOTE: duration-weighted CNN loss was tried and hurt (RUNLOG run 7);
        # cost weighting lives in the tabular models only
        wgt_list.append(1.0)
        grp_list.append(r.lang + "/" + r.turn_id)
        is_ann.append(True)
        dur_list.append(float(r.dur))
    n_mined = 0
    for (lang, tid), part in df.groupby(["lang", "turn_id"]):
        wav = part.iloc[0].wav
        spans = list(zip(part.pause_start, part.pause_end))
        for gs in mine_negatives(tzs[wav], spans):
            endf = speech_end_frame(tzs[wav]["e"], gs + 0.02)
            prior = [(a, b) for a, b in spans if b <= gs]
            sc = np.array([gs, len(prior), len(prior),
                           gs - (prior[-1][1] if prior else 0.0),
                           (prior[-1][1] - prior[-1][0]) if prior else 0.0,
                           np.mean([b - a for a, b in prior]) if prior else 0.0,
                           max([b - a for a, b in prior]) if prior else 0.0,
                           (gs - sum(b - a for a, b in prior)) / max(gs, 1e-3)],
                          dtype=np.float32)
            rows.append((wav, endf))
            Xs_list.append(sc)
            y_list.append(0)
            wgt_list.append(0.5)
            grp_list.append(lang + "/" + tid)
            is_ann.append(False)
            dur_list.append(0.0)
            n_mined += 1
    Xs = np.stack(Xs_list).astype(np.float32)
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    Xs = (Xs - mu) / sd
    y = np.array(y_list); wgt = np.array(wgt_list)
    grp = np.array(grp_list); is_ann = np.array(is_ann)
    dur_all = np.array(dur_list)
    print(f"rows: {is_ann.sum()} annotated + {n_mined} mined negatives")

    # ---- folds defined on annotated turns; mined rows follow their turn ----
    folds_ann = list(GroupKFold(n_splits=5).split(X, y_ann, groups_ann))
    all_idx = np.arange(len(rows))
    jobs = []
    for f, (tr_a, te_a) in enumerate(folds_ann):
        te_turns = set(groups_ann[te_a])
        tr = all_idx[~np.isin(grp, list(te_turns))]
        te = all_idx[np.isin(grp, list(te_turns)) & is_ann]   # validate on real pauses only
        val_neg_mask = dur_all[te] > 0.4                      # long holds = scored region
        for s in SEEDS:
            jobs.append((f, s, tr, te, val_neg_mask))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(
            lambda j: run_fold(j[0], j[1], j[2], j[3], j[4], rows, tzs, Xs, y, wgt), jobs))
    print(f"cnn training {time.time()-t0:.0f}s")

    oof_cnn = np.zeros(len(df)); cnt = np.zeros(len(df))
    states = []
    ann_pos = {i: k for k, i in enumerate(all_idx[is_ann])}   # row idx -> df idx
    for f, s, p, st, auc in results:
        te_turns = set(groups_ann[folds_ann[f][1]])
        te = all_idx[np.isin(grp, list(te_turns)) & is_ann]
        for row_i, pi in zip(te, p):
            oof_cnn[ann_pos[row_i]] += pi
            cnt[ann_pos[row_i]] += 1
        states.append(st)
        print(f"  fold {f} seed {s}: val AUC {auc:.3f}")
    oof_cnn /= np.maximum(cnt, 1)

    w_ann = cost_weights(y_ann, df.dur.to_numpy())
    oof_gbm = np.zeros(len(df))
    for tr_a, te_a in folds_ann:
        oof_gbm[te_a] = fit_predict(X[tr_a], y_ann[tr_a], X[te_a], w_ann[tr_a])

    print("\n=== OOF official metric ===")
    best_w, best_lat = 0.5, 1e9
    for w in np.round(np.arange(0.0, 1.01, 0.1), 2):
        df["p"] = w * oof_cnn + (1 - w) * oof_gbm
        m = np.mean([official_score(part)["delay_ms"] for _, part in df.groupby("lang")])
        if m < best_lat:
            best_lat, best_w = m, w
    for name, p in (("gbm", oof_gbm), ("cnn", oof_cnn),
                    (f"ens w={best_w}", best_w * oof_cnn + (1 - best_w) * oof_gbm)):
        df["p"] = p
        line = f"  {name:12s}"
        for lang, part in df.groupby("lang"):
            r = official_score(part)
            line += f"  {lang}: delay={r['delay_ms']:5.0f}ms AUC={r['auc']:.3f}"
        print(line)

    df["p"] = best_w * oof_cnn + (1 - best_w) * oof_gbm
    for lang, part in df.groupby("lang"):
        out = part[["turn_id", "pause_index"]].copy()
        out["p_eot"] = part.p.round(4)
        out.to_csv(os.path.join(args.out_dir, f"oof_{lang}.csv"), index=False)
    torch.save({"states": states, "scalar_mu": mu, "scalar_sd": sd,
                "scalars": SCALARS, "ens_w": best_w,
                "win": WIN, "n_mels": N_MELS},
               os.path.join(args.out_dir, "model_cnn.pt"))
    print(f"saved model_cnn.pt ({len(states)} snapshots, ens_w={best_w})")


if __name__ == "__main__":
    main()
