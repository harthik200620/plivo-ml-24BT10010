"""Train the EOT classifier and report HONEST (out-of-fold) dev scores.

    python train_model.py --data_root ../eot/eot_data --out_dir .

- Features: features_eot.py (strictly causal, see its docstring).
- Model: HistGradientBoosting + logistic regression, probability-averaged.
  Small data (496 pauses) -> shallow trees, strong regularization.
- Evaluation: 5-fold GroupKFold by turn_id, pooled over both languages.
  OOF predictions are scored with the official metric per language, so the
  number in RUNLOG.md estimates unseen-turn performance, not train fit.
- Cross-language: train on English only -> score Hindi (and vice versa),
  because the hidden test is mostly Hindi.
- Final artifact: model.pkl = both models fitted on ALL pooled data.
"""
import argparse
import os
import pickle
import time

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features_eot import FEATURE_NAMES, pause_features, wav_contours

TIMEOUT_S = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)


# ------------------------------------------------------------------
# official metric (verbatim semantics of starter/score.py)
# ------------------------------------------------------------------

def official_score(df, budget=0.05):
    """df: turn_id, dur, label, p  ->  (delay_ms, cutoff, thr, delay, auc)"""
    y = (df.label == "eot").to_numpy(int)
    p = df.p.to_numpy()
    dur = df.dur.to_numpy()
    tid = df.turn_id.to_numpy()
    n_turns = len(set(tid))
    best = None
    for t in THRESHOLDS:
        fires = p >= t
        for d in DELAYS:
            cut_turns = set(tid[(y == 0) & fires & (d < dur)])
            cut = len(cut_turns) / max(1, n_turns)
            if cut > budget:
                continue
            lat = np.where(fires[y == 1], d, TIMEOUT_S).mean()
            if best is None or lat < best[0]:
                best = (lat, cut, t, d)
    if best is None:
        best = (TIMEOUT_S, 0.0, 1.0, TIMEOUT_S)
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
    return {"delay_ms": best[0] * 1000, "cutoff": best[1],
            "thr": best[2], "delay_op": best[3], "auc": auc}


# ------------------------------------------------------------------
# dataset assembly
# ------------------------------------------------------------------

def load_labels(data_root):
    frames = []
    for lang in sorted(os.listdir(data_root)):
        p = os.path.join(data_root, lang, "labels.csv")
        if os.path.isfile(p):
            df = pd.read_csv(p)
            df["lang"] = lang
            df["wav"] = df["audio_file"].apply(
                lambda a: os.path.join(data_root, lang, a))
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["lang", "turn_id", "pause_index"]).reset_index(drop=True)


def build_matrix(df, cache_path=None, n_jobs=8):
    cache = {}
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)
    todo = [w for w in df.wav.unique() if w not in cache]
    if todo:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            got = list(ex.map(wav_contours, todo))
        cache.update(dict(zip(todo, got)))
        print(f"contours: {len(todo)} wavs in {time.time()-t0:.1f}s")
        if cache_path:
            with open(cache_path, "wb") as fh:
                pickle.dump(cache, fh)
    X = np.zeros((len(df), len(FEATURE_NAMES)), dtype=np.float32)
    prior_map = {}
    for i, r in enumerate(df.itertuples()):
        key = (r.lang, r.turn_id)
        prior = prior_map.setdefault(key, [])
        X[i] = pause_features(cache[r.wav], float(r.pause_start),
                              int(r.pause_index), prior)
        # this pause is now history for the NEXT pause of the same turn
        prior.append((float(r.pause_start), float(r.pause_end)))
    return X


def make_models():
    gbm = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.06, max_iter=220,
        l2_regularization=1.0, min_samples_leaf=20,
        max_bins=64, random_state=0)
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.3, class_weight="balanced"))
    return gbm, lr


def cost_weights(y, dur):
    """Deployment-aligned sample weights: firing on a hold only causes a
    cutoff if the hold outlasts the action delay, so long holds carry the
    cost; eots all cost the same. Labels are training supervision only —
    never a feature."""
    return np.where(y == 1, 1.0, np.clip(0.4 + dur, None, 1.5))


def fit_predict(Xtr, ytr, Xte, wtr=None):
    gbm, lr = make_models()
    gbm.fit(Xtr, ytr, sample_weight=wtr)
    lr.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
    return 0.5 * gbm.predict_proba(Xte)[:, 1] + 0.5 * lr.predict_proba(Xte)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = load_labels(args.data_root)
    df["dur"] = df.pause_end - df.pause_start          # scoring only, never a feature
    y = (df.label == "eot").to_numpy(int)
    groups = (df.lang + "/" + df.turn_id).to_numpy()
    X = build_matrix(df, cache_path=os.path.join(args.out_dir, "contours_cache.pkl"))

    # ---- pooled OOF ----
    w = cost_weights(y, df.dur.to_numpy())
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=args.folds).split(X, y, groups):
        oof[te] = fit_predict(X[tr], y[tr], X[te], w[tr])
    df["p"] = oof
    print("\n=== OOF (unseen-turn estimate), pooled training ===")
    for lang, part in df.groupby("lang"):
        r = official_score(part)
        print(f"  {lang:8s} delay={r['delay_ms']:5.0f} ms  cutoff={r['cutoff']*100:.1f}%  "
              f"AUC={r['auc']:.3f}  (thr={r['thr']}, act_delay={r['delay_op']*1000:.0f} ms)")
        part_out = part[["turn_id", "pause_index"]].copy()
        part_out["p_eot"] = part.p.round(4)
        part_out.to_csv(os.path.join(args.out_dir, f"oof_{lang}.csv"), index=False)

    # ---- cross-language transfer ----
    print("=== cross-language transfer ===")
    for src, dst in (("english", "hindi"), ("hindi", "english")):
        m_src, m_dst = df.lang == src, df.lang == dst
        part = df[m_dst].copy()
        part["p"] = fit_predict(X[m_src.to_numpy()], y[m_src.to_numpy()],
                                X[m_dst.to_numpy()])
        r = official_score(part)
        print(f"  {src:8s}->{dst:8s} delay={r['delay_ms']:5.0f} ms  AUC={r['auc']:.3f}")

    # ---- final model on everything ----
    gbm, lr = make_models()
    gbm.fit(X, y, sample_weight=w)
    lr.fit(X, y, logisticregression__sample_weight=w)
    with open(os.path.join(args.out_dir, "model.pkl"), "wb") as fh:
        pickle.dump({"gbm": gbm, "lr": lr, "features": FEATURE_NAMES}, fh)
    print(f"\nsaved model.pkl ({len(FEATURE_NAMES)} features, {len(df)} pauses)")

    # GBM importances (permutation would be better; this is the quick look)
    try:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(gbm, X, y, n_repeats=5, random_state=0,
                                     scoring="roc_auc")
        top = np.argsort(-imp.importances_mean)[:12]
        print("top features (perm. importance on train):")
        for i in top:
            print(f"  {FEATURE_NAMES[i]:18s} {imp.importances_mean[i]:+.4f}")
    except Exception as e:
        print("importance skipped:", e)


if __name__ == "__main__":
    main()
