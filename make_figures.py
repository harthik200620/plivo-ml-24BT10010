"""REPORT FIGURES ONLY — not part of the model pipeline.

predict.py / train_model.py / train_cnn.py use only the allowed libraries
(numpy, scipy, scikit-learn, pandas, librosa, PyTorch). This script exists
solely to render pictures for SUMMARY.html and may use matplotlib.

    python make_figures.py --data_root ../eot/eot_data
"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_model import load_labels, official_score

C_EOT, C_HOLD = "#2e7d32", "#ef6c00"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    args = ap.parse_args()

    df = load_labels(args.data_root)
    df["dur"] = df.pause_end - df.pause_start
    oof = pd.concat([pd.read_csv(f"oof_{l}.csv").assign(lang=l)
                     for l in ("english", "hindi")])
    df = df.merge(oof, on=["turn_id", "pause_index", "lang"])
    df["p"] = df["p_eot"]

    # ---- 1. delay vs interruption budget: model vs silence baseline ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    budgets = np.arange(0.01, 0.11, 0.01)
    for ax, lang in zip(axes, ("english", "hindi")):
        part = df[df.lang == lang]
        base = part.copy()
        base["p"] = 1.0
        for d, name, col in ((part, "model (OOF)", "#1565c0"),
                             (base, "silence-only baseline", "#9e9e9e")):
            ys = [official_score(d.assign(p=d.p), b)["delay_ms"] for b in budgets]
            ax.plot(budgets * 100, ys, "-o", ms=4, label=name, color=col)
        ax.axvline(5, ls=":", c="k", lw=1)
        ax.set_title(lang)
        ax.set_xlabel("interrupted-turn budget (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean response delay (ms)")
    axes[0].legend()
    fig.suptitle("Latency vs interruption budget (out-of-fold)")
    fig.tight_layout()
    fig.savefig("fig_delay_curve.png", dpi=110)

    # ---- 2. live-endpointer view of two turns ----
    with open("contours_cache.pkl", "rb") as fh:
        cache = pickle.load(fh)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6))
    picks = []
    for lang in ("english", "hindi"):
        part = df[df.lang == lang]
        multi = part.groupby("turn_id").size()
        tid = multi[multi >= 3].index[0]
        picks.append(part[part.turn_id == tid])
    for ax, part in zip(axes, picks):
        c = cache[part.iloc[0].wav]
        e = c["e_db"]
        t = np.arange(len(e)) * 0.01
        ax.plot(t, e, lw=0.5, color="#607d8b")
        ax.set_ylabel("energy (dB)")
        ax2 = ax.twinx()
        for r in part.itertuples():
            col = C_EOT if r.label == "eot" else C_HOLD
            ax.axvspan(r.pause_start, r.pause_end, alpha=0.18, color=col)
            ax2.plot([r.pause_start], [r.p], "o", color=col, ms=9,
                     markeredgecolor="k", zorder=5)
            ax2.annotate(f"{r.p:.2f}", (r.pause_start, r.p),
                         textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("p_eot at pause")
        ax.set_title(f"{part.iloc[0].turn_id} — model output at every pause "
                     f"(green span = true end, orange = user continues)", fontsize=10)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig("fig_streaming.png", dpi=110)

    # ---- 3. score progression ----
    hist = [("baseline\n(silence)", 1600, 850),
            ("v1 scalars\n+GBM/LR", 1235, 850),
            ("v2 offset\nanchor", 1366, 857),
            ("v3 offset\nshape", 1250, 857),
            ("CNN v1\n+ens", 1210, 850)]
    final_scores = {}
    for lang, part in df.groupby("lang"):
        final_scores[lang] = official_score(part.assign(p=part.p))["delay_ms"]
    hist.append(("final: CNN⊕GBM\n(mined negs)", final_scores["english"], final_scores["hindi"]))
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = np.arange(len(hist))
    ax.bar(xs - 0.18, [h[1] for h in hist], 0.36, label="english", color="#1565c0")
    ax.bar(xs + 0.18, [h[2] for h in hist], 0.36, label="hindi", color="#ef6c00")
    ax.set_xticks(xs, [h[0] for h in hist], fontsize=8)
    ax.set_ylabel("mean delay (ms) @ <=5% interrupts")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Iteration history (all scores out-of-fold)")
    fig.tight_layout()
    fig.savefig("fig_history.png", dpi=110)
    print("wrote fig_delay_curve.png fig_streaming.png fig_history.png")


if __name__ == "__main__":
    main()
