"""Causal prosodic features for end-of-turn detection.

CAUSALITY CONTRACT (the whole point):
  For a pause starting at `pause_start`, every feature is computed from
  audio[0 : pause_start] only. We enforce this at the frame level: a frame
  spanning [t, t + frame_len) is used only if t + frame_len <= pause_start.
  From labels.csv we read ONLY turn_id, audio_file, pause_index, pause_start.
  `pause_end` of the CURRENT pause (and hence its duration) is future
  information and is never touched. Prior pauses of the same turn ended
  before this pause started, so their durations ARE in the causal past and
  are used as turn-structure context.

Feature groups:
  A. Turn-so-far normalizers  - speaker/channel-relative baselines (median
     F0, speech energy level) computed on all causal frames, so the same
     features transfer across speakers and languages (hidden set is mostly
     Hindi; absolute Hz/dB would not transfer).
  B. Final-window prosody     - F0 slope/level, energy slope/level, voicing,
     over the last 0.3 / 0.6 / 1.2 s before the pause. Statement-final falls
     vs. continuation-level/rising intonation live here.
  C. Final voiced run         - phrase-final lengthening (duration of the
     trailing voiced stretch vs. the turn's median run), pitch flatness of
     that run (long flat low-variance runs = hesitation vowels, "uh"/"um"
     -> hold), its pitch level vs. the turn median (final lowering -> eot).
  D. Turn structure           - elapsed time, pause index, prior-pause
     durations, length of the just-finished speech segment, speech fraction.
"""
import librosa
import numpy as np
from scipy.signal import medfilt

SR = 16000
HOP_MS = 10
E_FRAME_MS = 25
F0_FRAME_MS = 40
F0_MIN, F0_MAX = 60.0, 400.0
VOICING_THRESH = 0.30

EPS = 1e-12


# ----------------------------------------------------------------------
# contour extraction (once per wav, then sliced causally per pause)
# ----------------------------------------------------------------------

def load_wav(path):
    # librosa (allowed) instead of soundfile: float32, mono, native rate
    x, sr = librosa.load(path, sr=None, mono=True)
    return x.astype(np.float32), sr


def _frame(x, sr, frame_ms, hop_ms=HOP_MS):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32), fl, hp
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx], fl, hp


def energy_contour_db(x, sr):
    """Short-time RMS energy in dB per 25ms frame (10ms hop)."""
    fr, fl, hp = _frame(x, sr, E_FRAME_MS)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + EPS)
    return 20 * np.log10(rms + EPS)


def f0_contour(x, sr):
    """Per-frame F0 (Hz), 0.0 where unvoiced.

    Same estimator as the starter kit's autocorr_f0 (normalized
    autocorrelation peak in the 60-400 Hz lag range, voicing threshold
    0.30), but vectorized over all frames with FFT autocorrelation so the
    full corpus takes seconds instead of minutes.
    """
    fr, fl, hp = _frame(x, sr, F0_FRAME_MS)
    if len(fr) == 0:
        return np.zeros(0, dtype=np.float32)
    fr = fr - fr.mean(axis=1, keepdims=True)
    amax = np.abs(fr).max(axis=1)
    nfft = 1 << int(np.ceil(np.log2(2 * fl)))
    spec = np.fft.rfft(fr, n=nfft, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=1)[:, :fl]
    ac0 = ac[:, 0].copy()
    ok = (ac0 > 0) & (amax >= 1e-4)
    ac = ac / np.where(ac0 > 0, ac0, 1.0)[:, None]
    lo = int(sr / F0_MAX)
    hi = min(int(sr / F0_MIN), fl - 1)
    lag = lo + np.argmax(ac[:, lo:hi], axis=1)
    peak = ac[np.arange(len(fr)), lag]
    f0 = np.where(ok & (peak >= VOICING_THRESH), sr / lag, 0.0)
    # median filter kills isolated octave errors / single-frame voicing blips
    return medfilt(f0.astype(np.float32), kernel_size=3)


def centroid_contour(x, sr):
    """Per-frame spectral centroid (Hz). Separates voiced trail-off (low)
    from trailing breath noise (broadband, high) at turn offsets."""
    fr, fl, hp = _frame(x, sr, E_FRAME_MS)
    if len(fr) == 0:
        return np.zeros(0, dtype=np.float32)
    w = np.hanning(fl).astype(np.float32)
    mag = np.abs(np.fft.rfft(fr * w, axis=1))
    freqs = np.fft.rfftfreq(fl, 1.0 / sr).astype(np.float32)
    den = mag.sum(axis=1) + EPS
    return ((mag * freqs).sum(axis=1) / den).astype(np.float32)


def wav_contours(path):
    """Everything per-wav that features need; computed once and cached."""
    x, sr = load_wav(path)
    return {
        "e_db": energy_contour_db(x, sr),
        "f0": f0_contour(x, sr),
        "sc": centroid_contour(x, sr),
        "sr": sr,
        "hop_s": HOP_MS / 1000.0,
        "e_frame_s": E_FRAME_MS / 1000.0,
        "f0_frame_s": F0_FRAME_MS / 1000.0,
        "dur_s": len(x) / sr,
    }


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _st(hz):
    """Hz -> semitones re 55 Hz (log pitch scale, speaker-comparable)."""
    return 12.0 * np.log2(np.maximum(hz, 1.0) / 55.0)


def _slope(t, v):
    """Robust-ish linear slope (units/s); 0 if under 4 points."""
    if len(v) < 4:
        return 0.0
    return float(np.polyfit(t, v, 1)[0])


def _runs(mask):
    """[(start_idx, end_idx_exclusive)] of contiguous True stretches."""
    out = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


# ----------------------------------------------------------------------
# per-pause features
# ----------------------------------------------------------------------

FEATURE_NAMES = [
    # anchor / offset shape (how speech DIES into this pause)
    "pre_sil", "fall_time", "offset_slope", "tail_max_rel", "tail_mean_rel",
    "tail_frames", "tail_sc", "tail_voiced",
    # B. energy before the true offset (turn-relative dB)
    "e_final_rel", "e_slope150", "e_slope300", "e_slope600", "e_drop600",
    # B. pitch, windows measured back from speech end (voiced AND loud only)
    "f0_slope300", "f0_slope600", "f0_slope1200", "f0_fall",
    "f0_last_rel", "f0_last_pctl", "f0_min600_rel", "creak_frac",
    "voiced_frac300", "voiced_frac600", "voiced_frac1200",
    # C. final voiced run
    "run_dur", "run_dur_ratio", "run_f0_rel", "run_f0_std",
    "run_f0_slope", "run_e_std", "run_gap_before",
    # rate / activity
    "nruns_1200", "voiced_frac_turn", "rate_delta",
    # D. turn structure
    "elapsed", "pause_index", "n_prior", "last_seg_dur",
    "prior_dur_last", "prior_dur_mean", "prior_dur_max", "speech_frac",
]


def pause_features(c, pause_start, pause_index, prior):
    """Features for one pause.

    c            : wav_contours(path) dict
    pause_start  : seconds; audio at t >= pause_start is NEVER used
    pause_index  : int, 0-based within the turn
    prior        : list of (start, end) of EARLIER pauses in this turn
                   (their ends precede this pause_start -> causal past)

    The labels' pause_start trails the true acoustic offset by a variable
    silence gap (measured median ~44 dB below speech in the last 100 ms),
    so all "final" windows are anchored to speech_end = the last causal
    frame still within 25 dB of the turn's loud speech level. speech_end
    <= pause_start by construction, so causality is preserved.
    """
    hop = c["hop_s"]
    # causal frame slices: frame i covers [i*hop, i*hop + frame_len)
    n_e = max(0, int(np.floor((pause_start - c["e_frame_s"]) / hop)) + 1)
    n_f = max(0, int(np.floor((pause_start - c["f0_frame_s"]) / hop)) + 1)
    e = c["e_db"][:n_e]
    f0 = c["f0"][:n_f]
    f = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    if len(e) < 8 or len(f0) < 8:
        return f  # almost no context; all-zero row, model learns the prior

    te = np.arange(len(e)) * hop           # frame start times
    tf = np.arange(len(f0)) * hop
    voiced = f0 > 0

    # --- turn-so-far normalizers (A) ---
    e_hi = np.percentile(e, 95)
    speech_mask = e > (e_hi - 20.0)        # within 20 dB of loud speech
    e_speech_med = np.median(e[speech_mask]) if speech_mask.any() else np.median(e)
    if voiced.any():
        f0v = _st(f0[voiced])
        f0_med, f0_sd = np.median(f0v), max(np.std(f0v), 0.5)
    else:
        f0v = np.zeros(1); f0_med, f0_sd = 0.0, 1.0

    sc = c["sc"][:n_e]
    F = {}

    # --- anchor: true acoustic speech offset before this pause ---
    loud_mask_e = e > (e_hi - 25.0)
    loud = np.where(loud_mask_e)[0]
    if len(loud):
        k = loud[-1]
        speech_end = te[k] + c["e_frame_s"]      # end time of last loud frame
    else:
        k = len(e) - 1
        speech_end = pause_start
    F["pre_sil"] = max(0.0, pause_start - speech_end)

    # --- offset shape: how the speech dies into THIS pause (all causal:
    #     the tail region [speech_end, pause_start] precedes the pause) ---
    dec = e[max(0, k - 40):min(len(e), k + int(0.3 / hop))] - e_hi
    above10 = np.where(dec >= -10.0)[0]
    below30 = np.where(dec <= -30.0)[0]
    if len(above10) and len(below30) and below30[-1] > above10[-1]:
        later = below30[below30 > above10[-1]]
        F["fall_time"] = (later[0] - above10[-1]) * hop if len(later) else 0.3
    else:
        F["fall_time"] = 0.3
    o0 = max(0, k - int(0.15 / hop)); o1 = min(len(e), k + int(0.15 / hop))
    F["offset_slope"] = _slope(te[o0:o1], e[o0:o1])
    tail = np.arange(k + 1, len(e))
    if len(tail):
        F["tail_max_rel"] = float(e[tail].max() - e_speech_med)
        F["tail_mean_rel"] = float(e[tail].mean() - e_speech_med)
        F["tail_frames"] = float(len(tail)) * hop
        F["tail_sc"] = float(np.log1p(sc[tail].mean()))
        vt = (tf >= te[k]) if len(tf) else np.zeros(0, bool)
        F["tail_voiced"] = float(voiced[vt].mean()) if vt.any() else 0.0
    else:
        F["tail_max_rel"] = F["tail_mean_rel"] = -60.0
        F["tail_frames"] = F["tail_sc"] = F["tail_voiced"] = 0.0

    # loudness gate for pitch: voiced frames must sit inside loud speech,
    # otherwise the tracker's noise-floor F0 pollutes every statistic
    n_shift = max(1, int((c["f0_frame_s"] - c["e_frame_s"]) / hop))
    loud_f = np.zeros(len(tf), dtype=bool)
    ne = min(len(e), len(tf))
    loud_f[:ne] = loud_mask_e[:ne]
    vl = voiced & loud_f

    def ewin(w):
        m = (te >= speech_end - w) & (te + c["e_frame_s"] <= speech_end)
        return te[m], e[m]

    def fwin(w):
        m = (tf >= speech_end - w) & (tf + c["f0_frame_s"] <= speech_end + 2 * hop) & vl
        return tf[m], _st(f0[m])

    v_ok = vl & (tf + c["f0_frame_s"] <= speech_end + 2 * hop)
    vidx = np.where(v_ok)[0]
    if vl.any():
        f0v = _st(f0[vl])
        f0_med, f0_sd = np.median(f0v), max(np.std(f0v), 0.5)

    # --- B. energy before the true offset ---
    t15, e15 = ewin(0.15); t3, e3 = ewin(0.3); t6, e6 = ewin(0.6)
    F["e_final_rel"] = e[max(0, k - 2):k + 1].mean() - e_speech_med
    F["e_slope150"] = _slope(t15, e15)
    F["e_slope300"] = _slope(t3, e3)
    F["e_slope600"] = _slope(t6, e6)
    F["e_drop600"] = (e6.max() - e6[-3:].mean()) if len(e6) >= 3 else 0.0

    # --- B. pitch ---
    for name, w in (("f0_slope300", 0.3), ("f0_slope600", 0.6), ("f0_slope1200", 1.2)):
        tt, vv = fwin(w)
        F[name] = _slope(tt, vv)
    tt5, vv5 = fwin(0.5)
    last_v = _st(f0[vidx[-3:]]).mean() if len(vidx) >= 1 else 0.0
    F["f0_fall"] = 0.0
    if len(vv5) >= 3:
        early = vv5[tt5 <= speech_end - 0.2]
        if len(early):
            F["f0_fall"] = last_v - early.mean()
    F["f0_last_rel"] = (last_v - f0_med) / f0_sd
    F["f0_last_pctl"] = float((f0v <= last_v).mean()) if len(vidx) else 0.5
    t6f, v6f = fwin(0.6)
    F["f0_min600_rel"] = (v6f.min() - f0_med) / f0_sd if len(v6f) else 0.0
    # creak: very low F0 in the final stretch signals finality (esp. English)
    F["creak_frac"] = float((f0[vidx[-8:]] < 100.0).mean()) if len(vidx) >= 3 else 0.0
    for name, w in (("voiced_frac300", 0.3), ("voiced_frac600", 0.6),
                    ("voiced_frac1200", 1.2)):
        m = (tf >= speech_end - w) & (tf + c["f0_frame_s"] <= speech_end)
        F[name] = float(vl[m].mean()) if m.any() else 0.0

    # --- C. final voiced run (last run ending by the speech offset) ---
    runs = [r for r in _runs(v_ok) if r[1] - r[0] >= 2]
    F["run_dur"] = F["run_dur_ratio"] = F["run_f0_rel"] = 0.0
    F["run_f0_std"] = F["run_f0_slope"] = F["run_e_std"] = F["run_gap_before"] = 0.0
    if runs:
        s, epos = runs[-1]
        run_t, run_f = tf[s:epos], _st(f0[s:epos])
        durs = [(b - a) * hop for a, b in runs]
        F["run_dur"] = durs[-1]
        F["run_dur_ratio"] = durs[-1] / max(np.median(durs), 1e-3)
        F["run_f0_rel"] = (run_f.mean() - f0_med) / f0_sd
        F["run_f0_std"] = float(run_f.std())
        F["run_f0_slope"] = _slope(run_t, run_f)
        i0, i1 = int(run_t[0] / hop), min(int(run_t[-1] / hop) + 1, len(e))
        F["run_e_std"] = float(e[i0:i1].std()) if i1 > i0 else 0.0
        F["run_gap_before"] = (tf[s] - tf[runs[-2][1] - 1]) if len(runs) >= 2 else tf[s]

    # --- rate / activity ---
    m12 = (tf >= speech_end - 1.2) & (tf + c["f0_frame_s"] <= speech_end)
    F["nruns_1200"] = float(sum(1 for a, b in _runs(vl[m12]) if b > a))
    F["voiced_frac_turn"] = float(vl.mean())
    F["rate_delta"] = F["voiced_frac1200"] - F["voiced_frac_turn"]

    # --- D. turn structure ---
    F["elapsed"] = pause_start
    F["pause_index"] = float(pause_index)
    F["n_prior"] = float(len(prior))
    prev_end = prior[-1][1] if prior else 0.0
    F["last_seg_dur"] = pause_start - prev_end
    F["prior_dur_last"] = F["prior_dur_mean"] = F["prior_dur_max"] = 0.0
    if prior:
        pd = np.array([b - a for a, b in prior], dtype=np.float32)
        F["prior_dur_last"], F["prior_dur_mean"], F["prior_dur_max"] = \
            float(pd[-1]), float(pd.mean()), float(pd.max())
    F["speech_frac"] = (pause_start - sum(b - a for a, b in prior)) / max(pause_start, 1e-3)

    for i, n in enumerate(FEATURE_NAMES):
        f[i] = F[n]
    return f
