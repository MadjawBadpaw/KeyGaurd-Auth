"""
features.py — KeyGuard behavioural biometric feature extractor
==============================================================

Design philosophy
-----------------
Absolute keystroke timing is NOT a stable identity signal.  A user's raw
typing speed varies wildly by mood, time of day, content difficulty, input
device, and cognitive load.

This extractor focuses on THREE classes of features that remain stable across
those environmental shifts:

  1. RELATIVE / NORMALISED features — ratios, not magnitudes.
     e.g.  dwell/interval ratio, pause rate, intra-session speed gradient.
     These are ~invariant to a 2× overall tempo shift.

  2. STRUCTURAL RHYTHM features — HOW a user modulates speed, not how fast.
     e.g.  burstiness, autocorrelation, entropy, rolling CV stability.
     A distracted typist and a focused typist have different *shapes*, even if
     the distracted session is globally slower.

  3. FINGER / HAND TRANSITION PROXIES — bigram-class statistics.
     Adjacent-key pairs fall into structural classes (same-finger repeat,
     same-hand alternation, cross-hand alternation).  Each class has a
     characteristic timing that is tied to motor memory, not session speed.
     We can estimate class membership from flight-time patterns even without
     knowing the actual keys pressed.

Feature count: 96  (FEATURE_DIM = 96)
"""

import numpy as np
from collections import deque
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Public constant
# ---------------------------------------------------------------------------
FEATURE_DIM = 96


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(events: List[Tuple[str, float]]) -> "np.ndarray | None":
    """
    events: list of ("press" | "release" | "backspace", timestamp_ms)
    Returns a float32 vector of length FEATURE_DIM, or None if too few keys.
    """
    presses         = [t for typ, t in events if typ == "press"]
    releases        = [t for typ, t in events if typ == "release"]
    backspace_count = sum(1 for typ, _ in events if typ == "backspace")

    if len(presses) < 8:
        return None

    dwells    = _compute_dwells(events)
    flights   = _compute_flights(events)
    intervals = np.diff(presses).astype(np.float32) if len(presses) > 1 else np.array([], dtype=np.float32)

    # ── session-level tempo baseline (used for normalisation throughout) ──
    # Median interval is a robust tempo estimate; not mean (skewed by pauses).
    tempo_ref = float(np.median(intervals)) if len(intervals) > 0 else 1.0
    tempo_ref = max(tempo_ref, 1.0)   # guard against degenerate sessions

    features: list = []

    # =======================================================================
    # BLOCK A — absolute stat blocks (kept for backward compatibility signal
    #            direction, but now supplemented by normalised counterparts)
    #            3 × 8 = 24 features
    # =======================================================================
    features += _stat_block(dwells,    8)
    features += _stat_block(flights,   8)
    features += _stat_block(intervals, 8)

    # =======================================================================
    # BLOCK B — normalised / ratio features (tempo-invariant)
    #           These divide raw timing by the session median so a globally
    #           slower session produces the same values.
    #           3 × 8 = 24 features
    # =======================================================================
    features += _stat_block(dwells    / tempo_ref, 8)
    features += _stat_block(flights   / tempo_ref if len(flights) > 0 else flights, 8)
    features += _stat_block(intervals / tempo_ref, 8)

    # =======================================================================
    # BLOCK C — speed & percentile landmarks (8)
    # =======================================================================
    total_time_s    = (presses[-1] - presses[0]) / 1000.0 if len(presses) > 1 else 1.0
    keys_per_second = len(presses) / max(total_time_s, 0.001)
    features += [
        keys_per_second,
        float(np.percentile(intervals, 10)) / tempo_ref if len(intervals) > 0 else 0.0,
        float(np.percentile(intervals, 25)) / tempo_ref if len(intervals) > 0 else 0.0,
        float(np.percentile(intervals, 50)) / tempo_ref if len(intervals) > 0 else 0.0,
        float(np.percentile(intervals, 75)) / tempo_ref if len(intervals) > 0 else 0.0,
        float(np.percentile(intervals, 90)) / tempo_ref if len(intervals) > 0 else 0.0,
        # IQR / median — shape metric, not speed
        (float(np.percentile(intervals, 75)) - float(np.percentile(intervals, 25))) / tempo_ref
            if len(intervals) > 0 else 0.0,
        # dwell-to-interval ratio (finger contact fraction)
        float(np.mean(dwells)) / tempo_ref if len(dwells) > 0 else 0.0,
    ]

    # =======================================================================
    # BLOCK D — pause statistics (8)
    # A "pause" is any interval > 2× the session median (not a fixed 500 ms).
    # This makes pause detection relative to the user's own tempo.
    # =======================================================================
    pause_thresh = tempo_ref * 2.0
    pauses       = intervals[intervals > pause_thresh] if len(intervals) > 0 else np.array([])
    micro_pauses = intervals[(intervals > tempo_ref * 1.2) & (intervals <= pause_thresh)] \
                   if len(intervals) > 0 else np.array([])
    pause_rate   = len(pauses)      / max(len(intervals), 1)
    micro_rate   = len(micro_pauses)/ max(len(intervals), 1)
    features += [
        pause_rate,
        micro_rate,
        float(np.mean(pauses))       / tempo_ref if len(pauses) > 0 else 0.0,
        float(np.std(pauses))        / tempo_ref if len(pauses) > 0 else 0.0,
        float(len(pauses)),
        float(np.mean(micro_pauses)) / tempo_ref if len(micro_pauses) > 0 else 0.0,
        # pause clustering: fraction of pauses that occur back-to-back
        _pause_clustering(intervals, pause_thresh),
        # recovery speed after pause (next interval vs median)
        _post_pause_recovery(intervals, pause_thresh, tempo_ref),
    ]

    # =======================================================================
    # BLOCK E — rhythm fingerprint (8)
    # =======================================================================
    cv         = float(np.std(intervals) / np.mean(intervals)) \
                 if len(intervals) > 0 and np.mean(intervals) > 0 else 0.0
    burst      = _burstiness(intervals)
    accel      = _mean_acceleration(presses) / tempo_ref   # normalised
    rhythm_reg = _rhythm_regularity(intervals)
    features  += [
        cv,
        burst,
        accel,
        rhythm_reg,
        _skewness(intervals),
        _kurtosis(intervals),
        _autocorr(intervals, lag=1),
        _autocorr(intervals, lag=2),
    ]

    # =======================================================================
    # BLOCK F — distribution shape (8)
    # =======================================================================
    features += [
        _skewness(dwells),
        _kurtosis(dwells),
        _skewness(flights),
        _kurtosis(flights),
        # normalised range: (max-min)/median — captures spread relative to tempo
        (float(np.max(intervals)) - float(np.min(intervals))) / tempo_ref
            if len(intervals) > 1 else 0.0,
        (float(np.max(dwells)) - float(np.min(dwells))) / tempo_ref
            if len(dwells) > 1 else 0.0,
        _median_absolute_deviation(intervals) / tempo_ref,
        _median_absolute_deviation(dwells)    / tempo_ref,
    ]

    # =======================================================================
    # BLOCK G — temporal autocorrelation & memory (4)
    # =======================================================================
    features += [
        _autocorr(intervals, lag=3),
        _autocorr(intervals, lag=4),
        _autocorr(dwells,    lag=1),
        # partial autocorrelation proxy: AC(2) - AC(1)^2
        _autocorr(intervals, lag=2) - _autocorr(intervals, lag=1) ** 2,
    ]

    # =======================================================================
    # BLOCK H — session warm-up / fatigue (4)
    # Ratio-based: first-quarter vs last-quarter normalised by median.
    # =======================================================================
    features += _warmup_features(presses)   # [ratio, slope, first_q_cv, last_q_cv]

    # =======================================================================
    # BLOCK I — rolling consistency (4)
    # =======================================================================
    features += [
        _rolling_cv_stability(intervals,   window=5),
        _rolling_mean_stability(intervals, window=5),
        _rolling_cv_stability(intervals,   window=10),
        _rolling_mean_stability(intervals, window=10),
    ]

    # =======================================================================
    # BLOCK J — error & correction signals (4)
    # =======================================================================
    total_keys    = max(len(presses), 1)
    error_rate    = backspace_count / total_keys
    # Burst error: backspaces occurring in rapid succession (< 1 s apart)
    bs_times      = [t for typ, t in events if typ == "backspace"]
    bs_bursts     = sum(1 for i in range(1, len(bs_times))
                        if bs_times[i] - bs_times[i-1] < 1000) / max(len(bs_times), 1) \
                    if len(bs_times) > 0 else 0.0
    features += [
        error_rate,
        bs_bursts,
        float(backspace_count),
        _median_absolute_deviation(intervals) / tempo_ref,   # robust spread (alias used here)
    ]

    # =======================================================================
    # BLOCK K — entropy (4)
    # =======================================================================
    features += [
        _interval_entropy(intervals, bin_frac=0.10),  # 10% of tempo_ref bin width
        _interval_entropy(dwells,    bin_frac=0.05),
        _interval_entropy(flights,   bin_frac=0.10),
        _bigram_class_entropy(intervals),              # structural key-transition entropy
    ]

    # =======================================================================
    # BLOCK L — bigram-class transition statistics (8)
    # We bucket consecutive intervals into 3 classes by comparing each pair
    # to the session median:
    #   fast-fast, fast-slow, slow-fast, slow-slow
    # Counts are normalised to fractions.  This captures motor memory patterns
    # (e.g. a touch typist executes many fast-fast transitions on common
    #  bigrams, then inserts deliberate pauses before unfamiliar trigrams).
    # =======================================================================
    ff, fs, sf, ss = _bigram_transition_fracs(intervals, tempo_ref)
    features += [ff, fs, sf, ss,
                 ff / max(ss, 1e-3),   # "flow ratio" — how often does speed sustain vs stall
                 (fs + sf) / max(ff + ss, 1e-3),   # alternation ratio
                 _run_length_mean(intervals, tempo_ref, fast=True),   # mean fast-run length
                 _run_length_mean(intervals, tempo_ref, fast=False),  # mean slow-run length
                 ]

    # =======================================================================
    # FINAL ASSEMBLY
    # =======================================================================
    vec = np.array(features, dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    if len(vec) < FEATURE_DIM:
        vec = np.pad(vec, (0, FEATURE_DIM - len(vec)))
    else:
        vec = vec[:FEATURE_DIM]
    return vec


# ---------------------------------------------------------------------------
# Core helpers (existing, mostly unchanged)
# ---------------------------------------------------------------------------

def _compute_dwells(events):
    press_times = deque()
    dwells = []
    for typ, t in events:
        if typ == "press":
            press_times.append(t)
        elif typ == "release" and press_times:
            p = press_times.popleft()
            d = t - p
            if 0 < d < 2000:
                dwells.append(d)
    return np.array(dwells, dtype=np.float32)


def _compute_flights(events):
    releases = [t for typ, t in events if typ == "release"]
    presses  = [t for typ, t in events if typ == "press"]
    flights  = []
    ri = pi = 0
    while ri < len(releases) and pi < len(presses):
        if presses[pi] > releases[ri]:
            f = presses[pi] - releases[ri]
            if 0 < f < 2000:
                flights.append(f)
            ri += 1
        else:
            pi += 1
    return np.array(flights, dtype=np.float32)


def _stat_block(arr, n):
    if len(arr) == 0:
        return [0.0] * n
    arr = np.asarray(arr, dtype=np.float32)
    stats = [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.median(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 75)),
        float(np.sum(arr > np.mean(arr)) / len(arr)),
    ]
    return stats[:n] + [0.0] * max(0, n - len(stats))


def _burstiness(intervals):
    if len(intervals) < 2:
        return 0.0
    mu, sigma = np.mean(intervals), np.std(intervals)
    return float((sigma - mu) / (sigma + mu)) if (sigma + mu) != 0 else 0.0


def _mean_acceleration(presses):
    if len(presses) < 3:
        return 0.0
    return float(np.mean(np.diff(np.diff(presses))))


def _rhythm_regularity(intervals):
    if len(intervals) < 4:
        return 0.0
    iqr = float(np.percentile(intervals, 75) - np.percentile(intervals, 25))
    return float(np.std(intervals) / iqr) if iqr != 0 else 0.0


def _skewness(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) < 3:
        return 0.0
    mu, std = np.mean(arr), np.std(arr)
    if std < 1e-9:
        return 0.0
    return float(np.mean(((arr - mu) / std) ** 3))


def _kurtosis(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) < 4:
        return 0.0
    mu, std = np.mean(arr), np.std(arr)
    if std < 1e-9:
        return 0.0
    return float(np.mean(((arr - mu) / std) ** 4) - 3.0)


def _autocorr(arr: np.ndarray, lag: int = 1) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) <= lag:
        return 0.0
    x, y = arr[:-lag], arr[lag:]
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _warmup_features(presses: list) -> list:
    """
    Returns [ratio, slope, first_q_cv, last_q_cv].
    All normalised/ratio-based — invariant to global tempo.
    """
    if len(presses) < 8:
        return [1.0, 0.0, 0.0, 0.0]

    intervals = np.diff(presses).astype(np.float32)
    q          = max(1, len(intervals) // 4)
    first_q    = intervals[:q]
    last_q     = intervals[-q:]

    first_mean = float(np.mean(first_q))
    last_mean  = float(np.mean(last_q))
    ratio      = float(np.clip(first_mean / max(last_mean, 1e-3), 0.0, 10.0))

    x = np.arange(len(intervals), dtype=np.float32)
    slope = float(np.polyfit(x, intervals / float(np.mean(intervals)), 1)[0]) \
            if len(x) >= 2 else 0.0

    cv = lambda a: float(np.std(a) / np.mean(a)) if np.mean(a) > 1e-3 else 0.0
    return [ratio, slope, cv(first_q), cv(last_q)]


def _rolling_cv_stability(arr: np.ndarray, window: int = 5) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) < window * 2:
        return 0.0
    cvs = []
    for i in range(len(arr) - window + 1):
        seg = arr[i:i + window]
        mu  = np.mean(seg)
        if mu > 1e-3:
            cvs.append(float(np.std(seg) / mu))
    return float(np.std(cvs)) if len(cvs) >= 2 else 0.0


def _rolling_mean_stability(arr: np.ndarray, window: int = 5) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) < window * 2:
        return 0.0
    means       = [float(np.mean(arr[i:i + window])) for i in range(len(arr) - window + 1)]
    global_mean = float(np.mean(arr))
    return float(np.std(means) / global_mean) if global_mean > 1e-3 else 0.0


def _median_absolute_deviation(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == 0:
        return 0.0
    return float(np.median(np.abs(arr - np.median(arr))))


def _interval_entropy(arr: np.ndarray, bin_frac: float = 0.10) -> float:
    """
    Shannon entropy with bin width = bin_frac × median(arr).
    Normalised to [0, 1].  Relative binning makes this tempo-invariant.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) < 4:
        return 0.0
    med    = float(np.median(arr))
    bin_ms = max(med * bin_frac, 1.0)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < bin_ms:
        return 0.0
    n_bins         = max(2, int((hi - lo) / bin_ms))
    counts, _      = np.histogram(arr, bins=n_bins)
    probs          = counts / counts.sum()
    probs          = probs[probs > 0]
    entropy        = -float(np.sum(probs * np.log2(probs)))
    max_entropy    = np.log2(n_bins)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


# ---------------------------------------------------------------------------
# New helpers
# ---------------------------------------------------------------------------

def _pause_clustering(intervals: np.ndarray, thresh: float) -> float:
    """
    Fraction of pauses that are immediately followed by another pause.
    High → user stops in multi-key bursts (thinker); low → isolated pauses.
    """
    if len(intervals) < 2:
        return 0.0
    is_pause = intervals > thresh
    count    = sum(1 for i in range(len(is_pause) - 1)
                   if is_pause[i] and is_pause[i + 1])
    total    = max(int(np.sum(is_pause)), 1)
    return count / total


def _post_pause_recovery(intervals: np.ndarray, thresh: float, tempo_ref: float) -> float:
    """
    Mean of the interval immediately following a pause, normalised by tempo_ref.
    A confident typist resumes quickly; a hesitant one stays slow.
    """
    recoveries = []
    for i in range(len(intervals) - 1):
        if intervals[i] > thresh:
            recoveries.append(intervals[i + 1])
    if not recoveries:
        return 1.0
    return float(np.mean(recoveries)) / tempo_ref


def _bigram_transition_fracs(intervals: np.ndarray, tempo_ref: float) -> tuple:
    """
    Classify consecutive interval pairs as fast-fast, fast-slow, slow-fast, slow-slow.
    Returns (ff, fs, sf, ss) as fractions summing to 1.
    """
    if len(intervals) < 2:
        return (0.25, 0.25, 0.25, 0.25)
    fast = intervals < tempo_ref
    ff = fs = sf = ss = 0
    for i in range(len(fast) - 1):
        if   fast[i] and     fast[i+1]: ff += 1
        elif fast[i] and not fast[i+1]: fs += 1
        elif not fast[i] and fast[i+1]: sf += 1
        else:                           ss += 1
    total = max(ff + fs + sf + ss, 1)
    return (ff/total, fs/total, sf/total, ss/total)


def _run_length_mean(intervals: np.ndarray, tempo_ref: float, fast: bool = True) -> float:
    """
    Mean length of consecutive fast (or slow) interval runs.
    Reflects how long a user sustains a rhythm before breaking it.
    """
    if len(intervals) == 0:
        return 0.0
    mask   = (intervals < tempo_ref) if fast else (intervals >= tempo_ref)
    runs   = []
    count  = 0
    for m in mask:
        if m:
            count += 1
        else:
            if count > 0:
                runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return float(np.mean(runs)) if runs else 0.0


def _bigram_class_entropy(intervals: np.ndarray) -> float:
    """
    Shannon entropy of the bigram-transition class distribution (ff/fs/sf/ss).
    A highly predictable typist has low entropy; an irregular one has high.
    Normalised to [0, 1] (max = log2(4) = 2 bits).
    """
    if len(intervals) < 2:
        return 0.0
    tempo_ref = max(float(np.median(intervals)), 1.0)
    ff, fs, sf, ss = _bigram_transition_fracs(intervals, tempo_ref)
    probs = np.array([ff, fs, sf, ss], dtype=np.float64)
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log2(probs)))
    return float(entropy / 2.0)   # normalise by log2(4)