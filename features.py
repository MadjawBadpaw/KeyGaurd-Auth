import numpy as np
from collections import deque
from typing import List, Tuple

# Increased from 40 → 56 with new feature blocks
FEATURE_DIM = 56

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(events: List[Tuple[str, float]]) -> "np.ndarray | None":
    """
    events: list of ("press"|"release"|"backspace", timestamp_ms)
    Returns a float32 vector of length FEATURE_DIM, or None if too few keys.
    """
    presses   = [t for typ, t in events if typ == "press"]
    releases  = [t for typ, t in events if typ == "release"]
    backspace_count = sum(1 for typ, _ in events if typ == "backspace")

    if len(presses) < 5:
        return None

    dwells    = _compute_dwells(events)
    flights   = _compute_flights(events)
    intervals = np.diff(presses) if len(presses) > 1 else np.array([])

    # ---- Block 1-3: core stat blocks (8 each = 24) -------------------------
    features  = []
    features += _stat_block(dwells,    8)   # hold-time distribution
    features += _stat_block(flights,   8)   # inter-key flight distribution
    features += _stat_block(intervals, 8)   # press-to-press distribution

    # ---- Block 4: speed & percentile landmarks (4) -------------------------
    total_time_s    = (presses[-1] - presses[0]) / 1000.0 if len(presses) > 1 else 1.0
    keys_per_second = len(presses) / max(total_time_s, 0.001)
    features += [
        keys_per_second,
        np.percentile(intervals, 25) if len(intervals) > 0 else 0,
        np.percentile(intervals, 75) if len(intervals) > 0 else 0,
        np.percentile(intervals, 90) if len(intervals) > 0 else 0,
    ]

    # ---- Block 5: pause statistics (4) -------------------------------------
    pauses     = intervals[intervals > 500] if len(intervals) > 0 else np.array([])
    pause_rate = len(pauses) / max(len(intervals), 1)
    features += [
        pause_rate,
        float(np.mean(pauses))  if len(pauses) > 0 else 0.0,
        float(np.std(pauses))   if len(pauses) > 0 else 0.0,
        float(len(pauses)),
    ]

    # ---- Block 6: rhythm fingerprint (4) -----------------------------------
    cv         = float(np.std(intervals) / np.mean(intervals)) \
                 if len(intervals) > 0 and np.mean(intervals) > 0 else 0.0
    burst      = _burstiness(intervals)
    accel      = _mean_acceleration(presses)
    rhythm_reg = _rhythm_regularity(intervals)
    features  += [cv, burst, accel, rhythm_reg]

    # ---- Block 7: extremes (4) ---------------------------------------------
    features += [
        float(np.max(intervals))  if len(intervals) > 0 else 0,
        float(np.min(intervals))  if len(intervals) > 0 else 0,
        float(np.max(dwells))     if len(dwells)    > 0 else 0,
        float(np.min(dwells))     if len(dwells)    > 0 else 0,
    ]

    # =========================================================================
    # NEW FEATURES (16)
    # =========================================================================

    # ---- Block 8: distribution shape — skewness & kurtosis (4) -------------
    # Captures asymmetry and tail-weight; two typists can share mean/std but
    # differ in shape (e.g. hunt-and-peck vs. touch-typist interval tails).
    features += [
        _skewness(intervals),
        _kurtosis(intervals),
        _skewness(dwells),
        _kurtosis(dwells),
    ]

    # ---- Block 9: temporal autocorrelation (2) ------------------------------
    # Lag-1: do fast intervals predict the next? Skilled typists show positive
    # autocorrelation (rhythmic runs); novices tend toward 0 or negative.
    # Lag-2: second-order rhythm (e.g. alternating-hand patterns).
    features += [
        _autocorr(intervals, lag=1),
        _autocorr(intervals, lag=2),
    ]

    # ---- Block 10: session warm-up effect (2) -------------------------------
    # Real users type faster/slower as they settle in.  Ratio > 1 → speeding
    # up; < 1 → slowing down.  The acceleration trajectory is user-specific.
    features += _warmup_features(presses)   # [warmup_ratio, warmup_slope]

    # ---- Block 11: rolling consistency (2) ----------------------------------
    # Std of rolling coefficient-of-variation over 5-key windows.
    # A steady typist has low rolling-CV variance; a distracted one has high.
    features += [
        _rolling_cv_stability(intervals, window=5),
        _rolling_mean_stability(intervals, window=5),
    ]

    # ---- Block 12: error & correction signal (2) ----------------------------
    # Backspace rate correlates with cognitive load / uncertainty and is highly
    # personal.  Also capture the median absolute deviation (robust spread).
    total_keys      = max(len(presses), 1)
    error_rate      = backspace_count / total_keys
    mad_intervals   = _median_absolute_deviation(intervals)
    features += [error_rate, mad_intervals]

    # ---- Block 13: interval entropy (2) ------------------------------------
    # Shannon entropy over 50 ms quantisation bins.  High-entropy → irregular,
    # low-entropy → mechanical / bot-like.  Normalised entropy ∈ [0, 1].
    # Also: entropy of dwell-time distribution.
    features += [
        _interval_entropy(intervals, bin_ms=50),
        _interval_entropy(dwells,    bin_ms=20),
    ]

    # =========================================================================
    # Final assembly
    # =========================================================================
    vec = np.array(features, dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    if len(vec) < FEATURE_DIM:
        vec = np.pad(vec, (0, FEATURE_DIM - len(vec)))
    else:
        vec = vec[:FEATURE_DIM]
    return vec


# ---------------------------------------------------------------------------
# Existing helpers (unchanged)
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
    stats = [
        float(np.mean(arr)),   float(np.std(arr)),    float(np.median(arr)),
        float(np.min(arr)),    float(np.max(arr)),
        float(np.percentile(arr, 25)), float(np.percentile(arr, 75)),
        float(np.sum(arr > np.mean(arr)) / len(arr)),   # fraction above mean
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


# ---------------------------------------------------------------------------
# New helpers
# ---------------------------------------------------------------------------

def _skewness(arr: np.ndarray) -> float:
    """Pearson's moment coefficient of skewness (scipy-free)."""
    if len(arr) < 3:
        return 0.0
    mu  = np.mean(arr)
    std = np.std(arr)
    if std < 1e-9:
        return 0.0
    return float(np.mean(((arr - mu) / std) ** 3))


def _kurtosis(arr: np.ndarray) -> float:
    """Excess kurtosis (Fisher definition; normal dist → 0)."""
    if len(arr) < 4:
        return 0.0
    mu  = np.mean(arr)
    std = np.std(arr)
    if std < 1e-9:
        return 0.0
    return float(np.mean(((arr - mu) / std) ** 4) - 3.0)


def _autocorr(arr: np.ndarray, lag: int = 1) -> float:
    """
    Pearson lag-k autocorrelation.
    Positive → rhythmic runs; negative → alternating fast/slow bursts.
    """
    if len(arr) <= lag:
        return 0.0
    x, y = arr[:-lag], arr[lag:]
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _warmup_features(presses: list) -> list:
    """
    Compare typing speed in the first quarter vs. last quarter of the session.
    Returns [ratio, slope]:
      ratio > 1 → user sped up (warmed up); ratio < 1 → slowed down (fatigued).
      slope     → linear trend of per-interval speed across the session.
    """
    if len(presses) < 8:
        return [1.0, 0.0]

    intervals = np.diff(presses)
    q          = max(1, len(intervals) // 4)
    first_mean = float(np.mean(intervals[:q]))
    last_mean  = float(np.mean(intervals[-q:]))

    ratio = (first_mean / last_mean) if last_mean > 1e-3 else 1.0
    ratio = float(np.clip(ratio, 0.0, 10.0))

    # Linear slope of interval over time (normalised by session length)
    x     = np.arange(len(intervals), dtype=np.float32)
    if len(x) >= 2:
        slope = float(np.polyfit(x, intervals, 1)[0])
    else:
        slope = 0.0

    return [ratio, slope]


def _rolling_cv_stability(arr: np.ndarray, window: int = 5) -> float:
    """
    Std of coefficient-of-variation computed in rolling windows.
    Low value → consistently paced; high value → variable bursts.
    """
    if len(arr) < window * 2:
        return 0.0
    cvs = []
    for i in range(len(arr) - window + 1):
        seg = arr[i:i + window]
        mu  = np.mean(seg)
        if mu > 1e-3:
            cvs.append(np.std(seg) / mu)
    return float(np.std(cvs)) if len(cvs) >= 2 else 0.0


def _rolling_mean_stability(arr: np.ndarray, window: int = 5) -> float:
    """
    Std of rolling window means, normalised by overall mean.
    Captures how uniformly a typist sustains their pace mid-session.
    """
    if len(arr) < window * 2:
        return 0.0
    means  = [float(np.mean(arr[i:i + window])) for i in range(len(arr) - window + 1)]
    global_mean = float(np.mean(arr))
    return float(np.std(means) / global_mean) if global_mean > 1e-3 else 0.0


def _median_absolute_deviation(arr: np.ndarray) -> float:
    """MAD: robust spread estimate, less sensitive to outlier keystrokes."""
    if len(arr) == 0:
        return 0.0
    return float(np.median(np.abs(arr - np.median(arr))))


def _interval_entropy(arr: np.ndarray, bin_ms: float = 50.0) -> float:
    """
    Shannon entropy of the interval distribution, normalised to [0, 1].
    Bots / replay attacks tend to produce very low entropy.
    """
    if len(arr) < 4:
        return 0.0
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < bin_ms:
        return 0.0
    n_bins = max(2, int((hi - lo) / bin_ms))
    counts, _ = np.histogram(arr, bins=n_bins)
    probs     = counts / counts.sum()
    probs     = probs[probs > 0]
    entropy   = -float(np.sum(probs * np.log2(probs)))
    max_entropy = np.log2(n_bins)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0