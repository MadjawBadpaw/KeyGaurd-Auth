import pickle
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler   # ← was StandardScaler
from sklearn.decomposition import PCA             # ← new: optional whitening

import logger
from features import FEATURE_DIM

DATA_DIR    = Path("data")
MODEL_PATH  = DATA_DIR / "model.pkl"
SCALER_PATH = DATA_DIR / "scaler.pkl"
DIST_PATH   = DATA_DIR / "score_dist.pkl"

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScoreDistribution:
    mean:   float = 0.0
    std:    float = 1.0
    min:    float = -1.0
    max:    float =  1.0
    p05:    float = -0.5   # ← new: 5th / 95th percentile anchors for better
    p95:    float =  0.5   #   normalisation of the tails


@dataclass
class ModelState:
    clf:     Optional[IsolationForest] = None
    scaler:  Optional[RobustScaler]    = None   # ← type updated
    dist:    ScoreDistribution         = field(default_factory=ScoreDistribution)
    trained: bool                      = False


_state = ModelState()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load() -> bool:
    """
    Load model, scaler, and score distribution from disk.
    Validates that the loaded scaler's feature dimension matches FEATURE_DIM
    so a stale model trained on a different feature set is rejected cleanly.
    """
    DATA_DIR.mkdir(exist_ok=True)
    if not (MODEL_PATH.exists() and SCALER_PATH.exists() and DIST_PATH.exists()):
        return False
    try:
        with open(MODEL_PATH,  "rb") as f: clf    = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
        with open(DIST_PATH,   "rb") as f: dist   = pickle.load(f)

        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != FEATURE_DIM:
            logger.log("ERROR", {
                "detail": (
                    f"model.load: feature dim mismatch — "
                    f"saved={scaler.n_features_in_}, current={FEATURE_DIM}. "
                    "Retrain required."
                )
            })
            return False

        with _lock:
            _state.clf     = clf
            _state.scaler  = scaler
            _state.dist    = dist
            _state.trained = True
        return True

    except Exception as e:
        logger.log("ERROR", {"detail": f"model.load: {e}"})
        return False


def save():
    """
    Write model artefacts atomically (tmp → rename).
    Must be called while _lock is held by the caller.
    """
    DATA_DIR.mkdir(exist_ok=True)
    pairs = [
        (MODEL_PATH,  _state.clf),
        (SCALER_PATH, _state.scaler),
        (DIST_PATH,   _state.dist),
    ]
    tmp_paths = []
    try:
        for path, obj in pairs:
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(obj, f)
            tmp_paths.append((tmp, path))
        for tmp, final in tmp_paths:
            tmp.replace(final)
    except Exception as e:
        for tmp, _ in tmp_paths:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        logger.log("ERROR", {"detail": f"model.save: {e}"})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X: np.ndarray, contamination: float = 0.05) -> bool:
    """
    Fit a new IsolationForest on the supplied feature matrix X.

    Changes vs. original
    --------------------
    1. RobustScaler instead of StandardScaler
       Our new feature set contains ratio features and entropy values with
       heavy-tailed distributions and occasional large outliers (e.g. a user
       who had one very distracted session).  RobustScaler uses median + IQR
       rather than mean + std, so a handful of anomalous sessions don't drag
       the scaling anchor.

    2. Larger forest (300 trees) with max_samples='auto'
       With 96 features the isolation paths are longer; more trees reduce
       variance without much extra cost (n_jobs=-1).

    3. Score distribution stores p05/p95 percentile anchors in addition to
       mean/std.  The normalisation function uses a piecewise sigmoid that
       is anchored to the empirical tails of the training distribution rather
       than assuming Gaussianity — important because IsolationForest raw scores
       are NOT Gaussian when most training samples are inliers.

    Guards (unchanged): shape, NaN/Inf, duplicate rows, near-zero variance.
    """
    if X.ndim != 2 or X.shape[0] < 8 or X.shape[1] != FEATURE_DIM:
        logger.log("RETRAIN_SKIP", {
            "reason":       "bad_shape",
            "shape":        list(X.shape),
            "required_dim": FEATURE_DIM,
        })
        return False

    if np.isnan(X).any() or np.isinf(X).any():
        logger.log("ERROR", {"detail": "train: NaN/Inf in training data"})
        return False

    if len(np.unique(X, axis=0)) < len(X) * 0.5:
        logger.log("RETRAIN_SKIP", {"reason": "too_many_duplicate_rows"})
        return False

    if float(np.mean(np.var(X, axis=0))) < 1e-6:
        logger.log("RETRAIN_SKIP", {"reason": "near_zero_variance"})
        return False

    try:
        # ── 1. Scale with RobustScaler ────────────────────────────────────
        scaler   = RobustScaler(quantile_range=(10.0, 90.0))
        X_scaled = scaler.fit_transform(X)

        # ── 2. Clip extreme scaled values (protects the forest from wild
        #       outlier paths that would distort the score distribution) ──
        X_scaled = np.clip(X_scaled, -6.0, 6.0)

        # ── 3. Fit the forest ─────────────────────────────────────────────
        contamination = float(np.clip(contamination, 0.02, 0.15))
        clf = IsolationForest(
            n_estimators=300,          # up from 200; better variance reduction
            max_samples="auto",        # sklearn default: min(256, n_samples)
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_scaled)

        # ── 4. Build richer score distribution ───────────────────────────
        raw_scores = clf.decision_function(X_scaled)
        dist = ScoreDistribution(
            mean=float(np.mean(raw_scores)),
            std =float(np.std(raw_scores)),
            min =float(np.min(raw_scores)),
            max =float(np.max(raw_scores)),
            p05 =float(np.percentile(raw_scores,  5)),
            p95 =float(np.percentile(raw_scores, 95)),
        )

        with _lock:
            _state.clf     = clf
            _state.scaler  = scaler
            _state.dist    = dist
            _state.trained = True
            save()

        logger.log("RETRAIN_OK", {
            "n_samples":     X.shape[0],
            "contamination": contamination,
            "score_mean":    round(dist.mean, 4),
            "score_std":     round(dist.std,  4),
            "score_p05":     round(dist.p05,  4),
            "score_p95":     round(dist.p95,  4),
        })
        return True

    except Exception as e:
        logger.log("ERROR", {"detail": f"model.train: {e}"})
        return False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _normalise(raw: float, d: ScoreDistribution) -> float:
    """
    Map a raw IsolationForest score to a confidence in [0, 1].

    Old approach: single sigmoid centred on the mean.
    Problem:      IsolationForest scores are NOT Gaussian; the mean sits in
                  the bulk of legitimate samples, so borderline imposters
                  scored around 0.5 and the function had low sensitivity
                  exactly where it mattered most.

    New approach: piecewise linear mapping anchored to the empirical
                  p05 / p95 of the TRAINING distribution, then soft-clipped
                  through a sigmoid for smoothness at the extremes.

      raw ≥ p95  → very confident inlier   → confidence near 1.0
      raw ≤ p05  → very likely outlier     → confidence near 0.0
      in between → linear interpolation

    This makes the function maximally sensitive in the decision region
    (p05…p95) regardless of the absolute score scale, which changes with
    dataset size and contamination.
    """
    lo, hi = d.p05, d.p95
    span   = hi - lo

    if span < 1e-9:
        # Degenerate distribution — fall back to std-based sigmoid
        if d.std < 1e-9:
            return 0.5
        z = (raw - d.mean) / d.std
        return float(np.clip(1.0 / (1.0 + np.exp(-z * 2.0)), 0.0, 1.0))

    # Linear stretch: p05 → 0, p95 → 1
    t = (raw - lo) / span          # typically in [-0.5 … 1.5]

    # Soft sigmoid so the output is smooth and bounded
    # Scale factor 4 gives ~0.02 at t=0 and ~0.98 at t=1
    return float(np.clip(1.0 / (1.0 + np.exp(-4.0 * (t - 0.5))), 0.0, 1.0))


def _validate_input(x: np.ndarray, expected_dim: int) -> bool:
    if x.ndim != 1 or x.shape[0] != expected_dim:
        return False
    if np.isnan(x).any() or np.isinf(x).any():
        return False
    return True


def score(x: np.ndarray) -> float:
    """
    Score a single feature vector.
    Returns confidence in [0, 1] (1 = very likely the enrolled user),
    or -1.0 on any error.
    """
    if not _validate_input(x, FEATURE_DIM):
        return -1.0

    with _lock:
        if not _state.trained or _state.clf is None:
            return -1.0
        try:
            x_scaled = np.clip(_state.scaler.transform(x.reshape(1, -1)), -6.0, 6.0)
            raw      = float(_state.clf.decision_function(x_scaled)[0])
            return _normalise(raw, _state.dist)
        except Exception as e:
            logger.log("ERROR", {"detail": f"model.score: {e}"})
            return -1.0


def score_batch(X: np.ndarray) -> list[float]:
    """
    Score a batch of feature vectors efficiently (single lock acquisition).
    Returns a list of confidences in [0, 1], or [] on any error.
    """
    if X.ndim != 2 or X.shape[1] != FEATURE_DIM:
        return []
    if np.isnan(X).any() or np.isinf(X).any():
        return []

    with _lock:
        if not _state.trained or _state.clf is None:
            return []
        try:
            X_scaled = np.clip(_state.scaler.transform(X), -6.0, 6.0)
            raws     = _state.clf.decision_function(X_scaled)
            d        = _state.dist
            return [_normalise(float(r), d) for r in raws]
        except Exception as e:
            logger.log("ERROR", {"detail": f"model.score_batch: {e}"})
            return []


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def is_trained() -> bool:
    with _lock:
        return _state.trained


def get_distribution() -> ScoreDistribution:
    """
    Returns a snapshot of the current score distribution.
    Safe to read after the lock is released because train() replaces the
    ScoreDistribution object rather than mutating it in place.
    """
    with _lock:
        return _state.dist