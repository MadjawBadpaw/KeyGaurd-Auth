import pickle
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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
    mean: float = 0.0
    std:  float = 1.0
    min:  float = -1.0
    max:  float =  1.0


@dataclass
class ModelState:
    clf:     Optional[IsolationForest] = None
    scaler:  Optional[StandardScaler]  = None
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

        # Reject models trained on a different feature dimension
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
    Write model artefacts atomically.
    Each file is written to a .tmp sibling first; all three are then renamed
    in one pass so a mid-write crash never leaves a partially-updated set.
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
        # All writes succeeded — atomically promote
        for tmp, final in tmp_paths:
            tmp.replace(final)      # atomic on POSIX; best-effort on Windows
    except Exception as e:
        # Clean up any .tmp files left behind
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

    Guards:
      - minimum sample count and correct feature dimension
      - NaN / Inf in training data
      - near-duplicate rows (e.g. copy-pasted feature vectors)
      - near-zero variance (constant features)
    """
    if X.ndim != 2 or X.shape[0] < 8 or X.shape[1] != FEATURE_DIM:
        logger.log("RETRAIN_SKIP", {
            "reason": "bad_shape",
            "shape":  list(X.shape),
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
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        contamination = float(np.clip(contamination, 0.02, 0.15))
        clf = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_scaled)

        raw_scores = clf.decision_function(X_scaled)
        dist = ScoreDistribution(
            mean=float(np.mean(raw_scores)),
            std =float(np.std(raw_scores)),
            min =float(np.min(raw_scores)),
            max =float(np.max(raw_scores)),
        )

        # Hold the lock for the entire state-update + save so no reader
        # can observe a half-updated state and no concurrent train() can
        # interleave its save() with ours.
        with _lock:
            _state.clf     = clf
            _state.scaler  = scaler
            _state.dist    = dist
            _state.trained = True
            save()          # called inside the lock — see save() docstring

        return True

    except Exception as e:
        logger.log("ERROR", {"detail": f"model.train: {e}"})
        return False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _normalise(raw: float, d: ScoreDistribution) -> float:
    """Map a raw IsolationForest score to a confidence in [0, 1]."""
    if d.std < 1e-9:
        return 0.5
    z = (raw - d.mean) / d.std
    return float(np.clip(1.0 / (1.0 + np.exp(-z * 1.5)), 0.0, 1.0))


def _validate_input(x: np.ndarray, expected_dim: int) -> bool:
    """Return False if the vector has wrong shape, NaNs, or Infs."""
    if x.ndim != 1 or x.shape[0] != expected_dim:
        return False
    if np.isnan(x).any() or np.isinf(x).any():
        return False
    return True


def score(x: np.ndarray) -> float:
    """
    Score a single feature vector.
    Returns a confidence in [0, 1] (1 = very likely the enrolled user),
    or -1.0 on any error.
    """
    if not _validate_input(x, FEATURE_DIM):
        return -1.0

    with _lock:
        if not _state.trained or _state.clf is None:
            return -1.0
        try:
            x_scaled = _state.scaler.transform(x.reshape(1, -1))
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
            X_scaled = _state.scaler.transform(X)
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