"""
Per-sample probability smoother for FOG detection.

Wraps HysteresisFilter and MajorityVotingFilter for both streaming
(sample-by-sample) and offline (full-array) use.
"""

import numpy as np
from .hysteresis_filter import HysteresisFilter


class MajorityVotingFilter:
    """Stateful causal majority-voting filter — call ``step()`` once per sample.

    Args:
        window_size: look-back window in samples (default 30)
    """

    def __init__(self, window_size=30):
        self.window_size = window_size
        self._buffer     = []

    def reset(self):
        """Reset filter state (call between trials)."""
        self._buffer = []

    def step(self, raw_pred):
        """Process one sample and return the majority-voted prediction.

        Args:
            raw_pred: int, 0 or 1

        Returns:
            voted: int8, 0 or 1
        """
        self._buffer.append(int(raw_pred))
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)
        return np.int8(1 if sum(self._buffer) > len(self._buffer) / 2 else 0)


def apply_hysteresis(binary_pred, enter_thresh=20, exit_thresh=5):
    """Apply causal hysteresis smoothing to a full binary prediction array (offline).

    Args:
        binary_pred:  (T,) binary array of raw thresholded predictions
        enter_thresh: consecutive 1s needed to enter FOG state
        exit_thresh:  consecutive 0s needed to exit  FOG state

    Returns:
        smoothed: (T,) binary int8 array
    """
    f = HysteresisFilter(enter_thresh=enter_thresh, exit_thresh=exit_thresh)
    return np.array([f.step(p) for p in binary_pred], dtype=np.int8)


def apply_majority_voting(labels_pred, window_size=30):
    """Apply causal majority-voting to a full binary prediction array (offline).

    Args:
        labels_pred: (T,) binary prediction array
        window_size: look-back window in samples

    Returns:
        output: (T,) filtered binary int8 array
    """
    if len(labels_pred) == 0:
        return np.array(labels_pred, dtype=np.int8)
    f = MajorityVotingFilter(window_size=window_size)
    return np.array([f.step(p) for p in labels_pred], dtype=np.int8)


def get_binary_predictions(probas, thresh=0.5,
                           use_hysteresis=True,
                           enter_thresh=20, exit_thresh=5,
                           use_majority_voting=False, majority_window=30):
    """Convert raw probabilities to binary FOG predictions (offline).

    Pipeline:
        1. Threshold probabilities → binary
        2a. Apply hysteresis smoothing  (default)
        2b. Apply majority-voting       (if use_majority_voting=True)

    For real-time use, instantiate HysteresisFilter directly and call
    step() per sample inside your inference loop.

    Args:
        probas:              (T,) probability array
        thresh:              binarisation threshold (default 0.5)
        use_hysteresis:      apply hysteresis (default True)
        enter_thresh:        consecutive 1s to enter FOG
        exit_thresh:         consecutive 0s to exit  FOG
        use_majority_voting: apply majority-voting instead (default False)
        majority_window:     look-back window for majority voting

    Returns:
        pred: (T,) binary int8 prediction array
    """
    pred = (np.asarray(probas) >= thresh).astype(int)

    if use_hysteresis:
        pred = apply_hysteresis(pred, enter_thresh=enter_thresh,
                                exit_thresh=exit_thresh)
    elif use_majority_voting:
        pred = apply_majority_voting(pred, window_size=majority_window)

    return pred
