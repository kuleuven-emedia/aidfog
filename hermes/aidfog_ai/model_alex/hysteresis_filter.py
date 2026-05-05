"""
Stateful hysteresis filter for FOG detection postprocessing.

Usage (streaming / real-time):
    f = HysteresisFilter(enter_thresh=20, exit_thresh=5)
    smoothed = f.step(raw_pred)   # call once per sample
"""

import numpy as np


class HysteresisFilter:
    """Stateful causal hysteresis filter — call ``step()`` once per sample.

    Designed to be instantiated once per trial and then called sample-by-sample
    inside the inference loop, so state carries across block boundaries exactly
    as it would in real-time deployment.

    Args:
        enter_thresh: consecutive 1s needed to enter FOG state (default 20)
        exit_thresh:  consecutive 0s needed to exit  FOG state (default 5)
    """

    def __init__(self, enter_thresh=20, exit_thresh=5):
        self.enter_thresh = enter_thresh
        self.exit_thresh  = exit_thresh
        self.in_fog       = False
        self.consec1      = 0
        self.consec0      = 0

    def reset(self):
        """Reset filter state (call between trials)."""
        self.in_fog  = False
        self.consec1 = 0
        self.consec0 = 0

    def step(self, raw_pred):
        """Process one sample and return the smoothed prediction.

        Args:
            raw_pred: int, 0 or 1 — raw thresholded model output

        Returns:
            smoothed: int8, 0 or 1
        """
        p = int(raw_pred)

        if not self.in_fog:
            if p == 1:
                self.consec1 += 1
            else:
                self.consec1 = 0
            if self.consec1 >= self.enter_thresh:
                self.in_fog  = True
                self.consec0 = 0
            return np.int8(1 if self.in_fog else 0)
        else:
            if p == 0:
                self.consec0 += 1
            else:
                self.consec0 = 0
            if self.consec0 >= self.exit_thresh:
                self.in_fog  = False
                self.consec1 = 0
            return np.int8(1)   # always 1 while in FOG, even on the exit sample
