"""
Real-time streaming inference pipeline for FOG detection.

Entry point: predict_streaming()

Input contract
--------------
- Sample rate  : 60 Hz  (resampled from raw 64 Hz before reaching this module)
- Window size  : 180 samples  (3 s × 60 Hz)
- Channels     : 30  (5 IMUs × 6 axes each)
                 Channel order per IMU: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
                 IMU order: Pelvic, Left_tibia, Left_talus, Right_tibia, Right_talus
- Normalisation: running zero-mean per channel over the last WINDOW_LEN samples
                 (normalise_streaming handles this internally — do NOT pre-normalise)

Model selection (LOSOCV)
------------------------
One .pt file exists per subject, named subject{id}.pt (e.g. subject010.pt).
Each model was trained with that subject held out (LOSOCV), so you MUST pass
the correct subject ID — this is what makes predictions valid for that subject.
Weights are expected in the weights/ folder next to this file, or pass weights_dir.

Production postprocessing hyperparameters
-----------------------------------------
- thresh       : 0.5
- enter_thresh : 20   (consecutive predicted-1s needed to enter FOG)
- exit_thresh  : 5    (consecutive predicted-0s needed to exit  FOG)
"""

import os
import numpy as np
import torch

from .tcn_model import TCNModel
from .hysteresis_filter import HysteresisFilter
from .smoothing import MajorityVotingFilter

# ── Constants (must match training pipeline) ──────────────────────────────────
TARGET_HZ  = 60
WINDOW_LEN = 180   # 3 s × 60 Hz
N_FEATURES = 30    # 5 IMUs × 6 axes
# ─────────────────────────────────────────────────────────────────────────────

_model_cache = {}


def _get_model(subject, weights_dir=None, n_features=N_FEATURES):
    """Load (and cache) the subject-specific TCNModel from disk.

    Each model was trained with its subject held out (LOSOCV), so you must
    pass the correct subject ID to get the model that never saw that subject.

    Filename convention: weights/subject{id}.pt  (e.g. subject010.pt)

    Args:
        subject:     subject ID string (e.g. '010')
        weights_dir: directory containing subject*.pt files.
                     Defaults to weights/ relative to this file.
        n_features:  number of input channels (default 30)

    Returns:
        model: TCNModel in eval mode
    """
    if weights_dir is None:
        weights_dir = os.path.join(os.path.dirname(__file__), 'weights')

    weights_path = os.path.join(weights_dir, f'subject{subject}.pt')

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f'No weights found for subject {subject} at {weights_path}. '
            f'Available files: {os.listdir(weights_dir)}'
        )

    if weights_path not in _model_cache:
        model = TCNModel(n_features)
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model.eval()
        _model_cache[weights_path] = model

    return _model_cache[weights_path]


def _normalise_streaming(block_raw, buffer):
    """Normalise a block using a rolling look-back buffer mean.

    For each timestep in the block, the mean is computed over the current
    buffer (past WINDOW_LEN samples). The buffer is updated sample by sample.

    Args:
        block_raw: (B, C) raw samples
        buffer:    (WINDOW_LEN, C) rolling buffer of past raw samples

    Returns:
        block_norm: (B, C) zero-centred block
        buffer:     updated rolling buffer
    """
    block_norm = np.zeros_like(block_raw)
    for t in range(len(block_raw)):
        block_norm[t] = block_raw[t] - np.mean(buffer, axis=0)
        buffer        = np.roll(buffer, -1, axis=0)
        buffer[-1]    = block_raw[t]
    return block_norm, buffer


def predict_streaming(trial_sequence, subject, weights_dir=None,
                      block_size=60, thresh=0.5,
                      use_hysteresis=True, enter_thresh=20, exit_thresh=5,
                      use_majority_voting=False, majority_window=30):
    """Run causal block-by-block streaming inference on a complete trial.

    Postprocessing (thresholding + hysteresis or majority voting) is applied
    sample-by-sample inside the inference loop, matching real-time deployment.

    Model selection (LOSOCV):
        One .pt file exists per subject, named subject{id}.pt (e.g. subject010.pt).
        Each model was trained with that subject held out, so you MUST pass the
        correct subject ID — this is what makes predictions valid for that subject.

    Args:
        trial_sequence:      (T, C) raw un-normalised feature array at 60 Hz.
                             C = 30  (5 IMUs × 6 axes).
                             Do NOT pre-normalise — normalisation is handled
                             internally via a rolling window mean.
        subject:             subject ID string (e.g. '010'). Selects
                             weights/subject{id}.pt.
        weights_dir:         directory containing subject*.pt files.
                             Defaults to weights/ relative to this file.
        block_size:          samples per inference block (default 60 = 1 s)
        thresh:              probability threshold for binarisation (default 0.5)
        use_hysteresis:      apply hysteresis smoothing (default True)
        enter_thresh:        consecutive 1s to enter FOG state (default 20)
        exit_thresh:         consecutive 0s to exit  FOG state (default 5)
        use_majority_voting: apply majority-voting instead of hysteresis (default False)
        majority_window:     look-back window for majority voting (default 30)

    Returns:
        probas: (T,) float32 — raw predicted FOG probability per timestep
        preds:  (T,) int8   — smoothed binary FOG prediction per timestep
    """
    model = _get_model(subject, weights_dir, n_features=trial_sequence.shape[1])
    model.tcn.reset_buffers()

    T, C   = trial_sequence.shape
    buffer = np.zeros((WINDOW_LEN, C), dtype=np.float32)
    probas = []
    preds  = []

    if use_hysteresis:
        postproc = HysteresisFilter(enter_thresh=enter_thresh, exit_thresh=exit_thresh)
    elif use_majority_voting:
        postproc = MajorityVotingFilter(window_size=majority_window)
    else:
        postproc = None

    with torch.no_grad():
        for i in range(0, T, block_size):
            block_raw          = trial_sequence[i:min(i + block_size, T)]
            block_norm, buffer = _normalise_streaming(block_raw, buffer)

            x   = torch.tensor(block_norm, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
            out = model(x, inference=True).squeeze().cpu().numpy()

            if out.ndim == 0:
                block_probas = [float(out)]
            elif out.ndim == 1:
                block_probas = out.tolist()
            else:
                raise ValueError(f'Unexpected model output shape: {out.shape}')

            for p in block_probas:
                probas.append(p)
                raw_pred = int(p >= thresh)
                if postproc is not None:
                    preds.append(postproc.step(raw_pred))
                else:
                    preds.append(raw_pred)

    return np.array(probas, dtype=np.float32), np.array(preds, dtype=np.int8)
