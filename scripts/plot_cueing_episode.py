"""
Plot a 5-track timeline of an aidfog trial:

  Track 1  FoG probability       (continuous 0-1, from aidfog_ai logits[1])
  Track 2  Smoothed prediction   (binary step, from aidfog_ai prediction)
  Track 3  Ground-truth label    (binary step, from aidfog_replay fog_label)
  Track 4  Cueing FSM status     (uint8 step, from aidfog cueing/status)
  Track 5  BLE GATT write events (vertical markers on all tracks)

Usage:
    python scripts/plot_cueing_episode.py path/to/trial_folder
    python scripts/plot_cueing_episode.py path/to/trial_folder --start 10 --end 30
    python scripts/plot_cueing_episode.py path/to/trial_folder --output custom.png

All times are rendered as seconds since the first recorded sample.
"""

import argparse
import json
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _load_stream(path: str, device: str, stream: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (process_time_s, values) with padding rows trimmed off."""
    with h5py.File(path, "r") as f:
        pt = f[f"{os.path.basename(path).split('.')[0]}/{device}/process_time_s"][:].flatten()
        v = f[f"{os.path.basename(path).split('.')[0]}/{device}/{stream}"][:]
    mask = pt > 0
    return pt[mask], v[mask]


def _exists(path: str) -> bool:
    if not os.path.exists(path):
        print(f"warning: missing {path}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_dir", help="Path to trial_<name> folder")
    parser.add_argument("--start", type=float, default=None,
                        help="Start of x-axis in seconds since trial start (default: 0)")
    parser.add_argument("--end", type=float, default=None,
                        help="End of x-axis in seconds since trial start (default: full trial)")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: trial_dir/cueing_episode.png)")
    args = parser.parse_args()

    ai_path = os.path.join(args.trial_dir, "aidfog_ai.hdf5")
    replay_path = os.path.join(args.trial_dir, "aidfog_replay.hdf5")
    cueing_path = os.path.join(args.trial_dir, "aidfog.hdf5")
    op_log_path = os.path.join(args.trial_dir, "aidfog_ble_op_log.json")

    # aidfog_ai.hdf5 is required; the others are optional.
    if not _exists(ai_path):
        sys.exit(1)

    with h5py.File(ai_path, "r") as f:
        ai_pt = f["aidfog_ai/pytorch-worker/process_time_s"][:].flatten()
        ai_logits = f["aidfog_ai/pytorch-worker/logits"][:]
        ai_pred = f["aidfog_ai/pytorch-worker/prediction"][:].flatten()
    ai_mask = ai_pt > 0
    ai_pt = ai_pt[ai_mask]
    ai_logits = ai_logits[ai_mask]
    ai_pred = ai_pred[ai_mask]

    # Raw logits can range anywhere; map logit[1] to [0,1] via softmax for the
    # "FoG probability" track. The underlying BudsPipeline still acts on the
    # raw logit, but for plotting a probability is more intuitive.
    lg = ai_logits
    shifted = lg - lg.max(axis=1, keepdims=True)
    expd = np.exp(shifted)
    softmax = expd / expd.sum(axis=1, keepdims=True)
    fog_prob = softmax[:, 1]

    t0 = ai_pt[0]

    # Ground-truth label is optional — aidfog_replay.hdf5 has a fog_label
    # sub-stream. Not present for live-IMU runs.
    replay_t = replay_v = None
    if _exists(replay_path):
        with h5py.File(replay_path, "r") as f:
            replay_t = f["aidfog_replay/dots-imu/process_time_s"][:].flatten()
            replay_v = f["aidfog_replay/dots-imu/fog_label"][:].flatten()
        m = replay_t > 0
        replay_t = replay_t[m]
        replay_v = replay_v[m]

    # Cueing FSM status history.
    cueing_t = cueing_v = None
    if _exists(cueing_path):
        with h5py.File(cueing_path, "r") as f:
            cueing_t = f["aidfog/cueing/process_time_s"][:].flatten()
            cueing_v = f["aidfog/cueing/status"][:].flatten()
        m = cueing_t > 0
        cueing_t = cueing_t[m]
        cueing_v = cueing_v[m]

    # BLE GATT writes — overlay as vertical markers.
    op_starts = []
    op_stops = []
    if _exists(op_log_path):
        with open(op_log_path) as f:
            ops = json.load(f)
        op_starts = [o["timestamp"] for o in ops
                     if o.get("operation") == "start_cue" and o.get("success")]
        op_stops = [o["timestamp"] for o in ops
                    if o.get("operation") == "stop_cue" and o.get("success")]

    fig, axes = plt.subplots(
        nrows=4, ncols=1, sharex=True, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
    )

    # Track 1 — FoG probability + AI raw prediction toggle.
    ax = axes[0]
    ax.plot(ai_pt - t0, fog_prob, color="tab:blue", lw=1.0, label="FoG prob (softmax of logits)")
    ax.axhline(0.7, color="tab:red", lw=0.8, ls="--", alpha=0.6, label="threshold_high (0.7)")
    ax.axhline(0.3, color="tab:orange", lw=0.8, ls="--", alpha=0.6, label="threshold_low (0.3)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("FoG prob")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Track 2 — smoothed binary prediction.
    ax = axes[1]
    ax.step(ai_pt - t0, ai_pred, where="post", color="tab:purple", lw=1.0)
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_ylabel("pred")
    ax.grid(True, alpha=0.3)

    # Track 3 — ground-truth label (if replay).
    ax = axes[2]
    if replay_t is not None:
        ax.step(replay_t - t0, replay_v, where="post", color="tab:green", lw=1.0)
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
    else:
        ax.text(0.5, 0.5, "no ground-truth label available",
                ha="center", va="center", transform=ax.transAxes, color="grey")
    ax.set_ylabel("truth")
    ax.grid(True, alpha=0.3)

    # Track 4 — cueing FSM status + BLE write markers.
    ax = axes[3]
    if cueing_t is not None:
        ax.step(cueing_t - t0, cueing_v, where="post", color="tab:red", lw=1.0)
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
    else:
        ax.text(0.5, 0.5, "no cueing status recorded",
                ha="center", va="center", transform=ax.transAxes, color="grey")
    ax.set_ylabel("cue\nstatus")
    ax.set_xlabel("time since trial start (s)")
    ax.grid(True, alpha=0.3)

    # BLE write markers overlaid on all tracks.
    for t in op_starts:
        for a in axes:
            a.axvline(t - t0, color="tab:green", lw=0.8, alpha=0.5)
    for t in op_stops:
        for a in axes:
            a.axvline(t - t0, color="tab:gray", lw=0.6, alpha=0.4, ls=":")

    # X-axis range.
    t_max = (ai_pt[-1] - t0)
    x_start = args.start if args.start is not None else 0.0
    x_end = args.end if args.end is not None else t_max
    axes[0].set_xlim(x_start, x_end)

    title = os.path.basename(os.path.normpath(args.trial_dir))
    fig.suptitle(f"Cueing episode — {title}", fontsize=12, y=0.995)
    fig.tight_layout()

    out_path = args.output or os.path.join(args.trial_dir, "cueing_episode.png")
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")

    # Also print a terse summary.
    total_s = t_max
    cue_starts = len(op_starts)
    cue_stops = len(op_stops)
    if replay_v is not None:
        gt_pos_frac = float(np.mean(replay_v > 0))
    else:
        gt_pos_frac = float("nan")
    pred_pos_frac = float(np.mean(ai_pred > 0))
    print(f"duration: {total_s:.1f}s  "
          f"start_cues: {cue_starts}  stop_cues: {cue_stops}  "
          f"gt FoG fraction: {gt_pos_frac:.3f}  pred FoG fraction: {pred_pos_frac:.3f}")


if __name__ == "__main__":
    main()
