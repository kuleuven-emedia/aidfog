"""
Side-by-side comparison of cueing FSM strategies on a recorded trial.

Reads aidfog_ai.hdf5 from a trial folder, replays the logit stream through
several candidate FSMs, and plots the resulting cue-state timelines so you
can see where each strategy fires/holds/releases. No hardware needed —
this is purely offline analysis to inform Phase 3 design.

Tracks (top to bottom):
  1. FoG probability (softmax of logits[1])
  2. Raw argmax — the noisy 1/0 stream the model emits per frame
  3. Smoothed prediction — what TorchPipeline's smooth() function outputs
  4. 2-state FSM (current production behaviour: enter @ prob>=0.7, exit @ prob<0.3)
  5. 4-state FSM (proposed: IDLE / CUEING / TAIL / COOLDOWN with sustained entry)

Usage:
    python scripts/compare_fsm_strategies.py path/to/trial_folder
    python scripts/compare_fsm_strategies.py path/to/trial_folder --start 0 --end 2
"""

import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np


# --- FSM implementations -----------------------------------------------------

def fsm_2state(fog_prob: np.ndarray, th_high=0.7, th_low=0.3) -> np.ndarray:
    """Current production FSM. State per frame: 0=IDLE, 1=CUEING."""
    state = np.zeros(len(fog_prob), dtype=np.uint8)
    cur = 0
    for i, p in enumerate(fog_prob):
        if cur == 0 and p >= th_high:
            cur = 1
        elif cur == 1 and p < th_low:
            cur = 0
        state[i] = cur
    return state


def fsm_4state(
    fog_prob: np.ndarray,
    th_high=0.7,
    th_low=0.3,
    entry_consec=3,
    tail_frames=30,        # ~500 ms at 60 Hz
    cooldown_frames=60,    # ~1 s at 60 Hz
) -> np.ndarray:
    """Proposed 4-state FSM. Per-frame state:
       0 = IDLE, 1 = CUEING, 2 = TAIL, 3 = COOLDOWN.
    """
    state = np.zeros(len(fog_prob), dtype=np.uint8)
    cur = 0
    consec_high = 0
    tail_remaining = 0
    cooldown_remaining = 0
    for i, p in enumerate(fog_prob):
        if cur == 0:  # IDLE
            consec_high = consec_high + 1 if p >= th_high else 0
            if consec_high >= entry_consec:
                cur = 1
                consec_high = 0
        elif cur == 1:  # CUEING
            if p < th_low:
                cur = 2
                tail_remaining = tail_frames
        elif cur == 2:  # TAIL
            if p >= th_high:
                cur = 1
            else:
                tail_remaining -= 1
                if tail_remaining <= 0:
                    cur = 3
                    cooldown_remaining = cooldown_frames
        elif cur == 3:  # COOLDOWN
            cooldown_remaining -= 1
            if cooldown_remaining <= 0:
                cur = 0
                consec_high = 0
        state[i] = cur
    return state


# --- main --------------------------------------------------------------------

def softmax_2class(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_dir")
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--th-high", type=float, default=0.7)
    parser.add_argument("--th-low", type=float, default=0.3)
    parser.add_argument("--entry-consec", type=int, default=3,
                        help="4-state: frames of sustained high-prob before entering CUEING")
    parser.add_argument("--tail-frames", type=int, default=30,
                        help="4-state: frames to hold cue after prob drops (default 30 = 500ms @60Hz)")
    parser.add_argument("--cooldown-frames", type=int, default=60,
                        help="4-state: lockout frames after TAIL ends (default 60 = 1s @60Hz)")
    args = parser.parse_args()

    ai_path = os.path.join(args.trial_dir, "aidfog_ai.hdf5")
    if not os.path.exists(ai_path):
        print(f"missing {ai_path}", file=sys.stderr)
        sys.exit(1)

    with h5py.File(ai_path, "r") as f:
        pt = f["aidfog_ai/pytorch-worker/process_time_s"][:].flatten()
        logits = f["aidfog_ai/pytorch-worker/logits"][:]
        smoothed = f["aidfog_ai/pytorch-worker/prediction"][:].flatten()
    mask = pt > 0
    pt = pt[mask]
    logits = logits[mask]
    smoothed = smoothed[mask]

    fog_prob = softmax_2class(logits)
    raw_argmax = (logits[:, 1] > logits[:, 0]).astype(np.uint8)

    state_2 = fsm_2state(fog_prob, args.th_high, args.th_low)
    state_4 = fsm_4state(fog_prob, args.th_high, args.th_low,
                        args.entry_consec, args.tail_frames, args.cooldown_frames)

    t = pt - pt[0]

    # Count cue events (rising edges into CUEING).
    edges_2 = int(np.sum((state_2[1:] == 1) & (state_2[:-1] != 1)))
    edges_4 = int(np.sum((state_4[1:] == 1) & (state_4[:-1] != 1)))
    cue_frames_2 = int(np.sum(state_2 == 1))
    cue_frames_4 = int(np.sum(state_4 == 1))
    print(f"trial duration: {t[-1]:.1f}s  ({len(t)} frames @ ~60 Hz)")
    print(f"raw argmax FoG fraction:    {raw_argmax.mean():.3f}")
    print(f"smoothed prediction frac:   {smoothed.mean():.3f}")
    print(f"2-state FSM: {edges_2:>3} cue events, {cue_frames_2} CUEING frames "
          f"({cue_frames_2/len(t)*100:.1f}%)")
    print(f"4-state FSM: {edges_4:>3} cue events, {cue_frames_4} CUEING frames "
          f"({cue_frames_4/len(t)*100:.1f}%)  (entry_consec={args.entry_consec}, "
          f"tail={args.tail_frames}, cooldown={args.cooldown_frames})")

    fig, axes = plt.subplots(
        nrows=5, ncols=1, sharex=True, figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 1, 1, 1.2, 1.4]},
    )

    ax = axes[0]
    ax.plot(t, fog_prob, color="tab:blue", lw=1.0)
    ax.axhline(args.th_high, color="tab:red", ls="--", lw=0.8, alpha=0.6,
               label=f"th_high={args.th_high}")
    ax.axhline(args.th_low, color="tab:orange", ls="--", lw=0.8, alpha=0.6,
               label=f"th_low={args.th_low}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("FoG prob")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.step(t, raw_argmax, where="post", color="tab:gray", lw=0.8)
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_ylabel("raw\nargmax")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.step(t, smoothed, where="post", color="tab:purple", lw=0.8)
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_ylabel("smoothed\npred")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.step(t, state_2, where="post", color="tab:red", lw=1.0)
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_ylabel("2-state\nFSM")
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.95, f"{edges_2} cue events",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="tab:red")

    ax = axes[4]
    ax.step(t, state_4, where="post", color="tab:green", lw=1.0)
    ax.set_ylim(-0.2, 3.2)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["IDLE", "CUEING", "TAIL", "COOLDOWN"])
    ax.set_ylabel("4-state\nFSM")
    ax.set_xlabel("time since trial start (s)")
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.95, f"{edges_4} cue events",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="tab:green")

    x_start = args.start if args.start is not None else 0.0
    x_end = args.end if args.end is not None else t[-1]
    axes[0].set_xlim(x_start, x_end)

    title = os.path.basename(os.path.normpath(args.trial_dir))
    fig.suptitle(f"FSM strategy comparison — {title}", fontsize=12, y=0.995)
    fig.tight_layout()

    out_path = args.output or os.path.join(args.trial_dir, "fsm_comparison.png")
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
