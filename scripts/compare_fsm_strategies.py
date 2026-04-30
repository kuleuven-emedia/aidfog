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
  5. 4-state FSM (proposed: IDLE / CUEING / CUEING_TAIL / REFRACTORY with sustained entry)

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
       0 = IDLE, 1 = CUEING, 2 = CUEING_TAIL, 3 = REFRACTORY.

    Argument names `tail_frames` / `cooldown_frames` kept for compatibility with
    earlier reports; in the runtime FSM they are `cueing_tail_frames` and
    `refractory_frames`.
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


def synthesize_stream(
    duration_s: float = 60.0,
    fs_hz: float = 60.0,
    n_episodes: int = 4,
    n_glitches: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """Generate a realistic-looking FoG probability stream for FSM testing.

    Background noise around 0.05; FoG episodes with smooth on/off ramps and
    intra-episode probability dips that the FSM's CUEING/TAIL states must
    handle gracefully; brief noise glitches the FSM's IDLE state's
    sustained-entry rule must reject.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs_hz)
    p = rng.normal(loc=0.05, scale=0.02, size=n).clip(0.0, 1.0).astype(np.float32)

    # Episodes: random duration 3-12 s, evenly spaced.
    spacing = duration_s / (n_episodes + 1)
    for k in range(n_episodes):
        center_s = spacing * (k + 1)
        ep_dur = float(rng.uniform(3.0, 12.0))
        ramp_s = 0.4
        ep_start_s = center_s - ep_dur / 2
        ep_end_s = ep_start_s + ep_dur
        # Smooth ramp up/down.
        for i in range(n):
            t = i / fs_hz
            if t < ep_start_s or t > ep_end_s + ramp_s:
                continue
            if t < ep_start_s + ramp_s:
                w = (t - ep_start_s) / ramp_s
            elif t > ep_end_s:
                w = max(0.0, 1.0 - (t - ep_end_s) / ramp_s)
            else:
                w = 1.0
            target = 0.88 + rng.normal(0.0, 0.03)
            p[i] = max(p[i], 0.05 + w * (target - 0.05))
        # Intra-episode dips: 1-2 short drops that should NOT release the cue.
        n_dips = int(rng.integers(1, 3))
        for _ in range(n_dips):
            dip_t = float(rng.uniform(ep_start_s + 0.5, ep_end_s - 0.5))
            dip_dur = float(rng.uniform(0.10, 0.25))
            dip_start = int(dip_t * fs_hz)
            dip_end = int((dip_t + dip_dur) * fs_hz)
            p[dip_start:dip_end] = rng.uniform(0.15, 0.35)

    # Random short glitches (single-frame to 5-frame spikes) outside episodes.
    for _ in range(n_glitches):
        g_idx = int(rng.uniform(0, n - 5))
        g_len = int(rng.integers(1, 5))
        p[g_idx:g_idx + g_len] = rng.uniform(0.75, 0.98)

    return p.clip(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_dir", nargs="?", default=None,
                        help="Path to trial folder; omit to use --synthetic")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate a synthetic probability stream instead of reading a trial")
    parser.add_argument("--syn-duration", type=float, default=60.0)
    parser.add_argument("--syn-episodes", type=int, default=4)
    parser.add_argument("--syn-glitches", type=int, default=6)
    parser.add_argument("--syn-seed", type=int, default=42)
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

    if args.synthetic or args.trial_dir is None:
        # Synthetic mode — bypass HDF5 entirely.
        fs = 60.0
        fog_prob = synthesize_stream(
            duration_s=args.syn_duration,
            fs_hz=fs,
            n_episodes=args.syn_episodes,
            n_glitches=args.syn_glitches,
            seed=args.syn_seed,
        )
        n = len(fog_prob)
        pt = np.linspace(0, args.syn_duration, n, endpoint=False)
        # Fake "logits" for the raw_argmax track and downstream softmax compat.
        raw_argmax = (fog_prob >= 0.5).astype(np.uint8)
        smoothed = np.zeros_like(raw_argmax)  # n/a in synthetic mode
        synthetic_out_dir = os.path.join(".", "data", "synthetic_fsm")
        os.makedirs(synthetic_out_dir, exist_ok=True)
        default_label = (f"synthetic_d{int(args.syn_duration)}s_"
                         f"e{args.syn_episodes}_g{args.syn_glitches}_s{args.syn_seed}")
        trial_label = default_label
    else:
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
        trial_label = os.path.basename(os.path.normpath(args.trial_dir))

    state_2 = fsm_2state(fog_prob, args.th_high, args.th_low)
    state_4 = fsm_4state(fog_prob, args.th_high, args.th_low,
                        args.entry_consec, args.tail_frames, args.cooldown_frames)

    t = pt if args.synthetic or args.trial_dir is None else (pt - pt[0])

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
    ax.set_yticklabels(["IDLE", "CUEING", "CUEING_TAIL", "REFRACTORY"])
    ax.set_ylabel("4-state\nFSM")
    ax.set_xlabel("time since trial start (s)")
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.95, f"{edges_4} cue events",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="tab:green")

    x_start = args.start if args.start is not None else 0.0
    x_end = args.end if args.end is not None else t[-1]
    axes[0].set_xlim(x_start, x_end)

    fig.suptitle(f"FSM strategy comparison — {trial_label}", fontsize=12, y=0.995)
    fig.tight_layout()

    if args.output:
        out_path = args.output
    elif args.synthetic or args.trial_dir is None:
        out_path = os.path.join(synthetic_out_dir, f"{trial_label}.png")
    else:
        out_path = os.path.join(args.trial_dir, "fsm_comparison.png")
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
