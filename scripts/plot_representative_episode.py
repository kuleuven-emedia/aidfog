"""
F9 — Representative-episode overlay (offline reconstruction).

Picks a FoG-rich trial from the LOSO data, runs Alex's TCN + HysteresisFilter
to reproduce the binary stream the cueing FSM would have seen, then simulates
both FSM A (DeFOG: 10 s cue + 5 s refractory) and FSM B (4-state: 1 s
CUEING_TAIL + 2 s REFRACTORY) on that stream. Plots six tracks:

  1. Probability (TCN softmax)
  2. Ground-truth FoG label
  3. Hysteresis-filter output (binary, post-debounce)
  4. FSM A — DeFOG-style cue policy
  5. FSM B — our 4-state cue policy
  6. Difference panel: where the two FSMs diverge

The figure makes the "different, not better" framing visible — FSM A keeps
cueing for 10 s after every detection, FSM B follows the freeze duration
plus a small tail.

Output: reports/F9_representative_episode.png
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from hermes.aidfog_ai.model_alex.streaming import predict_streaming
from hermes.aidfog_ai.model_alex.hysteresis_filter import HysteresisFilter
from analyze_fsm_thorough import (
    fsm_a_defog_traced, fsm_b_4state_traced, runs
)
from evaluate_loso import list_trials, load_trial, TARGET_HZ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="030")
    ap.add_argument("--trial-name", default="030_A_pm_TUG-ST_1",
                    help="Trial filename (without .npy)")
    ap.add_argument("--enter-thresh", type=int, default=20)
    ap.add_argument("--exit-thresh", type=int, default=5)
    ap.add_argument("--fsm-a-cue", type=int, default=10 * TARGET_HZ,
                    help="FSM A cue duration in frames (default 600 = 10 s)")
    ap.add_argument("--fsm-a-refr", type=int, default=5 * TARGET_HZ,
                    help="FSM A refractory in frames (default 300 = 5 s)")
    ap.add_argument("--fsm-b-tail", type=int, default=60,
                    help="FSM B cueing_tail_frames (default 60 = 1 s)")
    ap.add_argument("--fsm-b-cool", type=int, default=120,
                    help="FSM B refractory_frames (default 120 = 2 s)")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--out", default="reports/F9_representative_episode.png")
    args = ap.parse_args()

    # Load the trial
    trials = [t for t in list_trials("data/annotation", "data/imu", "TUG")
              if t[0] == args.subject]
    target = next((t for t in trials
                   if args.trial_name in os.path.basename(t[2])), None)
    if target is None:
        print(f"Trial {args.trial_name} not found for subject {args.subject}",
              file=sys.stderr)
        sys.exit(1)
    subj, imu_path, annot_path = target
    trial, gt = load_trial(imu_path, annot_path)
    print(f"Loaded {os.path.basename(annot_path)}: {len(gt)} samples = "
          f"{len(gt) / TARGET_HZ:.1f} s")

    # Run TCN + Hysteresis to get the binary stream
    weights_dir = os.path.join(ROOT, "hermes", "aidfog_ai", "model_alex", "weights")
    probas, _ = predict_streaming(
        trial, subject=subj, weights_dir=weights_dir,
        enter_thresh=args.enter_thresh, exit_thresh=args.exit_thresh,
    )
    n = min(len(probas), len(gt))
    probas = probas[:n]
    gt = gt[:n].astype(np.int8)

    # Apply hysteresis to thresholded probas
    thresholded = (probas >= 0.5).astype(np.int8)
    hf = HysteresisFilter(enter_thresh=args.enter_thresh, exit_thresh=args.exit_thresh)
    binary = np.array([hf.step(p) for p in thresholded], dtype=np.int8)

    # Simulate both FSMs
    cue_a, _ = fsm_a_defog_traced(binary, args.fsm_a_cue, args.fsm_a_refr)
    cue_b, _ = fsm_b_4state_traced(binary, args.fsm_b_tail, args.fsm_b_cool)

    # Time axis
    t = np.arange(n) / TARGET_HZ
    x_start = args.start if args.start is not None else 0.0
    x_end = args.end if args.end is not None else float(t[-1])

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        nrows=6, ncols=1, sharex=True, figsize=(15, 10), dpi=120,
        gridspec_kw={"height_ratios": [3, 1.2, 1.2, 1.4, 1.4, 1.0]},
    )
    fig.patch.set_facecolor("white")

    # Helper to shade GT episodes on every panel
    gt_eps = runs(gt)
    def shade_gt(ax, color="#fde2e2", alpha=0.65):
        for s, e in gt_eps:
            ax.axvspan(s / TARGET_HZ, e / TARGET_HZ,
                       color=color, alpha=alpha, zorder=0)

    # Track 1: probability
    ax = axes[0]
    shade_gt(ax)
    ax.plot(t, probas, color="#1a365d", lw=1.0, zorder=3)
    ax.axhline(0.5, color="#a02828", lw=0.8, ls="--", alpha=0.6,
               label="threshold = 0.5")
    ax.set_ylabel("FoG\nprobability", fontsize=10)
    ax.set_ylim(-0.04, 1.04)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Representative episode — {os.path.basename(annot_path).replace('.npy','')}, "
        f"subject {subj} (LOSO held-out)\n"
        f"Pink shading = GT FoG episodes  ·  "
        f"hysteresis (enter={args.enter_thresh}, exit={args.exit_thresh}, threshold=0.5)  ·  "
        f"FSM A (cue={args.fsm_a_cue/TARGET_HZ:.0f} s, refr={args.fsm_a_refr/TARGET_HZ:.0f} s)  ·  "
        f"FSM B (tail={args.fsm_b_tail/TARGET_HZ:.1f} s, refr={args.fsm_b_cool/TARGET_HZ:.1f} s)",
        fontsize=10.5, color="#222222",
    )

    # Track 2: ground truth
    ax = axes[1]
    shade_gt(ax)
    ax.fill_between(t, 0, gt, color="#a02828", alpha=0.7, step="post", zorder=3)
    ax.set_ylabel("GT\nFoG", fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Track 3: hysteresis filter output (the binary the FSM sees)
    ax = axes[2]
    shade_gt(ax)
    ax.fill_between(t, 0, binary, color="#5a8a5a", alpha=0.7, step="post", zorder=3)
    ax.set_ylabel("Hysteresis\noutput", fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Track 4: FSM A
    ax = axes[3]
    shade_gt(ax)
    ax.fill_between(t, 0, cue_a, color="#a02828", alpha=0.5, step="post", zorder=3,
                    label="FSM A cue active")
    a_eps = runs(cue_a)
    a_total_s = sum((e - s) for s, e in a_eps) / TARGET_HZ
    ax.set_ylabel("FSM A\n(DeFOG-\npublished)", fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.92, f"{len(a_eps)} cue events  ·  total cue {a_total_s:.1f} s",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="#7a2222", fontweight="semibold")

    # Track 5: FSM B
    ax = axes[4]
    shade_gt(ax)
    ax.fill_between(t, 0, cue_b, color="#1a5a1a", alpha=0.5, step="post", zorder=3,
                    label="FSM B cue active")
    b_eps = runs(cue_b)
    b_total_s = sum((e - s) for s, e in b_eps) / TARGET_HZ
    ax.set_ylabel("FSM B\n(ours,\n4-state)", fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.92, f"{len(b_eps)} cue events  ·  total cue {b_total_s:.1f} s",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="#1a5a1a", fontweight="semibold")

    # Track 6: difference (where FSM A and FSM B diverge)
    ax = axes[5]
    shade_gt(ax)
    diff = cue_a.astype(np.int16) - cue_b.astype(np.int16)
    only_a = (diff == 1).astype(np.int8)
    only_b = (diff == -1).astype(np.int8)
    ax.fill_between(t, 0, only_a, color="#a02828", alpha=0.55, step="post",
                    label="cued by A only")
    ax.fill_between(t, 0, -only_b, color="#1a5a1a", alpha=0.55, step="post",
                    label="cued by B only")
    ax.set_ylabel("FSM A − B", fontsize=10)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["B only", "agree", "A only"])
    ax.set_ylim(-1.15, 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_xlabel("time since trial start (s)", fontsize=11)

    axes[0].set_xlim(x_start, x_end)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Wrote {args.out}")
    print(f"\nSummary:")
    print(f"  GT FoG: {len(gt_eps)} episodes, {sum((e-s) for s,e in gt_eps)/TARGET_HZ:.1f} s total")
    print(f"  FSM A: {len(a_eps)} cue events, {a_total_s:.1f} s total cue")
    print(f"  FSM B: {len(b_eps)} cue events, {b_total_s:.1f} s total cue")
    print(f"  Cue overhead vs GT: A={a_total_s/(sum((e-s) for s,e in gt_eps)/TARGET_HZ):.1f}×, "
          f"B={b_total_s/(sum((e-s) for s,e in gt_eps)/TARGET_HZ):.1f}×")


if __name__ == "__main__":
    main()
