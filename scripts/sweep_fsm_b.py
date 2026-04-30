"""
FSM B Pareto sweep: cueing_tail × refractory grid vs FSM A (DeFOG) baseline.

For each (cueing_tail_frames, refractory_frames) point — kept as
`tail_frames` / `cooldown_frames` in this script's variable and CSV-column
names for backward compatibility with `reports/fsm_pareto.csv` — runs FSM B
on every weighted LOSO subject's TUG trials, computes per-subject means, and
reports the cohort mean. FSM A and SINGLE baselines computed once for reference.

State naming (Vayalet 2026-04-29):
  - `tail_frames` (here)  ↔ `cueing_tail_frames` (CueState.CUEING_TAIL)
  - `cooldown_frames` (here) ↔ `refractory_frames` (CueState.REFRACTORY)

Output:
  - reports/fsm_pareto.csv (one row per param point)
  - stdout: summary table + identification of param points that match or beat
    FSM A on key metrics

Usage:
    .venv/bin/python scripts/sweep_fsm_b.py
    .venv/bin/python scripts/sweep_fsm_b.py --task all --tail-grid 0,30,60,120,180,300,600 --cooldown-grid 0,60,120,240
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hermes.aidfog_ai.model_alex.streaming import predict_streaming  # noqa: E402
from evaluate_loso import list_trials, load_trial, TARGET_HZ  # noqa: E402
from compare_fsms import (  # noqa: E402
    fsm_a_defog,
    fsm_b_4state,
    single_threshold,
    cue_metrics,
)


def nanmean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def cohort_summary(by_subj: dict, fsm_label: str) -> dict:
    """Mean ± std across subjects, after per-subject mean over trials."""
    subj_aggs = []
    for subj in sorted(by_subj):
        ms = by_subj[subj].get(fsm_label, [])
        if not ms:
            continue
        subj_aggs.append({k: nanmean([m[k] for m in ms])
                         for k in ms[0] if isinstance(ms[0][k], float)})

    def musd(key, scale=1.0):
        xs = [a[key] for a in subj_aggs if a[key] == a[key]]
        if not xs:
            return float("nan"), float("nan"), 0
        mu = sum(xs) / len(xs)
        sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
        return mu * scale, sd * scale, len(xs)

    return dict(
        sens=musd("sensitivity", 100),
        ppv=musd("ppv", 100),
        fp_per_min=musd("fp_per_min"),
        onset=musd("onset_delay_median_s"),
        overshoot=musd("tail_overshoot_median_s"),
        cue_pct=musd("cue_pct"),
        n_subjects=len(subj_aggs),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu-root", default="data/imu")
    ap.add_argument("--annot-root", default="data/annotation")
    ap.add_argument("--weights-dir",
                    default=os.path.join(ROOT, "hermes", "aidfog_ai", "model_alex", "weights"))
    ap.add_argument("--task", default="TUG")
    ap.add_argument("--enter-thresh", type=int, default=20)
    ap.add_argument("--exit-thresh", type=int, default=5)
    ap.add_argument("--tail-grid", default="0,30,60,120,180,300,600",
                    help="comma-separated tail_frames values")
    ap.add_argument("--cooldown-grid", default="0,60,120,240,300",
                    help="comma-separated cooldown_frames values")
    ap.add_argument("--out", default="reports/fsm_pareto.csv")
    args = ap.parse_args()

    task_filter = None if args.task.lower() == "all" else args.task

    available = {fn.replace("subject", "").replace(".pt", "")
                 for fn in os.listdir(args.weights_dir)
                 if fn.startswith("subject") and fn.endswith(".pt")}
    trials = [t for t in list_trials(args.annot_root, args.imu_root, task_filter)
              if t[0] in available]

    tail_grid = [int(x) for x in args.tail_grid.split(",")]
    cooldown_grid = [int(x) for x in args.cooldown_grid.split(",")]

    print(f"Sweeping FSM B over {len(tail_grid)} × {len(cooldown_grid)} = "
          f"{len(tail_grid) * len(cooldown_grid)} points on {len(trials)} trials.")
    print(f"tail_grid     = {tail_grid}")
    print(f"cooldown_grid = {cooldown_grid}\n")

    # Cache binary detection per trial — predict_streaming is deterministic and
    # FSMs operate on the binary alone, so we only run inference once per trial.
    print("[1/3] Caching predict_streaming binary outputs...")
    trial_data: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for subj, imu_path, annot_path in trials:
        try:
            trial, gt = load_trial(imu_path, annot_path)
            _, binary = predict_streaming(
                trial, subject=subj, weights_dir=args.weights_dir,
                enter_thresh=args.enter_thresh, exit_thresh=args.exit_thresh,
            )
        except Exception as exc:
            print(f"  skip {subj}/{os.path.basename(annot_path)}: {exc}")
            continue
        n = min(len(binary), len(gt))
        trial_name = os.path.basename(annot_path).replace(".npy", "")
        trial_data.append((subj, trial_name, binary[:n].astype(np.int8), gt[:n].astype(np.int8)))
    print(f"  cached {len(trial_data)} trials\n")

    # Compute FSM A and SINGLE baselines once
    print("[2/3] Computing FSM A (DeFOG) and SINGLE baselines...")
    by_subj_baseline: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for subj, _, binary, gt in trial_data:
        for label, fn in [("FSM_A_DeFOG", fsm_a_defog),
                          ("SINGLE", single_threshold)]:
            cue = fn(binary)
            m = cue_metrics(cue, gt, TARGET_HZ)
            by_subj_baseline[subj][label].append(m)

    s_fsm_a = cohort_summary(by_subj_baseline, "FSM_A_DeFOG")
    s_single = cohort_summary(by_subj_baseline, "SINGLE")
    print(f"  FSM A:  sens={s_fsm_a['sens'][0]:.1f}±{s_fsm_a['sens'][1]:.1f}%  "
          f"ppv={s_fsm_a['ppv'][0]:.1f}%  FP/min={s_fsm_a['fp_per_min'][0]:.2f}  "
          f"overshoot={s_fsm_a['overshoot'][0]:.2f}s  cue%={s_fsm_a['cue_pct'][0]:.1f}")
    print(f"  SINGLE: sens={s_single['sens'][0]:.1f}±{s_single['sens'][1]:.1f}%  "
          f"ppv={s_single['ppv'][0]:.1f}%  FP/min={s_single['fp_per_min'][0]:.2f}  "
          f"overshoot={s_single['overshoot'][0]:.2f}s  cue%={s_single['cue_pct'][0]:.1f}\n")

    # Sweep FSM B
    print(f"[3/3] Sweeping FSM B over {len(tail_grid) * len(cooldown_grid)} points...\n")
    print(f"{'tail':>5} {'cool':>5} | {'sens':>6} {'ppv':>6} {'FP/m':>6} "
          f"{'onset':>6} {'over':>6} {'cue%':>5} | {'Δsens':>6} {'Δppv':>6} {'Δover':>6}")
    print("-" * 95)

    rows = []
    for tail in tail_grid:
        for cool in cooldown_grid:
            label = f"FSM_B_t{tail}_c{cool}"
            by_subj = defaultdict(lambda: defaultdict(list))
            for subj, _, binary, gt in trial_data:
                cue = fsm_b_4state(binary, tail_frames=tail, cooldown_frames=cool)
                m = cue_metrics(cue, gt, TARGET_HZ)
                by_subj[subj][label].append(m)
            s = cohort_summary(by_subj, label)
            d_sens = s["sens"][0] - s_fsm_a["sens"][0]
            d_ppv = s["ppv"][0] - s_fsm_a["ppv"][0]
            d_overshoot = s["overshoot"][0] - s_fsm_a["overshoot"][0]
            rows.append(dict(
                tail_frames=tail, cooldown_frames=cool,
                sens=s["sens"][0], sens_sd=s["sens"][1],
                ppv=s["ppv"][0], ppv_sd=s["ppv"][1],
                fp_per_min=s["fp_per_min"][0],
                onset=s["onset"][0], overshoot=s["overshoot"][0],
                cue_pct=s["cue_pct"][0],
                d_sens=d_sens, d_ppv=d_ppv, d_overshoot=d_overshoot,
            ))
            print(f"{tail:>5} {cool:>5} | "
                  f"{s['sens'][0]:5.1f} {s['ppv'][0]:5.1f} "
                  f"{s['fp_per_min'][0]:5.2f} "
                  f"{s['onset'][0]:5.2f} {s['overshoot'][0]:5.2f} "
                  f"{s['cue_pct'][0]:5.1f} | "
                  f"{d_sens:+5.1f} {d_ppv:+5.1f} {d_overshoot:+5.2f}")

    print()
    print(f"FSM A reference:  sens={s_fsm_a['sens'][0]:.1f}%  ppv={s_fsm_a['ppv'][0]:.1f}%  "
          f"overshoot={s_fsm_a['overshoot'][0]:.2f}s  cue%={s_fsm_a['cue_pct'][0]:.1f}")

    # Identify Pareto-interesting points
    print("\nFSM B operating points that match/beat FSM A:")
    for r in rows:
        wins = []
        if r["d_sens"] >= -1.0:
            wins.append(f"sens within 1pp ({r['d_sens']:+.1f})")
        if r["d_ppv"] >= -1.0:
            wins.append(f"ppv within 1pp ({r['d_ppv']:+.1f})")
        if r["d_overshoot"] <= -1.0:
            wins.append(f"overshoot {-r['d_overshoot']:.1f}s shorter")
        if len(wins) >= 2:
            print(f"  tail={r['tail_frames']:>3} cool={r['cooldown_frames']:>3}: "
                  f"{', '.join(wins)}")

    # CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tail_frames", "cooldown_frames",
                    "sens", "sens_sd", "ppv", "ppv_sd",
                    "fp_per_min", "onset", "overshoot", "cue_pct",
                    "d_sens_vs_fsm_a", "d_ppv_vs_fsm_a", "d_overshoot_vs_fsm_a"])
        for r in rows:
            w.writerow([r["tail_frames"], r["cooldown_frames"],
                        f"{r['sens']:.4f}", f"{r['sens_sd']:.4f}",
                        f"{r['ppv']:.4f}", f"{r['ppv_sd']:.4f}",
                        f"{r['fp_per_min']:.4f}",
                        f"{r['onset']:.4f}", f"{r['overshoot']:.4f}",
                        f"{r['cue_pct']:.4f}",
                        f"{r['d_sens']:.4f}", f"{r['d_ppv']:.4f}",
                        f"{r['d_overshoot']:.4f}"])
    print(f"\nWrote {len(rows)} param points to {args.out}")


if __name__ == "__main__":
    main()
