"""
Joint sweep over Alex's HysteresisFilter (enter_thresh, exit_thresh) AND
FSM controllers (DeFOG cue+refr; FSM-B tail+cool). Caches raw smoothed
probabilities per trial so we only run the TCN once per trial across the
entire sweep.

Addresses critique #8 from the deep audit (the upstream filter was previously
fixed at enter=20, exit=5; both are also tunable hyperparameters and should be
swept jointly with FSM parameters).

Output:
  - reports/full_pareto.csv  (one row per (enter, exit, controller) point)
  - stdout: Pareto front, top configs, comparison vs DeFOG-published

Usage:
    .venv/bin/python scripts/analyze_full_pareto.py
    .venv/bin/python scripts/analyze_full_pareto.py --enter-grid 5,10,20,30,60 --exit-grid 3,5,10
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
from hermes.aidfog_ai.model_alex.hysteresis_filter import HysteresisFilter  # noqa: E402
from evaluate_loso import list_trials, load_trial, runs, TARGET_HZ  # noqa: E402
from analyze_fsm_thorough import (  # noqa: E402
    fsm_a_defog_traced,
    fsm_b_4state_traced,
    single_threshold_traced,
    cue_metrics_thorough,
    cohort_agg,
    nanmean,
)


def apply_hysteresis(thresholded: np.ndarray, enter_thresh: int, exit_thresh: int) -> np.ndarray:
    """Apply Alex's HysteresisFilter to a thresholded binary stream."""
    f = HysteresisFilter(enter_thresh=enter_thresh, exit_thresh=exit_thresh)
    return np.array([f.step(p) for p in thresholded], dtype=np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu-root", default="data/imu")
    ap.add_argument("--annot-root", default="data/annotation")
    ap.add_argument("--weights-dir",
                    default=os.path.join(ROOT, "hermes", "aidfog_ai", "model_alex", "weights"))
    ap.add_argument("--task", default="TUG")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--enter-grid", default="5,10,15,20,30,60")
    ap.add_argument("--exit-grid", default="3,5,10,20")
    ap.add_argument("--defog-cue-grid", default="60,180,300,600")
    ap.add_argument("--defog-refr-grid", default="60,300")
    ap.add_argument("--fsmb-tail-grid", default="0,30,60,120,180,300,600")
    ap.add_argument("--fsmb-cool-grid", default="0,60,120,300")
    ap.add_argument("--out", default="reports/full_pareto.csv")
    args = ap.parse_args()

    task_filter = None if args.task.lower() == "all" else args.task

    available = {fn.replace("subject", "").replace(".pt", "")
                 for fn in os.listdir(args.weights_dir)
                 if fn.startswith("subject") and fn.endswith(".pt")}
    trials = [t for t in list_trials(args.annot_root, args.imu_root, task_filter)
              if t[0] in available]

    enter_grid = [int(x) for x in args.enter_grid.split(",")]
    exit_grid = [int(x) for x in args.exit_grid.split(",")]
    defog_cue_grid = [int(x) for x in args.defog_cue_grid.split(",")]
    defog_refr_grid = [int(x) for x in args.defog_refr_grid.split(",")]
    fsmb_tail_grid = [int(x) for x in args.fsmb_tail_grid.split(",")]
    fsmb_cool_grid = [int(x) for x in args.fsmb_cool_grid.split(",")]

    n_filter_pts = len(enter_grid) * len(exit_grid)
    n_ctrl_pts = (1 + len(defog_cue_grid) * len(defog_refr_grid)
                  + len(fsmb_tail_grid) * len(fsmb_cool_grid))
    print(f"Sweep: {len(enter_grid)} × {len(exit_grid)} = {n_filter_pts} filter points × "
          f"{n_ctrl_pts} controllers = {n_filter_pts * n_ctrl_pts} configs total.")
    print(f"Trials: {len(trials)} (task={args.task})\n")

    # ── Cache raw probabilities per trial (one TCN forward pass each) ────
    print(f"[1/3] Caching raw probabilities for {len(trials)} trials "
          f"(one TCN pass each)...")
    cache = []
    for subj, imu_path, annot_path in trials:
        try:
            trial, gt = load_trial(imu_path, annot_path)
            probas, _ = predict_streaming(
                trial, subject=subj, weights_dir=args.weights_dir,
                # Use defaults; we'll re-apply hysteresis post-hoc with each
                # (enter, exit) combo — what matters here is the prob stream.
                enter_thresh=20, exit_thresh=5,
            )
        except Exception as exc:
            print(f"  skip {subj}/{os.path.basename(annot_path)}: {exc}")
            continue
        n = min(len(probas), len(gt))
        trial_name = os.path.basename(annot_path).replace(".npy", "")
        thresholded = (probas[:n] >= args.thresh).astype(np.int8)
        cache.append((subj, trial_name, thresholded, gt[:n].astype(np.int8)))
    print(f"  cached {len(cache)} trials\n")

    # ── For each (enter, exit), compute hysteresis output once per trial ──
    print(f"[2/3] Applying hysteresis for {n_filter_pts} (enter, exit) points...")
    binaries: dict[tuple[int, int], list[np.ndarray]] = {}
    for enter in enter_grid:
        for exit_t in exit_grid:
            bs = []
            for subj, trial_name, thresholded, gt in cache:
                bs.append(apply_hysteresis(thresholded, enter, exit_t))
            binaries[(enter, exit_t)] = bs
    print(f"  applied {n_filter_pts * len(cache)} hysteresis passes\n")

    # ── For each filter × controller, compute metrics ─────────────────────
    print(f"[3/3] Evaluating {n_filter_pts * n_ctrl_pts} configurations...")

    def make_controllers():
        out = [("SINGLE", "single", None, None,
                lambda b: single_threshold_traced(b))]
        for cue_f in defog_cue_grid:
            for refr_f in defog_refr_grid:
                out.append((f"FSM_A_cue{cue_f}_refr{refr_f}", "defog",
                            cue_f, refr_f,
                            lambda b, c=cue_f, r=refr_f:
                            fsm_a_defog_traced(b, c, r)))
        for tail in fsmb_tail_grid:
            for cool in fsmb_cool_grid:
                out.append((f"FSM_B_tail{tail}_cool{cool}", "fsmb",
                            tail, cool,
                            lambda b, t=tail, c=cool:
                            fsm_b_4state_traced(b, t, c)))
        return out

    rows = []
    for (enter, exit_t), bs in binaries.items():
        for label, kind, p1, p2, fn in make_controllers():
            by_subj = defaultdict(list)
            for (subj, _, _, gt), binary in zip(cache, bs):
                cue, is_idle = fn(binary)
                m = cue_metrics_thorough(cue, gt, is_idle, TARGET_HZ)
                by_subj[subj].append(m)
            agg = cohort_agg(by_subj)
            row = dict(enter_thresh=enter, exit_thresh=exit_t,
                       label=label, kind=kind, p1=p1, p2=p2,
                       n_subjects=agg.get("sens_iou50", (0, 0, 0))[2])
            for k, (mu, sd, _) in agg.items():
                row[k] = mu
                row[k + "_sd"] = sd
            rows.append(row)
    print(f"  computed {len(rows)} configurations\n")

    # ── Print analysis ────────────────────────────────────────────────────
    valid = [r for r in rows if r.get("sens_iou50") == r.get("sens_iou50")]

    # Reference points
    print("Reference points (Alex's filter defaults: enter=20, exit=5):")
    print(f"{'controller':<32} {'sensA':>6} {'sensI':>6} {'PPV_I':>6} "
          f"{'in_fog':>7} {'fp_dens':>8} {'on_idle':>7}")
    print("-" * 90)

    def find(enter, exit_t, label):
        return next((r for r in valid
                     if r["enter_thresh"] == enter and r["exit_thresh"] == exit_t
                     and r["label"] == label), None)

    for label in ("SINGLE", "FSM_A_cue600_refr300", "FSM_B_tail60_cool120"):
        r = find(20, 5, label)
        if r:
            print(f"{label:<32} "
                  f"{100*r['sens_any']:5.1f} {100*r['sens_iou50']:5.1f} "
                  f"{100*r['ppv_iou50']:5.1f} "
                  f"{r['in_fog_ratio']:6.2f} {r['fp_density']:7.4f} "
                  f"{r['onset_idle_median_s']:6.2f}")

    # Best config per metric
    print("\nTop 10 configs by IoU-50 sensitivity (across full sweep):")
    print(f"{'enter':>5} {'exit':>4} {'controller':<28} "
          f"{'sensI':>6} {'PPV_I':>6} {'in_fog':>7} {'fp_dens':>8}")
    print("-" * 75)
    for r in sorted(valid, key=lambda x: -x["sens_iou50"])[:10]:
        print(f"{r['enter_thresh']:>5} {r['exit_thresh']:>4} {r['label']:<28} "
              f"{100*r['sens_iou50']:5.1f} {100*r['ppv_iou50']:5.1f} "
              f"{r['in_fog_ratio']:6.2f} {r['fp_density']:7.4f}")

    # Lowest fp_density at IoU-50 sens >= 30%
    print("\nLowest fp_density with IoU-50 sens >= 30%:")
    eligible = [r for r in valid
                if r["sens_iou50"] == r["sens_iou50"] and r["sens_iou50"] >= 0.30
                and r["fp_density"] == r["fp_density"]]
    for r in sorted(eligible, key=lambda x: x["fp_density"])[:8]:
        print(f"  enter={r['enter_thresh']:>3} exit={r['exit_thresh']:>3} "
              f"{r['label']:<28} "
              f"sensI={100*r['sens_iou50']:5.1f}% "
              f"in_fog={r['in_fog_ratio']:.2f} "
              f"fp_dens={r['fp_density']:.4f}")

    # 2D Pareto: minimize fp_density, maximize sens_iou50
    print("\nPareto front (sens_iou50 ↑ × fp_density ↓):")
    print(f"{'enter':>5} {'exit':>4} {'controller':<28} "
          f"{'sensI':>6} {'PPV_I':>6} {'in_fog':>7} {'fp_dens':>8}")
    print("-" * 75)
    front = []
    for r in valid:
        if r["sens_iou50"] != r["sens_iou50"] or r["fp_density"] != r["fp_density"]:
            continue
        dominated = False
        for r2 in valid:
            if r2 is r:
                continue
            if (r2["sens_iou50"] != r2["sens_iou50"] or
                r2["fp_density"] != r2["fp_density"]):
                continue
            if (r2["sens_iou50"] >= r["sens_iou50"] and
                r2["fp_density"] <= r["fp_density"] and
                (r2["sens_iou50"] > r["sens_iou50"] or
                 r2["fp_density"] < r["fp_density"])):
                dominated = True
                break
        if not dominated:
            front.append(r)
    for r in sorted(front, key=lambda x: x["fp_density"]):
        print(f"{r['enter_thresh']:>5} {r['exit_thresh']:>4} {r['label']:<28} "
              f"{100*r['sens_iou50']:5.1f} {100*r['ppv_iou50']:5.1f} "
              f"{r['in_fog_ratio']:6.2f} {r['fp_density']:7.4f}")
    print(f"\n{len(front)} non-dominated configurations on Pareto front.")

    # CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if rows:
        keys = list(rows[0].keys())
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {len(rows)} configs to {args.out}")


if __name__ == "__main__":
    main()
