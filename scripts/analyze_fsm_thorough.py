"""
Thorough FSM A vs FSM B analysis with four corrections to compare_fsms.py.

Naming note (Vayalet 2026-04-29): runtime FSM uses CUEING_TAIL / REFRACTORY.
This script keeps `tail_frames` / `cooldown_frames` as parameter and CSV-column
names for backward compatibility with `reports/fsm_thorough.csv` — they map to
`cueing_tail_frames` and `refractory_frames` respectively.


  #1  Joint sweep: DeFOG's cue_frames and refractory_frames are parameters too,
      not fixed at Zoetewei 2021's published 10s + 5s.
  #2  Two matching schemes: any-overlap (DeFOG convention) AND IoU≥50%
      (Alex's thesis convention). Long-cue strategies look better at any-overlap.
  #3  Onset delay only on episodes that begin while the FSM is in IDLE
      (mid-cue carryover doesn't count as "fast onset").
  #5  Cue-efficiency decomposed into:
        - in_fog_ratio  = in-FoG cue time / total FoG time     (overshoot)
        - fp_density   = out-of-FoG cue time / non-FoG time   (false-alarm density)

Output:
  - reports/fsm_thorough.csv
  - stdout: cohort summary + Pareto-relevant operating points

Usage:
    .venv/bin/python scripts/analyze_fsm_thorough.py
    .venv/bin/python scripts/analyze_fsm_thorough.py --task all
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hermes.aidfog_ai.model_alex.streaming import predict_streaming  # noqa: E402
from evaluate_loso import list_trials, load_trial, runs, TARGET_HZ  # noqa: E402

# ---------------------------------------------------------------------------
# FSM controllers — return per-frame cue state AND per-frame FSM state
# ---------------------------------------------------------------------------

def fsm_a_defog_traced(binary: np.ndarray, cue_frames: int, refractory_frames: int):
    """DeFOG controller; also returns per-frame state for onset-delay filter."""
    n = len(binary)
    cue = np.zeros(n, dtype=np.int8)
    is_idle = np.ones(n, dtype=bool)  # True if FSM is in IDLE at frame t
    state = "IDLE"
    cue_remaining = 0
    refr_remaining = 0
    for t in range(n):
        is_idle[t] = (state == "IDLE")
        if state == "IDLE":
            if binary[t] == 1:
                state = "CUEING"
                cue_remaining = cue_frames
                cue[t] = 1
        elif state == "CUEING":
            cue_remaining -= 1
            cue[t] = 1
            if cue_remaining <= 0:
                state = "REFRACTORY"
                refr_remaining = refractory_frames
        elif state == "REFRACTORY":
            refr_remaining -= 1
            if refr_remaining <= 0:
                state = "IDLE"
    return cue, is_idle


def fsm_b_4state_traced(binary: np.ndarray, tail_frames: int, cooldown_frames: int):
    """4-state controller (IDLE → CUEING → CUEING_TAIL → REFRACTORY); returns per-frame IDLE flag."""
    n = len(binary)
    cue = np.zeros(n, dtype=np.int8)
    is_idle = np.ones(n, dtype=bool)
    state = "IDLE"
    cueing_tail_remaining = 0
    refractory_remaining = 0
    for t in range(n):
        is_idle[t] = (state == "IDLE")
        if state == "IDLE":
            if binary[t] == 1:
                state = "CUEING"
                cue[t] = 1
        elif state == "CUEING":
            cue[t] = 1
            if binary[t] == 0:
                state = "CUEING_TAIL"
                cueing_tail_remaining = tail_frames
        elif state == "CUEING_TAIL":
            cue[t] = 1
            if binary[t] == 1:
                state = "CUEING"
            else:
                cueing_tail_remaining -= 1
                if cueing_tail_remaining <= 0:
                    state = "REFRACTORY"
                    refractory_remaining = cooldown_frames
        elif state == "REFRACTORY":
            refractory_remaining -= 1
            if refractory_remaining <= 0:
                state = "IDLE"
    return cue, is_idle


def single_threshold_traced(binary: np.ndarray):
    """Pass-through; IDLE flag is just (binary == 0) for symmetry."""
    cue = binary.astype(np.int8)
    is_idle = (binary == 0)
    return cue, is_idle


# ---------------------------------------------------------------------------
# Metrics — both matching schemes, IDLE-onset filter, decomposed efficiency
# ---------------------------------------------------------------------------

def event_iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_events(gt_events, cue_events, mode: str):
    """
    Return (matched_gt_idx, matched_cue_idx) sets.
    mode='any'  : any-frame overlap (DeFOG convention)
    mode='iou50': IoU >= 0.5  (Alex thesis convention)
    """
    matched_gt = set()
    matched_cue = set()
    if mode == "any":
        for gi, ge in enumerate(gt_events):
            for ci, ce in enumerate(cue_events):
                if ci in matched_cue:
                    continue
                if ce[1] > ge[0] and ce[0] < ge[1]:
                    matched_gt.add(gi)
                    matched_cue.add(ci)
                    break
        # second pass to mark leftover cues against any GT episode (for PPV)
        ppv_tp = 0
        for ci, ce in enumerate(cue_events):
            if any(ce[1] > ge[0] and ce[0] < ge[1] for ge in gt_events):
                ppv_tp += 1
        return matched_gt, matched_cue, ppv_tp
    elif mode == "iou50":
        for gi, ge in enumerate(gt_events):
            best_iou = 0.0
            best_ci = -1
            for ci, ce in enumerate(cue_events):
                if ci in matched_cue:
                    continue
                iou = event_iou(ge, ce)
                if iou > best_iou:
                    best_iou = iou
                    best_ci = ci
            if best_iou >= 0.5 and best_ci >= 0:
                matched_gt.add(gi)
                matched_cue.add(best_ci)
        ppv_tp = len(matched_cue)
        return matched_gt, matched_cue, ppv_tp


def cue_metrics_thorough(cue: np.ndarray, gt: np.ndarray, is_idle: np.ndarray,
                         sampling_hz: int) -> dict:
    n = len(cue)
    duration_s = n / sampling_hz
    gt_events = runs(gt)
    cue_events = runs(cue)

    # ── Decomposed cue efficiency (#5) ─────────────────────────────────────
    fog_seconds = float((gt == 1).sum()) / sampling_hz
    nonfog_seconds = float((gt == 0).sum()) / sampling_hz
    in_fog_cue_seconds = float(((cue == 1) & (gt == 1)).sum()) / sampling_hz
    out_fog_cue_seconds = float(((cue == 1) & (gt == 0)).sum()) / sampling_hz

    in_fog_ratio = in_fog_cue_seconds / fog_seconds if fog_seconds > 0 else float("nan")
    fp_density = out_fog_cue_seconds / nonfog_seconds if nonfog_seconds > 0 else float("nan")
    cue_pct = 100.0 * (cue == 1).sum() / n if n else 0.0
    pct_tf = 100.0 * (gt == 1).sum() / n if n else 0.0

    # ── Both matching schemes (#2) ─────────────────────────────────────────
    out = dict(duration_s=duration_s, pct_tf=pct_tf, cue_pct=cue_pct,
               n_episodes_gt=len(gt_events), n_cues=len(cue_events),
               fog_seconds=fog_seconds, nonfog_seconds=nonfog_seconds,
               in_fog_cue_seconds=in_fog_cue_seconds,
               out_fog_cue_seconds=out_fog_cue_seconds,
               in_fog_ratio=in_fog_ratio, fp_density=fp_density)

    for mode in ("any", "iou50"):
        matched_gt, matched_cue, ppv_tp = match_events(gt_events, cue_events, mode)
        tp_e = len(matched_gt)
        fn_e = len(gt_events) - tp_e
        fp_e = len(cue_events) - ppv_tp

        sens = tp_e / (tp_e + fn_e) if (tp_e + fn_e) else float("nan")
        ppv = ppv_tp / (ppv_tp + fp_e) if (ppv_tp + fp_e) else float("nan")
        non_fog_min = nonfog_seconds / 60.0
        fp_per_min = fp_e / non_fog_min if non_fog_min > 0 else float("nan")

        out[f"sens_{mode}"] = sens
        out[f"ppv_{mode}"] = ppv
        out[f"fp_per_min_{mode}"] = fp_per_min

    # ── Onset delay restricted to IDLE-entered episodes (#3) ───────────────
    onsets_idle = []
    onsets_all = []
    for ep_start, ep_end in gt_events:
        # First cue-on frame inside this episode
        inside = np.where(cue[ep_start:ep_end] == 1)[0]
        if len(inside) == 0:
            continue
        first = inside[0] / sampling_hz
        onsets_all.append(first)
        # Was the FSM in IDLE at the start of this episode?
        if ep_start < n and is_idle[ep_start]:
            onsets_idle.append(first)

    out["onset_all_median_s"] = float(np.median(onsets_all)) if onsets_all else float("nan")
    out["onset_idle_median_s"] = float(np.median(onsets_idle)) if onsets_idle else float("nan")
    out["n_episodes_idle_entry"] = len(onsets_idle)

    return out


# ---------------------------------------------------------------------------
# Cohort aggregation (per-subject means → cohort mean ± std)
# ---------------------------------------------------------------------------

def nanmean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def cohort_agg(by_subj_metrics: dict) -> dict:
    """by_subj_metrics: {subj: [m, m, ...]}; returns dict of (mean, std, n)."""
    subj_means = []
    keys = None
    for subj, ms in by_subj_metrics.items():
        if not ms:
            continue
        subj_means.append({k: nanmean([m[k] for m in ms])
                          for k in ms[0] if isinstance(ms[0][k], float)})
        keys = list(subj_means[0])

    out = {}
    for k in keys or []:
        xs = [a[k] for a in subj_means if a[k] == a[k]]
        if not xs:
            out[k] = (float("nan"), float("nan"), 0)
        else:
            mu = sum(xs) / len(xs)
            sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
            out[k] = (mu, sd, len(xs))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu-root", default="data/imu")
    ap.add_argument("--annot-root", default="data/annotation")
    ap.add_argument("--weights-dir",
                    default=os.path.join(ROOT, "hermes", "aidfog_ai", "model_alex", "weights"))
    ap.add_argument("--task", default="TUG")
    ap.add_argument("--enter-thresh", type=int, default=20)
    ap.add_argument("--exit-thresh", type=int, default=5)
    ap.add_argument(
        "--defog-cue-grid", default="60,120,180,300,450,600",
        help="DeFOG cue_frames sweep (60Hz: 1s..10s)")
    ap.add_argument(
        "--defog-refr-grid", default="0,60,120,300,600",
        help="DeFOG refractory_frames sweep")
    ap.add_argument(
        "--fsmb-tail-grid", default="0,30,60,120,180,300,600",
        help="FSM-B tail_frames sweep")
    ap.add_argument(
        "--fsmb-cool-grid", default="0,60,120,300",
        help="FSM-B cooldown_frames sweep")
    ap.add_argument("--out", default="reports/fsm_thorough.csv")
    args = ap.parse_args()

    task_filter = None if args.task.lower() == "all" else args.task

    available = {fn.replace("subject", "").replace(".pt", "")
                 for fn in os.listdir(args.weights_dir)
                 if fn.startswith("subject") and fn.endswith(".pt")}
    trials = [t for t in list_trials(args.annot_root, args.imu_root, task_filter)
              if t[0] in available]

    # Cache binary detection
    print(f"[1/3] Caching binary detection for {len(trials)} trials...")
    cache = []
    for subj, _, annot_path in trials:
        try:
            imu_path = annot_path.replace("/annotation/", "/imu/")
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
        cache.append((subj, trial_name,
                     binary[:n].astype(np.int8), gt[:n].astype(np.int8)))
    print(f"  cached {len(cache)} trials\n")

    # Build configurations
    cfgs = []
    cfgs.append(("SINGLE", "single", None, None, single_threshold_traced))
    for cue in [int(x) for x in args.defog_cue_grid.split(",")]:
        for refr in [int(x) for x in args.defog_refr_grid.split(",")]:
            cfgs.append((f"FSM_A_cue{cue}_refr{refr}", "defog", cue, refr,
                        lambda b, c=cue, r=refr: fsm_a_defog_traced(b, c, r)))
    for tail in [int(x) for x in args.fsmb_tail_grid.split(",")]:
        for cool in [int(x) for x in args.fsmb_cool_grid.split(",")]:
            cfgs.append((f"FSM_B_tail{tail}_cool{cool}", "fsmb", tail, cool,
                        lambda b, t=tail, c=cool: fsm_b_4state_traced(b, t, c)))
    print(f"[2/3] Evaluating {len(cfgs)} controller configurations...\n")

    rows = []
    for label, kind, p1, p2, fn in cfgs:
        by_subj = defaultdict(list)
        for subj, _, binary, gt in cache:
            cue, is_idle = fn(binary)
            m = cue_metrics_thorough(cue, gt, is_idle, TARGET_HZ)
            by_subj[subj].append(m)
        agg = cohort_agg(by_subj)
        row = dict(label=label, kind=kind, p1=p1, p2=p2,
                   n_subjects=agg.get("sens_any", (0, 0, 0))[2])
        for k, (mu, sd, n_) in agg.items():
            row[k] = mu
            row[k + "_sd"] = sd
        rows.append(row)

    # ── Print summary tables ──────────────────────────────────────────────
    def find(label):
        return next((r for r in rows if r["label"] == label), None)

    print("[3/3] Cohort summary (per-subject mean → cohort mean):\n")
    print("Reference points:")
    print(f"{'controller':<32} {'sensA':>6} {'PPV_A':>6} {'sensI':>6} {'PPV_I':>6} "
          f"{'FP/m':>5} {'in_fog':>7} {'fp_dens':>8} {'on_all':>6} {'on_idle':>7}")
    print("-" * 110)
    for label in ("SINGLE", "FSM_A_cue600_refr300",  # DeFOG published
                  "FSM_B_tail60_cool120"):  # FSM-B current default
        r = find(label)
        if not r:
            continue
        print(f"{label:<32} "
              f"{100*r['sens_any']:5.1f} {100*r['ppv_any']:5.1f} "
              f"{100*r['sens_iou50']:5.1f} {100*r['ppv_iou50']:5.1f} "
              f"{r['fp_per_min_any']:4.2f} "
              f"{r['in_fog_ratio']:6.2f} "
              f"{r['fp_density']:7.4f} "
              f"{r['onset_all_median_s']:5.2f} {r['onset_idle_median_s']:5.2f}")

    # Top performers per metric
    valid = [r for r in rows if r["sens_any"] == r["sens_any"]]

    print("\nTop FSM_B configs by IoU-50 sensitivity:")
    fsmb = [r for r in valid if r["kind"] == "fsmb"]
    for r in sorted(fsmb, key=lambda x: -x["sens_iou50"])[:5]:
        print(f"  {r['label']:<32} sens_iou50={100*r['sens_iou50']:5.1f}% "
              f"ppv_iou50={100*r['ppv_iou50']:5.1f}% "
              f"in_fog={r['in_fog_ratio']:.2f} fp_dens={r['fp_density']:.4f}")

    print("\nTop FSM_A (DeFOG-style) configs by IoU-50 sensitivity:")
    fsma = [r for r in valid if r["kind"] == "defog"]
    for r in sorted(fsma, key=lambda x: -x["sens_iou50"])[:5]:
        print(f"  {r['label']:<32} sens_iou50={100*r['sens_iou50']:5.1f}% "
              f"ppv_iou50={100*r['ppv_iou50']:5.1f}% "
              f"in_fog={r['in_fog_ratio']:.2f} fp_dens={r['fp_density']:.4f}")

    print("\nMost cue-efficient (lowest in_fog_ratio) configs with IoU-50 sens >= 30%:")
    eligible = [r for r in valid
                if r["sens_iou50"] == r["sens_iou50"] and r["sens_iou50"] >= 0.30
                and r["in_fog_ratio"] == r["in_fog_ratio"]]
    for r in sorted(eligible, key=lambda x: x["in_fog_ratio"])[:8]:
        print(f"  {r['label']:<32} in_fog={r['in_fog_ratio']:.2f} "
              f"sens_iou50={100*r['sens_iou50']:5.1f}% "
              f"ppv_iou50={100*r['ppv_iou50']:5.1f}% "
              f"fp_dens={r['fp_density']:.4f}")

    # Pareto-front of (in_fog_ratio, -sens_iou50): minimize ratio, maximize sens
    print("\nPareto front (low in_fog_ratio AND high sens_iou50, dominated configs removed):")
    front = []
    for r in valid:
        if r["sens_iou50"] != r["sens_iou50"] or r["in_fog_ratio"] != r["in_fog_ratio"]:
            continue
        dominated = False
        for r2 in valid:
            if r2 is r:
                continue
            if (r2["sens_iou50"] != r2["sens_iou50"] or
                r2["in_fog_ratio"] != r2["in_fog_ratio"]):
                continue
            if (r2["sens_iou50"] >= r["sens_iou50"] and
                r2["in_fog_ratio"] <= r["in_fog_ratio"] and
                (r2["sens_iou50"] > r["sens_iou50"] or
                 r2["in_fog_ratio"] < r["in_fog_ratio"])):
                dominated = True
                break
        if not dominated:
            front.append(r)
    for r in sorted(front, key=lambda x: x["in_fog_ratio"]):
        print(f"  {r['label']:<32} in_fog={r['in_fog_ratio']:5.2f} "
              f"sens_iou50={100*r['sens_iou50']:5.1f}% "
              f"ppv_iou50={100*r['ppv_iou50']:5.1f}%")

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
