"""
FSM A (DeFOG) vs FSM B (4-state) head-to-head on real LOSO predictions.

For each weighted subject's TUG trials:
  1. Run Alex's predict_streaming → binary detection at 60 Hz (post-HysteresisFilter)
  2. Feed binary into three controllers:
     - FSM A: DeFOG (Zoetewei 2021) — IDLE → CUEING (10s) → REFRACTORY (5s) → IDLE
     - FSM B: ours — IDLE → CUEING → CUEING_TAIL (1s) → REFRACTORY (2s) → IDLE; bounce-back re-fires
     - SINGLE: bare threshold pass-through (cue == binary)
  3. Compute clinical metrics per controller per trial vs ground truth.

Note (Vayalet 2026-04-29): the runtime FSM in `hermes/aidfog/utils/types.py` uses
the state names CUEING_TAIL and REFRACTORY. This script keeps the legacy
parameter names `tail_frames` / `cooldown_frames` for CSV-column compatibility
with `reports/fsm_comparison.csv`; they map to `cueing_tail_frames` and
`refractory_frames` respectively.

Output:
  - reports/fsm_comparison.csv  (per-trial × per-FSM)
  - stdout cohort summary

Cue→episode matching: any-overlap (≥1 frame) — DeFOG convention. Cue counts as
TP if it overlaps any FoG ground-truth episode; FP otherwise. FoG episode counts
as detected if any cue overlaps it.

Usage:
    .venv/bin/python scripts/compare_fsms.py
    .venv/bin/python scripts/compare_fsms.py --task all --tail-frames 60 --cooldown-frames 120
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

# Import data-loading helpers from evaluate_loso (sibling script)
from evaluate_loso import (  # noqa: E402
    list_trials,
    load_trial,
    runs,
    TARGET_HZ,
)

# ---------------------------------------------------------------------------
# FSM controllers — sample-by-sample, return per-frame cue state {0,1}
# ---------------------------------------------------------------------------

def fsm_a_defog(binary: np.ndarray, cue_frames: int = 10 * TARGET_HZ,
                refractory_frames: int = 5 * TARGET_HZ) -> np.ndarray:
    """DeFOG controller: 10s fixed cue, 5s refractory."""
    n = len(binary)
    cue = np.zeros(n, dtype=np.int8)
    state = "IDLE"
    cue_remaining = 0
    refr_remaining = 0
    for t in range(n):
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
    return cue


def fsm_b_4state(binary: np.ndarray, tail_frames: int = 60,
                 cooldown_frames: int = 120) -> np.ndarray:
    """Our 4-state controller: IDLE → CUEING → CUEING_TAIL → REFRACTORY → IDLE.

    Parameter names `tail_frames` / `cooldown_frames` are kept for CSV
    compatibility; they correspond to `cueing_tail_frames` and
    `refractory_frames` in the runtime FSM.
    """
    n = len(binary)
    cue = np.zeros(n, dtype=np.int8)
    state = "IDLE"
    cueing_tail_remaining = 0
    refractory_remaining = 0
    for t in range(n):
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
    return cue


def single_threshold(binary: np.ndarray) -> np.ndarray:
    """Bare pass-through: cue when binary==1, stop when binary==0."""
    return binary.astype(np.int8)


# ---------------------------------------------------------------------------
# Cue→episode metrics (any-overlap matching, DeFOG-style)
# ---------------------------------------------------------------------------

def cue_metrics(cue: np.ndarray, gt: np.ndarray, sampling_hz: int) -> dict:
    """Cue-event metrics vs ground-truth FoG episodes (any-overlap matching)."""
    n = len(cue)
    duration_s = n / sampling_hz
    gt_episodes = runs(gt)
    cue_events = runs(cue)

    if not gt_episodes and not cue_events:
        return dict(
            duration_s=duration_s, pct_tf=0.0,
            n_episodes_gt=0, n_episodes_cued=0, n_cues=0,
            sensitivity=float("nan"), ppv=float("nan"),
            fp_per_min=0.0, onset_delay_median_s=float("nan"),
            tail_overshoot_median_s=float("nan"),
            cue_pct=0.0,
        )

    # Per-episode: detected if any cue overlaps
    onset_delays = []
    tail_overshoots = []
    detected_episodes = 0
    for ep_start, ep_end in gt_episodes:
        any_overlap = False
        first_cue_in_ep = None
        last_cue_end_in_ep = None
        for c_start, c_end in cue_events:
            if c_end > ep_start and c_start < ep_end:
                any_overlap = True
                if first_cue_in_ep is None or c_start < first_cue_in_ep:
                    first_cue_in_ep = c_start
                if last_cue_end_in_ep is None or c_end > last_cue_end_in_ep:
                    last_cue_end_in_ep = c_end
        if any_overlap:
            detected_episodes += 1
            if first_cue_in_ep is not None:
                onset_delays.append(max(0, first_cue_in_ep - ep_start) / sampling_hz)
            if last_cue_end_in_ep is not None and last_cue_end_in_ep > ep_end:
                tail_overshoots.append((last_cue_end_in_ep - ep_end) / sampling_hz)

    # Per-cue: TP if it overlaps any FoG episode, else FP
    tp_cues = 0
    fp_cues = 0
    for c_start, c_end in cue_events:
        overlaps = any(c_end > es and c_start < ee for es, ee in gt_episodes)
        if overlaps:
            tp_cues += 1
        else:
            fp_cues += 1

    sensitivity = detected_episodes / len(gt_episodes) if gt_episodes else float("nan")
    ppv = tp_cues / (tp_cues + fp_cues) if (tp_cues + fp_cues) > 0 else float("nan")
    non_fog_min = ((gt == 0).sum() / sampling_hz) / 60.0
    fp_per_min = fp_cues / non_fog_min if non_fog_min > 0 else float("nan")
    pct_tf = 100.0 * (gt == 1).sum() / n if n else 0.0
    cue_pct = 100.0 * cue.sum() / n if n else 0.0

    return dict(
        duration_s=duration_s,
        pct_tf=pct_tf,
        n_episodes_gt=len(gt_episodes),
        n_episodes_cued=detected_episodes,
        n_cues=len(cue_events),
        sensitivity=sensitivity,
        ppv=ppv,
        fp_per_min=fp_per_min,
        onset_delay_median_s=float(np.median(onset_delays)) if onset_delays else float("nan"),
        tail_overshoot_median_s=float(np.median(tail_overshoots)) if tail_overshoots else float("nan"),
        cue_pct=cue_pct,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and x != x):
        return "  -  "
    if pct:
        return f"{100*x:5.1f}"
    return f"{x:5.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu-root", default="data/imu")
    ap.add_argument("--annot-root", default="data/annotation")
    ap.add_argument("--weights-dir",
                    default=os.path.join(ROOT, "hermes", "aidfog_ai", "model_alex", "weights"))
    ap.add_argument("--task", default="TUG")
    ap.add_argument("--enter-thresh", type=int, default=20)
    ap.add_argument("--exit-thresh", type=int, default=5)
    ap.add_argument("--tail-frames", type=int, default=60)
    ap.add_argument("--cooldown-frames", type=int, default=120)
    ap.add_argument("--out", default="reports/fsm_comparison.csv")
    args = ap.parse_args()

    task_filter = None if args.task.lower() == "all" else args.task

    available = {fn.replace("subject", "").replace(".pt", "")
                 for fn in os.listdir(args.weights_dir)
                 if fn.startswith("subject") and fn.endswith(".pt")}
    trials = [t for t in list_trials(args.annot_root, args.imu_root, task_filter)
              if t[0] in available]
    print(f"Evaluating {len(trials)} trials × 3 controllers (FSM A, FSM B, single)\n")

    fsm_specs = [
        ("FSM_A_DeFOG", lambda b: fsm_a_defog(b)),
        ("FSM_B_4state", lambda b: fsm_b_4state(b, args.tail_frames, args.cooldown_frames)),
        ("SINGLE", single_threshold),
    ]

    all_rows = []
    by_subj: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for subj, imu_path, annot_path in trials:
        try:
            trial, gt = load_trial(imu_path, annot_path)
        except Exception as exc:
            print(f"skip {os.path.basename(annot_path)}: {exc}")
            continue
        try:
            _, binary = predict_streaming(
                trial, subject=subj, weights_dir=args.weights_dir,
                enter_thresh=args.enter_thresh, exit_thresh=args.exit_thresh,
            )
        except Exception as exc:
            print(f"inference fail {subj}/{os.path.basename(annot_path)}: {exc}")
            continue
        n = min(len(binary), len(gt))
        binary = binary[:n].astype(np.int8)
        gt = gt[:n].astype(np.int8)

        trial_name = os.path.basename(annot_path).replace(".npy", "")
        for fsm_name, fsm_fn in fsm_specs:
            cue = fsm_fn(binary)
            m = cue_metrics(cue, gt, TARGET_HZ)
            row = dict(subject=subj, trial=trial_name, fsm=fsm_name, **m)
            all_rows.append(row)
            by_subj[subj][fsm_name].append(m)

    # Cohort summary per FSM
    print("Cohort summary per controller (mean ± std across N=10 subjects):\n")
    print(f"{'controller':<14} {'sens':>13} {'PPV':>13} {'FP/min':>13} "
          f"{'onset_s':>11} {'overshoot_s':>13} {'%cue':>7}")
    print("-" * 95)

    def nanmean(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    cohort_summary = {}
    for fsm_name, _ in fsm_specs:
        # Per subject: mean over their trials (NaN-aware)
        subj_aggs = []
        for subj in sorted(by_subj):
            ms = by_subj[subj][fsm_name]
            if not ms:
                continue
            subj_aggs.append({
                k: nanmean([m[k] for m in ms])
                for k in ms[0] if isinstance(ms[0][k], float)
            })

        def musd(key, scale=1.0):
            xs = [a[key] for a in subj_aggs if a[key] == a[key]]
            if not xs:
                return float("nan"), float("nan"), 0
            mu = sum(xs) / len(xs)
            sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
            return mu * scale, sd * scale, len(xs)

        sens_mu, sens_sd, sens_n = musd("sensitivity", 100)
        ppv_mu, ppv_sd, ppv_n = musd("ppv", 100)
        fp_mu, fp_sd, fp_n = musd("fp_per_min")
        onset_mu, onset_sd, onset_n = musd("onset_delay_median_s")
        over_mu, over_sd, over_n = musd("tail_overshoot_median_s")
        cuepct_mu, cuepct_sd, _ = musd("cue_pct")
        cohort_summary[fsm_name] = dict(
            sens=(sens_mu, sens_sd, sens_n), ppv=(ppv_mu, ppv_sd, ppv_n),
            fp=(fp_mu, fp_sd, fp_n), onset=(onset_mu, onset_sd, onset_n),
            over=(over_mu, over_sd, over_n), cue_pct=(cuepct_mu, cuepct_sd),
        )
        print(f"{fsm_name:<14} "
              f"{sens_mu:5.1f}±{sens_sd:4.1f} ({sens_n:>2})  "
              f"{ppv_mu:5.1f}±{ppv_sd:4.1f} ({ppv_n:>2})  "
              f"{fp_mu:5.1f}±{fp_sd:4.1f} ({fp_n:>2})  "
              f"{onset_mu:5.2f}±{onset_sd:4.2f}  "
              f"{over_mu:5.2f}±{over_sd:4.2f}    "
              f"{cuepct_mu:4.1f}%")

    # Pairwise deltas (FSM B vs FSM A; FSM B vs single)
    print("\nDeltas (FSM B vs FSM A; FSM B vs SINGLE) — per-subject paired means:\n")
    for ref in ("FSM_A_DeFOG", "SINGLE"):
        print(f"  vs {ref}:")
        deltas = defaultdict(list)
        for subj in sorted(by_subj):
            ms_b = by_subj[subj]["FSM_B_4state"]
            ms_ref = by_subj[subj][ref]
            if not ms_b or not ms_ref:
                continue
            for k in ("sensitivity", "ppv", "fp_per_min", "onset_delay_median_s",
                      "tail_overshoot_median_s", "cue_pct"):
                b_val = nanmean([m[k] for m in ms_b])
                r_val = nanmean([m[k] for m in ms_ref])
                if b_val == b_val and r_val == r_val:
                    deltas[k].append(b_val - r_val)
        for k, vs in deltas.items():
            if not vs:
                continue
            mu = sum(vs) / len(vs)
            sd = (sum((v - mu) ** 2 for v in vs) / len(vs)) ** 0.5
            unit = "%" if k in ("sensitivity", "ppv", "cue_pct") else ""
            scale = 100 if unit == "%" else 1
            print(f"    Δ {k:<25} {mu*scale:+6.2f} ± {sd*scale:5.2f} {unit}  (n={len(vs)})")

    # CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "trial", "fsm", "duration_s", "pct_tf",
                    "n_episodes_gt", "n_episodes_cued", "n_cues",
                    "sensitivity", "ppv", "fp_per_min",
                    "onset_delay_median_s", "tail_overshoot_median_s", "cue_pct"])
        for r in all_rows:
            w.writerow([r["subject"], r["trial"], r["fsm"],
                        f"{r['duration_s']:.2f}", f"{r['pct_tf']:.2f}",
                        r["n_episodes_gt"], r["n_episodes_cued"], r["n_cues"],
                        f"{r['sensitivity']:.4f}", f"{r['ppv']:.4f}",
                        f"{r['fp_per_min']:.4f}",
                        f"{r['onset_delay_median_s']:.4f}",
                        f"{r['tail_overshoot_median_s']:.4f}",
                        f"{r['cue_pct']:.4f}"])
    print(f"\nWrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
