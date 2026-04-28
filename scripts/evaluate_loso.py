"""
Offline LOSO evaluation of Alex's TCN + HysteresisFilter on the KU Leuven dataset.

For each subject with held-out weights in hermes/aidfog_ai/model_alex/weights/,
loads that subject's TUG trials, runs predict_streaming with the LOSO-trained
model, and computes sample- and event-wise clinical metrics against ground truth.

Output: per-trial CSV + per-subject aggregate + cohort summary printed to stdout.

Usage:
    .venv/bin/python scripts/evaluate_loso.py
    .venv/bin/python scripts/evaluate_loso.py --task TUG --out reports/loso_tug.csv
    .venv/bin/python scripts/evaluate_loso.py --task all  # all annotated trials

Mirrors the data-loading conventions of hermes.aidfog_replay.producer:
- sensor reindex [0, 5, 6, 2, 3] → (Pelvic, L_tibia, L_talus, R_tibia, R_talus)
- 64 Hz → 60 Hz resampling via scipy.signal.resample_poly
- annotation labels {1, 2} → 1 (trembling/akinetic FoG), {0, 4} → 0 (non-FoG)

Channel layout passed to predict_streaming: (T, 30), per-IMU
[acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z] — matches Alex's input contract.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample_poly

# Resolve hermes.aidfog_ai.model_alex import without HERMES side effects
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hermes.aidfog_ai.model_alex.streaming import predict_streaming  # noqa: E402

# Replay-producer conventions
SENSOR_INDICES = [0, 5, 6, 2, 3]
ORIGINAL_HZ = 64
TARGET_HZ = 60
SKIP_SUBJECTS = {f"{i:03d}" for i in range(31, 41)} | {
    "006", "008", "020", "021", "022", "024",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trial(imu_path: str, annot_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load + resample one trial; return (T, 30) float32 IMU and (T,) int8 labels."""
    imu = np.load(imu_path, allow_pickle=True)
    if imu.ndim != 3 or imu.shape[1] != 6:
        raise ValueError(f"unexpected IMU shape {imu.shape} in {imu_path}")
    imu = imu[SENSOR_INDICES, :, :]  # (5, 6, N)

    if ORIGINAL_HZ != TARGET_HZ:
        imu = resample_poly(imu, TARGET_HZ, ORIGINAL_HZ, axis=2)

    # (5, 6, T) -> (T, 30) per-IMU [accx,accy,accz,gyrx,gyry,gyrz]
    T = imu.shape[2]
    trial = np.transpose(imu, (2, 0, 1)).reshape(T, 30).astype(np.float32)

    lab = np.load(annot_path, allow_pickle=True)
    if lab.ndim > 1:
        lab = lab.flatten()
    lab = lab.astype(np.int64)
    lab[lab == 2] = 1
    lab[lab != 1] = 0

    if ORIGINAL_HZ != TARGET_HZ:
        lab_ds = resample_poly(
            lab[None, :].astype(np.float64), TARGET_HZ, ORIGINAL_HZ, axis=1
        )
        lab = (lab_ds >= 0.5).astype(np.int8).ravel()

    n = min(trial.shape[0], lab.shape[0])
    return trial[:n], lab[:n].astype(np.int8)


def list_trials(annot_root: str, imu_root: str, task_filter: str | None) -> list[tuple[str, str, str]]:
    """Return [(subject_id, imu_path, annot_path)] for all available trials."""
    out = []
    for subj in sorted(os.listdir(annot_root)):
        if subj in SKIP_SUBJECTS:
            continue
        subj_annot = os.path.join(annot_root, subj)
        if not os.path.isdir(subj_annot):
            continue
        for fn in sorted(os.listdir(subj_annot)):
            if task_filter and task_filter.lower() not in fn.lower():
                continue
            annot_path = os.path.join(subj_annot, fn)
            imu_path = os.path.join(imu_root, subj, fn)
            if not os.path.exists(imu_path):
                continue
            out.append((subj, imu_path, annot_path))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def runs(binary: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous 1-runs as [(start_idx, end_idx_exclusive)]."""
    if len(binary) == 0:
        return []
    diff = np.diff(np.r_[0, binary.astype(np.int8), 0])
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def event_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


@dataclass
class TrialMetrics:
    subject: str
    trial: str
    n_samples: int
    duration_s: float
    pct_tf: float

    sens_sample: float
    ppv_sample: float
    f1_sample: float
    fpr_sample: float

    sens_event: float
    ppv_event: float
    f1_event: float
    fp_per_min: float
    onset_delay_median_s: float

    n_episodes_gt: int
    n_episodes_pred: int


def compute_metrics(pred: np.ndarray, gt: np.ndarray, sampling_hz: int, iou_thresh: float = 0.5) -> dict:
    """Sample- and event-wise metrics with IoU-50 matching."""
    pred = pred.astype(np.int8)
    gt = gt.astype(np.int8)
    n = len(pred)
    duration_s = n / sampling_hz

    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())

    sens_sample = tp / (tp + fn) if (tp + fn) else float("nan")
    ppv_sample = tp / (tp + fp) if (tp + fp) else float("nan")
    f1_sample = (
        2 * sens_sample * ppv_sample / (sens_sample + ppv_sample)
        if sens_sample == sens_sample and ppv_sample == ppv_sample and (sens_sample + ppv_sample) > 0
        else float("nan")
    )
    fpr_sample = fp / (fp + tn) if (fp + tn) else float("nan")

    gt_events = runs(gt)
    pred_events = runs(pred)

    matched_pred = set()
    matched_gt = set()
    onset_delays = []
    for gi, ge in enumerate(gt_events):
        best_iou = 0.0
        best_pi = -1
        for pi, pe in enumerate(pred_events):
            if pi in matched_pred:
                continue
            iou = event_iou(ge, pe)
            if iou > best_iou:
                best_iou = iou
                best_pi = pi
        if best_iou >= iou_thresh and best_pi >= 0:
            matched_gt.add(gi)
            matched_pred.add(best_pi)
            # onset delay: first pred-1 frame inside this gt episode
            inside = pred[ge[0]:ge[1]]
            firsts = np.where(inside == 1)[0]
            if len(firsts):
                onset_delays.append(firsts[0] / sampling_hz)

    tp_event = len(matched_gt)
    fn_event = len(gt_events) - tp_event
    fp_event = len(pred_events) - len(matched_pred)

    sens_event = tp_event / (tp_event + fn_event) if (tp_event + fn_event) else float("nan")
    ppv_event = tp_event / (tp_event + fp_event) if (tp_event + fp_event) else float("nan")
    f1_event = (
        2 * sens_event * ppv_event / (sens_event + ppv_event)
        if sens_event == sens_event and ppv_event == ppv_event and (sens_event + ppv_event) > 0
        else float("nan")
    )

    non_fog_minutes = ((gt == 0).sum() / sampling_hz) / 60.0
    fp_per_min = fp_event / non_fog_minutes if non_fog_minutes > 0 else float("nan")

    pct_tf = 100.0 * (gt == 1).sum() / n if n else 0.0
    onset_median = float(np.median(onset_delays)) if onset_delays else float("nan")

    return dict(
        n_samples=n,
        duration_s=duration_s,
        pct_tf=pct_tf,
        sens_sample=sens_sample,
        ppv_sample=ppv_sample,
        f1_sample=f1_sample,
        fpr_sample=fpr_sample,
        sens_event=sens_event,
        ppv_event=ppv_event,
        f1_event=f1_event,
        fp_per_min=fp_per_min,
        onset_delay_median_s=onset_median,
        n_episodes_gt=len(gt_events),
        n_episodes_pred=len(pred_events),
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
    ap.add_argument(
        "--weights-dir",
        default=os.path.join(ROOT, "hermes", "aidfog_ai", "model_alex", "weights"),
    )
    ap.add_argument("--task", default="TUG", help="filename substring filter (e.g. TUG, 360Turn, all)")
    ap.add_argument("--enter-thresh", type=int, default=20)
    ap.add_argument("--exit-thresh", type=int, default=5)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out", default="reports/loso_evaluation.csv")
    args = ap.parse_args()

    task_filter = None if args.task.lower() == "all" else args.task

    available_subjects = {
        fn.replace("subject", "").replace(".pt", "")
        for fn in os.listdir(args.weights_dir)
        if fn.startswith("subject") and fn.endswith(".pt")
    }
    print(f"Weights available for subjects: {sorted(available_subjects)}")

    trials = list_trials(args.annot_root, args.imu_root, task_filter)
    trials = [t for t in trials if t[0] in available_subjects]
    print(f"Evaluating {len(trials)} trials across {len(set(t[0] for t in trials))} subjects "
          f"(task filter: {args.task})\n")

    results: list[TrialMetrics] = []
    by_subject: dict[str, list[dict]] = defaultdict(list)

    print(f"{'subject':>7} {'trial':<32} {'%TF':>5} "
          f"{'sens_s':>6} {'ppv_s':>6} {'F1_s':>6} {'FPR':>5} "
          f"{'sens_e':>6} {'ppv_e':>6} {'F1_e':>6} {'FP/min':>6} {'onset':>6}")
    print("-" * 124)

    for subj, imu_path, annot_path in trials:
        try:
            trial, gt = load_trial(imu_path, annot_path)
        except Exception as exc:
            print(f"  skip {os.path.basename(annot_path)}: {exc}")
            continue

        try:
            _, pred = predict_streaming(
                trial,
                subject=subj,
                weights_dir=args.weights_dir,
                thresh=args.thresh,
                enter_thresh=args.enter_thresh,
                exit_thresh=args.exit_thresh,
            )
        except Exception as exc:
            print(f"  inference failed on {subj}/{os.path.basename(annot_path)}: {exc}")
            continue

        n = min(len(pred), len(gt))
        m = compute_metrics(pred[:n], gt[:n], TARGET_HZ)

        trial_name = os.path.basename(annot_path).replace(".npy", "")
        tm = TrialMetrics(subject=subj, trial=trial_name, **m)
        results.append(tm)
        by_subject[subj].append(m)

        print(f"{subj:>7} {trial_name:<32} {m['pct_tf']:5.1f} "
              f"{fmt(m['sens_sample'], pct=True)} {fmt(m['ppv_sample'], pct=True)} "
              f"{fmt(m['f1_sample'], pct=True)} {fmt(m['fpr_sample'], pct=True)} "
              f"{fmt(m['sens_event'], pct=True)} {fmt(m['ppv_event'], pct=True)} "
              f"{fmt(m['f1_event'], pct=True)} {m['fp_per_min']:6.2f} "
              f"{fmt(m['onset_delay_median_s'])}")

    # Aggregate
    print()
    print("Per-subject means (over trials):")
    print(f"{'subject':>7} {'n':>3} {'%TF':>5} "
          f"{'sens_s':>6} {'ppv_s':>6} {'F1_s':>6} "
          f"{'sens_e':>6} {'ppv_e':>6} {'F1_e':>6} {'FP/min':>6} {'onset':>6}")
    print("-" * 100)

    def nanmean(xs):
        xs = [x for x in xs if x == x]  # drop NaN
        return sum(xs) / len(xs) if xs else float("nan")

    subj_aggs = []
    for subj in sorted(by_subject):
        ms = by_subject[subj]
        agg = {k: nanmean([m[k] for m in ms]) for k in ms[0] if isinstance(ms[0][k], float)}
        agg["n_trials"] = len(ms)
        subj_aggs.append((subj, agg))
        print(f"{subj:>7} {len(ms):>3} {agg['pct_tf']:5.1f} "
              f"{fmt(agg['sens_sample'], pct=True)} {fmt(agg['ppv_sample'], pct=True)} "
              f"{fmt(agg['f1_sample'], pct=True)} "
              f"{fmt(agg['sens_event'], pct=True)} {fmt(agg['ppv_event'], pct=True)} "
              f"{fmt(agg['f1_event'], pct=True)} {agg['fp_per_min']:6.2f} "
              f"{fmt(agg['onset_delay_median_s'])}")

    # Cohort summary
    print()
    print("Cohort (mean ± std across subjects):")
    for k in ("sens_sample", "ppv_sample", "f1_sample",
              "sens_event", "ppv_event", "f1_event",
              "fp_per_min", "onset_delay_median_s"):
        xs = [a[k] for _, a in subj_aggs if a[k] == a[k]]
        if not xs:
            continue
        mu = sum(xs) / len(xs)
        sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
        unit = "%" if k.startswith(("sens", "ppv", "f1")) else ""
        scale = 100 if unit == "%" else 1
        print(f"  {k:<25} {mu*scale:6.2f} ± {sd*scale:5.2f} {unit}  (n={len(xs)})")

    # CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "trial", "n_samples", "duration_s", "pct_tf",
                    "sens_sample", "ppv_sample", "f1_sample", "fpr_sample",
                    "sens_event", "ppv_event", "f1_event", "fp_per_min",
                    "onset_delay_median_s", "n_episodes_gt", "n_episodes_pred"])
        for r in results:
            w.writerow([
                r.subject, r.trial, r.n_samples, f"{r.duration_s:.2f}",
                f"{r.pct_tf:.2f}",
                f"{r.sens_sample:.4f}", f"{r.ppv_sample:.4f}", f"{r.f1_sample:.4f}",
                f"{r.fpr_sample:.4f}",
                f"{r.sens_event:.4f}", f"{r.ppv_event:.4f}", f"{r.f1_event:.4f}",
                f"{r.fp_per_min:.4f}", f"{r.onset_delay_median_s:.4f}",
                r.n_episodes_gt, r.n_episodes_pred,
            ])
    print(f"\nWrote {len(results)} trial rows to {args.out}")


if __name__ == "__main__":
    main()
