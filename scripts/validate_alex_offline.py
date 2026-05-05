"""Run Alex's predict_streaming offline on a recorded trial HDF5.

Purpose: confirm that Alex's canonical inference path correctly detects the
ground-truth FoG segments stored in the replay HDF5. If it does, the 2026-05-05
diagnosis stands — the live TorchPipeline's hand-rolled inference is the bug,
not the model itself.

Usage (from repo root):
    python scripts/validate_alex_offline.py
    python scripts/validate_alex_offline.py --hdf5 path/to/aidfog_ai.hdf5 --subject 010

The HDF5 must contain both /aidfog_replay/dots-imu/{acceleration,gyroscope,fog_label}
(produced by ImuReplayProducer) for this script to work.
"""

import argparse
import os
import sys

import h5py
import numpy as np

# Make hermes/aidfog_ai/model_alex importable as a top-level package.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "hermes", "aidfog_ai"))

from model_alex.streaming import predict_streaming  # noqa: E402


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def confusion(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt_b = gt.astype(bool)
    pr_b = pred.astype(bool)
    tp = int((gt_b & pr_b).sum())
    fp = int((~gt_b & pr_b).sum())
    fn = int((gt_b & ~pr_b).sum())
    tn = int((~gt_b & ~pr_b).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default=r".\data\project_AidFOG\trial_long_smoke2\aidfog_ai.hdf5")
    ap.add_argument("--subject", default="010",
                    help="LOSOCV subject ID. Available: 001, 004, 007, 009, 010, "
                         "011, 013, 015, 029, 030. Pick one whose data is NOT in the replay.")
    ap.add_argument("--enter-thresh", type=int, default=20)
    ap.add_argument("--exit-thresh", type=int, default=5)
    args = ap.parse_args()

    print(f"HDF5     : {args.hdf5}")
    print(f"Subject  : {args.subject}")
    print(f"Hysteresis enter/exit: {args.enter_thresh}/{args.exit_thresh}\n")

    with h5py.File(args.hdf5, "r") as f:
        acc = f["aidfog_replay/dots-imu/acceleration"][...]   # (T, 5, 3)
        gyr = f["aidfog_replay/dots-imu/gyroscope"][...]      # (T, 5, 3)
        labels = np.squeeze(f["aidfog_replay/dots-imu/fog_label"][...]).astype(np.int32)

    T = acc.shape[0]
    # Channel order per IMU: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
    # IMU order: same as the replay's sensor_indices selection.
    combined = np.concatenate([acc, gyr], axis=2)             # (T, 5, 6)
    trial_seq = combined.reshape(T, 30).astype(np.float32)    # (T, 30)
    print(f"Loaded trial: {T} frames @ 60 Hz ({T/60:.1f} s)")
    print(f"  acc range : [{acc.min():.3f}, {acc.max():.3f}]")
    print(f"  gyr range : [{gyr.min():.3f}, {gyr.max():.3f}]")
    n_fog = int((labels == 1).sum())
    print(f"  GT fog    : {n_fog}/{T} frames ({100.0*n_fog/T:.1f}%), "
          f"longest segment {longest_run(labels == 1)} frames\n")

    print("Running predict_streaming...")
    probas, preds = predict_streaming(
        trial_seq,
        subject=args.subject,
        enter_thresh=args.enter_thresh,
        exit_thresh=args.exit_thresh,
    )
    print(f"  probas range  : [{probas.min():.3f}, {probas.max():.3f}], mean={probas.mean():.3f}")
    print(f"  raw frames p>=0.5     : {int((probas>=0.5).sum())}/{T}")
    print(f"  hysteresis-on frames  : {int((preds==1).sum())}/{T}")
    print(f"  longest p>=0.5 run    : {longest_run(probas>=0.5)} frames")
    print(f"  longest pred==1 run   : {longest_run(preds==1)} frames\n")

    cm = confusion(labels, preds)
    print("Frame-level vs ground truth (post-hysteresis):")
    print(f"  TP={cm['tp']}  FP={cm['fp']}  FN={cm['fn']}  TN={cm['tn']}")
    print(f"  precision={cm['precision']:.3f}  recall={cm['recall']:.3f}  F1={cm['f1']:.3f}")

    cm_raw = confusion(labels, (probas >= 0.5).astype(np.int32))
    print("\nFrame-level vs ground truth (raw p>=0.5, no hysteresis):")
    print(f"  TP={cm_raw['tp']}  FP={cm_raw['fp']}  FN={cm_raw['fn']}  TN={cm_raw['tn']}")
    print(f"  precision={cm_raw['precision']:.3f}  recall={cm_raw['recall']:.3f}  F1={cm_raw['f1']:.3f}")


if __name__ == "__main__":
    main()
