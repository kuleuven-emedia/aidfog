"""
Scan Alex's IMU dataset and report which subjects/trials contain FoG.

Reads .npy annotation files under <annot_root>/<subj>/, prints per-subject
totals, per-trial breakdown, and recommends the trial(s) most useful for
plotting/characterising the cueing FSM.

Usage:
    python scripts/scan_dataset.py
    python scripts/scan_dataset.py --annot-root .\data\annotation
    python scripts/scan_dataset.py --top 5
"""

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annot-root", default="./data/annotation")
    parser.add_argument("--task", default="TUG",
                        help="Filename substring to filter by (default: TUG)")
    parser.add_argument("--top", type=int, default=5,
                        help="How many top FoG-richest trials to highlight")
    parser.add_argument("--rate-hz", type=float, default=64.0,
                        help="Original annotation sample rate (default: 64)")
    args = parser.parse_args()

    skip_list = set([f"{i:03d}" for i in range(31, 41)] +
                    ["006", "008", "020", "021", "022", "024"])

    rows = []
    for subj in sorted(os.listdir(args.annot_root)):
        subj_path = os.path.join(args.annot_root, subj)
        if subj in skip_list or not os.path.isdir(subj_path):
            continue
        for fn in sorted(os.listdir(subj_path)):
            if args.task not in fn:
                continue
            p = os.path.join(subj_path, fn)
            lab = np.load(p, allow_pickle=True)
            if lab.ndim > 1:
                lab = lab.flatten()
            lab_remap = lab.copy()
            lab_remap[lab_remap == 2] = 1
            lab_remap[lab_remap != 1] = 0
            n_frames = len(lab_remap)
            n_fog = int(lab_remap.sum())
            duration_s = n_frames / args.rate_hz
            fog_s = n_fog / args.rate_hz
            rows.append({
                "subj": subj, "trial": fn,
                "duration_s": duration_s,
                "fog_s": fog_s,
                "fog_frac": n_fog / n_frames if n_frames else 0.0,
            })

    if not rows:
        print(f"no {args.task} trials found under {args.annot_root}")
        return

    # Per-subject totals.
    by_subj = {}
    for r in rows:
        s = by_subj.setdefault(r["subj"], {"duration_s": 0.0, "fog_s": 0.0, "n_trials": 0})
        s["duration_s"] += r["duration_s"]
        s["fog_s"] += r["fog_s"]
        s["n_trials"] += 1

    print(f"=== Per-subject totals ({args.task}) ===")
    print(f"{'subj':>5}  {'trials':>6}  {'duration':>10}  {'FoG':>10}  {'frac':>6}")
    for subj in sorted(by_subj):
        s = by_subj[subj]
        frac = s["fog_s"] / s["duration_s"] if s["duration_s"] else 0.0
        print(f"{subj:>5}  {s['n_trials']:>6}  {s['duration_s']:>9.1f}s  "
              f"{s['fog_s']:>9.1f}s  {frac:>6.1%}")

    # Top-N FoG-richest trials.
    print(f"\n=== Top {args.top} FoG-richest single trials ({args.task}) ===")
    print(f"{'fog_s':>8}  {'frac':>6}  {'duration':>10}  trial")
    rows_sorted = sorted(rows, key=lambda r: r["fog_s"], reverse=True)
    for r in rows_sorted[:args.top]:
        print(f"{r['fog_s']:>7.1f}s  {r['fog_frac']:>6.1%}  "
              f"{r['duration_s']:>9.1f}s  {r['subj']}/{r['trial']}")

    # Recommendation: a trial with at least 5 s of FoG and not too short overall.
    candidates = [r for r in rows
                  if r["fog_s"] >= 5.0 and r["duration_s"] >= 30.0]
    candidates.sort(key=lambda r: r["fog_s"], reverse=True)
    print()
    if candidates:
        best = candidates[0]
        print(f"Recommended for FSM design plot: subject {best['subj']}, "
              f"trial {best['trial']}")
        print(f"  duration {best['duration_s']:.1f}s, "
              f"FoG {best['fog_s']:.1f}s ({best['fog_frac']:.1%})")
    else:
        print("no trials with >=5s of FoG and >=30s duration found")


if __name__ == "__main__":
    main()
