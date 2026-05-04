"""
enter_thresh sweep on the cueing side (Vayalet 2026-04-29 §4).

Slices reports/full_pareto.csv at fixed (controller, exit_thresh) and plots
all five cueing metrics across enter_thresh ∈ {5, 10, 15, 20, 30, 60}, with
mean ± SD across the LOSO subjects (population SD, N=10, matching Alex).

Default slice: SINGLE controller at exit_thresh=5 — the "no cueing-side FSM"
baseline that isolates the detector-parameter effect. Other controllers can
be selected via --label.

Output:
  - reports/enter_thresh_sweep_<label>_exit<exit>.png
  - stdout: table of mean ± SD per enter_thresh

Companion to scripts/analyze_full_pareto.py; consumes the cohort CSV (no
recomputation needed).

Usage:
    .venv/bin/python scripts/plot_enter_thresh_sweep.py
    .venv/bin/python scripts/plot_enter_thresh_sweep.py --label FSM_B_tail60_cool120
"""

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("sens_any", "Sensitivity (any-overlap)", 100, "%"),
    ("sens_iou50", "Sensitivity (IoU≥50)", 100, "%"),
    ("ppv_iou50", "PPV (IoU≥50)", 100, "%"),
    ("f1_segment_alex", "F1 (Alex segment, IoU≥50)", 1, ""),
    ("fp_density", "Wasted-audio fraction", 1, ""),
    ("in_fog_ratio", "In-FoG ratio (cue covers freeze)", 1, ""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reports/full_pareto.csv")
    ap.add_argument("--label", default="SINGLE",
                    help="controller label, e.g. SINGLE / FSM_B_tail60_cool120 / FSM_A_cue600_refr300")
    ap.add_argument("--exit-thresh", type=int, default=5)
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    sliced = [r for r in rows
              if r["label"] == args.label
              and int(r["exit_thresh"]) == args.exit_thresh]
    if not sliced:
        print(f"No rows match label={args.label} exit_thresh={args.exit_thresh}",
              file=sys.stderr)
        sys.exit(1)

    sliced.sort(key=lambda r: int(r["enter_thresh"]))
    enters = [int(r["enter_thresh"]) for r in sliced]

    def col(r, k, default=float("nan")):
        v = r.get(k, "")
        if v in ("", "nan"):
            return default
        return float(v)

    print(f"\nenter_thresh sweep — {args.label}, exit_thresh={args.exit_thresh}, "
          f"n_subjects={sliced[0].get('n_subjects', '?')}\n")
    header = f"{'enter':>6} | " + " | ".join(f"{name[:18]:>20}" for _, name, _, _ in METRICS)
    print(header)
    print("-" * len(header))
    for r in sliced:
        cells = [f"{int(r['enter_thresh']):>6}"]
        for k, _, scale, unit in METRICS:
            mu = col(r, k) * scale
            sd = col(r, k + "_sd") * scale
            cells.append(f"{mu:6.2f}{unit} ±{sd:5.2f}".rjust(20))
        print(" | ".join(cells))

    # 2x3 grid of metric plots
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axes = axes.flatten()
    for ax, (k, name, scale, unit) in zip(axes, METRICS):
        mus = np.array([col(r, k) * scale for r in sliced])
        sds = np.array([col(r, k + "_sd") * scale for r in sliced])
        ax.errorbar(enters, mus, yerr=sds, fmt="o-", capsize=4,
                    color="#2c5282", ecolor="#9ca3af", linewidth=2,
                    markersize=7)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("enter_thresh (frames @ 60 Hz)", fontsize=9)
        ax.set_ylabel(unit if unit else "value", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(enters)

    fig.suptitle(
        f"enter_thresh sweep — {args.label}, exit_thresh={args.exit_thresh}\n"
        f"Mean ± SD across N={sliced[0].get('n_subjects', '?')} LOSO subjects (KU Leuven, TUG)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir,
                       f"enter_thresh_sweep_{args.label}_exit{args.exit_thresh}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
