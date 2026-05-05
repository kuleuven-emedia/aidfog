"""
F7 — Joint Pareto scatter: extended_cueing_fraction vs sens_iou50, all 888 configs.

Marker shape encodes controller family (SINGLE, FSM_A=DeFOG-style fixed-duration,
FSM_B=4-state with cueing-tail + refractory). Colour encodes enter_thresh.
Highlights two reference points: Alex's published default (enter=20, exit=5,
SINGLE) and DeFOG-published (enter=20, exit=5, FSM_A cue=10s, refr=5s).

Output: reports/F7_pareto_scatter.png
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reports/full_pareto.csv")
    ap.add_argument("--out", default="reports/F7_pareto_scatter.png")
    ap.add_argument("--y-metric", default="sens_iou50",
                    choices=["sens_iou50", "sens_any", "f1_segment_alex"])
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))

    def f(r, k):
        v = r.get(k, "")
        if v in ("", "nan"):
            return float("nan")
        return float(v)

    families = {
        "SINGLE": dict(marker="o", label="SINGLE (no cueing-side FSM)", size=42),
        "FSM_A": dict(marker="s", label="FSM A (DeFOG-style: fixed cue + refractory)", size=36),
        "FSM_B": dict(marker="^", label="FSM B (4-state: CUEING_TAIL + REFRACTORY)", size=36),
    }

    enter_vals = sorted(set(int(r["enter_thresh"]) for r in rows))
    norm = Normalize(vmin=min(enter_vals), vmax=max(enter_vals))
    cmap = cm.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(13, 8), dpi=120)
    fig.patch.set_facecolor("white")

    # Compute Pareto frontier (max sens for each X bin) — extended_cueing low + sens high
    valid = [r for r in rows
             if f(r, args.y_metric) == f(r, args.y_metric)
             and f(r, "extended_cueing_fraction") == f(r, "extended_cueing_fraction")]
    front = []
    for r in valid:
        dominated = False
        for r2 in valid:
            if r2 is r:
                continue
            if (f(r2, args.y_metric) >= f(r, args.y_metric)
                and f(r2, "extended_cueing_fraction") <= f(r, "extended_cueing_fraction")
                and (f(r2, args.y_metric) > f(r, args.y_metric)
                     or f(r2, "extended_cueing_fraction") < f(r, "extended_cueing_fraction"))):
                dominated = True
                break
        if not dominated:
            front.append(r)
    front.sort(key=lambda r: f(r, "extended_cueing_fraction"))
    fx = [f(r, "extended_cueing_fraction") for r in front]
    fy = [f(r, args.y_metric) * 100 for r in front]
    ax.plot(fx, fy, color="#a02828", lw=1.6, ls="-", alpha=0.5,
            zorder=2, label=f"Pareto front (n={len(front)} non-dominated)")

    # Plot all configs by family, colored by enter_thresh
    for fam_key, fam_info in families.items():
        if fam_key == "SINGLE":
            fam_rows = [r for r in rows if r["label"] == "SINGLE"]
        else:
            fam_rows = [r for r in rows if r["label"].startswith(fam_key)]

        x = [f(r, "extended_cueing_fraction") for r in fam_rows]
        y = [f(r, args.y_metric) * 100 for r in fam_rows]  # to %
        c = [int(r["enter_thresh"]) for r in fam_rows]
        ax.scatter(x, y, c=c, cmap=cmap, norm=norm,
                   marker=fam_info["marker"], s=fam_info["size"],
                   alpha=0.65, edgecolor="#222222", linewidth=0.4,
                   label=None, zorder=3)

    # Highlight reference points
    def find_pt(label, enter="20", exit_t="5"):
        return next((r for r in rows
                     if r["label"] == label
                     and r["enter_thresh"] == enter
                     and r["exit_thresh"] == exit_t), None)

    references = [
        (find_pt("SINGLE"),
         "Alex production default\n(SINGLE, enter=20, exit=5)",
         "o", "#1a365d", (40, 40)),
        (find_pt("FSM_A_cue600_refr300"),
         "DeFOG-published\n(cue=10 s, refr=5 s; pilot-derived)",
         "s", "#a02828", (40, -10)),
        (find_pt("FSM_B_tail60_cool120"),
         "FSM B reference\n(tail=1 s, refr=2 s)",
         "^", "#2a6f3a", (-160, 25)),
    ]
    for r, name, marker, color, offset in references:
        if r is None:
            continue
        xv = f(r, "extended_cueing_fraction")
        yv = f(r, args.y_metric) * 100
        ax.scatter([xv], [yv], marker=marker, s=180,
                   facecolor="none", edgecolor=color, linewidth=2.5, zorder=5)
        ax.annotate(name, xy=(xv, yv), xytext=offset,
                    textcoords="offset points", fontsize=8.5,
                    color=color, fontweight="semibold",
                    arrowprops=dict(arrowstyle="-", color=color,
                                    lw=1.0, shrinkA=8, shrinkB=8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=color, lw=1.0, alpha=0.95))

    # Axes
    y_label_map = {
        "sens_iou50": "Strict-match sensitivity (IoU≥50, %)",
        "sens_any": "Any-overlap sensitivity (%)",
        "f1_segment_alex": "F1 (Alex segment, IoU≥50)",
    }
    ax.set_xlabel("Extended cueing fraction\n(out-of-FoG cue time / non-FoG time)", fontsize=11)
    ax.set_ylabel(y_label_map[args.y_metric], fontsize=11)
    ax.set_title(
        f"Joint Pareto surface — 888 configurations on KU Leuven AID-FOG TUG\n"
        f"Detector hysteresis (enter, exit) × controller architecture × controller params · "
        f"per-subject mean across N=10 LOSO subjects",
        fontsize=11, color="#222222",
    )
    ax.grid(True, alpha=0.3, zorder=1)

    # Custom legend (markers only, colour described separately) — bottom-right
    legend_handles = []
    # Pareto-front line entry first
    pareto_line = plt.Line2D([], [], color="#a02828", lw=1.6, ls="-", alpha=0.7,
                              label=f"Pareto front (n={len(front)} non-dominated)")
    legend_handles.append(pareto_line)
    for fam_key, fam_info in families.items():
        h = ax.scatter([], [], marker=fam_info["marker"], s=60,
                       color="#666666", edgecolor="#222222", linewidth=0.6,
                       label=fam_info["label"])
        legend_handles.append(h)
    leg = ax.legend(handles=legend_handles, loc="upper right",
                    fontsize=9, framealpha=0.95, title="Controller family")
    leg.get_title().set_fontsize(9)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("enter_thresh (frames @ 60 Hz)", fontsize=10)
    cbar.set_ticks(enter_vals)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
