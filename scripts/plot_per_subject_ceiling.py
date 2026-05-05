"""
F8 — Per-subject ceiling chart (best each subject can reach across all 888 configs).

Reads reports/full_pareto_per_subject.csv and computes, for each LOSO subject,
the best sens_any, sens_iou50, and F1 reachable across the full configuration
sweep. Annotated with %TF (FoG fraction in that subject's trials).

Demonstrates Vayalet 2026-04-29 §5: there is no universal best operating point.
Subject 007 ceiling 35% any-overlap is the strongest piece of evidence for
the patient-tunable framing.

Output: reports/F8_per_subject_ceiling.png
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    rows = list(csv.DictReader(open("reports/full_pareto_per_subject.csv")))

    def f(r, k):
        v = r.get(k, "")
        if v in ("", "nan"):
            return float("nan")
        return float(v)

    # Group by subject
    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["subject"], []).append(r)

    subjects = sorted(by_subj)

    # For each subject: best each metric, plus %TF (constant across configs)
    data = []
    for subj in subjects:
        rs = by_subj[subj]
        best_any = max((f(r, "sens_any") for r in rs if f(r, "sens_any") == f(r, "sens_any")),
                       default=float("nan"))
        best_iou = max((f(r, "sens_iou50") for r in rs if f(r, "sens_iou50") == f(r, "sens_iou50")),
                       default=float("nan"))
        best_f1 = max((f(r, "f1_segment_alex") for r in rs if f(r, "f1_segment_alex") == f(r, "f1_segment_alex")),
                      default=float("nan"))
        # %TF is in the data; take any row
        pct_tf = f(rs[0], "pct_tf")
        data.append((subj, best_any * 100, best_iou * 100, best_f1, pct_tf))

    # Sort by best sens_any so the visual story (007 lowest) reads left-to-right
    data.sort(key=lambda d: d[1])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=120,
                             gridspec_kw={"width_ratios": [3, 2], "wspace": 0.18})
    fig.patch.set_facecolor("white")

    # LEFT — bar chart of per-subject ceilings
    ax = axes[0]
    x = np.arange(len(data))
    w = 0.27
    bars_any = ax.bar(x - w, [d[1] for d in data], w,
                      color="#5072a8", edgecolor="#1a365d", linewidth=0.8,
                      label="Best sens_any (any-overlap, %)")
    bars_iou = ax.bar(x, [d[2] for d in data], w,
                      color="#d2a14a", edgecolor="#7a5000", linewidth=0.8,
                      label="Best sens_iou50 (strict, %)")
    bars_f1 = ax.bar(x + w, [d[3] * 100 for d in data], w,
                     color="#5a8a5a", edgecolor="#2a6f3a", linewidth=0.8,
                     label="Best F1 × 100 (Alex segment)")

    # Annotate %TF below each subject label
    labels = [f"{d[0]}\n%TF={d[4]:.1f}" for d in data]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)

    # Cohort floor lines for reference
    ax.axhline(50, color="#a02828", lw=0.8, ls=":", alpha=0.7,
               label="Reference: 50% sens_any")
    ax.axhline(30, color="#7a2222", lw=0.8, ls=":", alpha=0.5)

    ax.set_ylabel("Best reachable across 888 configs (%)", fontsize=11)
    ax.set_xlabel("Subject (sorted by best any-overlap sensitivity)", fontsize=11)
    ax.set_title(
        "Per-subject ceiling — best reachable across 888 configurations\n"
        "Optimal config differs per subject; no operating point is universally best",
        fontsize=11, color="#222222",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, 110)

    # RIGHT — cohort-floor table summary
    ax2 = axes[1]
    ax2.axis("off")

    # Compute cohort statistics that go in the box
    sens_any_floor_30 = sum(1 for d in data if d[1] >= 30)
    sens_any_floor_50 = sum(1 for d in data if d[1] >= 50)
    sens_any_floor_70 = sum(1 for d in data if d[1] >= 70)

    # Cohort-mean numbers at Alex's production default (enter=20, exit=5, SINGLE)
    full_rows = list(csv.DictReader(open("reports/full_pareto.csv")))
    prod = next((r for r in full_rows
                 if r["enter_thresh"] == "20" and r["exit_thresh"] == "5"
                 and r["label"] == "SINGLE"), None)

    summary_text = (
        "Cohort summary — KU Leuven AID-FOG TUG\n"
        "(N=10 LOSO subjects, 219 trials, 514 GT FoG episodes,\n"
        " median episode duration 1.60 s)\n\n"
        "BEST-CASE (per subject, optimised across 888 configs):\n"
        f"  Min sens_any (worst subject):   {min(d[1] for d in data):.1f}%   ← subject 007\n"
        f"  Max sens_any (best subject):    {max(d[1] for d in data):.1f}%\n"
        f"  Subjects ≥30% sens_any:         {sens_any_floor_30}/10\n"
        f"  Subjects ≥50% sens_any:         {sens_any_floor_50}/10\n"
        f"  Subjects ≥70% sens_any:         {sens_any_floor_70}/10\n\n"
        "AT ALEX'S PRODUCTION DEFAULT (enter=20, exit=5, SINGLE):\n"
        f"  cohort sens_any   = {100*float(prod['sens_any']):.1f}% ± {100*float(prod['sens_any_sd']):.1f}\n"
        f"  cohort sens_iou50 = {100*float(prod['sens_iou50']):.1f}% ± {100*float(prod['sens_iou50_sd']):.1f}\n"
        f"  cohort F1         = {float(prod['f1_segment_alex']):.2f} ± {float(prod['f1_segment_alex_sd']):.2f}\n\n"
        "ACROSS THE FULL 888-CONFIG SWEEP:\n"
        f"  0 of 888 configs reach sens_any ≥50% for every subject\n"
        f"  59 of 888 configs reach sens_any ≥30% for every subject\n\n"
        "→ No universal best operating point.\n"
        "→ Patient-tunable parameter dial is the deployable contribution."
    )

    ax2.text(
        0.0, 0.98, summary_text,
        transform=ax2.transAxes, va="top", ha="left",
        fontsize=9.5, fontfamily="monospace", color="#222222",
        bbox=dict(boxstyle="round,pad=0.6", fc="#f7fafc",
                  ec="#cbd5e0", linewidth=1.0, alpha=0.95),
    )

    fig.tight_layout()
    out = "reports/F8_per_subject_ceiling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
