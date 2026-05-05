"""
F5 — End-to-end acoustic latency histogram (n=44).

Reads reports/acoustic_latency.csv, filters to 'detected' rows from the
2026-04-27 NT-USB Mini run 1 session (the n=44 valid trials Vayalet asked us
to recompute on), and produces a publication-ready histogram + summary stats.

Output: reports/F5_acoustic_latency_histogram.png
"""

import csv
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    rows = list(csv.DictReader(open("reports/acoustic_latency.csv")))
    valid = [r for r in rows
             if r["session"] == "nt_usb_mini_n50_raised"
             and r["classification"] == "detected"]
    lat = sorted(float(r["first_cross_ms"]) for r in valid)
    n = len(lat)
    mean = statistics.mean(lat)
    sd = statistics.pstdev(lat)
    median = statistics.median(lat)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
    fig.patch.set_facecolor("white")

    bins = np.arange(40, 140 + 5, 5)
    counts, edges, patches = ax.hist(
        lat, bins=bins, color="#5072a8", edgecolor="#1a365d",
        linewidth=1.0, alpha=0.85, zorder=3,
    )

    # Mean ± SD shading and lines
    ax.axvspan(mean - sd, mean + sd, alpha=0.12, color="#1a365d",
               zorder=2, label=f"mean ± SD: {mean:.1f} ± {sd:.1f} ms")
    ax.axvline(mean, color="#1a365d", lw=2.0, linestyle="-",
               zorder=4, label=f"mean = {mean:.1f} ms")
    ax.axvline(median, color="#a02828", lw=1.6, linestyle="--",
               zorder=4, label=f"median = {median:.1f} ms")

    ax.set_xlabel("End-to-end latency (ms)", fontsize=12)
    ax.set_ylabel(f"Count (out of n={n})", fontsize=12)
    ax.set_title(
        "End-to-end acoustic latency — model logit emission to audio at microphone\n"
        f"NT-USB Mini, acoustic room, raised position, 2026-04-27   "
        f"(n={n} valid; 50 - 4 sub-10 ms - 1 negative - 1 undetected)",
        fontsize=11, color="#222222",
    )

    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3, zorder=1)
    ax.set_xlim(40, 140)

    # Inline numeric summary
    ax.text(
        0.015, 0.97,
        f"n = {n}\n"
        f"mean   = {mean:.1f} ms\n"
        f"SD     = {sd:.1f} ms\n"
        f"median = {median:.1f} ms\n"
        f"range  = {min(lat):.1f}–{max(lat):.1f} ms",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9.5, fontfamily="monospace", color="#222222",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f7fafc",
                  ec="#cbd5e0", linewidth=1.0, alpha=0.95),
    )

    fig.tight_layout()
    out = "reports/F5_acoustic_latency_histogram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}  (n={n}, mean={mean:.1f} ms, SD={sd:.1f} ms)")


if __name__ == "__main__":
    main()
