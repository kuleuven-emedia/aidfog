"""
F1 — System architecture block diagram.

End-to-end pipeline: IMU → ZMQ → TCN model → ZMQ → cueing FSM → mp.Queue →
BLE handler → GATT write → custom firmware → audio. Side branch to mic for
latency measurement only (not part of patient signal path).

Output: reports/F1_architecture_diagram.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    FONT = "DejaVu Sans"
    EDGE = "#333333"

    # Colour palette per subsystem
    C_INPUT = "#C8D8F8"   # blue — sensors / IMU
    C_AI = "#D9C8F8"      # purple — model
    C_CUE = "#C8EEC8"     # green — cueing FSM
    C_BLE = "#FFE4A0"     # amber — BLE handler
    C_FW = "#F5C8C8"      # rose — firmware
    C_MEAS = "#E2E8F0"    # grey — measurement (not patient path)

    def block(x, y, w, h, label, color, sub=None, fontsize=10.5,
              code_ref=None):
        rect = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.18",
            linewidth=1.8, edgecolor=EDGE, facecolor=color, zorder=4,
        )
        ax.add_patch(rect)
        ax.text(x, y + (0.2 if sub else 0), label,
                ha="center", va="center", fontsize=fontsize,
                fontfamily=FONT, fontweight="bold",
                color="#111111", zorder=5)
        if sub:
            ax.text(x, y - 0.32, sub,
                    ha="center", va="center", fontsize=8,
                    fontfamily=FONT, color="#444444",
                    style="italic", zorder=5)
        if code_ref:
            ax.text(x, y - h / 2 - 0.18, code_ref,
                    ha="center", va="top", fontsize=7,
                    fontfamily="monospace", color="#666666", zorder=5)

    def arrow(x0, y0, x1, y1, label="", color="#222222",
              lw=1.8, label_offset=(0, 0), label_fontsize=8.5,
              label_color="#1a1a5a", curved=False):
        if curved:
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle="arc3,rad=-0.3"),
                zorder=6,
            )
        else:
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw),
                zorder=6,
            )
        if label:
            mx = (x0 + x1) / 2 + label_offset[0]
            my = (y0 + y1) / 2 + label_offset[1]
            ax.text(mx, my, label, ha="center", va="center",
                    fontsize=label_fontsize, fontfamily=FONT,
                    color=label_color, fontweight="semibold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#cccccc", lw=0.7, alpha=0.92),
                    zorder=7)

    # ── Patient signal path (top row, 7.0) ───────────────────────────────
    BW, BH = 2.2, 1.2
    y_main = 6.5

    block(1.5, y_main, BW, BH,
          "IMU sensors", C_INPUT,
          sub="5 × 6-axis MoveIt @ 60 Hz",
          code_ref="hermes.aidfog_replay")

    block(5.0, y_main, BW, BH,
          "TCN detector", C_AI,
          sub="causal, [5, 6] @ 60 Hz",
          code_ref="hermes.aidfog_ai\n(model_alex / streaming.py)")

    block(8.5, y_main, BW, BH,
          "Cueing FSM", C_CUE,
          sub="4-state: IDLE → CUEING\n→ CUEING_TAIL → REFRACTORY",
          code_ref="hermes.aidfog\n(BudsPipeline)")

    block(12.0, y_main, BW, BH,
          "BLE handler", C_BLE,
          sub="asyncio + bleak,\nGATT writes",
          code_ref="BudsHandler\n(subprocess)")

    block(15.0, y_main, 1.7, BH,
          "Earbud", C_FW,
          sub="custom firmware\nBES2300YP",
          code_ref="buds_firmware/")

    # Patient-path arrows
    arrow(1.5 + BW/2, y_main, 5.0 - BW/2, y_main,
          label="ZMQ\nIMU samples", color="#1a1a5a")
    arrow(5.0 + BW/2, y_main, 8.5 - BW/2, y_main,
          label="ZMQ\nbinary 0/1\n+ probability", color="#5a2a8a")
    arrow(8.5 + BW/2, y_main, 12.0 - BW/2, y_main,
          label="mp.Queue\nstart / stop\ncommands", color="#2a6f3a")
    arrow(12.0 + BW/2, y_main, 15.0 - 1.7/2, y_main,
          label="BLE GATT\nwrite (3 B)", color="#7a5000")

    # ── Audio output ────────────────────────────────────────────────────
    block(15.0, 4.0, 1.7, 0.9,
          "Audio cue", "#FFEDD5",
          sub="pre-stored tone, DAC")
    arrow(15.0, y_main - BH/2, 15.0, 4.0 + 0.45,
          label="firmware\nfetch + DAC", color="#8B0000",
          label_offset=(0.85, 0))

    block(15.0, 2.3, 1.7, 0.9,
          "Patient ear", "#FFFFFF", fontsize=11)
    arrow(15.0, 4.0 - 0.45, 15.0, 2.3 + 0.45,
          color="#222222", lw=2.2)

    # ── Side branch: latency measurement (lower row, not on patient path) ─
    y_meas = 1.0

    block(11.5, y_meas, 1.9, 0.85,
          "USB microphone", C_MEAS,
          sub="Rode NT-USB Mini",
          fontsize=9.5)
    block(13.8, y_meas, 1.9, 0.85,
          "Audio capture", C_MEAS,
          sub="ffmpeg → MP3",
          fontsize=9.5)
    block(15.0, 0.3, 1.7, 0.5,
          "latency CSV", C_MEAS, fontsize=9)

    # Side-branch arrows
    arrow(15.0, 2.3 - 0.45 - 0.05, 12.4, y_meas + 0.4,
          color="#888888", lw=1.4, curved=True)
    arrow(11.5 + 0.95, y_meas, 13.8 - 0.95, y_meas,
          label="audio + RMS", color="#666666",
          label_fontsize=7.5)
    arrow(13.8, y_meas - 0.4, 15.0, 0.55,
          color="#666666", lw=1.4)

    # Annotations on side branch
    ax.text(12.65, 2.0, "Acoustic capture for latency\nmeasurement only —\nnot in patient signal path",
            ha="center", va="center", fontsize=8.5, fontfamily=FONT,
            color="#555555", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f7fafc",
                      ec="#cbd5e0", linewidth=0.8, alpha=0.92))

    # ── HERMES framework label ──────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (0.4, 4.6), 13.8, 3.6,
        boxstyle="round,pad=0.0,rounding_size=0.25",
        linewidth=1.8, edgecolor="#8a8a8a", facecolor="none",
        linestyle=(0, (4, 3)), zorder=2,
    ))
    ax.text(0.6, 8.0, "HERMES framework  (ZMQ broker + multiprocess pipeline)",
            ha="left", va="center", fontsize=10, fontfamily=FONT,
            fontweight="bold", color="#555555", zorder=3,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#8a8a8a", linewidth=0.8))

    # ── Title and legend ────────────────────────────────────────────────
    ax.text(8.0, 8.6,
            "AidFOG closed-loop pipeline — IMU to audio cue",
            ha="center", va="center", fontsize=14, fontfamily=FONT,
            fontweight="bold", color="#111111")
    ax.text(8.0, 8.25,
            "Detection (TCN) and cueing (4-state FSM) run as separate processes; "
            "all parameters tunable via YAML config.",
            ha="center", va="center", fontsize=10, fontfamily=FONT,
            color="#555555", style="italic")

    # Subsystem legend
    legend_items = [
        mpatches.Patch(facecolor=C_INPUT, edgecolor=EDGE, label="Input (IMU replay or live)"),
        mpatches.Patch(facecolor=C_AI, edgecolor=EDGE, label="AI detection (TCN + hysteresis)"),
        mpatches.Patch(facecolor=C_CUE, edgecolor=EDGE, label="Cueing policy (4-state FSM)"),
        mpatches.Patch(facecolor=C_BLE, edgecolor=EDGE, label="BLE transport"),
        mpatches.Patch(facecolor=C_FW, edgecolor=EDGE, label="Custom firmware (BES2300YP)"),
        mpatches.Patch(facecolor=C_MEAS, edgecolor=EDGE, label="Measurement instrumentation"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              bbox_to_anchor=(0.0, -0.03), fontsize=8.5, framealpha=0.95,
              ncol=3, edgecolor="#bbbbbb")

    fig.tight_layout()
    out = "reports/F1_architecture_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
