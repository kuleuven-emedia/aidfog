"""
Convert one or more `calibrate_latency.py` output folders into a single CSV
of per-cue measurements, suitable for §4.3 thesis tables and figures.

For each calibration folder:
  - Reads the BLE op log (every start_cue with timestamp)
  - Decodes the captured MP3 to PCM
  - Computes the envelope and per-cue first-crossing latency
  - Writes one row per cue with: session, cue_idx, ble_time, win_max,
    first_cross_ms, classification (detected / undetected / outlier)

Output columns:
  session, cue_idx, ble_time_s, win_max, first_cross_ms,
  classification, baseline_median, baseline_mad, threshold,
  threshold_mads, sustained_blocks, mic_label, n_total

Classification rules:
  - "undetected"     : no envelope crossing within search window
  - "outlier_sub10ms": crossing < 10 ms (likely noise before cue onset)
  - "outlier_negative": crossing at or before BLE write timestamp
  - "detected"       : 10 ms <= crossing <= search_window_ms

Usage:
    .venv/bin/python scripts/latency_to_csv.py \\
        --session "nt_usb_mini_n50_raised:NT-USB Mini (raised)" \\
        --session "run_n10_open_air_plain:Rode NT-USB (plain)" \\
        --threshold-mads 3 --sustained-blocks 3 \\
        --out reports/acoustic_latency.csv
"""

import argparse
import csv
import json
import os
import sys

import ffmpeg
import numpy as np


def decode_mp3(path: str, sampling_rate_hz: int) -> np.ndarray:
    out, _ = (
        ffmpeg.input(path)
        .output("pipe:", format="s16le", acodec="pcm_s16le",
                ar=sampling_rate_hz, ac=1)
        .global_args("-hide_banner", "-loglevel", "error")
        .run(capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(out, dtype=np.int16)


def envelope_rms(pcm: np.ndarray, window_samples: int) -> np.ndarray:
    n_full = (len(pcm) // window_samples) * window_samples
    pcm = pcm[:n_full].astype(np.float32)
    return np.sqrt(np.mean(pcm.reshape(-1, window_samples) ** 2, axis=1))


def analyse_session(session_dir: str, mic_label: str,
                    threshold_mads: float, sustained_blocks: int,
                    window_ms: float = 1.0,
                    search_window_ms: float = 200.0) -> list[dict]:
    """Re-run the analyze_latency logic, return a per-cue row list."""
    meta_path = os.path.join(session_dir, "aidfog_audio_meta.json")
    mp3_path = os.path.join(session_dir, "aidfog_audio_microphone.mp3")
    op_log_path = os.path.join(session_dir, "aidfog_ble_op_log.json")
    for p in (meta_path, mp3_path, op_log_path):
        if not os.path.exists(p):
            print(f"  skip {session_dir}: missing {os.path.basename(p)}",
                  file=sys.stderr)
            return []

    with open(meta_path) as f:
        meta = json.load(f)
    with open(op_log_path) as f:
        op_log = json.load(f)

    fs = int(meta["sampling_rate_hz"])
    t_ffmpeg_start = float(meta["t_ffmpeg_start_s"])

    pcm = decode_mp3(mp3_path, fs)
    window_samples = max(1, int(fs * window_ms / 1000.0))
    env = envelope_rms(pcm, window_samples)
    seconds_per_block = window_samples / fs

    median = float(np.median(env))
    mad = float(np.median(np.abs(env - median)))
    threshold = median + threshold_mads * mad

    starts = [e for e in op_log
              if e.get("operation") == "start_cue" and e.get("success")]
    if not starts:
        print(f"  {session_dir}: no successful start_cue entries",
              file=sys.stderr)
        return []

    search_window_blocks = int(search_window_ms / window_ms)
    K = max(1, sustained_blocks)
    session_name = os.path.basename(session_dir.rstrip("/\\"))
    rows = []

    for i, s in enumerate(starts):
        t_ble = s["timestamp"]
        start_block = int((t_ble - t_ffmpeg_start) / seconds_per_block)
        end_block = min(start_block + search_window_blocks, len(env))

        if start_block < 0 or start_block >= len(env):
            rows.append(dict(
                session=session_name, cue_idx=i, ble_time_s=t_ble,
                win_max=None, first_cross_ms=None,
                classification="outside_mp3",
                baseline_median=median, baseline_mad=mad,
                threshold=threshold, threshold_mads=threshold_mads,
                sustained_blocks=sustained_blocks,
                mic_label=mic_label, n_total=len(starts),
            ))
            continue

        w = env[start_block:end_block]
        win_max = float(w.max())

        first = -1
        for j in range(len(w) - K + 1):
            if np.all(w[j:j + K] > threshold):
                first = j
                break

        if first < 0:
            classification = "undetected"
            first_cross_ms = None
        else:
            t_audio = t_ffmpeg_start + (start_block + first) * seconds_per_block
            first_cross_ms = (t_audio - t_ble) * 1000
            if first_cross_ms <= 0:
                classification = "outlier_negative"
            elif first_cross_ms < 10:
                classification = "outlier_sub10ms"
            else:
                classification = "detected"

        rows.append(dict(
            session=session_name, cue_idx=i, ble_time_s=t_ble,
            win_max=win_max, first_cross_ms=first_cross_ms,
            classification=classification,
            baseline_median=median, baseline_mad=mad,
            threshold=threshold, threshold_mads=threshold_mads,
            sustained_blocks=sustained_blocks,
            mic_label=mic_label, n_total=len(starts),
        ))

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", action="append", required=True,
                    help="`<folder_name>:<mic_label>` (relative to data/calibration). "
                         "May be passed multiple times.")
    ap.add_argument("--data-root", default="data/calibration")
    ap.add_argument("--threshold-mads", type=float, default=3.0)
    ap.add_argument("--sustained-blocks", type=int, default=3)
    ap.add_argument("--out", default="reports/acoustic_latency.csv")
    args = ap.parse_args()

    all_rows = []
    for spec in args.session:
        if ":" not in spec:
            print(f"bad --session '{spec}': use folder:label", file=sys.stderr)
            sys.exit(2)
        folder, label = spec.split(":", 1)
        path = os.path.join(args.data_root, folder)
        if not os.path.isdir(path):
            print(f"missing session folder: {path}", file=sys.stderr)
            continue
        print(f"  analysing {folder} ({label})...")
        rows = analyse_session(
            path, label,
            threshold_mads=args.threshold_mads,
            sustained_blocks=args.sustained_blocks,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("no rows produced", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    keys = list(all_rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nWrote {len(all_rows)} rows to {args.out}")

    # Per-session summary to stdout
    print()
    sessions = sorted(set(r["session"] for r in all_rows))
    for sess in sessions:
        sess_rows = [r for r in all_rows if r["session"] == sess]
        det = [r["first_cross_ms"] for r in sess_rows
               if r["classification"] == "detected"]
        n_total = sess_rows[0]["n_total"] if sess_rows else 0
        n_detected = len(det)
        if det:
            med = float(np.median(det))
            mn, mx = min(det), max(det)
            print(f"  {sess:<40}  n={n_detected}/{n_total}  "
                  f"median={med:.1f} ms  range={mn:.1f}-{mx:.1f} ms")
        else:
            print(f"  {sess:<40}  n=0/{n_total}  no detections")


if __name__ == "__main__":
    main()
