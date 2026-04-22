"""
Compute BLE-write-to-audio-onset latency for an aidfog trial.

Reads three artifacts from the trial folder:
  - aidfog_audio_meta.json      → t_ffmpeg_start_s, sampling_rate_hz
  - aidfog_audio_microphone.mp3 → decoded with ffmpeg-python to int16 PCM
  - aidfog_ble_op_log.json      → wall-clock timestamps of every GATT write

Finds onsets in the audio (envelope > baseline + N*MAD), pairs each with the
nearest preceding start_cue BLE write, and prints the deltas.

Usage:
    python scripts/analyze_latency.py path/to/trial_folder
    python scripts/analyze_latency.py path/to/trial_folder --window-ms 1.0 --threshold-mads 8
"""

import argparse
import json
import os
import sys

import ffmpeg
import numpy as np


def decode_mp3_to_pcm(path: str, sampling_rate_hz: int, num_channels: int = 1) -> np.ndarray:
    """Decode an MP3 to a 1-D int16 numpy array of mono samples."""
    out, _ = (
        ffmpeg.input(path)
        .output("pipe:", format="s16le", acodec="pcm_s16le",
                ar=sampling_rate_hz, ac=num_channels)
        .global_args("-hide_banner", "-loglevel", "error")
        .run(capture_stdout=True, capture_stderr=True)
    )
    arr = np.frombuffer(out, dtype=np.int16)
    return arr


def envelope_rms(pcm: np.ndarray, window_samples: int) -> np.ndarray:
    """Block-wise RMS, one value per window of `window_samples`."""
    n_full = (len(pcm) // window_samples) * window_samples
    pcm = pcm[:n_full].astype(np.float32)
    blocks = pcm.reshape(-1, window_samples)
    return np.sqrt(np.mean(blocks ** 2, axis=1))


def find_onsets(env: np.ndarray, threshold: float, refractory_blocks: int) -> list[int]:
    """Return block indices where env crosses threshold, deduplicated by refractory."""
    above = env > threshold
    onsets = []
    last = -refractory_blocks
    for i in range(len(env)):
        if above[i] and (i - last) >= refractory_blocks:
            onsets.append(i)
            last = i
    return onsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_dir", help="Path to trial_<name> folder")
    parser.add_argument("--window-ms", type=float, default=1.0,
                        help="RMS window in ms (default 1.0)")
    parser.add_argument("--threshold-mads", type=float, default=8.0,
                        help="Onset threshold in MADs above median (default 8)")
    parser.add_argument("--refractory-ms", type=float, default=50.0,
                        help="Min gap between onsets in ms (default 50)")
    args = parser.parse_args()

    meta_path = os.path.join(args.trial_dir, "aidfog_audio_meta.json")
    mp3_path = os.path.join(args.trial_dir, "aidfog_audio_microphone.mp3")
    op_log_path = os.path.join(args.trial_dir, "aidfog_ble_op_log.json")

    for p in (meta_path, mp3_path, op_log_path):
        if not os.path.exists(p):
            print(f"missing required file: {p}", file=sys.stderr)
            sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    with open(op_log_path) as f:
        op_log = json.load(f)

    fs = int(meta["sampling_rate_hz"])
    num_channels = int(meta.get("num_channels", 1))
    t_ffmpeg_start = float(meta["t_ffmpeg_start_s"])

    print(f"=== {args.trial_dir} ===")
    print(f"FFmpeg capture started at wall-clock t = {t_ffmpeg_start:.6f}")
    print(f"Sample rate: {fs} Hz, channels: {num_channels}")

    pcm = decode_mp3_to_pcm(mp3_path, fs, num_channels)
    duration_s = len(pcm) / fs
    print(f"Decoded MP3: {len(pcm)} samples = {duration_s:.2f} s")

    window_samples = max(1, int(fs * args.window_ms / 1000.0))
    refractory_blocks = max(1, int(args.refractory_ms / args.window_ms))
    env = envelope_rms(pcm, window_samples)
    seconds_per_block = window_samples / fs

    median = float(np.median(env))
    mad = float(np.median(np.abs(env - median)))
    threshold = median + args.threshold_mads * mad
    print(f"Envelope: {len(env)} blocks of {window_samples} samples "
          f"({1000*seconds_per_block:.2f} ms each)")
    print(f"Baseline median={median:.1f} MAD={mad:.1f} → threshold={threshold:.1f}")

    onsets = find_onsets(env, threshold, refractory_blocks)
    print(f"Detected {len(onsets)} onsets")
    print()

    starts = [e for e in op_log if e.get("operation") == "start_cue" and e.get("success")]
    if not starts:
        print("no successful start_cue entries in op log")
        return

    print(f"{'#':>3}  {'T5 (BLE write)':>20}  {'T7 (audio onset)':>20}"
          f"  {'Δ (T7-T5) ms':>14}  {'envelope':>10}")
    print("-" * 75)
    for i, onset_idx in enumerate(onsets):
        t_audio = t_ffmpeg_start + onset_idx * seconds_per_block
        # Pair with the closest start_cue at or just before the audio onset.
        candidates = [s for s in starts if s["timestamp"] <= t_audio + 0.05]
        if not candidates:
            continue
        match = max(candidates, key=lambda s: s["timestamp"])
        t_ble = match["timestamp"]
        delta_ms = (t_audio - t_ble) * 1000
        print(f"{i:>3}  {t_ble:>20.6f}  {t_audio:>20.6f}"
              f"  {delta_ms:>14.2f}  {env[onset_idx]:>10.1f}")

    # Summary stats over all paired deltas.
    deltas = []
    for onset_idx in onsets:
        t_audio = t_ffmpeg_start + onset_idx * seconds_per_block
        candidates = [s for s in starts if s["timestamp"] <= t_audio + 0.05]
        if candidates:
            match = max(candidates, key=lambda s: s["timestamp"])
            deltas.append(1000 * (t_audio - match["timestamp"]))
    if deltas:
        print()
        print(f"summary: n={len(deltas)} "
              f"min={min(deltas):.2f} ms  "
              f"median={float(np.median(deltas)):.2f} ms  "
              f"max={max(deltas):.2f} ms")


if __name__ == "__main__":
    main()
