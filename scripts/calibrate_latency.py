"""
Standalone BLE-to-audio latency calibration — bypasses HERMES entirely.

Captures Rode mic audio via FFmpeg and fires N isolated cues spaced
`--interval` seconds apart. Each cue's BLE write timestamp is recorded.
Saves the MP3 and the timestamps next to each other so analyze_latency.py
can correlate them.

Usage (Windows, mic touching one earbud, earbud out of case and on):
    python scripts/calibrate_latency.py
    python scripts/calibrate_latency.py --num 10 --interval 2.5 --volume 100
    python scripts/calibrate_latency.py --output-dir .\data\calibration\run1
"""

import argparse
import asyncio
import json
import os
import time

import ffmpeg
from bleak import BleakClient

from hermes.aidfog.utils.types import (
    CUE_CMD_CHAR_UUID,
    CUE_CMD_START,
    CUE_CMD_STOP,
)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="12:34:56:C2:A2:30")
    parser.add_argument("--num", type=int, default=10,
                        help="Number of cue events (default 10)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between cues (default 2.0)")
    parser.add_argument("--volume", type=int, default=100,
                        help="Cue volume 0-100 (default 100)")
    parser.add_argument("--tone", type=int, default=0)
    parser.add_argument("--device", default="Microphone (RODE NT-USB)",
                        help="FFmpeg dshow device name")
    parser.add_argument("--rate", type=int, default=48000)
    parser.add_argument("--output-dir", default=None,
                        help="Output folder; defaults to .\\data\\calibration\\<timestamp>")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(
        ".", "data", "calibration", time.strftime("run_%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)
    mp3_path = os.path.join(out_dir, "aidfog_audio_microphone.mp3")
    meta_path = os.path.join(out_dir, "aidfog_audio_meta.json")
    op_log_path = os.path.join(out_dir, "aidfog_ble_op_log.json")

    print(f"Output: {out_dir}")
    print(f"Will fire {args.num} cues, {args.interval}s apart, volume={args.volume}")

    # 1. Start FFmpeg encoder writing to MP3 directly (no Storage involvement).
    print("Starting FFmpeg mic capture...")
    t_ffmpeg_start_s = time.time()
    ff_proc = (
        ffmpeg
        .input(f"audio={args.device}", format="dshow",
               ar=args.rate, ac=1, rtbufsize="64M")
        .output(mp3_path, format="mp3", acodec="libmp3lame",
                audio_bitrate="128k")
        .global_args("-hide_banner", "-loglevel", "error", "-y")
        .run_async(pipe_stdin=True)
    )
    with open(meta_path, "w") as f:
        json.dump({
            "t_ffmpeg_start_s": t_ffmpeg_start_s,
            "sampling_rate_hz": args.rate,
            "num_channels": 1,
            "chunk_size": 1024,
            "sample_format": "s16",
            "device_name": args.device,
            "ffmpeg_backend": "dshow",
        }, f, indent=2)

    # Give FFmpeg a moment to actually start grabbing samples.
    await asyncio.sleep(0.5)

    op_log = []
    cue_start = bytes([CUE_CMD_START, args.tone & 0xFF, args.volume & 0xFF])
    cue_stop = bytes([CUE_CMD_STOP])

    print(f"Connecting to {args.address}...")
    async with BleakClient(args.address, timeout=15.0) as client:
        print(f"Connected. MTU: {client.mtu_size}. Settling 1s before first cue...")
        await asyncio.sleep(1.0)

        for i in range(args.num):
            t_before = time.time()
            t_perf = time.perf_counter()
            await client.write_gatt_char(CUE_CMD_CHAR_UUID, cue_start, response=False)
            elapsed_ms = (time.perf_counter() - t_perf) * 1000
            op_log.append({
                "timestamp": t_before,
                "perf_counter": t_perf,
                "operation": "start_cue",
                "success": True,
                "latency_ms": elapsed_ms,
            })
            print(f"  cue {i+1}/{args.num} fired at t={t_before:.6f} ({elapsed_ms:.2f} ms BLE)")
            # Firmware auto-stops at ~500 ms; spacing between cues should
            # exceed firmware duration so each beep is isolated in the audio.
            await asyncio.sleep(args.interval)

        # One final stop just to be tidy.
        await client.write_gatt_char(CUE_CMD_CHAR_UUID, cue_stop, response=False)

        # Trailing capture so the last cue's audio is fully recorded.
        await asyncio.sleep(0.5)

    # Stop capture cleanly.
    print("Stopping FFmpeg...")
    try:
        ff_proc.stdin.write(b"q")
        ff_proc.stdin.close()
    except Exception:
        pass
    ff_proc.wait(timeout=5)

    with open(op_log_path, "w") as f:
        json.dump(op_log, f, indent=2)

    print(f"\nDone. Now analyze with:")
    print(f"  python scripts/analyze_latency.py {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
