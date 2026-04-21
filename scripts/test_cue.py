"""
Minimal BLE cueing smoke test — bypasses HERMES entirely.

Connects to the PineBuds Pro via bleak, sends one START cue, waits,
sends STOP. Use this to confirm earbud + firmware + BLE stack work
before running the full pipeline.

Usage (from repo root, on Windows with earbuds out of case and on):
    python scripts/test_cue.py
    python scripts/test_cue.py --address 12:34:56:C2:A2:30
    python scripts/test_cue.py --duration 1.5 --volume 80
"""

import argparse
import asyncio
import time

from bleak import BleakClient, BleakScanner

from hermes.aidfog.utils.types import (
    CUE_CMD_CHAR_UUID,
    CUE_STATUS_CHAR_UUID,
    CUE_CMD_START,
    CUE_CMD_STOP,
    DEFAULT_DEVICE_NAME,
    STATUS_NAMES,
)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="12:34:56:C2:A2:30",
                        help="BLE MAC address; if omitted, scan by name")
    parser.add_argument("--name", default=DEFAULT_DEVICE_NAME,
                        help="Device name used when scanning")
    parser.add_argument("--duration", type=float, default=1.5,
                        help="Seconds between START and STOP")
    parser.add_argument("--volume", type=int, default=80, help="0-100")
    parser.add_argument("--tone", type=int, default=0, help="tone_id")
    args = parser.parse_args()

    if args.address:
        target = args.address
        print(f"Connecting directly to {target}...")
    else:
        print(f"Scanning for '{args.name}'...")
        device = await BleakScanner.find_device_by_name(args.name, timeout=10.0)
        if device is None:
            print(f"Device '{args.name}' not found.")
            return
        target = device
        print(f"Found {device.name} [{device.address}]")

    def on_status(_sender, data: bytearray):
        if data:
            name = STATUS_NAMES.get(data[0], f"0x{data[0]:02x}")
            print(f"  <- STATUS: {name}")

    async with BleakClient(target, timeout=15.0) as client:
        print(f"Connected. MTU: {client.mtu_size}")
        await client.start_notify(CUE_STATUS_CHAR_UUID, on_status)

        cmd_start = bytes([CUE_CMD_START, args.tone & 0xFF, args.volume & 0xFF])
        t0 = time.perf_counter()
        await client.write_gatt_char(CUE_CMD_CHAR_UUID, cmd_start, response=False)
        t1 = time.perf_counter()
        print(f"-> START (tone={args.tone} vol={args.volume}) in {(t1-t0)*1000:.1f} ms")

        await asyncio.sleep(args.duration)

        cmd_stop = bytes([CUE_CMD_STOP])
        t2 = time.perf_counter()
        await client.write_gatt_char(CUE_CMD_CHAR_UUID, cmd_stop, response=False)
        t3 = time.perf_counter()
        print(f"-> STOP in {(t3-t2)*1000:.1f} ms")

        await asyncio.sleep(0.3)  # give firmware time to send final STATUS
        await client.stop_notify(CUE_STATUS_CHAR_UUID)
        print("Disconnecting.")


if __name__ == "__main__":
    asyncio.run(main())
