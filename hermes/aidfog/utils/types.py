"""
Type definitions and constants for the PineBuds Pro Audio Cueing system.

UUIDs must match the firmware at:
  services/ble_profiles/cueing/cueingps/src/cueingps.c
"""

import struct
from dataclasses import dataclass
from enum import Enum


# --- BLE GATT UUIDs (128-bit, base: AC0000xx-CAFE-B0BA-F001-DEADBEEF0000) ---

CUEING_SERVICE_UUID = "ac000001-cafe-b0ba-f001-deadbeef0000"
CUE_CMD_CHAR_UUID = "ac000002-cafe-b0ba-f001-deadbeef0000"
CUE_STATUS_CHAR_UUID = "ac000003-cafe-b0ba-f001-deadbeef0000"
CUE_CONFIG_CHAR_UUID = "ac000004-cafe-b0ba-f001-deadbeef0000"

# --- Command bytes (host → earbud) ---

CUE_CMD_START = 0x01
CUE_CMD_STOP = 0x02
CUE_CMD_CONFIGURE = 0x03

# --- Status bytes (earbud → host) ---

CUE_STATUS_IDLE = 0x00
CUE_STATUS_CUEING = 0x01
CUE_STATUS_ERROR = 0xFF

STATUS_NAMES = {
    CUE_STATUS_IDLE: "IDLE",
    CUE_STATUS_CUEING: "CUEING",
    CUE_STATUS_ERROR: "ERROR",
}

# --- Default earbud device name (used for BLE scanning when no address is provided) ---

DEFAULT_DEVICE_NAME = "D&D TECH"


# --- Data structures ---

@dataclass
class CueingConfig:
    """Cueing configuration (matches firmware cueing_config_t, 7 bytes LE packed)."""
    tone_id: int = 0
    volume: int = 80
    duration_ms: int = 500
    burst_count: int = 1
    burst_gap_ms: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(
            "<BBHBH",
            self.tone_id & 0xFF,
            self.volume & 0xFF,
            self.duration_ms & 0xFFFF,
            self.burst_count & 0xFF,
            self.burst_gap_ms & 0xFFFF,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "CueingConfig":
        if len(data) < 7:
            return cls()
        tone_id, volume, duration_ms, burst_count, burst_gap_ms = struct.unpack(
            "<BBHBH", data[:7]
        )
        return cls(
            tone_id=tone_id,
            volume=volume,
            duration_ms=duration_ms,
            burst_count=burst_count,
            burst_gap_ms=burst_gap_ms,
        )


class CueState(Enum):
    IDLE = "IDLE"
    CUEING = "CUEING"
    CUEING_TAIL = "CUEING_TAIL"  # prob dropped, hold cue briefly in case it's a glitch
    REFRACTORY = "REFRACTORY"    # episode ended, lock out re-trigger (DeFOG terminology)


@dataclass
class CueingControlConfig:
    """Tunable parameters for the 4-state cueing FSM.

    Defaults: informed by Borzì et al. 2022 episode distribution (50% < 5 s,
    90% < 20 s) and the project's existing 2-state thresholds. All durations
    in frames at the AI's sampling rate (typically 60 Hz), so 30 frames ≈ 500 ms.

    State names follow Vayalet 2026-04-29 convention: "Cueing tail" (still cueing
    during this state) and "Refractory" (matches DeFOG vocabulary).
    """
    th_high: float = 0.7
    th_low: float = 0.3
    entry_consec: int = 3                 # frames ≥ th_high before IDLE → CUEING
    cueing_tail_frames: int = 30          # frames to hold in CUEING_TAIL before → REFRACTORY (~500 ms @ 60 Hz)
    refractory_frames: int = 60           # frames to lock out after CUEING_TAIL (~1 s @ 60 Hz)

    def __init__(self, th_high: float = 0.7, th_low: float = 0.3,
                 entry_consec: int = 3,
                 cueing_tail_frames: int | None = None,
                 refractory_frames: int | None = None,
                 tail_frames: int | None = None,
                 cooldown_frames: int | None = None) -> None:
        # Back-compat: accept legacy kwarg names from older configs/scripts
        # (`tail_frames` → `cueing_tail_frames`, `cooldown_frames` → `refractory_frames`).
        self.th_high = th_high
        self.th_low = th_low
        self.entry_consec = entry_consec
        self.cueing_tail_frames = cueing_tail_frames if cueing_tail_frames is not None else (tail_frames if tail_frames is not None else 30)
        self.refractory_frames = refractory_frames if refractory_frames is not None else (cooldown_frames if cooldown_frames is not None else 60)


@dataclass
class CueEvent:
    """Record of a single cueing event for post-hoc analysis."""
    start_time: float = 0.0
    stop_time: float = 0.0
    trigger_probability: float = 0.0
    was_false_positive: bool = False


@dataclass
class CueingAction:
    """A cueing command to be sent to the earbuds."""
    action: str  # "start", "stop", or "configure"
    tone_id: int = 0
    volume: int = 80
    duration_ms: int = 500
    burst_count: int = 1
    burst_gap_ms: int = 0

    def to_dict(self) -> dict:
        if self.action == "stop":
            return {"action": "stop"}
        elif self.action == "start":
            return {"action": "start", "tone_id": self.tone_id, "volume": self.volume}
        else:
            return {
                "action": "configure",
                "tone_id": self.tone_id,
                "volume": self.volume,
                "duration_ms": self.duration_ms,
                "burst_count": self.burst_count,
                "burst_gap_ms": self.burst_gap_ms,
            }
