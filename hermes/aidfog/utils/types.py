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
    COOLDOWN = "COOLDOWN"


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
