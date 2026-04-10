"""
BLE backend for PineBuds Pro audio cueing.

Manages the BLE connection lifecycle (discover, connect, reconnect, cleanup)
and exposes async methods to send cueing commands and receive status notifications.
Follows the same Backend pattern as the original NiclaBleBackend.
"""

import asyncio
import logging
import time
from typing import Optional

from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

from ..utils.types import (
    CUE_CMD_CHAR_UUID,
    CUE_STATUS_CHAR_UUID,
    CUE_CONFIG_CHAR_UUID,
    CUE_CMD_START,
    CUE_CMD_STOP,
    CUE_CMD_CONFIGURE,
    CUE_STATUS_IDLE,
    CueingConfig,
    STATUS_NAMES,
    DEFAULT_DEVICE_NAME,
)

logger = logging.getLogger(__name__)


class BudsBleBackend:
    """
    Async BLE backend for a single PineBuds Pro earbud.

    Handles discovery, connection, status notifications, and command writes.
    Supports both name-based scanning and direct MAC address connection.
    """

    def __init__(
        self,
        device_name: str = DEFAULT_DEVICE_NAME,
        address: Optional[str] = None,
        scan_timeout: float = 10.0,
        max_reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0,
        reconnect_backoff: float = 1.5,
    ):
        self._device_name = device_name
        self._address = address
        self._scan_timeout = scan_timeout
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._reconnect_backoff = reconnect_backoff

        self._client: Optional[BleakClient] = None
        self._connected = False
        self._current_status: int = CUE_STATUS_IDLE
        self._status_event = asyncio.Event()
        self._reconnecting = False
        self._running = False
        self._op_log: list[dict] = []

    @property
    def is_connected(self) -> bool:
        return (
            self._connected
            and self._client is not None
            and self._client.is_connected
        )

    @property
    def current_status(self) -> int:
        return self._current_status

    @property
    def current_status_name(self) -> str:
        return STATUS_NAMES.get(self._current_status, f"UNKNOWN(0x{self._current_status:02x})")

    @property
    def operation_log(self) -> list[dict]:
        return list(self._op_log)

    def _log_op(self, op: str, latency_ms: Optional[float] = None, success: bool = True):
        entry = {
            "timestamp": time.time(),
            "perf_counter": time.perf_counter(),
            "operation": op,
            "success": success,
        }
        if latency_ms is not None:
            entry["latency_ms"] = latency_ms
        self._op_log.append(entry)

    def _status_callback(self, sender, data: bytearray):
        if len(data) > 0:
            self._current_status = data[0]
        self._status_event.set()
        logger.debug("Status notification: %s", self.current_status_name)

    def _on_disconnect(self, client: BleakClient):
        logger.warning("BLE disconnected")
        self._connected = False
        if self._running and not self._reconnecting:
            asyncio.create_task(self._auto_reconnect())

    async def _auto_reconnect(self):
        if self._reconnecting:
            return
        self._reconnecting = True
        delay = self._reconnect_delay

        for attempt in range(1, self._max_reconnect_attempts + 1):
            logger.info("Reconnect attempt %d/%d (delay %.1fs)",
                        attempt, self._max_reconnect_attempts, delay)
            await asyncio.sleep(delay)
            try:
                if await self._do_connect():
                    logger.info("Reconnected on attempt %d", attempt)
                    self._log_op("reconnect", success=True)
                    self._reconnecting = False
                    return
            except Exception as e:
                logger.warning("Reconnect attempt %d failed: %s", attempt, e)
            delay = min(delay * self._reconnect_backoff, 30.0)

        logger.error("Failed to reconnect after %d attempts", self._max_reconnect_attempts)
        self._log_op("reconnect", success=False)
        self._reconnecting = False

    async def _do_connect(self) -> bool:
        if self._address:
            logger.info("Connecting directly to %s...", self._address)
            target = self._address
        else:
            device = await BleakScanner.find_device_by_name(
                self._device_name, timeout=self._scan_timeout
            )
            if device is None:
                logger.warning("Device '%s' not found", self._device_name)
                return False
            logger.info("Found %s [%s]", device.name, device.address)
            target = device

        self._client = BleakClient(
            target,
            disconnected_callback=self._on_disconnect,
            timeout=15.0,
        )
        await self._client.connect()
        self._connected = True

        await self._client.start_notify(CUE_STATUS_CHAR_UUID, self._status_callback)
        logger.info("Connected and subscribed. MTU: %d", self._client.mtu_size)
        return True

    # --- Public API ---

    async def connect(self) -> bool:
        """Discover and connect to the earbud."""
        try:
            self._running = True
            return await self._do_connect()
        except BleakError as e:
            logger.error("Connection error: %s", e)
            self._connected = False
            return False

    async def run(self, is_cleanup_event):
        """Background reconnection loop (matches NiclaBleBackend.run pattern)."""
        while not is_cleanup_event.is_set():
            if not self.is_connected and not self._reconnecting:
                await self._auto_reconnect()
            await asyncio.sleep(1.5)

    async def cleanup(self):
        """Disconnect and release resources."""
        self._running = False
        self._reconnecting = True  # prevent _on_disconnect from triggering reconnect
        if self._client and self._client.is_connected:
            try:
                await self._client.stop_notify(CUE_STATUS_CHAR_UUID)
            except BleakError:
                pass
            await self._client.disconnect()
        self._connected = False
        logger.info("BLE cleanup done")

    async def _ensure_connected(self) -> bool:
        if self.is_connected:
            return True
        logger.warning("Not connected, reconnecting...")
        try:
            return await self._do_connect()
        except BleakError as e:
            logger.error("Reconnect failed: %s", e)
            return False

    async def start_cue(self, tone_id: int = 0, volume: int = 80) -> bool:
        if not await self._ensure_connected():
            self._log_op("start_cue", success=False)
            return False

        cmd = bytes([CUE_CMD_START, tone_id & 0xFF, volume & 0xFF])
        try:
            self._status_event.clear()
            t0 = time.perf_counter()
            await self._client.write_gatt_char(CUE_CMD_CHAR_UUID, cmd, response=False)
            elapsed = (time.perf_counter() - t0) * 1000
            self._log_op("start_cue", latency_ms=elapsed)
            return True
        except BleakError as e:
            logger.error("Failed to send START: %s", e)
            self._log_op("start_cue", success=False)
            self._connected = False
            return False

    async def stop_cue(self) -> bool:
        if not await self._ensure_connected():
            self._log_op("stop_cue", success=False)
            return False

        cmd = bytes([CUE_CMD_STOP])
        try:
            self._status_event.clear()
            t0 = time.perf_counter()
            await self._client.write_gatt_char(CUE_CMD_CHAR_UUID, cmd, response=False)
            elapsed = (time.perf_counter() - t0) * 1000
            self._log_op("stop_cue", latency_ms=elapsed)
            return True
        except BleakError as e:
            logger.error("Failed to send STOP: %s", e)
            self._log_op("stop_cue", success=False)
            self._connected = False
            return False

    async def configure(self, config: CueingConfig) -> bool:
        if not await self._ensure_connected():
            self._log_op("configure", success=False)
            return False

        cmd = bytes([CUE_CMD_CONFIGURE]) + config.to_bytes()
        try:
            t0 = time.perf_counter()
            await self._client.write_gatt_char(CUE_CMD_CHAR_UUID, cmd, response=False)
            elapsed = (time.perf_counter() - t0) * 1000
            self._log_op("configure", latency_ms=elapsed)
            return True
        except BleakError as e:
            logger.error("Failed to send CONFIGURE: %s", e)
            self._log_op("configure", success=False)
            self._connected = False
            return False

    async def read_config(self) -> Optional[CueingConfig]:
        if not await self._ensure_connected():
            return None
        try:
            data = await self._client.read_gatt_char(CUE_CONFIG_CHAR_UUID)
            return CueingConfig.from_bytes(data)
        except BleakError as e:
            logger.error("Failed to read config: %s", e)
            return None

    async def read_status(self) -> Optional[int]:
        if not await self._ensure_connected():
            return None
        try:
            data = await self._client.read_gatt_char(CUE_STATUS_CHAR_UUID)
            return data[0] if data else None
        except BleakError as e:
            logger.error("Failed to read status: %s", e)
            return None

    async def wait_for_status(self, timeout: float = 2.0) -> Optional[int]:
        self._status_event.clear()
        try:
            await asyncio.wait_for(self._status_event.wait(), timeout=timeout)
            return self._current_status
        except asyncio.TimeoutError:
            return None
