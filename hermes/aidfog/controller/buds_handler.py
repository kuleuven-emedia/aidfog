"""
HERMES Handler for PineBuds Pro audio cueing.

Runs in a background process. Manages BLE connection to the earbuds and
processes cueing commands received from the Pipeline via a multiprocessing Queue.
"""

import asyncio
import logging
from multiprocessing.synchronize import Event as _Event
from queue import Queue, Empty

from hermes.utils.time_utils import get_time, init_time

from .buds_facade import BudsBleBackend
from ..utils.types import CueingConfig, DEFAULT_DEVICE_NAME

logger = logging.getLogger(__name__)


class BudsHandler:
    def __init__(
        self,
        buds: dict,
        ref_time_s: float,
        is_ready_event: _Event,
        is_keep_data_event: _Event,
        is_stop_new_data_event: _Event,
        is_cleanup_event: _Event,
        is_finished_event: _Event,
        cueing_command_queue: "Queue[dict]",
        cueing_status_queue: "Queue[tuple[float, int]]",
        dt: float = 0.01,
    ):
        self._ref_time_s = ref_time_s
        self._dt = dt

        self._cueing_command_queue = cueing_command_queue
        self._cueing_status_queue = cueing_status_queue

        self._is_ready_event = is_ready_event
        self._is_keep_data_event = is_keep_data_event
        self._is_stop_new_data_event = is_stop_new_data_event
        self._is_cleanup_event = is_cleanup_event
        self._is_finished_event = is_finished_event

        device_name = buds.get("device_name", DEFAULT_DEVICE_NAME)
        address = buds.get("address", None)

        self._buds_backend = BudsBleBackend(
            device_name=device_name,
            address=address,
        )

    async def _process_commands(self):
        """Poll the command queue and forward cueing commands to the earbud."""
        loop = asyncio.get_event_loop()
        while not self._is_cleanup_event.is_set():
            try:
                cmd = await loop.run_in_executor(
                    None, self._cueing_command_queue.get, True, 0.1
                )
            except Empty:
                continue
            except Exception as e:
                logger.error("Command queue error: %s", e)
                continue

            action = cmd.get("action", "").lower()
            if action == "start":
                await self._buds_backend.start_cue(
                    tone_id=cmd.get("tone_id", 0),
                    volume=cmd.get("volume", 80),
                )
            elif action == "stop":
                await self._buds_backend.stop_cue()
            elif action == "configure":
                config = CueingConfig(
                    tone_id=cmd.get("tone_id", 0),
                    volume=cmd.get("volume", 80),
                    duration_ms=cmd.get("duration_ms", 500),
                    burst_count=cmd.get("burst_count", 1),
                    burst_gap_ms=cmd.get("burst_gap_ms", 0),
                )
                await self._buds_backend.configure(config)

            if (
                self._is_keep_data_event.is_set()
                and not self._is_stop_new_data_event.is_set()
            ):
                toa_s = get_time()
                self._cueing_status_queue.put(
                    (toa_s, self._buds_backend.current_status)
                )

    async def _watch_status(self):
        """Periodically read earbud status and push to the status queue."""
        while not self._is_cleanup_event.is_set():
            if self._buds_backend.is_connected:
                status = self._buds_backend.current_status
                if (
                    self._is_keep_data_event.is_set()
                    and not self._is_stop_new_data_event.is_set()
                ):
                    toa_s = get_time()
                    self._cueing_status_queue.put((toa_s, status))
            await asyncio.sleep(self._dt)

    async def _cleanup(self) -> None:
        logger.info("Cleaning up BudsHandler.")
        self._buds_backend._running = False
        self._buds_backend._reconnecting = True
        await self._buds_backend.stop_cue()
        await self._buds_backend.cleanup()

    async def _wait_for_connection_then_signal(self) -> None:
        """Wait until BLE connects (initial or via reconnect), then set is_ready_event."""
        while not self._is_cleanup_event.is_set():
            if self._buds_backend.is_connected:
                logger.info("BudsHandler ready, signaling pipeline")
                self._is_ready_event.set()
                return
            await asyncio.sleep(0.5)

    async def main(self) -> None:
        init_time(ref_time=self._ref_time_s)

        connected = await self._buds_backend.connect()
        if connected:
            logger.info("BudsHandler ready, signaling pipeline")
            self._is_ready_event.set()

        await asyncio.gather(
            self._process_commands(),
            self._watch_status(),
            self._buds_backend.run(self._is_cleanup_event),
            *([] if connected else [self._wait_for_connection_then_signal()]),
        )

        await self._cleanup()
        self._is_finished_event.set()

    def __call__(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="[buds_handler] %(levelname)s %(name)s: %(message)s",
            force=True,
        )
        asyncio.run(self.main())
