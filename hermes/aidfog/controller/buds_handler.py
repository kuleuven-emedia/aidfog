import asyncio
from multiprocessing import Process
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event as _Event
from queue import Queue, Empty
import can
import numpy as np
from collections import deque
from dataclasses import fields

from hermes.utils.time_utils import get_time, init_time
from hermes.utils.mp_utils import launch_handler


class BudsHandler:
    def __init__(
        self,
        niclas: dict,
        nicla_data_queue: "Queue[tuple[str, float, NiclaData]]",

        ref_time_s: float,
        is_ready_event: _Event,
        is_keep_data_event: _Event,
        is_stop_new_data_event: _Event,
        is_cleanup_event: _Event,
        is_finished_event: _Event,
        input_queue: "Queue[tuple[float, str]]",
        dt: float = 0.01,
    ):
        self._ref_time_s = ref_time_s
        self._dt = dt

        ##### Inter-process communication related variables.
        self._input_queue = input_queue

        self._is_ready_event = is_ready_event
        self._is_keep_data_event = is_keep_data_event
        self._is_stop_new_data_event = is_stop_new_data_event
        self._is_cleanup_event = is_cleanup_event
        self._is_finished_event = is_finished_event

        ##### Nicla Sense ME related variables.
        nicla_mapping: dict[str, dict] = niclas["device_mapping"]
        self._nicla_name_mapping = ExoNiclaMapping(
            **dict(zip(nicla_mapping.keys(), nicla_mapping.keys()))
        )  # validates input mapping.
        self._nicla_latest_data: dict[str, deque[NiclaData]] = dict(
            map(
                lambda field: (field.name, deque(maxlen=2)),
                fields(self._nicla_name_mapping),
            )
        )
        self._offsets: dict[str, float] = dict(
            map(lambda field: (field.name, 0.0), fields(self._nicla_name_mapping))
        )
        self._offsets_lock = asyncio.Lock()

        self._nicla_backend = NiclaBleBackend(
            niclas=niclas,
            nicla_latest_data=self._nicla_latest_data,
            nicla_data_queue=nicla_data_queue,
            is_keep_data_event=is_keep_data_event,
            is_stop_new_data_event=is_stop_new_data_event,
            is_cleanup_event=is_cleanup_event,
        )

    async def _measure_offsets(self, duration: float = 2.0) -> None:
        """Measure average torso, thigh, and knee offsets over given seconds."""

        print("Measuring offsets... Please stand still.", flush=True)

        samples = dict(
            map(lambda field: (field.name, []), fields(self._nicla_name_mapping))
        )

        end_s = asyncio.get_event_loop().time() + duration
        while asyncio.get_event_loop().time() < end_s:
            try:
                for device_name, device_data in self._nicla_latest_data.items():
                    samples[device_name].append(
                        wrap_angle(device_data[-1].euler[0], 90)
                    )
            except (KeyError, IndexError) as e:
                print(
                    f"Error getting offset sample for {device_name}:\n", e, flush=True
                )
                await asyncio.sleep(self._dt)
                continue
            await asyncio.sleep(self._dt)

        async with self._offsets_lock:
            print("Offsets measured:", flush=True)
            for device_name, device_samples in samples.items():
                self._offsets[device_name] = np.mean(device_samples)
                print(f"{device_name}: {self._offsets[device_name]:.2f}")

    async def _watch_for_offset_recalibration(self):
        while not self._is_cleanup_event.is_set():
            await self._recv_calibration_trigger()
            await asyncio.sleep(5)

    async def _recv_calibration_trigger(self) -> bool:
        loop = asyncio.get_event_loop()
        try:
            toa_s, user_input = await loop.run_in_executor(
                None,
                self._input_queue.get,
                True,
                0.1
            )
            if user_input == "m":
                await self._measure_offsets()
                return True
        except Empty:
            pass
        except Exception as e:
            print(f"Failed to connect: {e}", flush=True)

    async def _run(self):
        next_period_s = get_time()
        while not self._is_cleanup_event.is_set():
            next_period_s += self._dt
            # TODO: do smth useful

            end_time_s = get_time()
            if (sleep_s := next_period_s - end_time_s) > 0:
                await asyncio.sleep(sleep_s)

    async def _cleanup(self) -> None:
        print("Cleaning up Buds.", flush=True)
        await self._buds_backend.cleanup()

    async def main(self) -> None:
        # Initialize time utils for temporal alignment (data synchronization) with other HERMES components and networked host devices.
        init_time(ref_time=self._ref_time_s)

        # 1) Connect to the Nicla Sense ME sensors.
        await self._buds_backend.connect()

        # 2) Perform initial calibration.
        print("Press 'm' for initial offset calibration.", flush=True)
        is_calibrated = False
        while not is_calibrated:
            is_calibrated = await self._recv_calibration_trigger()

        # 3) Indicate to `Pipeline` that handler finished connecting and exo calibrated.
        # NOTE: Begins streaming IMU and motor data, but stores only after upstream HERMES node triggers saving via `_is_keep_data_event` event.
        self._is_ready_event.set()

        # 4) Main working loop.
        # NOTE: Loops until upstream HERMES node triggers closure via `_is_cleanup_event` event.
        # TODO: Add a coroutine with `watchdog` of the motors gains file.
        await asyncio.gather(
            self._run(),
            self._watch_for_offset_recalibration(),
            self._buds_backend.run(),
        )
        # Indicate to `BudsPipeline` that no new data will be produced.
        self._is_finished_event.set()

        # 5) Cleanup on exit.
        await self._cleanup()

    def __call__(self) -> None:
        asyncio.run(self.main())
