"""
HERMES Pipeline for PineBuds Pro audio cueing.

Receives FoG detection results from the upstream AI node and translates
them into cueing commands for the BudsHandler running in a background process.
"""

import os
from multiprocessing import Process, Queue, Event
import numpy as np

from hermes.base.nodes.pipeline import Pipeline
from hermes.utils.types import LoggingSpec
from hermes.utils.time_utils import get_time
from hermes.utils.mp_utils import launch_handler
from hermes.utils.zmq_utils import (
    PORT_BACKEND,
    PORT_FRONTEND,
    PORT_SYNC_HOST,
    PORT_KILL,
)

from .controller import BudsHandler
from .stream import BudsStream
from .utils.types import CueState, CueingControlConfig


class BudsPipeline(Pipeline):
    @classmethod
    def _log_source_tag(cls) -> str:
        return "aidfog"

    def __init__(
        self,
        host_ip: str,
        stream_out_spec: dict,
        stream_in_specs: list[dict],
        logging_spec: LoggingSpec,
        port_pub: str = PORT_BACKEND,
        port_sub: str = PORT_FRONTEND,
        port_sync: str = PORT_SYNC_HOST,
        port_killsig: str = PORT_KILL,
        **_,
    ):
        buds: dict = stream_out_spec["buds"]
        dt: float = buds.get("dt", 0.01)

        # Resolve address from device_mapping (follows the Nicla pattern).
        # Use the first entry in device_mapping if present; fall back to
        # an explicit 'address' key or None (triggers name-based BLE scan).
        device_mapping: dict = buds.get("device_mapping", {})
        address = list(device_mapping.values())[0] if device_mapping else buds.get("address", None)
        buds = {**buds, "address": address}

        self._cueing_command_queue: Queue[dict] = Queue()
        self._cueing_status_queue: Queue[tuple[float, int]] = Queue()

        # 4-state cueing FSM with literature-informed defaults (see
        # CueingControlConfig docstring). Overridable per-run via the
        # `cueing_control` block in buds.yml.
        ctrl_kwargs = buds.get("cueing_control", {})
        # Back-compat: promote legacy threshold_high/threshold_low keys.
        if "threshold_high" in buds and "th_high" not in ctrl_kwargs:
            ctrl_kwargs.setdefault("th_high", buds["threshold_high"])
        if "threshold_low" in buds and "th_low" not in ctrl_kwargs:
            ctrl_kwargs.setdefault("th_low", buds["threshold_low"])
        self._ctrl = CueingControlConfig(**ctrl_kwargs)
        self._cue_state = CueState.IDLE
        self._consec_high = 0
        self._cueing_tail_remaining = 0
        self._refractory_remaining = 0

        self._is_ready_event = Event()
        self._is_keep_data_event = Event()
        self._is_stop_new_data_event = Event()
        self._is_cleanup_event = Event()
        self._is_finished_event = Event()

        hermes_kwargs = {
            "ref_time_s": logging_spec.ref_time_s,
            "is_ready_event": self._is_ready_event,
            "is_keep_data_event": self._is_keep_data_event,
            "is_stop_new_data_event": self._is_stop_new_data_event,
            "is_cleanup_event": self._is_cleanup_event,
            "is_finished_event": self._is_finished_event,
        }

        op_log_path = os.path.join(
            logging_spec.log_dir,
            "%s_ble_op_log.json" % self._log_source_tag(),
        )

        self._handler_proc = Process(
            target=launch_handler,
            args=(BudsHandler,),
            kwargs={
                "buds": buds,
                "cueing_command_queue": self._cueing_command_queue,
                "cueing_status_queue": self._cueing_status_queue,
                **hermes_kwargs,
                "dt": dt,
                "op_log_path": op_log_path,
            },
        )
        self._handler_proc.start()
        self._is_ready_event.wait()

        stream_out_spec = {"buds": buds}

        super().__init__(
            host_ip=host_ip,
            stream_out_spec=stream_out_spec,
            stream_in_specs=stream_in_specs,
            logging_spec=logging_spec,
            is_async_generate=True,
            port_pub=port_pub,
            port_sub=port_sub,
            port_sync=port_sync,
            port_killsig=port_killsig,
        )

    @classmethod
    def create_stream(cls, stream_spec: dict) -> BudsStream:
        return BudsStream(**stream_spec)

    def _keep_samples(self) -> None:
        self._is_keep_data_event.set()

    def _process_data(self, topic: str, msg: dict) -> None:
        """4-state cueing FSM driven by Alex's post-hysteresis binary stream.

        States: IDLE → CUEING → CUEING_TAIL → REFRACTORY → IDLE.
          IDLE: enter CUEING immediately on binary==1 (Alex's HysteresisFilter
                upstream already does the entry debouncing — `entry_consec`
                defaults to 1 to avoid stacking another debounce).
          CUEING: send START once on entry; remain while binary==1.
          CUEING_TAIL: binary dropped to 0; hold for `cueing_tail_frames` before
                releasing. If binary rises back to 1, fall back to CUEING. The
                patient is still being cued in this state.
          REFRACTORY: send STOP on entry; lock out re-trigger for
                `refractory_frames` (DeFOG terminology).
        """
        data = msg.get("data", {})
        pytorch_data = data.get("pytorch-worker", {})
        binary = int(pytorch_data.get("binary", 0))

        if self._cue_state == CueState.IDLE:
            self._consec_high = self._consec_high + 1 if binary == 1 else 0
            if self._consec_high >= self._ctrl.entry_consec:
                self._cue_state = CueState.CUEING
                self._consec_high = 0
                self._cueing_command_queue.put(
                    {"action": "start", "tone_id": 0, "volume": 80}
                )
        elif self._cue_state == CueState.CUEING:
            if binary == 0:
                self._cue_state = CueState.CUEING_TAIL
                self._cueing_tail_remaining = self._ctrl.cueing_tail_frames
        elif self._cue_state == CueState.CUEING_TAIL:
            if binary == 1:
                # Hysteresis re-asserted FoG before CUEING_TAIL expired —
                # restart cue. Firmware auto-stops at duration_ms (default
                # 500 ms); without re-firing we'd go silent on long dips.
                self._cue_state = CueState.CUEING
                self._cueing_command_queue.put(
                    {"action": "start", "tone_id": 0, "volume": 80}
                )
            else:
                self._cueing_tail_remaining -= 1
                if self._cueing_tail_remaining <= 0:
                    self._cue_state = CueState.REFRACTORY
                    self._refractory_remaining = self._ctrl.refractory_frames
                    self._cueing_command_queue.put({"action": "stop"})
        elif self._cue_state == CueState.REFRACTORY:
            self._refractory_remaining -= 1
            if self._refractory_remaining <= 0:
                self._cue_state = CueState.IDLE
                self._consec_high = 0

    def _generate_data(self) -> None:
        """Forward cueing status data to HERMES middleware for logging."""
        process_time_s = get_time()
        tag: str = "%s.data" % self._log_source_tag()
        output = {}

        status_data: list[tuple[float, int]] = []
        while not self._cueing_status_queue.empty():
            status_data.append(self._cueing_status_queue.get_nowait())

        if status_data:
            output["cueing"] = {
                "toa_s": np.array(
                    [[t for t, _ in status_data]], dtype=np.float64
                ).transpose((1, 0)),
                "status": np.array(
                    [[s for _, s in status_data]], dtype=np.uint8
                ).transpose((1, 0)),
            }

        if output:
            self._publish(tag, process_time_s=process_time_s, data=output)
        elif (
            self._is_finished_event.is_set()
            and self._cueing_status_queue.empty()
        ):
            self._notify_no_more_data_out()

    def _stop_new_data(self):
        self._is_cleanup_event.set()
        self._is_stop_new_data_event.set()

    def _cleanup(self) -> None:
        self._handler_proc.join()
        super()._cleanup()
