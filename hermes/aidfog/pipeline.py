"""
HERMES Pipeline for PineBuds Pro audio cueing.

Receives FoG detection results from the upstream AI node and translates
them into cueing commands for the BudsHandler running in a background process.
"""

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
from .utils.types import CueState


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
        dt: float = stream_out_spec.get("dt", 0.01)

        self._cueing_command_queue: Queue[dict] = Queue()
        self._cueing_status_queue: Queue[tuple[float, int]] = Queue()

        # FSM state for cueing control.
        self._cue_state = CueState.IDLE
        self._threshold_high: float = buds.get("threshold_high", 0.7)
        self._threshold_low: float = buds.get("threshold_low", 0.3)

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

        self._handler_proc = Process(
            target=launch_handler,
            args=(BudsHandler,),
            kwargs={
                "buds": buds,
                "cueing_command_queue": self._cueing_command_queue,
                "cueing_status_queue": self._cueing_status_queue,
                **hermes_kwargs,
                "dt": dt,
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
        """Receive FoG detection results from upstream AI node."""
        data = msg.get("data", {})
        fog_prob = data.get("fog_probability", 0.0)

        if self._cue_state == CueState.IDLE:
            if fog_prob >= self._threshold_high:
                self._cue_state = CueState.CUEING
                self._cueing_command_queue.put({
                    "action": "start",
                    "tone_id": 0,
                    "volume": 80,
                })
        elif self._cue_state == CueState.CUEING:
            if fog_prob < self._threshold_low:
                self._cue_state = CueState.IDLE
                self._cueing_command_queue.put({"action": "stop"})

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
                "count": len(status_data),
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
