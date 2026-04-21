"""
HERMES Pipeline for PineBuds Pro audio cueing.

Receives FoG detection results from the upstream AI node and translates
them into cueing commands for the BudsHandler running in a background process.
"""

import sys
from multiprocessing import Process, Queue, Event
import numpy as np


def _trace(msg: str) -> None:
    """Unbuffered trace to stderr (stdout from Windows subprocesses is buffered)."""
    sys.stderr.write("[buds_pipeline] %s\n" % msg)
    sys.stderr.flush()

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
        dt: float = buds.get("dt", 0.01)

        # Resolve address from device_mapping (follows the Nicla pattern).
        # Use the first entry in device_mapping if present; fall back to
        # an explicit 'address' key or None (triggers name-based BLE scan).
        device_mapping: dict = buds.get("device_mapping", {})
        address = list(device_mapping.values())[0] if device_mapping else buds.get("address", None)
        buds = {**buds, "address": address}

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
        _trace("entered BudsPipeline.__init__")
        _trace("launching BudsHandler subprocess...")
        self._handler_proc.start()
        _trace("waiting for is_ready_event (BLE connect)...")
        self._is_ready_event.wait()
        _trace("is_ready_event set, continuing init")

        stream_out_spec = {"buds": buds}

        import time as _t
        import threading as _threading
        from collections import OrderedDict as _OrderedDict
        from hermes.base.nodes.node import Node as _Node
        from hermes.base.storage.storage import Storage as _Storage
        from hermes.utils.di_utils import search_module_class as _search

        _trace("pre: importing aidfog_ai to warm torch...")
        _t0 = _t.time()
        import hermes.aidfog_ai  # noqa
        _trace(f"aidfog_ai imported in {_t.time()-_t0:.1f}s")

        # Replicate Pipeline.__init__ step by step with traces.
        _trace("step 1: calling Node.__init__...")
        _t0 = _t.time()
        _Node.__init__(
            self,
            host_ip=host_ip,
            port_sync=port_sync,
            port_killsig=port_killsig,
            ref_time=logging_spec.ref_time_s,
        )
        _trace(f"step 1 done in {_t.time()-_t0:.1f}s")

        _trace("step 2: setting instance attrs...")
        self._port_pub = port_pub
        self._port_sub = port_sub
        self._is_async_generate = True
        self._is_more_data_in = True
        self._is_more_data_out = True
        self._publish_fn = lambda tag, **kwargs: None
        _trace("step 2 done")

        _trace("step 3: creating out Stream (BudsStream)...")
        _t0 = _t.time()
        self._out_stream = self.create_stream(stream_out_spec)
        _trace(f"step 3 done in {_t.time()-_t0:.1f}s")

        _trace("step 4: setting up in_streams dicts and poll fns...")
        self._in_streams = _OrderedDict()
        self._poll_data_fn = self._poll_data_packets
        self._on_poll_fn = self._on_poll_in_out if self._is_async_generate else self._on_poll_in_only
        self._is_producer_ended = _OrderedDict()
        _trace("step 4 done")

        _trace(f"step 5: creating {len(stream_in_specs)} in-stream(s)...")
        for i, stream_spec in enumerate(stream_in_specs):
            module_name = stream_spec["package"]
            class_name = stream_spec["class"]
            specs = stream_spec["settings"]
            _trace(f"  5.{i}a: search_module_class({module_name}, {class_name})...")
            _t0 = _t.time()
            class_type = _search(module_name, class_name)
            _trace(f"  5.{i}a done in {_t.time()-_t0:.1f}s")
            _trace(f"  5.{i}b: {class_name}.create_stream(...)...")
            _t0 = _t.time()
            try:
                class_object = class_type.create_stream(specs)
            except Exception as e:
                # Windows multiprocessing + spawn swallows tracebacks from child
                # subprocesses. Log explicitly so a missing YAML setting is visible.
                _trace(f"  5.{i}b FAILED: {type(e).__name__}: {e}")
                raise
            _trace(f"  5.{i}b done in {_t.time()-_t0:.1f}s")
            self._in_streams.setdefault(class_type._log_source_tag(), class_object)
            self._is_producer_ended.setdefault(class_type._log_source_tag(), False)
        _trace("step 5 done")

        _trace("step 6: creating Storage...")
        _t0 = _t.time()
        self._storage = _Storage(self._log_source_tag(), logging_spec)
        _trace(f"step 6 done in {_t.time()-_t0:.1f}s")

        _trace("step 7: starting storage thread...")
        _t0 = _t.time()
        self._storage_thread = _threading.Thread(
            target=self._storage,
            args=(
                _OrderedDict([
                    (self._log_source_tag(), self._out_stream),
                    *self._in_streams.items(),
                ]),
            ),
        )
        self._storage_thread.start()
        _trace(f"step 7 done in {_t.time()-_t0:.1f}s; pipeline ctor finished")

    @classmethod
    def create_stream(cls, stream_spec: dict) -> BudsStream:
        return BudsStream(**stream_spec)

    def _keep_samples(self) -> None:
        self._is_keep_data_event.set()

    def _process_data(self, topic: str, msg: dict) -> None:
        """Receive FoG detection results from upstream AI node."""
        data = msg.get("data", {})
        pytorch_data = data.get("pytorch-worker", {})
        logits = pytorch_data.get("logits", [0.0, 0.0])
        fog_prob = float(logits[1])

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
