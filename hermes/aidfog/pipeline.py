from multiprocessing import Process, Queue, Event, Value
from multiprocessing.sharedctypes import Synchronized
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
        niclas: dict = stream_out_spec["niclas"]
        motors: dict = stream_out_spec["motors"]
        pmu: dict = stream_out_spec["pmu"]
        dt: float = stream_out_spec["dt"]

        self._nicla_mapping: dict[str, dict] = niclas["device_mapping"]
        self._motor_mapping: dict[str, dict] = motors["device_mapping"]
        self._nicla_payload_mode = NiclaPayloadMode(
            is_acc=niclas["is_acc"],
            is_gyr=niclas["is_gyr"],
            is_mag=niclas["is_mag"],
            is_euler=niclas["is_euler"],
            is_quat=niclas["is_quat"],
            is_temp=niclas["is_temp"],
            is_baro=niclas["is_baro"],
            is_hum=niclas["is_hum"],
        )

        # Keyboard stdin input queue.
        self._input_queue: Queue[tuple[float, str]] = _["input_queue"]

        # Onboard data.
        self._nicla_data_queue: Queue[tuple[str, float, NiclaData]] = Queue()

        # Shared controls for exo handler.
        self._next_mode: Synchronized[int] = Value("i")
        self._next_fatigue: Synchronized[float] = Value("f")
        self._next_mode_sequence_id: Synchronized[int] = Value("i")
        self._next_fatigue_sequence_id: Synchronized[int] = Value("i")

        # Synchronization primitives between background exo handler and foreground HERMES procs.
        self._is_ready_event = Event()
        self._is_keep_data_event = Event()
        self._is_stop_new_data_event = Event()
        self._is_cleanup_event = Event()
        self._is_finished_event = Event()

        # Outgoing onboard data.
        telemetry_kwargs = {
            "nicla_data_queue": self._nicla_data_queue,
        }

        # Incoming AI controls.
        ai_kwargs = {
            "next_mode": self._next_mode,
            "next_mode_sequence_id": self._next_mode_sequence_id,
        }

        hermes_kwargs = {
            "ref_time_s": logging_spec.ref_time_s,
            "is_ready_event": self._is_ready_event,
            "is_keep_data_event": self._is_keep_data_event,
            "is_stop_new_data_event": self._is_stop_new_data_event,
            "is_cleanup_event": self._is_cleanup_event,
            "is_finished_event": self._is_finished_event,
            "input_queue": self._input_queue,
        }

        self._handler_proc = Process(
            target=launch_handler,
            args=(BudsHandler,),
            kwargs={
                "niclas": niclas,
                **telemetry_kwargs,
                **ai_kwargs,
                **hermes_kwargs,
                "dt": dt,
            },
        )
        self._handler_proc.start()
        self._is_ready_event.wait()

        stream_out_spec = {
            "niclas": niclas,
        }

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
        if "intent" in msg["data"]:
            # Passes to the top-level exo module the next state to choose internally when to switch to.
            # NOTE: AI component will provide `int` matching one of the ModeStateEnum values.
            self._next_mode.value = msg["data"]["intent"]["mode"]
            self._next_mode_sequence_id.value = msg["data"]["intent"]["sequence_id"]
        elif "fatigue" in msg["data"]:
            # Passes to the top-level exo module the next fatigue percentage to choose internally to scale torques.
            # NOTE: AI component will provide `float` in range [0, 1].
            self._next_fatigue.value = msg["data"]["fatigue"]["level"]
            self._next_fatigue_sequence_id.value = msg["data"]["fatigue"]["sequence_id"]

    def _generate_data(self) -> None:
        # Pass internally generated data to the middleware.
        process_time_s = get_time()
        tag: str = "%s.data" % self._log_source_tag()
        output = {}

        # Nicla data.
        nicla_data: dict[str, list[NiclaData]] = {
            name: [] for name in self._nicla_mapping.keys()
        }
        nicla_toa: dict[str, list[float]] = {
            name: [] for name in self._nicla_mapping.keys()
        }
        while not self._nicla_data_queue.empty():
            nicla_name, toa_s, nicla_sample = self._nicla_data_queue.get_nowait()
            nicla_toa[nicla_name].append(toa_s)
            nicla_data[nicla_name].append(nicla_sample)
        for nicla_name, data in nicla_data.items():
            if data:
                output[f"nicla_{nicla_name}"] = {
                    "toa_s": np.array(
                        [nicla_toa[nicla_name]], dtype=np.float64
                    ).transpose((1, 0)),
                    "sequence_id": np.array(
                        [list(map(lambda n: n.sequence_id, data))], dtype=np.uint32
                    ).transpose((1, 0)),
                    "timestamp": np.array(
                        [list(map(lambda n: n.timestamp, data))], dtype=np.uint32
                    ).transpose((1, 0)),
                    "count": len(data),
                }
                for (
                    data_name,
                    data_getter,
                ) in self._nicla_payload_mode.get_data_getters().items():
                    output[f"nicla_{nicla_name}"][data_name] = data_getter(data)

        # Motor command data.
        motor_command_data: dict[str, tuple[str, list[MotorCommand]]] = {
            motor_spec["can_id"]: (motor_name, [])
            for motor_name, motor_spec in self._motor_mapping.items()
        }
        while not self._motor_command_queue.empty():
            motor_command = self._motor_command_queue.get_nowait()
            motor_command_data[motor_command.motor_id][1].append(motor_command)
        for motor_name, data in motor_command_data.values():
            if data:
                output[f"command_{motor_name}"] = {
                    "timestamp": np.array(
                        [list(map(lambda m: m.timestamp, data))], dtype=np.float64
                    ).transpose((1, 0)),
                    "data": np.array(
                        [list(map(lambda m: bytes(m.data), data))]
                    ).transpose((1, 0)),
                    "control_mode": np.array(
                        [list(map(lambda m: m.control_mode, data))], dtype=np.uint8
                    ).transpose((1, 0)),
                    "count": len(data),
                }

        if output:
            self._publish(tag, process_time_s=process_time_s, data=output)
        elif (
            self._is_finished_event.is_set()
            and self._motor_data_queue.empty()
            and self._nicla_data_queue.empty()
            and self._mode_changed_queue.empty()
            and self._state_changed_queue.empty()
            and self._phase_estimate_queue.empty()
        ):
            self._notify_no_more_data_out()

    def _stop_new_data(self):
        # Trigger exo handler to stop adding data to the timestamp alignment buffer for the AI model to consumer.
        self._is_cleanup_event.set()
        self._is_stop_new_data_event.set()

    def _cleanup(self) -> None:
        self._handler_proc.join()
        super()._cleanup()
