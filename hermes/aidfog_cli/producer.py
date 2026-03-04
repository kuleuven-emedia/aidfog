from collections import defaultdict
from functools import partial
from queue import Queue
import queue
from typing import Callable

from hermes.utils.time_utils import get_time
from hermes.utils.zmq_utils import PORT_BACKEND, PORT_KILL, PORT_SYNC_HOST
from hermes.utils.types import LoggingSpec

from hermes.base.nodes.producer import Producer

from ..aidfog.utils.types import ModeEnum
from .stream import AidfogCliStream


class AidfogCliProducer(Producer):
    @classmethod
    def _log_source_tag(cls) -> str:
        return "aidfog_cli"

    def __init__(
        self,
        host_ip: str,
        logging_spec: LoggingSpec,
        port_pub: str = PORT_BACKEND,
        port_sync: str = PORT_SYNC_HOST,
        port_killsig: str = PORT_KILL,
        **_,
    ):
        self._input_queue: Queue[tuple[float, str]] = _["input_queue"]
        self._intent_sequence_id = 0
        self._fatigue_sequence_id = 0

        def intent_callback(
            mode: ModeEnum, toa_s: float, process_time_s: float, sequence_id: int
        ) -> None:
            print(f"User selected transition to: {mode.value.text}", flush=True)
            self._publish(
                "%s.data" % self._log_source_tag(),
                process_time_s=process_time_s,
                data={
                    "intent": {
                        "toa_s": toa_s,
                        "mode": mode.value.id,
                        "sequence_id": sequence_id,
                    }
                },
            )

        def fatigue_callback(
            fatigue: float, toa_s: float, process_time_s: float, sequence_id: int
        ) -> None:
            print(f"User set assistance to: {fatigue}", flush=True)
            self._publish(
                "%s.data" % self._log_source_tag(),
                process_time_s=process_time_s,
                data={
                    "fatigue": {
                        "toa_s": toa_s,
                        "level": fatigue,
                        "sequence_id": sequence_id,
                    }
                },
            )

        self._intent_keyboard_mapper: defaultdict[str, Callable[[float, float, int], None]] = (
            defaultdict(lambda: (lambda toa_s, process_time_s, sequence_id: None))
        )
        for mode in ModeEnum:
            self._intent_keyboard_mapper[str(mode.value.id)] = partial(intent_callback, mode)

        self._fatigue_keyboard_mapper = fatigue_callback

        stream_out_spec = {}

        super().__init__(
            host_ip=host_ip,
            stream_out_spec=stream_out_spec,
            logging_spec=logging_spec,
            port_pub=port_pub,
            port_sync=port_sync,
            port_killsig=port_killsig,
        )

    @classmethod
    def create_stream(cls, stream_spec: dict) -> AidfogCliStream:
        return AidfogCliStream(**stream_spec)

    def _ping_device(self) -> None:
        return None

    def _connect(self) -> bool:
        return True

    def _keep_samples(self) -> None:
        pass

    def _process_data(self) -> None:
        if self._is_continue_capture:
            try:
                toa_s, user_input = self._input_queue.get(timeout=5)
                process_time_s = get_time()
                if user_input[0] == "%":
                    user_lvl: int = int(user_input[1:]) 
                    fatigue: float = user_lvl if 0 <= user_lvl <= 100 else (100 if user_lvl > 100 else 0)
                    self._fatigue_keyboard_mapper(
                        fatigue, toa_s, process_time_s, self._fatigue_sequence_id
                    )
                    self._fatigue_sequence_id += 1
                else:
                    self._intent_keyboard_mapper[user_input](
                        toa_s, process_time_s, self._intent_sequence_id
                    )
                    self._intent_sequence_id += 1
            except queue.Empty:
                pass
            except Exception as e:
                print(e, flush=True)
                pass
        else:
            self._send_end_packet()

    def _stop_new_data(self):
        pass

    def _cleanup(self) -> None:
        super()._cleanup()
