"""HERMES AI inference pipeline — wires Alex's TCNModel + HysteresisFilter
into the live closed-loop system.

Inference contract (matches hermes/aidfog_ai/model_alex/streaming.py):
- Input         : (T, 30) at 60 Hz; channel order per IMU [acc_x, acc_y, acc_z,
                  gyr_x, gyr_y, gyr_z]; IMU order [Pelvic, L_tibia, L_talus,
                  R_tibia, R_talus].
- Normalisation : rolling 180-sample mean subtraction (zero-centring only,
                  no division). Handled here; do not pre-normalise upstream.
- Model         : TCNModel (sigmoid output) — per-subject LOSOCV weights at
                  hermes/aidfog_ai/model_alex/weights/subject{id}.pt.
- Postproc      : HysteresisFilter (enter/exit thresholds in samples).

Published payload schema (per frame):
- prob_smoothed             : float, sigmoid-bounded FoG probability
- binary                    : int8, post-hysteresis FoG state (0 or 1)
- inference_latency_s       : float
- delay_since_first_sensor_s: float
- delay_since_snapshot_ready_s: float

Subject ID is read from the `subject_id` setting in buds.yml. The legacy
`model_path` setting is accepted but ignored with a deprecation warning.
"""

import logging
import os
import warnings

import numpy as np
import torch

from hermes.base.nodes.pipeline import Pipeline
from hermes.utils.time_utils import get_time
from hermes.utils.zmq_utils import (
    PORT_BACKEND, PORT_FRONTEND, PORT_SYNC_HOST, PORT_KILL,
)

from hermes.aidfog_ai.stream import TorchStream
from hermes.aidfog_ai.model_alex.tcn_model import TCNModel
from hermes.aidfog_ai.model_alex.hysteresis_filter import HysteresisFilter

logger = logging.getLogger(__name__)

WINDOW_LEN = 180   # 3 s × 60 Hz — must match model_alex/streaming.py
N_FEATURES = 30    # 5 IMUs × 6 axes


class TorchPipeline(Pipeline):
    @classmethod
    def _log_source_tag(cls) -> str:
        return "aidfog_ai"

    def __init__(
        self,
        host_ip: str,
        stream_in_specs: list[dict],
        input_size: tuple[int, int],
        output_classes: list[str],
        sampling_rate_hz: int,
        logging_spec: dict,
        subject_id: str | None = None,
        enter_thresh: int = 20,
        exit_thresh: int = 5,
        thresh: float = 0.5,
        weights_dir: str | None = None,
        model_path: str | None = None,
        port_pub: str = PORT_BACKEND,
        port_sub: str = PORT_FRONTEND,
        port_sync: str = PORT_SYNC_HOST,
        port_killsig: str = PORT_KILL,
        **_,
    ):
        if model_path is not None:
            warnings.warn(
                "TorchPipeline: 'model_path' is deprecated; loading is now driven "
                "by 'subject_id' (LOSOCV weights at model_alex/weights/subject{id}.pt). "
                "The 'model_path' setting is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        if subject_id is None:
            raise ValueError(
                "TorchPipeline requires 'subject_id' (e.g. '001') to select the "
                "matching LOSOCV weights file. Add `subject_id: \"<id>\"` under "
                "the aidfog_ai pipeline settings in your YAML."
            )

        if weights_dir is None:
            weights_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "model_alex", "weights",
            )
        weights_path = os.path.join(weights_dir, f"subject{subject_id}.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"No weights file at {weights_path}. Available subjects: "
                f"{sorted(f for f in os.listdir(weights_dir) if f.startswith('subject'))}"
            )

        n_features = input_size[0] * input_size[1]
        self._model = TCNModel(n_features=n_features)
        self._model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self._model.eval()
        self._model.tcn.reset_buffers()
        torch.set_grad_enabled(False)

        self._buffer = np.zeros((WINDOW_LEN, n_features), dtype=np.float32)
        self._hysteresis = HysteresisFilter(
            enter_thresh=enter_thresh, exit_thresh=exit_thresh,
        )
        self._thresh = float(thresh)

        self._n_sensors = input_size[0]
        self._n_axes = input_size[1]

        logger.info(
            "TorchPipeline ready — subject=%s, enter_thresh=%d, exit_thresh=%d, "
            "thresh=%.2f, weights=%s",
            subject_id, enter_thresh, exit_thresh, thresh, weights_path,
        )

        stream_out_spec = {
            "classes": output_classes,
            "sampling_rate_hz": sampling_rate_hz,
        }

        super().__init__(
            host_ip=host_ip,
            stream_out_spec=stream_out_spec,
            stream_in_specs=stream_in_specs,
            logging_spec=logging_spec,
            port_pub=port_pub,
            port_sub=port_sub,
            port_sync=port_sync,
            port_killsig=port_killsig,
        )

    @classmethod
    def create_stream(cls, stream_spec: dict) -> TorchStream:
        return TorchStream(**stream_spec)

    def _normalise_step(self, sample: np.ndarray) -> np.ndarray:
        """Zero-centre the new sample using the current buffer mean, then
        roll the buffer to include it. Matches `_normalise_streaming` in
        model_alex/streaming.py for a single sample."""
        centred = sample - self._buffer.mean(axis=0)
        self._buffer = np.roll(self._buffer, -1, axis=0)
        self._buffer[-1] = sample
        return centred

    def _generate_prediction(self, centred: np.ndarray) -> tuple[float, int]:
        # TCN expects (batch, channels, time). Single sample → time dim = 1.
        x = torch.from_numpy(centred).float().reshape(1, self._n_sensors * self._n_axes, 1)
        prob = float(self._model(x, inference=True).squeeze().item())
        raw_pred = int(prob >= self._thresh)
        binary = int(self._hysteresis.step(raw_pred))
        return prob, binary

    def _process_data(self, topic: str, msg: dict) -> None:
        acc = msg["data"]["dots-imu"]["acceleration"]    # (5, 3)
        gyr = msg["data"]["dots-imu"]["gyroscope"]       # (5, 3)
        toa_s = msg["data"]["dots-imu"]["toa_s"]

        # (5, 6) → (30,) with channel order [ax,ay,az,gx,gy,gz] per IMU.
        sample = np.concatenate((acc, gyr), axis=1).astype(np.float32).reshape(-1)
        if np.isnan(sample).any():
            # Drop NaN samples without advancing the buffer; matches old
            # behavior of skipping NaN-rows in the deque buffer.
            return

        centred = self._normalise_step(sample)
        start_time_s: float = get_time()
        prob, binary = self._generate_prediction(centred)
        end_time_s: float = get_time()

        data = {
            "prob_smoothed": prob,
            "binary": binary,
            "inference_latency_s": end_time_s - start_time_s,
            "delay_since_first_sensor_s": start_time_s - float(np.min(toa_s)),
            "delay_since_snapshot_ready_s": start_time_s - msg.get("process_time_s", start_time_s),
        }

        tag: str = "%s.data" % self._log_source_tag()
        self._publish(tag, process_time_s=end_time_s, data={"pytorch-worker": data})

    def _keep_samples(self) -> None:
        pass

    def _generate_data(self) -> None:
        pass

    def _stop_new_data(self):
        pass

    def _cleanup(self) -> None:
        super()._cleanup()
