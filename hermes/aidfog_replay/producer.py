"""
HERMES Producer for replaying recorded IMU datasets.

Loads .npy IMU files from the KU Leuven FoG dataset, resamples to the
target rate, and streams them frame-by-frame into the HERMES pipeline
as if they were live sensor data. TorchPipeline._process_data() can
consume this without any changes.

Data loading logic follows Alex's load_trials_by_subject():
- Sensor reindexing from raw layout to 5-sensor order
- Resampling from 64 Hz to 60 Hz via resample_poly
- NO high-pass filtering (TorchPipeline.normalize() handles that)
- FoG labels loaded, resampled, and published alongside IMU data
"""

import os
import time
import logging

import numpy as np
from scipy.signal import resample_poly

from hermes.utils.time_utils import get_time
from hermes.utils.zmq_utils import PORT_BACKEND, PORT_KILL, PORT_SYNC_HOST
from hermes.utils.types import LoggingSpec

from hermes.aidfog_replay.stream import ImuReplayStream
from hermes.base.nodes.producer import Producer

logger = logging.getLogger(__name__)


class ImuReplayProducer(Producer):
    """Replays recorded IMU data from .npy files as a HERMES Producer."""

    @classmethod
    def _log_source_tag(cls) -> str:
        return "aidfog_replay"

    def __init__(
        self,
        host_ip: str,
        logging_spec: LoggingSpec,
        imu_root: str = "./data/imu",
        annot_root: str = "./data/annot",
        sampling_rate_hz: int = 60,
        original_rate_hz: int = 64,
        sensor_indices: list = None,
        loop: bool = True,
        port_pub: str = PORT_BACKEND,
        port_sync: str = PORT_SYNC_HOST,
        port_killsig: str = PORT_KILL,
        **_,
    ):
        if sensor_indices is None:
            sensor_indices = [0, 5, 6, 2, 3]

        self._imu_root = imu_root
        self._annot_root = annot_root
        self._target_hz = sampling_rate_hz
        self._original_hz = original_rate_hz
        self._sensor_indices = sensor_indices
        self._loop = loop
        self._period = 1.0 / sampling_rate_hz
        self._tag: str = "%s.data" % self._log_source_tag()
        self._next_period: float = 0.0

        self._frames: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._frame_idx: int = 0
        self._total_frames: int = 0

        self._load_dataset()

        num_sensors = len(sensor_indices)

        stream_out_spec = {
            "num_sensors": num_sensors,
            "sampling_rate_hz": sampling_rate_hz,
        }

        super().__init__(
            host_ip=host_ip,
            stream_out_spec=stream_out_spec,
            logging_spec=logging_spec,
            port_pub=port_pub,
            port_sync=port_sync,
            port_killsig=port_killsig,
        )

    def _load_dataset(self) -> None:
        """Load all TUG trials from the dataset, resample, and flatten into frames."""
        skip_list = [f"{i:03d}" for i in range(31, 41)] + [
            "006", "008", "020", "021", "022", "024",
        ]

        all_imu = []
        all_labels = []
        trial_count = 0

        for subj in sorted(os.listdir(self._annot_root)):
            subj_annot_path = os.path.join(self._annot_root, subj)
            if subj in skip_list or not os.path.isdir(subj_annot_path):
                continue

            for fn in sorted(os.listdir(subj_annot_path)):
                if "TUG" not in fn:
                    continue

                annot_file = os.path.join(subj_annot_path, fn)
                imu_file = os.path.join(self._imu_root, subj, fn)

                if not os.path.exists(imu_file):
                    logger.warning("IMU file not found: %s", imu_file)
                    continue
                if not os.path.exists(annot_file):
                    logger.warning("Annotation file not found: %s", annot_file)
                    continue

                # Load IMU data: raw shape (N_sensors_raw, 6, n_samples)
                data = np.load(imu_file, allow_pickle=True)
                data = data[self._sensor_indices, :, :]  # (5, 6, n_samples)

                # Resample from original_hz to target_hz (no filtering — TorchPipeline does it)
                if self._original_hz != self._target_hz:
                    data = resample_poly(data, self._target_hz, self._original_hz, axis=2)

                # Load and resample labels
                lab = np.load(annot_file, allow_pickle=True)
                if lab.ndim > 1:
                    lab = lab.flatten()
                lab[lab == 2] = 1
                lab[lab != 1] = 0

                if self._original_hz != self._target_hz:
                    lab_ds = resample_poly(
                        lab[None, :].astype(float),
                        self._target_hz,
                        self._original_hz,
                        axis=1,
                    )
                    lab = (lab_ds >= 0.5).astype(np.int64).ravel()

                # Trim to match lengths after resampling
                n_frames = min(data.shape[2], len(lab))
                all_imu.append(data[:, :, :n_frames])
                all_labels.append(lab[:n_frames])
                trial_count += 1

        if not all_imu:
            logger.error("No trials loaded from %s / %s", self._imu_root, self._annot_root)
            self._total_frames = 0
            return

        # Concatenate all trials along the time axis
        self._imu_data = np.concatenate(all_imu, axis=2)  # (5, 6, total_frames)
        self._label_data = np.concatenate(all_labels)       # (total_frames,)
        self._total_frames = self._imu_data.shape[2]

        logger.info(
            "Loaded %d trials, %d total frames (%.1f seconds at %d Hz)",
            trial_count,
            self._total_frames,
            self._total_frames / self._target_hz,
            self._target_hz,
        )

    @classmethod
    def create_stream(cls, stream_spec: dict) -> ImuReplayStream:
        return ImuReplayStream(**stream_spec)

    def _ping_device(self) -> None:
        return None

    def _connect(self) -> bool:
        return self._total_frames > 0

    def _keep_samples(self) -> None:
        self._next_period = get_time() + self._period

    def _process_data(self) -> None:
        if not self._is_continue_capture:
            self._send_end_packet()
            return

        if self._frame_idx >= self._total_frames:
            if self._loop:
                self._frame_idx = 0
                logger.info("Dataset replay looping back to start")
            else:
                self._send_end_packet()
                return

        # Rate limiting — match DummyProducer's timing pattern
        process_time_s = get_time()
        time_to_wait = self._next_period - process_time_s
        if time_to_wait > 0:
            time.sleep(time_to_wait * 0.9)
            while (process_time_s := get_time()) < self._next_period:
                pass

        t = self._frame_idx
        frame = self._imu_data[:, :, t]  # (5, 6)
        label = int(self._label_data[t])

        # Split into acceleration and gyroscope, matching TorchPipeline expectations
        acc = frame[:, :3]  # (5, 3)
        gyr = frame[:, 3:]  # (5, 3)

        # Synthetic timestamps — all sensors "arrive" at the same time
        toa_s = np.full(acc.shape[0], t / self._target_hz, dtype=np.float64)

        self._publish(
            self._tag,
            process_time_s=process_time_s,
            data={
                "dots-imu": {
                    "acceleration": acc,
                    "gyroscope": gyr,
                    "toa_s": toa_s,
                    "fog_label": label,
                }
            },
        )

        self._frame_idx += 1
        self._next_period += self._period

    def _stop_new_data(self):
        pass

    def _cleanup(self) -> None:
        super()._cleanup()
