"""
HERMES Stream definition for IMU data replayed from a recorded dataset.

Declares the same sub-streams that real IMU sensors would publish,
so TorchPipeline._process_data() can consume the data without changes.
"""

from collections import OrderedDict

from hermes.base.stream import Stream


class ImuReplayStream(Stream):
    """Stream for replayed IMU data matching the dots-imu device format."""

    def __init__(
        self,
        num_sensors: int = 5,
        sampling_rate_hz: int = 60,
        **_,
    ) -> None:
        super().__init__()

        self.add_stream(
            device_name="dots-imu",
            stream_name="acceleration",
            data_type="float64",
            sample_size=(num_sensors, 3),
            sampling_rate_hz=sampling_rate_hz,
            is_measure_rate_hz=True,
            data_notes=OrderedDict([
                ("Description", "3-axis accelerometer per sensor (replayed from dataset)"),
                ("Units", "m/s^2"),
            ]),
        )

        self.add_stream(
            device_name="dots-imu",
            stream_name="gyroscope",
            data_type="float64",
            sample_size=(num_sensors, 3),
            sampling_rate_hz=sampling_rate_hz,
            data_notes=OrderedDict([
                ("Description", "3-axis gyroscope per sensor (replayed from dataset)"),
                ("Units", "rad/s"),
            ]),
        )

        self.add_stream(
            device_name="dots-imu",
            stream_name="toa_s",
            data_type="float64",
            sample_size=(num_sensors,),
            sampling_rate_hz=sampling_rate_hz,
            data_notes=OrderedDict([
                ("Description", "Synthetic time-of-arrival per sensor"),
                ("Units", "seconds"),
            ]),
        )

        self.add_stream(
            device_name="dots-imu",
            stream_name="fog_label",
            data_type="uint8",
            sample_size=(1,),
            sampling_rate_hz=sampling_rate_hz,
            data_notes=OrderedDict([
                ("Description", "Ground-truth FoG label from dataset annotation"),
                ("Values", "0=no FoG, 1=FoG"),
            ]),
        )

    def get_fps(self) -> dict[str, float | None]:
        return {"dots-imu": super()._get_fps("dots-imu", "acceleration")}
