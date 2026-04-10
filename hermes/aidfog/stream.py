"""
HERMES Stream definition for PineBuds Pro cueing data.

Defines the HDF5 streams for logging cueing commands, status changes,
and latency measurements during recording sessions.
"""

from hermes.base.stream import Stream


class BudsStream(Stream):
    def __init__(self, buds: dict, **_) -> None:
        super().__init__()

        sample_rate_hz = buds.get("sampling_rate_hz", 100)

        self.add_stream(
            device_name="cueing",
            stream_name="toa_s",
            data_type="float64",
            sample_size=(1,),
            sampling_rate_hz=sample_rate_hz,
            is_measure_rate_hz=True,
        )

        self.add_stream(
            device_name="cueing",
            stream_name="status",
            data_type="uint8",
            sample_size=(1,),
            sampling_rate_hz=sample_rate_hz,
        )

    def get_fps(self) -> dict[str, float | None]:
        return {
            "cueing": super()._get_fps("cueing", "toa_s"),
        }
