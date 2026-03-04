from hermes.base.stream import Stream


class AidfogCliStream(Stream):
    def __init__(self, **_) -> None:
        super().__init__()

        self.add_stream(
            device_name="intent",
            stream_name="mode",
            data_type="int8",
            sample_size=[1],
        )
        self.add_stream(
            device_name="intent",
            stream_name="sequence_id",
            data_type="int32",
            sample_size=[1],
        )
        self.add_stream(
            device_name="intent",
            stream_name="toa_s",
            data_type="float64",
            sample_size=[1],
        )

        self.add_stream(
            device_name="fatigue",
            stream_name="level",
            data_type="float32",
            sample_size=[1],
        )
        self.add_stream(
            device_name="fatigue",
            stream_name="sequence_id",
            data_type="int32",
            sample_size=[1],
        )
        self.add_stream(
            device_name="fatigue",
            stream_name="toa_s",
            data_type="float64",
            sample_size=[1],
        )

    def get_fps(self) -> dict[str, float | None]:
        return {"intent": super()._get_fps("intent", "toa_s")}
