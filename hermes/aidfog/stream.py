from hermes.base.stream import Stream


class BudsStream(Stream):
    def __init__(
        self,
        niclas: dict,
        motors: dict,
        pmu: dict,
        **_,
    ) -> None:
        super().__init__()

        # TODO: use details to add HDF5 metadata.
        sample_rate_niclas = niclas["sampling_rate_hz"]
        sample_rate_motors = motors["sampling_rate_hz"]
        sample_rate_pmu = pmu["sampling_rate_hz"]

        # Nicla Sense ME data.
        nicla_specs: dict = niclas["device_mapping"]
        for nicla_name in nicla_specs.keys():
            self.add_stream(
                device_name=f"nicla_{nicla_name}",
                stream_name="toa_s",
                data_type="float64",
                sample_size=(1,),
                sampling_rate_hz=sample_rate_niclas,
                is_measure_rate_hz=True,
            )
            self.add_stream(
                device_name=f"nicla_{nicla_name}",
                stream_name="timestamp",
                data_type="uint32",
                sample_size=(1,),
                sampling_rate_hz=sample_rate_niclas,
            )
            self.add_stream(
                device_name=f"nicla_{nicla_name}",
                stream_name="sequence_id",
                data_type="uint32",
                sample_size=(1,),
                sampling_rate_hz=sample_rate_niclas,
            )
            self.add_stream(
                device_name=f"nicla_{nicla_name}",
                stream_name="count",
                data_type="uint16",
                sample_size=(1,),
                sampling_rate_hz=sample_rate_niclas,
            )
            if niclas["is_acc"]:
                self.add_stream(
                    device_name=f"nicla_{nicla_name}",
                    stream_name="acceleration",
                    data_type="int16",
                    sample_size=(3,),
                    sampling_rate_hz=sample_rate_niclas,
                )
            if niclas["is_gyr"]:
                self.add_stream(
                    device_name=f"nicla_{nicla_name}",
                    stream_name="gyroscope",
                    data_type="int16",
                    sample_size=(3,),
                    sampling_rate_hz=sample_rate_niclas,
                )
            if niclas["is_mag"]:
                self.add_stream(
                    device_name=f"nicla_{nicla_name}",
                    stream_name="magnetometer",
                    data_type="int16",
                    sample_size=(3,),
                    sampling_rate_hz=sample_rate_niclas,
                )
            if niclas["is_euler"]:
                self.add_stream(
                    device_name=f"nicla_{nicla_name}",
                    stream_name="euler",
                    data_type="float32",
                    sample_size=(3,),
                    sampling_rate_hz=sample_rate_niclas,
                )
            if niclas["is_quat"]:
                self.add_stream(
                    device_name=f"nicla_{nicla_name}",
                    stream_name="quaternion",
                    data_type="float32",
                    sample_size=(4,),
                    sampling_rate_hz=sample_rate_niclas,
                )
            if niclas["is_temp"]:
                self.add_stream(
                    device_name=f"nicla_{nicla_name}",
                    stream_name="temperature",
                    data_type="float32",
                    sample_size=(1,),
                    sampling_rate_hz=sample_rate_niclas,
                )
            if niclas["is_baro"]:
                self.add_stream(
                    device_name=f"nicla_{nicla_name}",
                    stream_name="pressure",
                    data_type="float32",
                    sample_size=(1,),
                    sampling_rate_hz=sample_rate_niclas,
                )
            if niclas["is_hum"]:
                self.add_stream(
                    device_name=f"nicla_{nicla_name}",
                    stream_name="humidity",
                    data_type="float32",
                    sample_size=(1,),
                    sampling_rate_hz=sample_rate_niclas,
                )

        # Motor data.
        motor_specs: dict = motors["device_mapping"]
        for motor_name in motor_specs.keys():
            self.add_stream(
                device_name=f"motor_{motor_name}",
                stream_name="timestamp",
                data_type="float64",
                sample_size=[1],
                sampling_rate_hz=sample_rate_motors,
                is_measure_rate_hz=True,
            )
            self.add_stream(
                device_name=f"motor_{motor_name}",
                stream_name="position",
                data_type="float32",
                sample_size=[1],
                sampling_rate_hz=sample_rate_motors,
            )
            self.add_stream(
                device_name=f"motor_{motor_name}",
                stream_name="velocity",
                data_type="float32",
                sample_size=[1],
                sampling_rate_hz=sample_rate_motors,
            )
            self.add_stream(
                device_name=f"motor_{motor_name}",
                stream_name="current",
                data_type="float32",
                sample_size=[1],
                sampling_rate_hz=sample_rate_motors,
            )
            self.add_stream(
                device_name=f"motor_{motor_name}",
                stream_name="temperature",
                data_type="int8",
                sample_size=[1],
                sampling_rate_hz=sample_rate_motors,
            )
            self.add_stream(
                device_name=f"motor_{motor_name}",
                stream_name="error",
                data_type="uint8",
                sample_size=[1],
                sampling_rate_hz=sample_rate_motors,
            )
            self.add_stream(
                device_name=f"motor_{motor_name}",
                stream_name="count",
                data_type="uint16",
                sample_size=[1],
                sampling_rate_hz=sample_rate_motors,
            )

        # Motor commands.
        for motor_name in motor_specs.keys():
            self.add_stream(
                device_name=f"command_{motor_name}",
                stream_name="timestamp",
                data_type="float64",
                sample_size=[1],
            )
            self.add_stream(
                device_name=f"command_{motor_name}",
                stream_name="control_mode",
                data_type=f"uint8",
                sample_size=[1],
            )
            self.add_stream(
                device_name=f"command_{motor_name}",
                stream_name="data",
                data_type=f"S{8}",  # CubeMars motors driver receives up to 8 bytes payloads for control
                sample_size=[1],
            )
            self.add_stream(
                device_name=f"command_{motor_name}",
                stream_name="count",
                data_type="uint16",
                sample_size=[1],
            )

        # Locomotion mode transitions, w/ reference to upstream switch command sequence id.
        self.add_stream(
            device_name="mode",
            stream_name="timestamp",
            data_type="float64",
            sample_size=[1],
        )
        self.add_stream(
            device_name="mode",
            stream_name="mode",
            data_type="uint8",
            sample_size=[1],
        )
        self.add_stream(
            device_name="mode",
            stream_name="sequence_id",
            data_type="uint32",
            sample_size=[1],
        )
        self.add_stream(
            device_name="mode",
            stream_name="count",
            data_type="uint16",
            sample_size=[1],
        )

        # Power monitor.
        self.add_stream(
            device_name="power_monitor",
            stream_name="timestamp",
            data_type="float64",
            sample_size=[1],
            sampling_rate_hz=sample_rate_pmu,
        )
        self.add_stream(
            device_name="power_monitor",
            stream_name="temperature",
            data_type="float16",
            sample_size=[1],
            sampling_rate_hz=sample_rate_pmu,
        )
        self.add_stream(
            device_name="power_monitor",
            stream_name="voltage",
            data_type="float16",
            sample_size=[1],
            sampling_rate_hz=sample_rate_pmu,
        )
        self.add_stream(
            device_name="power_monitor",
            stream_name="current",
            data_type="float16",
            sample_size=[1],
            sampling_rate_hz=sample_rate_pmu,
        )
        self.add_stream(
            device_name="power_monitor",
            stream_name="power",
            data_type="float32",
            sample_size=[1],
            sampling_rate_hz=sample_rate_pmu,
        )
        self.add_stream(
            device_name="power_monitor",
            stream_name="count",
            data_type="uint16",
            sample_size=[1],
            sampling_rate_hz=sample_rate_pmu,
        )

        # State machine transitions (intra-mode).
        self.add_stream(
            device_name="state",
            stream_name="timestamp",
            data_type="float64",
            sample_size=[1],
        )
        self.add_stream(
            device_name="state",
            stream_name="state",
            data_type="uint8",
            sample_size=[1],
        )
        self.add_stream(
            device_name="state",
            stream_name="count",
            data_type="uint16",
            sample_size=[1],
        )

        # State machine transitions (intra-mode).
        self.add_stream(
            device_name="phase",
            stream_name="timestamp",
            data_type="float64",
            sample_size=[1],
        )
        self.add_stream(
            device_name="phase",
            stream_name="phase",
            data_type="float32",
            sample_size=[1],
        )
        self.add_stream(
            device_name="phase",
            stream_name="count",
            data_type="uint16",
            sample_size=[1],
        )

    def get_fps(self) -> dict[str, float | None]:
        return {
            "nicla": super()._get_fps("nicla", "toa_s"),
            "motor": super()._get_fps("motor", "timestamp"),
        }
