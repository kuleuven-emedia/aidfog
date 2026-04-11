"""
HERMES Stream definition for microphone audio capture.

Declares two sub-streams under the "microphone" device tree:
- audio: raw PCM bytes, marked is_audio=True so Storage pipes to FFmpeg for MP3 encoding
- rms: float64 scalar per chunk, stored to HDF5 for post-hoc onset detection
"""

from collections import OrderedDict

from hermes.base.stream import Stream


class AudioStream(Stream):
    """Stream for continuous microphone audio capture."""

    def __init__(
        self,
        sampling_rate_hz: int = 44100,
        num_channels: int = 1,
        chunk_size: int = 1024,
        sample_format: str = "s16",
        **_,
    ) -> None:
        super().__init__()

        bytes_per_chunk = chunk_size * 2 * num_channels  # s16 = 2 bytes per sample

        # Raw PCM audio stream — Storage writes this to MP3 via its FFmpeg audio pipeline.
        self.add_stream(
            device_name="microphone",
            stream_name="audio",
            data_type=f"S{bytes_per_chunk}",
            sample_size=[1],
            sampling_rate_hz=sampling_rate_hz,
            is_audio=True,
            is_measure_rate_hz=True,
            data_notes=OrderedDict([
                ("Description", "Raw PCM audio from microphone"),
                ("Format", f"s16le, {num_channels}ch, {sampling_rate_hz}Hz"),
                ("Chunk size", f"{chunk_size} samples ({bytes_per_chunk} bytes)"),
            ]),
        )
        # Set audio-specific metadata that Storage._init_files_audio() expects.
        # These are not accepted as add_stream() parameters directly — same pattern
        # as how video streams set "format" and "color" after add_stream().
        self._streams_info["microphone"]["audio"]["num_channels"] = num_channels
        self._streams_info["microphone"]["audio"]["sample_format"] = sample_format

        # RMS level per chunk — stored to HDF5 for onset detection analysis.
        self.add_stream(
            device_name="microphone",
            stream_name="rms",
            data_type="float64",
            sample_size=(1,),
            sampling_rate_hz=sampling_rate_hz / chunk_size,
            data_notes=OrderedDict([
                ("Description", "RMS amplitude of each audio chunk"),
                ("Units", "PCM amplitude (int16 scale)"),
            ]),
        )

    def get_fps(self) -> dict[str, float | None]:
        return {"microphone": super()._get_fps("microphone", "audio")}
