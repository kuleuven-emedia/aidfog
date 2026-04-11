"""
HERMES Producer for microphone audio capture via FFmpeg.

Captures raw PCM audio from a system microphone using ffmpeg-python
and publishes it into the HERMES pipeline. The Storage layer handles
writing the audio to an MP3 file via its own FFmpeg subprocess.

Follows the DummyProducer pattern: same lifecycle hooks, same _process_data
loop, same _send_end_packet shutdown sequence.

Supported FFmpeg backends (set via ffmpeg_backend in buds.yml):
  - "dshow"       : Windows (DirectShow)
  - "avfoundation": macOS
  - "alsa"        : Linux (ALSA)
  - "pulse"       : Linux (PulseAudio)
"""

import logging

import numpy as np

try:
    import ffmpeg
except ImportError as e:
    print(
        e,
        "\nffmpeg-python not installed. Install with: pip install ffmpeg-python",
        flush=True,
    )

from hermes.utils.time_utils import get_time
from hermes.utils.zmq_utils import PORT_BACKEND, PORT_KILL, PORT_SYNC_HOST
from hermes.utils.types import LoggingSpec

from hermes.aidfog_audio.stream import AudioStream
from hermes.base.nodes.producer import Producer

logger = logging.getLogger(__name__)


class AudioProducer(Producer):
    """Captures audio from a microphone via FFmpeg and publishes PCM + RMS to HERMES."""

    @classmethod
    def _log_source_tag(cls) -> str:
        return "aidfog_audio"

    def __init__(
        self,
        host_ip: str,
        logging_spec: LoggingSpec,
        ffmpeg_backend: str = "dshow",
        device_name: str = "Microphone (Jabra Engage 50)",
        sampling_rate_hz: int = 44100,
        num_channels: int = 1,
        chunk_size: int = 1024,
        sample_format: str = "s16",
        port_pub: str = PORT_BACKEND,
        port_sync: str = PORT_SYNC_HOST,
        port_killsig: str = PORT_KILL,
        **_,
    ):
        self._ffmpeg_backend = ffmpeg_backend
        self._device_name = device_name
        self._audio_sampling_rate_hz = sampling_rate_hz
        self._num_channels = num_channels
        self._chunk_size = chunk_size
        self._sample_format = sample_format
        self._bytes_per_sample = 2  # s16le = 2 bytes per sample
        self._read_size = chunk_size * self._bytes_per_sample * num_channels
        self._tag: str = "%s.data" % self._log_source_tag()

        self._ffmpeg_process = None

        stream_out_spec = {
            "sampling_rate_hz": sampling_rate_hz,
            "num_channels": num_channels,
            "chunk_size": chunk_size,
            "sample_format": sample_format,
        }

        super().__init__(
            host_ip=host_ip,
            stream_out_spec=stream_out_spec,
            logging_spec=logging_spec,
            port_pub=port_pub,
            port_sync=port_sync,
            port_killsig=port_killsig,
        )

    @classmethod
    def create_stream(cls, stream_spec: dict) -> AudioStream:
        return AudioStream(**stream_spec)

    def _connect(self) -> bool:
        """Start the FFmpeg capture subprocess."""
        # Build the input device name per FFmpeg backend convention.
        if self._ffmpeg_backend == "dshow":
            input_name = "audio=%s" % self._device_name
        elif self._ffmpeg_backend == "avfoundation":
            input_name = ":%s" % self._device_name
        else:
            # alsa, pulse — device name is used directly
            input_name = self._device_name

        try:
            self._ffmpeg_process = (
                ffmpeg
                .input(
                    input_name,
                    format=self._ffmpeg_backend,
                    ar=self._audio_sampling_rate_hz,
                    ac=self._num_channels,
                )
                .output(
                    "pipe:",
                    format="s16le",
                    acodec="pcm_s16le",
                )
                .global_args("-hide_banner", "-loglevel", "error")
                .run_async(pipe_stdout=True)
            )
            logger.info(
                "FFmpeg capture started: %s via %s at %d Hz",
                self._device_name,
                self._ffmpeg_backend,
                self._audio_sampling_rate_hz,
            )
            return True
        except Exception as e:
            logger.error("Failed to start FFmpeg capture: %s", e)
            return False

    def _ping_device(self) -> None:
        return None

    def _keep_samples(self) -> None:
        pass

    def _process_data(self) -> None:
        if not self._is_continue_capture:
            self._send_end_packet()
            return

        raw = self._ffmpeg_process.stdout.read(self._read_size)

        if not raw or len(raw) < self._read_size:
            # FFmpeg ended (device disconnected or terminated during shutdown).
            logger.warning("FFmpeg audio read returned incomplete data, stopping")
            self._send_end_packet()
            return

        process_time_s = get_time()

        # Compute RMS amplitude for onset detection (post-hoc analysis).
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2)))

        self._publish(
            self._tag,
            process_time_s=process_time_s,
            data={
                "microphone": {
                    "audio": raw,
                    "rms": rms,
                }
            },
        )

    def _stop_new_data(self):
        # Terminate FFmpeg so stdout.read() returns empty on next call,
        # unblocking _process_data if it's mid-read.
        if self._ffmpeg_process and self._ffmpeg_process.poll() is None:
            self._ffmpeg_process.terminate()
            logger.info("FFmpeg capture process terminated")

    def _cleanup(self) -> None:
        if self._ffmpeg_process:
            # Ensure process is fully stopped.
            if self._ffmpeg_process.poll() is None:
                self._ffmpeg_process.terminate()
            if self._ffmpeg_process.stdout:
                self._ffmpeg_process.stdout.close()
            self._ffmpeg_process.wait()
            self._ffmpeg_process = None
            logger.info("FFmpeg process cleaned up")
        super()._cleanup()
