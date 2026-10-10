"""Audio transcoding to 16kHz mono PCM using FFmpeg."""

import asyncio
import io
import shutil
import subprocess
import wave

import numpy as np


class AudioFormatError(Exception):
    """Raised when audio cannot be transcoded or format is invalid."""


def check_ffmpeg_available() -> bool:
    """Return True if ffmpeg executable is present in PATH."""
    return shutil.which("ffmpeg") is not None


def transcode_to_pcm16_wav(
    audio_bytes: bytes,
    target_sample_rate: int = 16000,
    timeout_seconds: float = 10.0,
) -> tuple[bytes, np.ndarray, int]:
    """Transcode raw audio bytes (WebM, MP4, WAV, MP3, etc.) to 16kHz mono PCM WAV.

    Returns:
        wav_bytes: Standalone WAV formatted bytes.
        samples: 1D numpy array of float32 samples in range [-1.0, 1.0].
        sample_rate: 16000.

    Raises:
        AudioFormatError: If FFmpeg fails or audio format is unsupported.
    """
    if not audio_bytes:
        raise AudioFormatError("unsupported format: empty audio payload received")

    if not check_ffmpeg_available():
        # Fallback for minimal environments without FFmpeg when input is already clean WAV
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())
                if sampwidth == 2:
                    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    if n_channels > 1:
                        data = data.reshape(-1, n_channels).mean(axis=1)
                    return audio_bytes, data, sr
        except Exception as exc:
            raise AudioFormatError(
                "unsupported format: ffmpeg is required to transcode audio input"
            ) from exc

    # FFmpeg command: stdin -> 16kHz mono 16-bit PCM WAV on stdout
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        "pipe:0",
        "-vn",
        "-ar",
        str(target_sample_rate),
        "-ac",
        "1",
        "-f",
        "wav",
        "-acodec",
        "pcm_s16le",
        "pipe:1",
    ]

    try:
        process = subprocess.run(
            cmd,
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioFormatError("unsupported format: audio transcoding timed out") from exc
    except Exception as exc:
        raise AudioFormatError(f"unsupported format: failed to execute ffmpeg ({exc})") from exc

    if process.returncode != 0:
        err_msg = process.stderr.decode("utf-8", errors="replace").strip()
        raise AudioFormatError(
            f"unsupported format: ffmpeg could not decode input ({err_msg or 'invalid codec/stream'})"
        )

    wav_bytes = process.stdout
    if len(wav_bytes) < 44:
        raise AudioFormatError("unsupported format: transcoded audio is empty or corrupted")

    # Read WAV into float32 numpy array
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
            samples_int16 = np.frombuffer(frames, dtype=np.int16)
            samples = samples_int16.astype(np.float32) / 32768.0
            actual_sr = wf.getframerate()
            return wav_bytes, samples, actual_sr
    except Exception as exc:
        raise AudioFormatError(f"unsupported format: invalid WAV output header ({exc})") from exc


async def transcode_to_pcm16_wav_async(
    audio_bytes: bytes,
    target_sample_rate: int = 16000,
    timeout_seconds: float = 10.0,
) -> tuple[bytes, np.ndarray, int]:
    """Async wrapper for transcode_to_pcm16_wav to avoid blocking the event loop."""
    return await asyncio.to_thread(
        transcode_to_pcm16_wav, audio_bytes, target_sample_rate, timeout_seconds
    )
