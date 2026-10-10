"""Audio fixtures generator for silence, noise, clipping, and browser containers."""

import io
import subprocess
import wave

import numpy as np


def samples_to_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert float32 samples [-1.0, 1.0] to 16-bit mono WAV bytes."""
    samples_clipped = np.clip(samples, -1.0, 1.0)
    int16_samples = (samples_clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_samples.tobytes())
    return buf.getvalue()


def generate_speech_like_wav(duration_s: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate audio resembling speech formants (multi-tone with syllable modulation)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Fundamental frequency + formant harmonics
    signal = (
        0.4 * np.sin(2 * np.pi * 180 * t)
        + 0.25 * np.sin(2 * np.pi * 700 * t)
        + 0.15 * np.sin(2 * np.pi * 2400 * t)
    )
    # Syllable modulation (3-4 Hz envelope)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.5 * t))
    # Add gentle pauses
    silence_gate = np.where(np.sin(2 * np.pi * 0.8 * t) > -0.2, 1.0, 0.05)
    samples = signal * envelope * silence_gate
    # Add light room ambient noise (30 dB SNR)
    ambient_noise = np.random.normal(0, 0.005, len(samples))
    samples = samples + ambient_noise
    return samples_to_wav_bytes(samples.astype(np.float32), sample_rate)


def generate_silence_wav(duration_s: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate silent WAV audio."""
    samples = np.zeros(int(sample_rate * duration_s), dtype=np.float32)
    return samples_to_wav_bytes(samples, sample_rate)


def generate_noisy_wav(duration_s: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate continuous loud white/pink noise with no intelligible SNR."""
    noise = np.random.normal(0, 0.35, int(sample_rate * duration_s)).astype(np.float32)
    return samples_to_wav_bytes(noise, sample_rate)


def generate_clipped_wav(duration_s: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate severely clipped and distorted audio."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    loud_signal = 3.5 * np.sin(2 * np.pi * 440 * t)  # Heavily overdriven
    clipped = np.clip(loud_signal, -1.0, 1.0).astype(np.float32)
    return samples_to_wav_bytes(clipped, sample_rate)


def generate_short_wav(duration_s: float = 0.15, sample_rate: int = 16000) -> bytes:
    """Generate audio that is below the minimum duration threshold."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    samples = 0.3 * np.sin(2 * np.pi * 440 * t)
    return samples_to_wav_bytes(samples.astype(np.float32), sample_rate)


def generate_webm_audio(duration_s: float = 1.5, sample_rate: int = 16000) -> bytes:
    """Generate valid browser WebM/Opus audio using FFmpeg."""
    wav_bytes = generate_speech_like_wav(duration_s, sample_rate)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-b:a",
            "32k",
            "-f",
            "webm",
            "pipe:1",
        ],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def generate_mp4_audio(duration_s: float = 1.5, sample_rate: int = 16000) -> bytes:
    """Generate valid browser MP4/AAC audio using FFmpeg."""
    wav_bytes = generate_speech_like_wav(duration_s, sample_rate)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-f",
            "mp4",
            "-movflags",
            "frag_keyframe+empty_moov",
            "pipe:1",
        ],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout
