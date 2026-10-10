"""Defensive audio validation: VAD, SNR, clipping, duration, and silence checks."""

from dataclasses import dataclass

import numpy as np


class AudioValidationError(Exception):
    """Base exception for audio validation failures."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class NoSpeechError(AudioValidationError):
    def __init__(self, message: str = "no speech: audio is silent or unintelligible"):
        super().__init__("no_speech", message)


class TooNoisyError(AudioValidationError):
    def __init__(self, message: str = "too noisy: signal-to-noise ratio is too low"):
        super().__init__("too_noisy", message)


class TooShortError(AudioValidationError):
    def __init__(self, message: str = "too short: audio must contain at least 0.5 seconds of speech"):
        super().__init__("too_short", message)


class TooLongError(AudioValidationError):
    def __init__(self, message: str = "too long: audio exceeds maximum allowed duration"):
        super().__init__("too_long", message)


class ClippingError(AudioValidationError):
    def __init__(self, message: str = "clipping: audio input is severely clipped or distorted"):
        super().__init__("clipping", message)


class UnsupportedFormatError(AudioValidationError):
    def __init__(self, message: str = "unsupported format: please provide valid audio (WAV, WebM, MP3, MP4)"):
        super().__init__("unsupported_format", message)


class PermissionDeniedError(AudioValidationError):
    def __init__(self, message: str = "permission denied: microphone access was blocked"):
        super().__init__("permission_denied", message)


@dataclass
class AudioMetrics:
    """Metrics computed during defensive audio validation."""

    duration_seconds: float
    speech_duration_seconds: float
    silence_ratio: float
    snr_db: float
    clipping_ratio: float
    is_valid: bool = True


def validate_audio_samples(
    samples: np.ndarray,
    sample_rate: int = 16000,
    min_speech_duration_seconds: float = 0.5,
    max_audio_seconds: float = 30.0,
    min_snr_db: float = 10.0,
    max_silence_ratio: float = 0.85,
    max_clipping_ratio: float = 0.05,
) -> AudioMetrics:
    """Validate 1D normalized float32 audio samples before passing to STT.

    Raises:
        TooShortError: If overall length < 0.2s or speech < min_speech_duration_seconds.
        TooLongError: If audio duration exceeds max_audio_seconds.
        ClippingError: If clipping ratio exceeds max_clipping_ratio.
        NoSpeechError: If audio is silent or unintelligible.
        TooNoisyError: If signal-to-noise ratio is below min_snr_db.
    """
    if len(samples) == 0:
        raise TooShortError("too short: empty audio buffer")

    duration_seconds = len(samples) / float(sample_rate)

    if duration_seconds < 0.2:
        raise TooShortError(f"too short: total duration {duration_seconds:.2f}s is under minimum")

    if duration_seconds > max_audio_seconds:
        raise TooLongError(
            f"too long: audio duration {duration_seconds:.1f}s exceeds limit of {max_audio_seconds:.1f}s"
        )

    # 1. Clipping check
    clipping_ratio = float(np.mean(np.abs(samples) >= 0.98))
    if clipping_ratio > max_clipping_ratio:
        raise ClippingError(
            f"clipping: audio is clipped ({clipping_ratio * 100:.1f}% samples distorted)"
        )

    # 2. Peak amplitude and absolute silence check
    max_abs = float(np.max(np.abs(samples)))
    if max_abs < 0.015:  # -36 dBFS
        raise NoSpeechError("no speech: audio is silent or unintelligible")

    # 3. Short-time frame energy analysis for VAD and SNR
    frame_size = int(sample_rate * 0.025)  # 25ms
    hop_size = int(sample_rate * 0.010)    # 10ms

    if len(samples) < frame_size:
        raise TooShortError("too short: audio is shorter than analysis window")

    num_frames = 1 + (len(samples) - frame_size) // hop_size
    frame_energies = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_size
        frame = samples[start : start + frame_size]
        # RMS energy squared
        energy = np.mean(frame**2)
        frame_energies[i] = energy

    # Avoid log of zero
    energies_db = 10.0 * np.log10(np.maximum(frame_energies, 1e-10))

    # Noise floor (approx 10th percentile of frame energy)
    noise_floor_db = float(np.percentile(energies_db, 10))
    # Speech level (approx 90th percentile)
    peak_speech_db = float(np.percentile(energies_db, 90))

    snr_db = float(peak_speech_db - noise_floor_db)

    # Pure noise check: If max energy is moderate/high but dynamic range (SNR) is too flat
    if snr_db < min_snr_db:
        # If the sound has significant volume, it's noise/static
        if noise_floor_db > -45.0:
            raise TooNoisyError(
                f"too noisy: signal-to-noise ratio is too low ({snr_db:.1f} dB < {min_snr_db:.1f} dB)"
            )
        else:
            raise NoSpeechError("no speech: audio is silent or unintelligible")

    # VAD: Frames above noise floor + 6dB and above absolute threshold -42 dB
    speech_threshold_db = max(noise_floor_db + 6.0, -42.0)
    speech_mask = energies_db >= speech_threshold_db
    speech_frames = int(np.sum(speech_mask))

    hop_seconds = hop_size / float(sample_rate)
    speech_duration_seconds = speech_frames * hop_seconds
    silence_ratio = float(1.0 - (speech_frames / float(num_frames)))

    if speech_frames == 0 or silence_ratio > max_silence_ratio or speech_duration_seconds < min_speech_duration_seconds:
        if noise_floor_db > -35.0:
            raise TooNoisyError("too noisy: background noise dominates speech")
        raise NoSpeechError("no speech: audio is silent or unintelligible")

    return AudioMetrics(
        duration_seconds=duration_seconds,
        speech_duration_seconds=speech_duration_seconds,
        silence_ratio=silence_ratio,
        snr_db=snr_db,
        clipping_ratio=clipping_ratio,
        is_valid=True,
    )
