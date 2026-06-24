# EXPERIMENTAL — theoretical extension to audio domain
# Not yet empirically validated
# See images.py for the validated implementation

import numpy as np
import librosa
import allantools
from scipy.signal import welch
from scipy.fft import fft, ifft

from coupling import (THRESHOLD, MINIMUM_POINTS,
                      compute_marginals, compute_gaps,
                      verdict, which_pairs_broke, surviving_pairs)
from binary_search import binary_search


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _load_audio(path: str):
    """
    Load audio preserving original sample rate.
    Returns (waveform, sr): waveform shape (n_samples,), sr in Hz.
    """
    waveform, sr = librosa.load(path, sr=None, mono=True)
    return waveform.astype(np.float32), sr


# ---------------------------------------------------------------------------
# Extractions
# ---------------------------------------------------------------------------

def _extract_noise(waveform: np.ndarray, sr: int,
                   n_samples: int = 1000) -> np.ndarray:
    """
    Energy read: spectral subtraction on noise floor segments.
    Returns shape (n_samples, 1).
    """
    # Find non-silent intervals, invert to get silent (noise floor) segments
    intervals = librosa.effects.split(waveform, top_db=30)
    silent_mask = np.ones(len(waveform), dtype=bool)
    for start, end in intervals:
        silent_mask[start:end] = False

    if silent_mask.sum() > 0:
        noise_segment = waveform[silent_mask]
    else:
        # Fall back to lowest 10% amplitude frames
        frame_energy = np.array([
            np.abs(waveform[i:i+512]).mean()
            for i in range(0, len(waveform) - 512, 512)
        ])
        threshold = np.percentile(frame_energy, 10)
        low_frames = np.where(frame_energy <= threshold)[0]
        indices = np.concatenate([
            np.arange(f * 512, min((f + 1) * 512, len(waveform)))
            for f in low_frames
        ])
        noise_segment = waveform[indices] if len(indices) > 0 else waveform[:512]

    # Spectral subtraction
    noise_fft = fft(noise_segment)
    mean_noise_spectrum = np.abs(noise_fft).mean()

    full_fft = fft(waveform)
    subtracted = full_fft - mean_noise_spectrum
    noise_residual = np.real(ifft(subtracted)).astype(np.float32)

    flat = noise_residual.flatten()
    samples = flat[:n_samples] if len(flat) >= n_samples else np.pad(
        flat, (0, n_samples - len(flat)), constant_values=0.0)
    return samples.reshape(-1, 1)


def _extract_geometry(waveform: np.ndarray, sr: int,
                      n_samples: int = 500) -> np.ndarray:
    """
    Space read: deviation of frequency response from flat reference.
    Models microphone frequency response nonlinearity.
    Returns shape (n_samples, 1).
    """
    frequencies, psd = welch(waveform, fs=sr, nperseg=1024)
    flat_reference = np.full_like(psd, psd.mean())
    deviation = np.abs(psd - flat_reference).astype(np.float32)

    if len(deviation) >= n_samples:
        indices = np.linspace(0, len(deviation) - 1, n_samples, dtype=int)
        samples = deviation[indices]
    else:
        samples = np.pad(deviation, (0, n_samples - len(deviation)),
                         constant_values=0.0)
    return samples.reshape(-1, 1)


def _extract_clock(waveform: np.ndarray, sr: int,
                   n_samples: int = 500) -> np.ndarray:
    """
    Time read: overlapping Allan deviation of sample timestamps.
    Measures sample clock jitter.
    Returns shape (n_samples, 1).
    """
    timestamps = np.arange(len(waveform), dtype=np.float64) / sr
    try:
        taus, adev, errors, ns = allantools.oadev(
            timestamps, rate=float(sr), data_type='phase'
        )
        adev = np.array(adev, dtype=np.float32)
    except Exception:
        # Fallback: use timestamp differences as a proxy
        adev = np.diff(timestamps).astype(np.float32)

    if len(adev) >= n_samples:
        samples = adev[:n_samples]
    else:
        samples = np.pad(adev, (0, n_samples - len(adev)),
                         constant_values=0.0)
    return samples.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Region-level extraction for binary search
# ---------------------------------------------------------------------------

def _make_extract_fn(waveform: np.ndarray, sr: int):
    """
    Returns a callable that extracts reads from a temporal segment.
    Region format: (sample_start, sample_end)
    Returns None if segment produces fewer than MINIMUM_POINTS.
    """
    def extract_fn(region):
        s0, s1 = region
        segment = waveform[s0:s1]
        if len(segment) < MINIMUM_POINTS:
            return None
        n_samples = min(1000, len(segment))
        g_samples = min(500, len(segment))
        noise = _extract_noise(segment, sr, n_samples=n_samples)
        geom  = _extract_geometry(segment, sr, n_samples=g_samples)
        clock = _extract_clock(segment, sr, n_samples=g_samples)
        if any(len(r) < 2 for r in [noise, geom, clock]):
            return None
        return noise, geom, clock
    return extract_fn


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_audio(path: str) -> dict:
    """
    Run physical causality verification on an audio file.

    Returns
    -------
    dict with keys:
        noise_geometry_gap, noise_clock_gap,
        geometry_clock_gap, joint_all_gap,
        verdict
        — and if manipulated —
        location (start_time, end_time in seconds),
        which_pair_broke, surviving_pairs, magnitude
    """
    waveform, sr = _load_audio(path)

    noise_read    = _extract_noise(waveform, sr)
    geometry_read = _extract_geometry(waveform, sr)
    clock_read    = _extract_clock(waveform, sr)

    marginals = compute_marginals(noise_read, geometry_read, clock_read)
    gaps      = compute_gaps(noise_read, geometry_read, clock_read, marginals)
    v         = verdict(gaps)

    result = {**gaps, "verdict": v}

    if v == "manipulated":
        extract_fn = _make_extract_fn(waveform, sr)
        initial_region = (0, len(waveform))
        location_data = binary_search(extract_fn, initial_region)

        raw_loc = location_data["location"]
        start_time = raw_loc[0] / sr
        end_time   = raw_loc[1] / sr

        result["location"]         = {"start_seconds": start_time,
                                      "end_seconds": end_time}
        result["which_pair_broke"] = which_pairs_broke(gaps)
        result["surviving_pairs"]  = surviving_pairs(gaps)
        result["magnitude"]        = location_data["magnitude"]

    return result
