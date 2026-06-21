import numpy as np
import cv2
import ffmpeg
from skimage.restoration import denoise_wavelet
from scipy.fft import fft

from coupling import (THRESHOLD, MINIMUM_POINTS,
                      compute_marginals, compute_gaps,
                      verdict, which_pairs_broke, surviving_pairs)
from binary_search import binary_search


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _load_video(path: str):
    """
    Load all frames as grayscale float32.
    Returns (frames, fps, frame_times):
        frames shape (n_frames, H, W)
        fps float
        frame_times shape (n_frames,) in seconds
    """
    probe = ffmpeg.probe(path)
    video_stream = next(
        s for s in probe["streams"] if s["codec_type"] == "video"
    )

    fps_raw = video_stream.get("r_frame_rate", "25/1")
    num, den = fps_raw.split("/")
    fps = float(num) / float(den)

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frames.append(gray)
    cap.release()

    frames = np.stack(frames, axis=0)  # (n_frames, H, W)
    n_frames = len(frames)
    frame_times = np.arange(n_frames, dtype=np.float64) / fps

    # Attempt to get presentation timestamps from ffprobe
    try:
        pts_list = []
        for packet in ffmpeg.probe(path, select_streams="v",
                                    show_entries="packet=pts_time",
                                    **{"of": "csv=p=0"})["packets"]:
            pts_list.append(float(packet["pts_time"]))
        if len(pts_list) == n_frames:
            frame_times = np.array(pts_list, dtype=np.float64)
    except Exception:
        pass  # fall back to uniform timestamps computed above

    return frames, fps, frame_times


# ---------------------------------------------------------------------------
# Extractions
# ---------------------------------------------------------------------------

def _extract_noise(frames: np.ndarray, n_samples: int = 1000) -> np.ndarray:
    """
    Energy read: per-frame wavelet noise residual + temporal correlation.
    Returns shape (n_samples, 1).
    """
    residuals = []
    for frame in frames:
        denoised = denoise_wavelet(frame, rescale_sigma=True)
        residuals.append(frame - denoised)

    residuals = np.stack(residuals, axis=0)  # (n_frames, H, W)

    # Temporal noise correlation: std of noise across frames per pixel
    temporal_corr = np.std(residuals, axis=0).flatten()

    # Spatial residuals: flatten all per-frame residuals
    spatial_flat = residuals.flatten()

    combined = np.concatenate([spatial_flat, temporal_corr])
    samples = combined[:n_samples] if len(combined) >= n_samples else np.pad(
        combined, (0, n_samples - len(combined)), constant_values=0.0)
    return samples.astype(np.float32).reshape(-1, 1)


def _extract_geometry(frames: np.ndarray, n_samples: int = 500) -> np.ndarray:
    """
    Space read: per-frame edge deviation from ideal lines
    plus cross-frame consistency (variance of deviation across frames).
    Returns shape (n_samples, 1).
    """
    all_deviations = []

    for frame in frames:
        uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        edges = cv2.Canny(uint8, 50, 150)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                 threshold=50, minLineLength=10, maxLineGap=5)
        if lines is None:
            continue
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length == 0:
                continue
            dx, dy = (x2 - x1) / length, (y2 - y1) / length
            n_pts = max(int(length), 2)
            xs = np.clip(np.linspace(x1, x2, n_pts).astype(int),
                         0, frame.shape[1] - 1)
            ys = np.clip(np.linspace(y1, y2, n_pts).astype(int),
                         0, frame.shape[0] - 1)
            for i, (x, y) in enumerate(zip(xs, ys)):
                ideal_x = x1 + i * dx
                ideal_y = y1 + i * dy
                dev = np.sqrt((x - ideal_x) ** 2 + (y - ideal_y) ** 2)
                all_deviations.append(dev)

    if len(all_deviations) == 0:
        all_deviations = [0.0] * n_samples

    arr = np.array(all_deviations, dtype=np.float32)

    # Cross-frame variance: variance of deviations
    cross_frame_var = np.var(arr) * np.ones(min(50, n_samples // 10),
                                             dtype=np.float32)
    combined = np.concatenate([arr, cross_frame_var])

    if len(combined) >= n_samples:
        samples = combined[:n_samples]
    else:
        samples = np.pad(combined, (0, n_samples - len(combined)),
                         constant_values=0.0)
    return samples.reshape(-1, 1)


def _extract_clock(frame_times: np.ndarray, fps: float,
                   n_samples: int = 500) -> np.ndarray:
    """
    Time read: inter-frame interval jitter.
    Deviation of actual frame intervals from expected 1/fps.
    Returns shape (n_samples, 1).
    """
    inter_frame = np.diff(frame_times).astype(np.float32)
    expected = 1.0 / fps
    jitter = inter_frame - expected

    if len(jitter) >= n_samples:
        samples = jitter[:n_samples]
    elif len(jitter) > 1:
        # Interpolate up to n_samples
        x_old = np.linspace(0, 1, len(jitter))
        x_new = np.linspace(0, 1, n_samples)
        samples = np.interp(x_new, x_old, jitter).astype(np.float32)
    else:
        samples = np.zeros(n_samples, dtype=np.float32)
    return samples.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Region-level extraction for binary search
# ---------------------------------------------------------------------------

def _make_extract_fn(frames: np.ndarray, fps: float,
                     frame_times: np.ndarray):
    """
    Returns a callable that extracts reads from a frame range.
    Region format: (frame_start, frame_end)
    Returns None if segment produces fewer than MINIMUM_POINTS.
    """
    def extract_fn(region):
        f0, f1 = region
        seg_frames = frames[f0:f1]
        seg_times  = frame_times[f0:f1]
        if len(seg_frames) < 2:
            return None
        n_pixels = len(seg_frames) * seg_frames[0].size
        if n_pixels < MINIMUM_POINTS:
            return None
        n_samples = min(1000, n_pixels)
        g_samples = min(500, n_pixels)
        noise = _extract_noise(seg_frames, n_samples=n_samples)
        geom  = _extract_geometry(seg_frames, n_samples=g_samples)
        clock = _extract_clock(seg_times, fps, n_samples=g_samples)
        if any(len(r) < 2 for r in [noise, geom, clock]):
            return None
        return noise, geom, clock
    return extract_fn


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_video(path: str) -> dict:
    """
    Run physical causality verification on a video file.

    Returns
    -------
    dict with keys:
        noise_geometry_gap, noise_clock_gap,
        geometry_clock_gap, joint_all_gap,
        verdict
        — and if manipulated —
        location (start_frame, end_frame, start_time, end_time),
        which_pair_broke, surviving_pairs, magnitude
    """
    frames, fps, frame_times = _load_video(path)

    noise_read    = _extract_noise(frames)
    geometry_read = _extract_geometry(frames)
    clock_read    = _extract_clock(frame_times, fps)

    marginals = compute_marginals(noise_read, geometry_read, clock_read)
    gaps      = compute_gaps(noise_read, geometry_read, clock_read, marginals)
    v         = verdict(gaps)

    result = {**gaps, "verdict": v}

    if v == "manipulated":
        extract_fn = _make_extract_fn(frames, fps, frame_times)
        initial_region = (0, len(frames))
        location_data = binary_search(extract_fn, initial_region)

        raw_loc = location_data["location"]
        f0, f1 = raw_loc
        result["location"] = {
            "start_frame": f0,
            "end_frame":   f1,
            "start_time":  f0 / fps,
            "end_time":    f1 / fps,
        }
        result["which_pair_broke"] = which_pairs_broke(gaps)
        result["surviving_pairs"]  = surviving_pairs(gaps)
        result["magnitude"]        = location_data["magnitude"]

    return result
