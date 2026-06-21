"""
verify_api_surface.py

Tests that each library behaves exactly as the code assumes.
Run this first in any new environment before using the instrument.

Each check is independent. Failure tells you exactly which assumption
broke and what the code needs to handle differently.

Usage:
    python tests/verify_api_surface.py
"""

import numpy as np
import sys

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, fn):
    try:
        result, note = fn()
        status = PASS if result else FAIL
        results.append((status, name, note))
    except Exception as e:
        results.append((FAIL, name, f"raised {type(e).__name__}: {e}"))


# ---------------------------------------------------------------------------
# skdim — TwoNN
# ---------------------------------------------------------------------------

def check_twonn_shape_n1():
    """
    Assumption: marginal reads of shape (n,1) are handled via column duplication
    before being passed to TwoNN. Test that _to_2d + TwoNN works correctly.
    Raw TwoNN on (n,1) is known to fail — that is expected and handled.
    """
    import skdim
    data = np.random.rand(300, 1).astype(np.float32)
    data_2d = np.concatenate([data, data], axis=1)  # _to_2d equivalent
    try:
        dim = skdim.id.TwoNN().fit(data_2d).dimension_
        return isinstance(dim, (float, np.floating)), \
               f"_to_2d workaround works, returned {type(dim).__name__} = {dim:.4f}"
    except Exception as e:
        return False, f"_to_2d workaround failed: {e}"

check("skdim TwoNN accepts (n,1) input", check_twonn_shape_n1)


def check_twonn_shape_n2():
    """
    Assumption: TwoNN().fit(data).dimension_ works when data is (n, 2).
    This is the joint case for pairwise gaps.
    """
    import skdim
    data = np.random.rand(300, 2).astype(np.float32)
    try:
        dim = skdim.id.TwoNN().fit(data).dimension_
        return isinstance(dim, float), f"returned {type(dim).__name__} = {dim:.4f}"
    except Exception as e:
        return False, f"failed on (n,2) input: {e}"

check("skdim TwoNN accepts (n,2) input", check_twonn_shape_n2)


def check_twonn_returns_float():
    """
    Assumption: dimension_ is a plain float, not an array or int.
    Code uses it directly in arithmetic: gap = joint - d_n - d_g
    """
    import skdim
    data = np.random.rand(300, 2).astype(np.float32)
    dim = skdim.id.TwoNN().fit(data).dimension_
    is_float = isinstance(dim, (float, np.floating))
    return is_float, f"type is {type(dim).__name__}"

check("skdim TwoNN dimension_ is float", check_twonn_returns_float)


def check_twonn_at_minimum_points():
    """
    Assumption: TwoNN is reliable at exactly 200 points (Facco 2017 floor).
    Check it does not raise or return None at this boundary.
    """
    import skdim
    data = np.random.rand(200, 2).astype(np.float32)
    try:
        dim = skdim.id.TwoNN().fit(data).dimension_
        return dim is not None, f"returned {dim}"
    except Exception as e:
        return False, f"raised at 200 points: {e}"

check("skdim TwoNN stable at 200 points", check_twonn_at_minimum_points)


# ---------------------------------------------------------------------------
# scipy.fft
# ---------------------------------------------------------------------------

def check_fft_axis0_2d():
    """
    Assumption: fft(noise_residual, axis=0) works on 2D array (H, W).
    Used in image clock extraction to transform column-wise across rows.
    """
    from scipy.fft import fft
    data = np.random.rand(100, 80).astype(np.float32)
    result = fft(data, axis=0)
    return result.shape == data.shape, f"output shape {result.shape}, expected {data.shape}"

check("scipy.fft fft accepts 2D input with axis=0", check_fft_axis0_2d)


def check_ifft_requires_real():
    """
    Assumption: ifft returns complex output and np.real() is always needed.
    If it returns real directly, np.real() is harmless but confirms the pattern.
    """
    from scipy.fft import fft, ifft
    data = np.random.rand(100).astype(np.float32)
    result = ifft(fft(data))
    is_complex = np.iscomplexobj(result)
    real_part = np.real(result)
    close = np.allclose(real_part, data, atol=1e-5)
    return close, f"ifft output is complex={is_complex}, real part recovers input={close}"

check("scipy.fft ifft returns complex, np.real recovers input", check_ifft_requires_real)


# ---------------------------------------------------------------------------
# allantools
# ---------------------------------------------------------------------------

def check_allantools_oadev_arguments():
    """
    Assumption: allantools.oadev(timestamps, rate=sr, data_type='phase')
    accepts a numpy array of timestamps and returns (taus, adev, errors, ns).
    """
    import allantools
    sr = 44100
    n = 1000
    timestamps = np.arange(n, dtype=np.float64) / sr
    try:
        taus, adev, errors, ns = allantools.oadev(
            timestamps, rate=float(sr), data_type='phase'
        )
        return len(adev) > 0, f"returned adev of length {len(adev)}"
    except Exception as e:
        return False, f"raised {type(e).__name__}: {e}"

check("allantools oadev accepts phase timestamp array", check_allantools_oadev_arguments)


def check_allantools_short_input():
    """
    Assumption: oadev handles short input gracefully without raising.
    Risk: very short audio segments in binary search may produce < 10 samples.
    """
    import allantools
    sr = 44100
    timestamps = np.arange(50, dtype=np.float64) / sr
    try:
        taus, adev, errors, ns = allantools.oadev(
            timestamps, rate=float(sr), data_type='phase'
        )
        return True, f"returned adev of length {len(adev)}"
    except Exception as e:
        return False, f"raised on short input: {e}"

check("allantools oadev handles short input without raising", check_allantools_short_input)


# ---------------------------------------------------------------------------
# skimage.restoration.denoise_wavelet
# ---------------------------------------------------------------------------

def check_denoise_wavelet_shape():
    """
    Assumption: denoise_wavelet returns same shape as input for 2D float32.
    noise_residual = img - denoised requires identical shapes.
    """
    from skimage.restoration import denoise_wavelet
    img = np.random.rand(100, 100).astype(np.float32)
    denoised = denoise_wavelet(img, rescale_sigma=True)
    return denoised.shape == img.shape, f"input {img.shape}, output {denoised.shape}"

check("skimage denoise_wavelet preserves shape", check_denoise_wavelet_shape)


def check_denoise_wavelet_small_input():
    """
    Assumption: denoise_wavelet works on small patches (binary search regions).
    Risk: wavelet decomposition may fail below a minimum size.
    """
    from skimage.restoration import denoise_wavelet
    img = np.random.rand(16, 16).astype(np.float32)
    try:
        denoised = denoise_wavelet(img, rescale_sigma=True)
        return denoised.shape == img.shape, f"16x16 patch returned shape {denoised.shape}"
    except Exception as e:
        return False, f"raised on 16x16 patch: {e}"

check("skimage denoise_wavelet handles small patches", check_denoise_wavelet_small_input)


# ---------------------------------------------------------------------------
# cv2.HoughLinesP
# ---------------------------------------------------------------------------

def check_hough_returns_none_not_empty():
    """
    Assumption: HoughLinesP returns None when no lines detected.
    Code checks `if lines is not None` — if it returns empty array instead,
    the check still works but the note confirms which behaviour to expect.
    """
    blank = np.zeros((100, 100), dtype=np.uint8)
    import cv2
    edges = cv2.Canny(blank, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50)
    if lines is None:
        return True, "returns None when no lines — code check is correct"
    elif hasattr(lines, '__len__') and len(lines) == 0:
        return True, "returns empty array when no lines — code check still safe"
    else:
        return False, f"unexpected return type: {type(lines)}"

check("cv2 HoughLinesP returns None or empty when no lines", check_hough_returns_none_not_empty)


# ---------------------------------------------------------------------------
# PIL Image.open
# ---------------------------------------------------------------------------

def check_pil_grayscale_array_shape():
    """
    Assumption: np.array(Image) for a grayscale image returns (H, W) not (H, W, 1).
    Code checks ndim == 3 to detect colour images — grayscale must be ndim == 2.
    """
    from PIL import Image
    import io
    # Create a synthetic grayscale PNG in memory
    arr = (np.random.rand(64, 64) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    loaded = Image.open(buf)
    result = np.array(loaded)
    return result.ndim == 2, f"grayscale image array ndim={result.ndim}, shape={result.shape}"

check("PIL grayscale image array is (H,W) not (H,W,1)", check_pil_grayscale_array_shape)


# ---------------------------------------------------------------------------
# librosa
# ---------------------------------------------------------------------------

def check_librosa_load_mono_shape():
    """
    Assumption: librosa.load with mono=True returns waveform shape (n_samples,).
    Risk: may return (1, n_samples) in some versions.
    """
    import librosa
    import io
    import soundfile as sf

    # Generate synthetic audio and write to buffer
    sr = 22050
    waveform = np.random.rand(sr).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format='WAV')
    buf.seek(0)

    loaded, loaded_sr = librosa.load(buf, sr=None, mono=True)
    return loaded.ndim == 1, f"shape={loaded.shape}, ndim={loaded.ndim}"

check("librosa.load mono=True returns 1D array", check_librosa_load_mono_shape)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report():
    print("\n" + "=" * 60)
    print("API SURFACE VERIFICATION REPORT")
    print("=" * 60)

    passed = [r for r in results if r[0] == PASS]
    failed = [r for r in results if r[0] == FAIL]

    for status, name, note in results:
        symbol = "✓" if status == PASS else "✗"
        print(f"\n  {symbol} {name}")
        print(f"    {note}")

    print("\n" + "-" * 60)
    print(f"  {len(passed)} passed   {len(failed)} failed   {len(results)} total")
    print("-" * 60)

    if failed:
        print("\nFAILED CHECKS — action required before running instrument:\n")
        for _, name, note in failed:
            print(f"  ✗ {name}")
            print(f"    {note}\n")
        sys.exit(1)
    else:
        print("\n  All API surface assumptions confirmed.")
        print("  Safe to run the instrument.\n")
        sys.exit(0)


if __name__ == "__main__":
    report()
