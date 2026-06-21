# Physical Causality Verification

A media-agnostic instrument for detecting synthetic and manipulated files.

---

## The core idea

Every file captured by a real device carries a mathematical trace of that capture. A single physical event — photons hitting a sensor, sound waves moving a diaphragm, a clock ticking — simultaneously causes structure in the noise field, the geometry field, and the timing field of the file. Those three fields are coupled because they share one cause.

Synthetic files have no single physical cause. Each field is generated independently. The coupling is absent.

This instrument measures that coupling directly. It produces four numbers for any file. No training data. No generator-specific signatures. No empirical tuning.

---

## Four numbers

```
gap(noise, geometry)    Energy-Space coupling
gap(noise, clock)       Energy-Time coupling
gap(geometry, clock)    Space-Time coupling
gap(all three)          Joint coupling across all domains
```

Each number is the difference between joint intrinsic dimensionality and the sum of marginal intrinsic dimensionalities, computed using the TwoNN estimator (Facco et al. 2017).

**All four large** → single physical cause throughout → file is real

**Any gap near zero** → causal chain broke at a location → file is manipulated → binary search activates to localise the break

**All four near zero** → no physical cause anywhere → file is fully synthetic

---

## Works on images, audio, and video without modification

The instrument operates below the level where media type exists. Energy, Space, and Time are physical dimensions, not media-specific quantities. The four numbers mean the same thing regardless of file format.

---

## Why this is different from every other forensic tool

Every other tool answers: *does this file look like files I was trained to recognise as synthetic?*

That question can always be falsified by a new generator.

This instrument answers: *does this file carry the mathematical trace of a single physical cause?*

That question has an analytical answer. No generator, however sophisticated, produces physical causality it does not have. There is no escape hatch for new generators because the instrument does not detect generators. It detects physical causality.

If the instrument fails, one of its four foundational facts must be false — and each of those facts is either a mathematical identity or a physical law.

---

## Installation

```bash
pip install numpy scikit-image opencv-python Pillow librosa scipy allantools ffmpeg-python scikit-dimension
```

---

## Usage

```python
from src.images import analyse_image
from src.audio import analyse_audio
from src.video import analyse_video

# Returns dictionary of four gap numbers
# plus localisation data if binary search activated
result = analyse_image("path/to/image.jpg")
result = analyse_audio("path/to/audio.wav")
result = analyse_video("path/to/video.mp4")

print(result)
```

---

## Output format

```python
{
    "noise_geometry_gap": float,
    "noise_clock_gap": float,
    "geometry_clock_gap": float,
    "joint_all_gap": float,

    # present only if any gap collapsed
    "location": ...,          # pixel region / time range / frame range
    "which_pair_broke": str,
    "surviving_pairs": list,
    "magnitude": float
}
```

---

## Theoretical foundation

The instrument assembles six existing results across six independent fields into one instrument. No new mathematics. No new experiments.

| Paper | Original purpose | Role here |
|---|---|---|
| Fridrich et al. (2006) | Camera identification via PRNU | Energy read — validated noise extraction |
| Facco et al. (2017) | Intrinsic dimensionality estimation | The measurement operation |
| Ma et al. (2013) | Magnetic resonance fingerprinting | Cross-domain consistency logic |
| Shannon (1959) | Rate-distortion theory | Analytical operating envelope |
| Goodfellow et al. (2014) + Ho et al. (2020) | GAN and diffusion model fidelity | Proof coupling survives any faithful pipeline |
| Wang et al. (2004) | SSIM pipeline fidelity | Calibration curves — already published |

Full derivation in [preprint](./preprint.md).

---

## Limitations

The limitations are analytically derived, not empirically discovered.

**Rate-distortion floor** — files compressed below the Shannon limit are outside the operating envelope. The boundary is known analytically.

**TwoNN reliability floor** — below 200 sample points per read the estimator is unreliable (Facco et al. 2017). Binary search stops here.

**Extraction implementation fidelity** — the theory requires each extraction function to correctly read its physical quantity. This is an engineering constraint per domain, bounded in scope, not a theoretical weakness.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

You may use, modify, and distribute this software freely under AGPL terms. If you deploy this software as a network service, you must make your complete source code available under AGPL.

**Commercial license available** for organisations that require proprietary deployment.
Contact: 8amogh@gmail.com

---

## Citation

If you use this instrument in research, please cite:

```
G K, A. (2026). Physical Causality Verification: A Media-Agnostic Instrument 
for Detecting Synthetic and Manipulated Files. Zenodo. 
https://doi.org/10.5281/zenodo.20786125
```

---

## Status

Theory complete. Implementation in progress.
