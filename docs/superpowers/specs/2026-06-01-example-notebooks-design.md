# Example notebooks for recent features — design

Date: 2026-06-01

## Goal

Add a comprehensive, tracked set of example notebooks under `notebooks/`, focused
on the features added since the saturation-flag work. Today `notebooks/` only
tracks `01_example.ipynb` (quickstart) and `02_saturation_flag.ipynb`; the rest of
the recent work lives only in gitignored `notebooks_scratch/`.

## Approach

Each notebook is built from a Python builder script and executed + verified
end-to-end via the `notebook-demo` workflow, so plots and outputs are real and the
notebook runs clean top-to-bottom. Each notebook is self-contained:

1. imports + a shared scene/sensor setup,
2. a feature walkthrough with plots,
3. at least one sanity-check / cross-check cell with an asserted or printed
   expectation,
4. short markdown narration framing each step.

Continue the existing `0N_` numbering convention.

## Notebooks

### `03_from_sensorfilter.ipynb` — auto PSF selection by sensor:filter
- `Simulation.from_sensorfilter('zwo:r', scene)` and
  `ImageSimulator.from_sensorfilter(...)` auto-select the PSF from the filter's
  `focus_level`.
- Contrast in-focus (`zwo:r`, 0wave → `AiryPSF`) vs defocused (`zwo:r+1` 1wave,
  `zwo:bb2` 2wave → `DefocusPSF`). Plot the auto-selected PSFs side by side.
- Show `get_image_snr()` using the stored `_default_psf` when no `psf=` is passed.
- Demonstrate the guards: `ValueError` for an unknown label, `NotImplementedError`
  for a label with no throughput curve (e.g. `zwo:halpha`).

### `04_psf_and_image_snr.ipynb` — PSF sim + PSF-aware SNR
- `AiryPSF` / `DefocusPSF(DEFOCUS_1WAVE_PATH | DEFOCUS_2WAVE_PATH)`.
- `ImageSimulator.from_sensor_and_scene(...).simulate(...)` with Poisson + read
  noise, jitter, saturation mask; show `image_e`, `image_clean`, `to_adu()`.
- `get_image_snr` vs analytic `get_snr` cross-check (~1% for Airy).
- Defocus SNR penalty; `optimize=True` best-aperture.

### `05_n_reads_exptime.ipynb` — n_reads and exptime inverses
- `get_exptime_for_snr` / `get_image_exptime_for_snr` round-trip against
  `get_snr` / `get_image_snr`.
- SNR and required exposure time vs `n_reads`.
- Per-frame saturation flip as `n_reads` changes.

### `06_source_spectra.ipynb` — parametric source spectra
- `get_scene(name=...)` for blackbody / flat / powerlaw / emission: spectra plots.
- Analytic physics checks (Planck/Wien, F_ν↔F_λ, λ^α, Gaussian line).
- SNR-vs-mag; `sim.update(source__teff=...)` rebuilding the spectrum and changing
  SNR.

## Also

Add a short "Example notebooks" table to `README.md` so the set is discoverable.

## Out of scope
- Saturation flag (already `02_saturation_flag.ipynb`).
- Any `src/` changes.
- The empty / legacy scratch notebooks.
