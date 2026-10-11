# Bundled data and third-party code

Every dataset shipped inside the `wcc_etc` wheel, where it came from, and what to cite.
Paths are relative to `src/wcc_etc/data/`.

## Astrophysical templates

| Path | What | Source and citation |
|---|---|---|
| `astr_obj_models/stars/pickles_models/dat_uvk/` | 131 stellar spectra, 1150–25000 Å | Pickles (1998), PASP 110, 863, via the [STScI Pickles Atlas](https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/pickles-atlas). `pickles_mapping.csv` maps spectral type to file. |
| `astr_obj_models/galaxies/brown/` | 129 galaxy SEDs, UV to mid-IR | Brown et al. (2014), ApJS 212, 18. See `AA_README_BROWN` in that directory. |
| `astrophysics/hsiao07.dat` | SN Ia template, phase 0 d (B maximum) slice only | Hsiao et al. (2007), ApJ 663, 1187, as distributed in the [SALTShaker](https://github.com/djones1040/SALTShaker/blob/main/saltshaker/initfiles/hsiao07.dat) repository. |
| `astrophysics/zodiacal_model_001.fits` | Zodiacal-light spectral template | HST Exposure Time Calculator background model (`newzodi.fits`, STScI), based on Leinert et al. (1998), A&AS 127, 1. The normalization (V = 22.5 mag/arcsec², see #98) is set in `scene.py`. |

The Vega reference spectrum (CALSPEC `alpha_lyr_stis_011.fits`) and the Johnson V
curve (`johnson_v_004_syn.fits`) are fetched by `synphot` from STScI at first use
and cached locally; they are not in the wheel.

## Filter profiles

| Path | What | Source and citation |
|---|---|---|
| `throughput/other_filters/SLOAN_SDSS.*prime_filter.dat` | SDSS u'g'r'i'z' transmission | [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/svo/theory/fps/); Rodrigo, Solano & Bayo (2012), IVOA Working Draft; Rodrigo & Solano (2020), SEA XIV. |
| `throughput/20260304_wcc_throughputs/` | End-of-life telescope + WCC throughput per filter and sensor, including the three Hα bands | Lazuli / Project Pearl optical throughput model, 2026-03-04 delivery. Internal product of the Schmidt Observatory System. |

## Detector data

| Path | What | Source |
|---|---|---|
| `sensors/ZWO_ASI6200MM/` | Gain, read noise, well depth and dark current vs setting; QE curve (Sony IMX455) | Digitized by Aaron Goldtooth from the manufacturer plots on the [ZWO product page](https://astronomy-imaging-camera.com/product/asi6200mm-pro-mono/). The QE curve is identical to the IMX571 (verified by Patrick Ingraham, 2023-09-18). See `readme.md` there. |
| `sensors/qCMOS/` | HWK4123 (Hamamatsu ORCA-Quest qCMOS) parameters, QE, non-linearity | Manufacturer values, plus an MIT dark-current measurement at −20 °C and gain-mode numbers recorded in `HWK4123_sensor_parameters.txt`. |
| `sensors/sony_imx992/sony_qe_digitized.csv` | Digitized relative QE of the Sony IMX992 | Source not recorded in the file header (open item in #101). Not used by any bundled configuration. |

## Point-spread functions

| Path | What | Source |
|---|---|---|
| `psfs/CAD_{1,2}-waves-defocus_500nm_Huygens-PSF-Data_Linear.fits` | WCC PSF at 500 nm with 1 and 2 waves of defocus | Zemax Huygens-PSF exports of the Lazuli WCC optical design, converted with `psfsim.convert_zemax_txt_to_fits`. The `.txt` exports are kept in the repository but not in the wheel. |

## Third-party code

- `src/wcc_etc/radial_data.py` is adapted from `radial_data()` by Ian Crossfield (2005,
  Jet Propulsion Laboratory), distributed in his public Python astronomy library.
  Redistributed here with attribution; the original carried no explicit license.
  Modifications: `numpy`/`scipy` updates only.

## Citing wcc-etc

See `CITATION.cff`.
