"""Guard the PSF reference wavelength: pivot, not peak-transmission (wpeak).

The monochromatic diffraction PSF is rendered at Sensor.wavelength. That used to
be bandpass.wpeak() (the wavelength of maximum throughput), which for a roughly
flat-topped filter is essentially arbitrary within the band and biases the PSF
size (the Airy scale is linear in wavelength). It must be the pivot wavelength,
the photometrically meaningful effective wavelength of the bandpass.
"""

import warnings

import astropy.units as u
import pytest

from wcc_etc.sensor import Sensor

warnings.simplefilter("ignore")


@pytest.mark.parametrize("sensorfilter", ["sony:r", "sony:bb", "sony:u", "qcmos:bb"])
def test_wavelength_is_pivot_not_wpeak(sensorfilter):
    s = Sensor.from_name(sensorfilter)
    pivot = s.bandpass.pivot().to(u.nm).value
    assert s.wavelength.to(u.nm).value == pytest.approx(pivot, rel=1e-9)


def test_wavelength_differs_from_wpeak_for_broadband():
    # For the broadband filter wpeak and pivot are far apart (~17%); this is the
    # case the old code got materially wrong, so pin that they are NOT equal.
    s = Sensor.from_name("sony:bb")
    pivot = s.bandpass.pivot().to(u.nm).value
    wpeak = s.bandpass.wpeak().to(u.nm).value
    assert abs(pivot - wpeak) / pivot > 0.1
    assert s.wavelength.to(u.nm).value == pytest.approx(pivot, rel=1e-9)


def test_wavelength_drives_psf_scale():
    # Sanity that the reference wavelength actually sets the rendered PSF scale:
    # a longer wavelength produces a wider Airy core (lower peak-pixel fraction).
    from wcc_etc import airy

    s = Sensor.from_name("sony:bb")
    common = dict(
        fnum=15.0,
        D=3.065,
        pixel_size=s.pixel_size.value,
        jitter_sigma_mas=0,
        n_pixels=63,
        oversample=11,
    )
    psf_short, _ = airy.render_detector_psf(wavelength=400e-9, **common)
    psf_long, _ = airy.render_detector_psf(wavelength=800e-9, **common)
    assert psf_long.max() < psf_short.max()
