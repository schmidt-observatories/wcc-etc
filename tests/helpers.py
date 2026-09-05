"""Shared factory helpers for the wcc_etc test suite."""

import wcc_etc


def make_scene(
    name="G5V", mag=15, bandpass="johnson_r", background="zodi", background_mag=22.5
):
    """Build a standard scene with a zodi background."""
    return wcc_etc.get_scene(
        name=name,
        mag=mag,
        background=background,
        bandpass=bandpass,
        background_prop={"bandpass": bandpass, "mag": background_mag},
    )


def make_simulation(mag=15, sensor="sony:r", name="G5V", bandpass="johnson_r"):
    """Build a Simulation from a standard zodi-background scene."""
    scene = make_scene(name=name, mag=mag, bandpass=bandpass)
    return wcc_etc.Simulation.from_sensor_and_scene(sensor, scene)


def make_broadband_simulation(name="G5V", mag=12, sensor="sony:bb"):
    """Build a broad-band Simulation of one spectral type.

    The broad band is where the source SED matters most for the PSF: the filter
    pivot is 582 nm regardless of colour, while the photons actually delivered
    average 541 nm (O5V) to 716 nm (M5V). See issue #65.
    """
    return wcc_etc.Simulation.from_sensor_and_scene(
        sensor, make_scene(name=name, mag=mag)
    )
