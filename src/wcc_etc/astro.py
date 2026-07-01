from __future__ import annotations

from numpy.typing import NDArray
import numpy as np

def get_moon_magnitude(
    phase: float | NDArray[np.float64],
    phase_type: str = 'fraction',
    distance_km: float | None = 384400.0,
    m_full: float = -12.74,
) -> float | NDArray[np.float64]:
    """
    Return an approximate V-band magnitude of the Moon.

    Parameters
    - phase: illuminated fraction (0..1) if phase_type='fraction', or phase angle in degrees (0=full, 180=new) if phase_type='angle'.
    - phase_type: 'fraction' or 'angle'
    - distance_km: Earth-Moon distance in km (default mean 384400 km). Magnitude scaled by inverse-square relative to this distance.
    - m_full: reference full-moon magnitude (default -12.74)

    Notes
    - This is a simple empirical approximation: it assumes brightness scales with illuminated fraction and with inverse-square of distance.
    - For very small illuminated fractions the function clips fraction to avoid -inf magnitudes.

    Example usage
        print("Example moon magnitudes:")
        print(" new    (frac=0.0):", get_moon_magnitude(0.0))
        print(" first  (frac=0.5):", get_moon_magnitude(0.5))
        print(" full   (frac=1.0):", get_moon_magnitude(1.0))

    Using phase angles: 0=full, 90=quarter, 180=new
        for angle in (0, 90, 180):
            print(f"angle {angle} deg -> mag = {get_moon_magnitude(angle, phase_type='angle'):.2f}")
    """
    if phase_type == 'angle':
        # convert phase angle (0=full, 180=new) to illuminated fraction
        phi = np.deg2rad(phase)
        frac = (1.0 + np.cos(phi)) / 2.0
    else:
        frac = phase

    # avoid log(0)
    frac = np.clip(frac, 1e-6, 1.0)

    # scale from full-moon magnitude by illuminated fraction
    mag = m_full - 2.5 * np.log10(frac)

    # scale for distance (inverse-square law). d0 = 384400 km
    if distance_km is not None:
        mag += 5.0 * np.log10(distance_km / 384400.0)

    return mag