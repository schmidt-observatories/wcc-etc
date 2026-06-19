# src/wcc_etc/lightcurve.py
"""WCC light-curve simulation.

`LightCurveSimulator` turns a FluxModel + a Simulation into a synthetic
observed WCC light curve, using one ETC photometric SNR measurement at
baseline brightness as the per-point error. The flux models themselves
(`FluxModel`, `TransitModel`) live in the instrument-agnostic `lazuli_transit`
package and are re-exported here for backward compatibility.
"""
import numpy as np

from lazuli_transit import FluxModel, TransitModel  # noqa: F401  (re-export)


class LightCurve:
    """Result of a light-curve simulation.

    Attributes
    ----------
    time : ndarray
    flux : ndarray            # noisy realization
    flux_clean : ndarray      # noiseless model
    flux_err : float          # per-point sigma (= 1 / snr)
    exptime : float           # per-point exposure time (s)
    snr : float               # baseline photometric SNR
    """

    def __init__(self, time, flux, flux_clean, flux_err, exptime, snr):
        self.time = np.asarray(time, dtype=float)
        self.flux = np.asarray(flux, dtype=float)
        self.flux_clean = np.asarray(flux_clean, dtype=float)
        self.flux_err = float(flux_err)
        self.exptime = float(exptime)
        self.snr = float(snr)

    def __repr__(self):
        return (f"LightCurve(n={self.time.size}, snr={self.snr:.1f}, "
                f"flux_err={self.flux_err:.3g}, exptime={self.exptime:g}s)")

    def plot(self, backend="mpl", **kw):
        """Plot this light curve. Wired to plotting.py (lazy import to avoid
        an import cycle, mirroring SimulatedImage)."""
        from . import plotting
        if backend == "mpl":
            return plotting.plot_lightcurve_mpl(self, **kw)
        if backend == "bokeh":
            return plotting.plot_lightcurve_bokeh(self, **kw)
        raise ValueError("backend must be 'mpl' or 'bokeh'")


class LightCurveSimulator:
    """Turn a FluxModel into a WCC-observed light curve via the ETC."""

    def __init__(self, sim, model):
        self.sim = sim
        self.model = model

    def simulate(self, time, exptime, *, r_aper_mas=None, ee_frac=None,
                 psf=None, jitter_sigma_mas=None, npix=128, seed=None):
        time = np.asarray(time, dtype=float)
        result = self.sim.get_image_snr(
            time=float(exptime), n_reads=1, r_aper_mas=r_aper_mas,
            ee_frac=ee_frac, psf=psf, jitter_sigma_mas=jitter_sigma_mas,
            npix=npix)
        snr = float(result["snr"])
        sigma = 1.0 / snr
        flux_clean = self.model.relative_flux(time)
        rng = np.random.default_rng(seed)
        flux = flux_clean + rng.normal(0.0, sigma, size=time.shape)
        return LightCurve(time, flux, flux_clean, sigma, float(exptime), snr)
