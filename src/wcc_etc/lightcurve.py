# src/wcc_etc/lightcurve.py
"""WCC light-curve simulation.

`LightCurveSimulator` turns a FluxModel + a Simulation into a synthetic
observed WCC light curve, using one ETC photometric SNR measurement at
baseline brightness as the per-point error. The flux models themselves
(`FluxModel`, `TransitModel`) live in the instrument-agnostic `lazuli_transit`
package and are re-exported here for backward compatibility.

Two conventions matter for interpreting the output:

**Timestamps are mid-exposure.** Each entry of ``time`` is the centre of an
``exptime``-long window, and the model is averaged over that window
(supersampled, see `supersample_factor`) so that finite-cadence smearing of
ingress and egress is captured. Feeding exposure *start* times shifts the
recovered ephemeris by half an exposure.

**Photometric error is flux-independent.** ``sigma = 1 / SNR`` is computed
once from the out-of-transit (baseline) brightness and applied at every point,
in transit and out. In reality the source shot-noise term scales as
``sqrt(f)``, so the true in-transit error is smaller than this by roughly
``1 - depth / 2`` in the source-limited regime, and by less than that whenever
sky, host, dark or read noise contribute. The approximation is therefore
conservative and accurate to first order in the transit depth: for a 1% transit
it overstates the in-transit error by <0.5%. It becomes noticeably wrong for
deep (>~5%) transits of source-noise-dominated targets; model those with an
external, per-point error budget.
"""

import numpy as np
from lazuli_transit import FluxModel, TransitModel  # noqa: F401  (re-export)

# Sub-exposure sampling step for the supersampling default (seconds).
SUBSAMPLE_SECONDS = 10.0


def supersample_factor(exptime, target_s=SUBSAMPLE_SECONDS):
    """Odd number of sub-samples spanning an ``exptime``-second exposure.

    Chosen so that each sub-sample is no longer than ``target_s`` seconds, and
    forced odd so one sub-sample lands on the mid-exposure timestamp itself.
    Returns 1 for exposures at or below ``target_s``, i.e. short exposures are
    not oversampled.
    """
    n = max(1, int(np.ceil(float(exptime) / target_s)))
    return n if n % 2 else n + 1


class LightCurve:
    """Result of a light-curve simulation.

    Attributes
    ----------
    time : ndarray            # mid-exposure timestamps (days)
    flux : ndarray            # noisy realization
    flux_clean : ndarray      # noiseless model, averaged over each exposure
    flux_err : float          # per-point sigma (= 1 / snr), baseline value
                              # applied in and out of transit (see module docs)
    exptime : float           # per-point exposure time (s)
    snr : float               # baseline photometric SNR
    supersample : int         # sub-samples averaged per exposure
    """

    def __init__(self, time, flux, flux_clean, flux_err, exptime, snr, supersample=1):
        self.time = np.asarray(time, dtype=float)
        self.flux = np.asarray(flux, dtype=float)
        self.flux_clean = np.asarray(flux_clean, dtype=float)
        self.flux_err = float(flux_err)
        self.exptime = float(exptime)
        self.snr = float(snr)
        self.supersample = int(supersample)

    def __repr__(self):
        return (
            f"LightCurve(n={self.time.size}, snr={self.snr:.1f}, "
            f"flux_err={self.flux_err:.3g}, exptime={self.exptime:g}s)"
        )

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

    def simulate(
        self,
        time,
        exptime,
        *,
        r_aper_mas=None,
        ee_frac=None,
        psf=None,
        jitter_sigma_mas=None,
        npix=128,
        seed=None,
        supersample=None,
    ):
        """Simulate an observed light curve.

        ``time`` holds **mid-exposure** timestamps in days; the model is
        averaged over the ``exptime``-long window centred on each one. Pass
        ``supersample`` to override the number of sub-samples per exposure
        (default: `supersample_factor(exptime)`); ``supersample=1`` restores
        the instantaneous sampling used before this was added.

        The returned ``flux_err`` is the baseline (out-of-transit) sigma and is
        applied at every point -- see the module docstring for the validity
        range of that approximation.
        """
        time = np.asarray(time, dtype=float)
        result = self.sim.get_image_snr(
            time=float(exptime),
            n_reads=1,
            r_aper_mas=r_aper_mas,
            ee_frac=ee_frac,
            psf=psf,
            jitter_sigma_mas=jitter_sigma_mas,
            npix=npix,
        )
        snr = float(result["snr"])
        sigma = 1.0 / snr
        n = supersample_factor(exptime) if supersample is None else int(supersample)
        if n < 1:
            raise ValueError("supersample must be >= 1")
        if n == 1:
            flux_clean = np.asarray(self.model.relative_flux(time), dtype=float)
        else:
            # Centred sub-exposure offsets, so `time` is the exposure midpoint.
            offsets = (np.arange(n) - (n - 1) / 2) * (float(exptime) / 86400.0) / n
            t_sub = time[:, None] + offsets[None, :]
            flux_clean = (
                np.asarray(self.model.relative_flux(t_sub.ravel()), dtype=float)
                .reshape(t_sub.shape)
                .mean(axis=1)
            )
        rng = np.random.default_rng(seed)
        flux = flux_clean + rng.normal(0.0, sigma, size=time.shape)
        return LightCurve(time, flux, flux_clean, sigma, float(exptime), snr, n)
