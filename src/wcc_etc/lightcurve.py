# src/wcc_etc/lightcurve.py
"""Time-series light-curve simulation for the WCC ETC.

`FluxModel` maps time to normalized relative flux; `TransitModel` wraps the
`batman` package. `LightCurveSimulator` turns a model + a Simulation into a
synthetic observed light curve, using one ETC photometric SNR measurement at
baseline brightness as the per-point error. Future scenarios (moon transits,
Cepheids) subclass `FluxModel` and work with the simulator and plotters
unchanged.
"""
import numpy as np

_BATMAN_HINT = (
    "TransitModel requires the 'batman' package. "
    "Install it with: pip install wcc-etc[lightcurve]"
)


class FluxModel:
    """Base class: map time -> relative flux, normalized to 1.0 out of event."""

    def relative_flux(self, time):
        raise NotImplementedError


class TransitModel(FluxModel):
    """Exoplanet transit light-curve model backed by `batman`.

    Parameters mirror batman.TransitParams: t0 (center), per (period),
    rp (Rp/R*), a (a/R*), inc (deg), ecc, w (deg), limb_dark, u (coeffs).
    """

    def __init__(self, t0=0.0, per=1.0, rp=0.1, a=15.0, inc=87.0,
                 ecc=0.0, w=90.0, limb_dark="quadratic", u=(0.1, 0.3)):
        self.t0, self.per, self.rp, self.a = t0, per, rp, a
        self.inc, self.ecc, self.w = inc, ecc, w
        self.limb_dark, self.u = limb_dark, list(u)

    def _params(self):
        try:
            import batman
        except ImportError as exc:  # pragma: no cover - exercised via hint
            raise ImportError(_BATMAN_HINT) from exc
        p = batman.TransitParams()
        p.t0, p.per, p.rp, p.a = self.t0, self.per, self.rp, self.a
        p.inc, p.ecc, p.w = self.inc, self.ecc, self.w
        p.limb_dark, p.u = self.limb_dark, list(self.u)
        return batman, p

    def relative_flux(self, time):
        time = np.asarray(time, dtype=float)
        batman, params = self._params()
        return batman.TransitModel(params, time).light_curve(params)
