"""Instrument-agnostic transit / flux models.

`FluxModel` maps time to normalized relative flux; `TransitModel` wraps the
`jaxoplanet` package. These models carry no dependency on any specific
instrument: a WCC or IFS simulator consumes a FluxModel through its
`relative_flux(time)` interface.
"""

import numpy as np

_JAXOPLANET_HINT = (
    "TransitModel requires the 'jaxoplanet' package. "
    "Install it with: pip install lazuli-transit[jaxoplanet]"
)

# jaxoplanet implements polynomial limb darkening (Agol, Luger &
# Foreman-Mackey 2020); law name -> required coefficient count.
_POLY_LIMB_DARK = {"uniform": 0, "linear": 1, "quadratic": 2}


class FluxModel:
    """Base class: map time -> relative flux, normalized to 1.0 out of event."""

    def relative_flux(self, time):
        raise NotImplementedError


class TransitModel(FluxModel):
    """Exoplanet transit light-curve model backed by `jaxoplanet`.

    Parameters follow the batman.TransitParams convention: t0 (center),
    per (period), rp (Rp/R*), a (a/R*), inc (deg), ecc, w (deg), limb_dark,
    u (coeffs). Only polynomial limb-darkening laws are supported
    ("uniform", "linear", "quadratic").
    """

    def __init__(
        self,
        t0=0.0,
        per=1.0,
        rp=0.1,
        a=15.0,
        inc=87.0,
        ecc=0.0,
        w=90.0,
        limb_dark="quadratic",
        u=(0.1, 0.3),
    ):
        self.t0, self.per, self.rp, self.a = t0, per, rp, a
        self.inc, self.ecc, self.w = inc, ecc, w
        self.limb_dark, self.u = limb_dark, list(u)

    @classmethod
    def from_planet(
        cls, name, df=None, require_transit=True, limb_dark="quadratic", u=(0.1, 0.3)
    ):
        """Build a TransitModel from a NASA Exoplanet Archive (PSCompPars) row.

        Looks up ``name`` in ``df`` (or the local cache via
        load_exoplanet_archive if None) and maps archive columns to
        transit-model parameters. ``pl_orbper`` is required; other fields
        fall back to circular-orbit / direct-ratio defaults when missing.
        Limb-darkening is not in the archive, so it stays a user argument.

        If ``require_transit`` is True (default), a planet flagged as
        non-transiting in the archive (``tran_flag == 0``) raises ValueError —
        a transit model is not meaningful for it. Pass ``require_transit=False``
        to build one anyway. When ``tran_flag`` is absent/unknown, the check is
        skipped.

        Note: ``t0`` is taken from ``pl_tranmid`` (absolute BJD). For a
        relative-time light curve, pass a ``time`` array spanning that epoch or
        set ``t0=0`` after construction.
        """
        if df is None:
            from .archive import load_exoplanet_archive

            df = load_exoplanet_archive()

        def _norm(s):
            return str(s).lower().replace(" ", "")

        matches = df[df["pl_name"].map(_norm) == _norm(name)]
        if len(matches) == 0:
            raise ValueError(f"Planet {name!r} not found in archive table")
        row = matches.iloc[0]

        if require_transit:
            tran = row.get("tran_flag")
            known = tran is not None and not (
                isinstance(tran, float) and np.isnan(tran)
            )
            if known and int(tran) == 0:
                raise ValueError(
                    f"Planet {name!r} is not flagged as transiting "
                    "(tran_flag=0); pass require_transit=False to build a "
                    "model anyway"
                )

        def _val(col, default):
            v = row.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return float(v)

        per = row.get("pl_orbper")
        if per is None or (isinstance(per, float) and np.isnan(per)):
            raise ValueError(
                f"Planet {name!r} has no pl_orbper; cannot build a transit model"
            )

        import astropy.constants as const
        import astropy.units as units

        rjup_rsun = float((const.R_jup / const.R_sun).decompose().value)
        au_rsun = float((1 * units.au).to(units.R_sun).value)

        rp = _val("pl_ratror", np.nan)
        if np.isnan(rp):
            rp = _val("pl_radj", np.nan) * rjup_rsun / _val("st_rad", np.nan)

        a = _val("pl_ratdor", np.nan)
        if np.isnan(a):
            a = _val("pl_orbsmax", np.nan) * au_rsun / _val("st_rad", np.nan)

        return cls(
            t0=_val("pl_tranmid", 0.0),
            per=float(per),
            rp=rp,
            a=a,
            inc=_val("pl_orbincl", 90.0),
            ecc=_val("pl_orbeccen", 0.0),
            w=_val("pl_orblper", 90.0),
            limb_dark=limb_dark,
            u=u,
        )

    def build_system(self):
        """Return ``(system, light_curve_fn)`` for the jaxoplanet backend.

        Note this enables jax's float64 mode globally: without it jax runs in
        float32 and the light curve is only accurate to ~1e-4.
        """
        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            from jaxoplanet.light_curves import limb_dark_light_curve
            from jaxoplanet.orbits.keplerian import Central, System
        except ImportError as exc:  # pragma: no cover - exercised via hint
            raise ImportError(_JAXOPLANET_HINT) from exc

        # Fix the star at R* = 1 so a/R* and Rp/R* come out as given.
        central = Central.from_orbital_properties(
            period=self.per, semimajor=self.a, radius=1.0
        )
        body = dict(
            period=self.per,
            time_transit=self.t0,
            inclination=np.deg2rad(self.inc),
            radius=self.rp,
        )
        if self.ecc != 0.0:
            body.update(eccentricity=self.ecc, omega_peri=np.deg2rad(self.w))
        return System(central).add_body(**body), limb_dark_light_curve

    def check_limb_dark(self):
        """Validate ``limb_dark`` and ``u`` against the supported polynomial laws."""
        n_coeff = _POLY_LIMB_DARK.get(self.limb_dark)
        if n_coeff is None:
            raise ValueError(
                f"limb_dark={self.limb_dark!r} is not supported: the "
                "jaxoplanet backend implements polynomial laws only "
                f"({', '.join(_POLY_LIMB_DARK)})"
            )
        if len(self.u) != n_coeff:
            raise ValueError(
                f"limb_dark={self.limb_dark!r} needs {n_coeff} "
                f"coefficient(s), got {len(self.u)}"
            )

    def relative_flux(self, time):
        time = np.asarray(time, dtype=float)
        self.check_limb_dark()
        system, limb_dark_light_curve = self.build_system()
        flux = limb_dark_light_curve(system, *self.u)(time)
        return 1.0 + np.asarray(flux).reshape(time.shape)
