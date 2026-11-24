"""Low-resolution slitless grating in a WCC filter slot.

Model of the two-zone R ~ 20 blazed transmission grating of the WCC grating
trade study (G. Furesz, 2026-11-16): a 16-level fused-silica staircase 70 mm in
front of the sensor, 4.414 lines/mm, with one passband per half so that first
and second orders do not overlap:

- ``"blue"``: 350-700 nm behind a 700 nm short-pass coating, blaze 430 nm,
  bands u, g, r;
- ``"red"``: 600-1100 nm behind a 600 nm long-pass coating, blaze 775 nm,
  bands i, z.

The throughput is the ``nofilter`` sensor curve (telescope x WCC contamination
and reserve x detector QE, end of life) times the edge coating times the scalar
diffraction efficiency of the staircase. :class:`GratingSimulation` renders the
dispersed image of the scene source (all requested orders) and extracts
per-band signal-to-noise in a box along the first order.

Not modelled: static wavefront error (pass ``psf=`` for a non-Airy PSF), FSM
smear, the +-3.6% dispersion change across the tilted sensor, extended sources.
"""

import dataclasses
import warnings

import numpy as np
from astropy import units as u
from scipy.integrate import trapezoid
from scipy.signal import fftconvolve
from synphot import units as su

from .psfsim import (
    AiryPSF,
    ImageSimulator,
    SimulatedImage,
    saturation_mask_from_image_e,
    solve_time_for_snr,
)
from .simulation import Simulation

__all__ = [
    "GRATING_HALVES",
    "GratingSimulation",
    "edge_coating",
    "fused_silica_index",
    "staircase_efficiency",
]

#: Design of each half: blaze wavelength, coating edge (nm; the sign of
#: ``edge_sign`` is +1 for a short-pass, -1 for a long-pass), and the first-order
#: extraction bands in nm.
GRATING_HALVES = {
    "blue": {
        "blaze_nm": 430.0,
        "edge_nm": 700.0,
        "edge_sign": 1,
        "bands": {"u": (350, 395), "g": (400, 550), "r": (555, 690)},
    },
    "red": {
        "blaze_nm": 775.0,
        "edge_nm": 600.0,
        "edge_sign": -1,
        "bands": {"i": (700, 850), "z": (835, 1100)},
    },
}


def fused_silica_index(wave_nm):
    """Refractive index of fused silica (Malitson 1965 Sellmeier)."""
    w2 = (np.asarray(wave_nm, dtype=float) / 1000.0) ** 2
    return np.sqrt(
        1
        + 0.6961663 * w2 / (w2 - 0.0684043**2)
        + 0.4079426 * w2 / (w2 - 0.1162414**2)
        + 0.8974794 * w2 / (w2 - 9.896161**2)
    )


def staircase_efficiency(wave_nm, order, blaze_nm, levels=16):
    """Scalar diffraction efficiency of an M-level staircase blazed grating.

    For a staircase that approximates a 2 pi blaze at ``blaze_nm`` the phase
    depth at another wavelength is ``alpha = blaze [n(wave)-1] / (wave
    [n(blaze)-1])`` periods and the efficiency of order m is (Swanson 1989)::

        sinc^2(m / M) * [sin(pi (alpha - m)) / (M sin(pi (alpha - m) / M))]^2

    which tends to ``sinc^2(alpha - m)`` for a continuous blaze.
    """
    wave_nm = np.asarray(wave_nm, dtype=float)
    alpha = (
        blaze_nm
        * (fused_silica_index(wave_nm) - 1)
        / (wave_nm * (fused_silica_index(blaze_nm) - 1))
    )
    x = np.pi * (alpha - order)
    den = levels * np.sin(x / levels)
    with np.errstate(divide="ignore", invalid="ignore"):
        array_factor = np.where(np.abs(den) < 1e-12, 1.0, np.sin(x) / den) ** 2
    return np.sinc(order / levels) ** 2 * array_factor


def edge_coating(wave_nm, half, peak=0.95, edge_width_nm=15.0):
    """Transmission of the order-sorting edge coating of a half.

    ``peak`` in band, with a logistic edge whose 10-90% width is
    ``edge_width_nm`` (the memo's assumption; no vendor curve yet).
    """
    design = GRATING_HALVES[half]
    scale = edge_width_nm / (2 * np.log(9))
    x = design["edge_sign"] * (np.asarray(wave_nm, dtype=float) - design["edge_nm"])
    return peak / (1 + np.exp(np.clip(x / scale, -50, 50)))


def box_weights(n, center, half_width):
    """Fractional coverage of each of ``n`` pixels by ``center +- half_width``."""
    idx = np.arange(n)
    lo, hi = center - half_width, center + half_width
    return np.clip(np.minimum(idx + 0.5, hi) - np.maximum(idx - 0.5, lo), 0, 1)


class GratingSimulation:
    """ETC for one half of the slitless grating on one WCC sensor.

    Parameters
    ----------
    simulation : Simulation
        A simulation whose sensor bandpass has no filter (``"zwo:nofilter"`` or
        ``"qcmos:nofilter"``); see :meth:`from_sensor_and_scene`.
    half : str
        ``"blue"`` (350-700 nm; u, g, r) or ``"red"`` (600-1100 nm; i, z).
    orders : tuple of int
        Diffraction orders rendered in the image.
    lines_per_mm, distance_mm, levels : float
        Groove density, grating-to-sensor distance and staircase levels.
    coating_peak, edge_width_nm : float
        In-band transmission and 10-90% edge width of the edge coating.
    psf : PSFSource, optional
        PSF rendered at each wavelength. Default :class:`~wcc_etc.AiryPSF`.
    psf_npix : int
        Side of the (odd) PSF stamp in pixels.
    aperture_mas : float
        Half-width of the extraction box across the dispersion.

    Images are returned with the dispersion along x (pixel rows) for the blue
    half and along y (columns) for the red half, as in the baseline layout.
    """

    def __init__(
        self,
        simulation,
        half="blue",
        orders=(-1, 0, 1, 2),
        lines_per_mm=4.414,
        distance_mm=70.0,
        levels=16,
        coating_peak=0.95,
        edge_width_nm=15.0,
        psf=None,
        psf_npix=65,
        aperture_mas=70.0,
    ):
        if half not in GRATING_HALVES:
            raise ValueError(
                f"half must be one of {sorted(GRATING_HALVES)}, got {half!r}"
            )
        if any(p is not None for p in simulation.scene.call_down("profile")):
            raise NotImplementedError(
                "extended (profiled) scene elements are not supported by the grating"
            )
        self.sim = simulation
        self.half = half
        self.design = GRATING_HALVES[half]
        self.orders = tuple(int(m) for m in orders)
        self.lines_per_mm = float(lines_per_mm)
        self.distance_mm = float(distance_mm)
        self.levels = int(levels)
        self.coating_peak = float(coating_peak)
        self.edge_width_nm = float(edge_width_nm)
        self.psf = AiryPSF() if psf is None else psf
        self.psf_npix = int(psf_npix) | 1
        self.aperture_mas = float(aperture_mas)
        self.wave_nm = np.arange(300.0, 1100.5, 1.0)
        self.cache = {}

    @classmethod
    def from_sensor_and_scene(cls, kind, scene, half="blue", gain_mode=None, **kwargs):
        """Build from a sensor kind (``"zwo"``/``"imx"`` or ``"qcmos"``) and a Scene."""
        sim = Simulation.from_sensor_and_scene(
            f"{kind.split(':')[0]}:nofilter", scene, gain_mode=gain_mode
        )
        return cls(sim, half=half, **kwargs)

    # ---------------------------------------------------------------- geometry
    @property
    def plate_scale_mas(self):
        """Detector plate scale in mas/pixel."""
        sim = self.sim
        return sim.sensor.get_plate_scale(sim.telescope).to("arcsec/pix").value * 1000.0

    @property
    def dispersion_px_per_nm(self):
        """First-order dispersion at the sensor in pixels per nm."""
        pixel_um = self.sim.sensor.pixel_size.value
        return self.distance_mm * self.lines_per_mm * 1e-3 / pixel_um

    def offset_px(self, wave_nm, order=1):
        """Distance from the zeroth-order image along the dispersion, in pixels."""
        return order * self.dispersion_px_per_nm * np.asarray(wave_nm, dtype=float)

    @property
    def resolving_power(self):
        """R = illuminated lines / 1.03 (diffraction-limited, any wavelength)."""
        footprint_mm = self.distance_mm / self.sim.telescope.f_num
        return footprint_mm * self.lines_per_mm / 1.03

    # -------------------------------------------------------------- throughput
    def efficiency(self, order=1, wave_nm=None):
        """Edge coating times staircase efficiency of ``order`` (no telescope/QE)."""
        wave_nm = self.wave_nm if wave_nm is None else wave_nm
        return edge_coating(
            wave_nm, self.half, self.coating_peak, self.edge_width_nm
        ) * staircase_efficiency(wave_nm, order, self.design["blaze_nm"], self.levels)

    def spectral_rates(self):
        """Scene rates before the grating, on ``self.wave_nm``.

        Returns ``(point, sky)``: e-/s/nm of the source plus unresolved
        contaminants, and e-/s/nm/pixel of the surface-brightness elements.
        """
        if "rates" in self.cache:
            return self.cache["rates"]
        sim = self.sim
        pixel_area = (self.plate_scale_mas / 1000.0) ** 2
        area_cm2 = sim.telescope.surface.to_value(u.cm**2)
        wave = self.wave_nm * u.nm
        point = np.zeros_like(self.wave_nm)
        sky = np.zeros_like(self.wave_nm)
        sb = sim.scene.call_down("mag_is_surface_brightness", as_dict=True)
        obs = sim.scene.get_observation(
            band=sim.sensor.bandpass, area=pixel_area, as_dict=True
        )
        for name, o in obs.items():
            if o is None:
                continue
            # PHOTLAM = photon / s / cm^2 / Angstrom, after the no-filter bandpass
            rate = o(wave, flux_unit=su.PHOTLAM).value * area_cm2 * 10.0
            if sb.get(name, False):
                sky += rate
            else:
                point += rate
        self.cache["rates"] = (point, sky)
        return point, sky

    def background_rate_per_pix(self):
        """Sky e-/s/pixel: slitless, so every pixel sees the whole passband.

        All orders land on the sensor and their efficiencies sum to one in
        scalar theory, so only the edge coating applies.
        """
        _, sky = self.spectral_rates()
        coating = edge_coating(
            self.wave_nm, self.half, self.coating_peak, self.edge_width_nm
        )
        return float(trapezoid(sky * coating, self.wave_nm))

    # ------------------------------------------------------------------- image
    def canvas(self):
        """Geometry of the dispersion-along-x image: ``(ny, nx, x0, y0)``.

        ``(x0, y0)`` is the pixel of the zeroth-order image. The canvas spans
        every requested order over the wavelengths the coating passes.
        """
        passed = self.wave_nm[edge_coating(self.wave_nm, self.half) > 1e-3]
        ends = [self.offset_px(w, m) for m in self.orders for w in passed[[0, -1]]]
        pad = self.psf_npix // 2
        x0 = int(np.ceil(-min(ends + [0.0]))) + pad
        nx = x0 + int(np.ceil(max(ends + [0.0]))) + pad + 1
        return self.psf_npix, nx, x0, pad

    def rate_image(self, orders=None, psf_step_nm=25.0):
        """Noiseless e-/s image of the point sources, dispersion along x.

        The trace of each ``psf_step_nm`` wavelength bin is laid down at its
        sub-pixel offsets for every order and convolved with the PSF rendered
        at the bin centre (with the telescope jitter).
        """
        orders = self.orders if orders is None else tuple(orders)
        key = ("image", orders, float(psf_step_nm))
        if key in self.cache:
            return self.cache[key]
        ny, nx, x0, y0 = self.canvas()
        point, _ = self.spectral_rates()
        ctx = ImageSimulator(self.sim, npix=self.psf_npix)._context()
        image = np.zeros((ny, nx))
        bins = np.floor((self.wave_nm - self.wave_nm[0]) / psf_step_nm).astype(int)
        dw = np.gradient(self.wave_nm)
        for b in np.unique(bins):
            sel = bins == b
            wave = self.wave_nm[sel]
            row = np.zeros(nx + 1)
            for m in orders:
                flux = point[sel] * self.efficiency(m, wave) * dw[sel]  # e-/s
                x = x0 + self.offset_px(wave, m)
                ok = (x >= 0) & (x < nx - 1)
                i = np.floor(x[ok]).astype(int)
                frac = x[ok] - i
                np.add.at(row, i, flux[ok] * (1 - frac))
                np.add.at(row, i + 1, flux[ok] * frac)
            if not row.any():
                continue
            trace = np.zeros((ny, nx))
            trace[y0] = row[:nx]
            psf = self.psf.render(
                dataclasses.replace(ctx, wavelength_m=float(wave.mean()) * 1e-9)
            )
            image += fftconvolve(trace, psf, mode="same")
        image = np.clip(image, 0.0, None)
        self.cache[key] = image
        return image

    def oriented(self, image):
        """Rotate a dispersion-along-x image to the half's detector orientation."""
        return image if self.half == "blue" else image.T

    def simulate(self, time, n_reads=1, add_noise=True, seed=None):
        """Detector image of one frame stack of total exposure ``time`` (s).

        Returns a :class:`~wcc_etc.SimulatedImage` (rectangular; ``npix`` is the
        long side). The saturation mask is evaluated per frame (``time /
        n_reads``); read noise is added ``n_reads`` times.
        """
        t = u.Quantity(time, u.s).value
        sensor = self.sim.sensor
        dark = sensor.dark_current.to(u.electron / (u.s * u.pix)).value
        clean = self.oriented(self.rate_image()) + self.background_rate_per_pix() + dark
        clean = clean * t
        if add_noise:
            rng = np.random.default_rng(seed)
            read_noise = sensor.read_noise.to(u.electron / u.pix).value
            image = rng.poisson(clean).astype(float) + rng.normal(
                0.0, read_noise * np.sqrt(n_reads), size=clean.shape
            )
        else:
            image = clean.copy()
        return SimulatedImage(
            image_e=image,
            image_clean=clean,
            saturation_mask=saturation_mask_from_image_e(sensor, image / n_reads),
            gain=sensor.gain.to(u.electron / u.ct).value,
            bias_level=sensor.bias_level.to(u.ct).value,
            npix=max(image.shape),
            pixel_scale_mas=self.plate_scale_mas,
            psf=self.psf,
        )

    # -------------------------------------------------------------- saturation
    def get_peak_pixel(self, time, n_reads=1):
        """Brightest-pixel charge per frame (electrons), all orders included."""
        tf = u.Quantity(time, u.s).value / n_reads
        dark = self.sim.sensor.dark_current.to(u.electron / (u.s * u.pix)).value
        rate = self.rate_image().max() + self.background_rate_per_pix() + dark
        return rate * tf * u.electron

    def is_saturated(self, time, n_reads=1):
        """Whether the brightest pixel of a frame reaches ``sensor.saturation_e``."""
        return self.get_peak_pixel(time, n_reads) >= self.sim.sensor.saturation_e

    # --------------------------------------------------------------------- SNR
    def band_coefficients(self, n_reads=1):
        """Per band ``(A, B, C)`` with ``SNR = A t / sqrt(B t + C)``.

        The box is ``+-aperture_mas`` across the dispersion by the band length
        along the first order. A is the first-order signal rate in the box; B
        adds the light of the other orders, sky and dark in the box; C is the
        read-noise variance of the box pixels times ``n_reads``.
        """
        ny, nx, x0, y0 = self.canvas()
        first = self.rate_image(orders=(1,))
        total = self.rate_image()
        sensor = self.sim.sensor
        dark = sensor.dark_current.to(u.electron / (u.s * u.pix)).value
        read_noise = sensor.read_noise.to(u.electron / u.pix).value
        per_pix = self.background_rate_per_pix() + dark
        wy = box_weights(ny, y0, self.aperture_mas / self.plate_scale_mas)
        out = {}
        for band, (lo, hi) in self.design["bands"].items():
            a, b = x0 + self.offset_px(lo), x0 + self.offset_px(hi)
            w = np.outer(wy, box_weights(nx, (a + b) / 2, (b - a) / 2))
            n_pix = w.sum()
            out[band] = (
                float((first * w).sum()),
                float((total * w).sum() + n_pix * per_pix),
                float(n_pix * read_noise**2 * n_reads),
            )
        return out

    def get_band_snr(self, time, n_reads=1):
        """SNR per band (dict) for a total ``time`` (s) in ``n_reads`` frames."""
        t = u.Quantity(time, u.s).value
        if self.is_saturated(t, n_reads):
            warnings.warn(
                "the brightest pixel saturates; use more reads or a shorter exposure",
                stacklevel=2,
            )
        return {
            band: A * t / np.sqrt(B * t + C)
            for band, (A, B, C) in self.band_coefficients(n_reads).items()
        }

    def get_band_exptime_for_snr(self, snr, n_reads=1):
        """Total exposure time (s) per band (dict) to reach ``snr``."""
        return {
            band: solve_time_for_snr(snr, A, B, C)
            for band, (A, B, C) in self.band_coefficients(n_reads).items()
        }
