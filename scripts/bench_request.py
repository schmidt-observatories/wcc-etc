"""Time one WCC ETC web request (warm, best of N).

The request is Simulation.from_sensorfilter + get_snr + psf_profile +
get_image_snr (monochromatic and polychromatic) + ImageSimulator.simulate +
the 256/5 PSF profile-curve render. Run before and after a change::

    python scripts/bench_request.py              # qcmos:bb (Airy)
    python scripts/bench_request.py --all        # one sensorfilter per focus level
    python scripts/bench_request.py -n 10 zwo:r+1
"""

import argparse
import time
import warnings

import wcc_etc
from wcc_etc import airy
from wcc_etc.io import SENSORFILTER_FOCUS
from wcc_etc.psfsim import ImageSimulator
from wcc_etc.simulation import Simulation

TIME_S = 100


def make_scene():
    return wcc_etc.get_scene(
        name="G5V",
        mag=15,
        background="zodi",
        bandpass="johnson_r",
        background_prop={"bandpass": "johnson_r", "mag": 22.5},
    )


def request(sensorfilter):
    sim = Simulation.from_sensorfilter(sensorfilter, make_scene())
    sim.get_snr(TIME_S)
    sim.psf_profile
    sim.get_image_snr(TIME_S)
    sim.get_image_snr(TIME_S, psf=sim.polychromatic_psf())
    ImageSimulator(sim).simulate(TIME_S, seed=0)
    airy.render_detector_psf(
        sim.effective_wavelength.to("m").value,
        sim.telescope.f_num,
        sim.telescope.diameter_primary.value,
        sim.sensor.pixel_size.value,
        jitter_sigma_mas=sim.telescope.jitter_sigma.to("mas").value,
        n_pixels=256,
        oversample=5,
    )


def bench(sensorfilter, n):
    request(sensorfilter)  # warm: imports, file reads, caches
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        request(sensorfilter)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sensorfilter", nargs="?", default="qcmos:bb")
    ap.add_argument("-n", type=int, default=5, help="repeats (best of N)")
    ap.add_argument(
        "--all", action="store_true", help="one sensorfilter per focus level"
    )
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    if args.all:
        first = {}
        for sf, focus in SENSORFILTER_FOCUS.items():
            first.setdefault(focus, sf)
        targets = [first[k] for k in sorted(first)]
    else:
        targets = [args.sensorfilter]
    for sf in targets:
        best = bench(sf, args.n)
        print(f"{sf:>10s} ({SENSORFILTER_FOCUS[sf]}): best of {args.n} = {best:.3f} s")


if __name__ == "__main__":
    main()
