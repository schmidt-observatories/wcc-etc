"""Patch the get_snr usages in tracked notebooks for the 2D-default get_snr.

get_snr now runs the 2D image simulation and returns a dict; ["snr"] is the SNR.
The legacy analytic path is get_snr_airy (deprecated). Cells that are explicitly
about the *analytic* value / round-trip use get_snr_airy; plain SNR values use
get_snr(...)["snr"].
"""

import json
import os

NBDIR = "notebooks"


def load(nb):
    with open(os.path.join(NBDIR, f"{nb}.ipynb")) as f:
        return json.load(f)


def save(nb, d):
    with open(os.path.join(NBDIR, f"{nb}.ipynb"), "w") as f:
        json.dump(d, f, indent=1)
        f.write("\n")


def cell_src(d, i):
    return "".join(d["cells"][i]["source"])


def set_cell(d, i, text):
    # store as a single string list element split on lines preserving newlines
    lines = text.splitlines(keepends=True)
    d["cells"][i]["source"] = lines


def replace_in_cell(d, i, old, new, count=1):
    src = cell_src(d, i)
    n = src.count(old)
    assert n == count, f"cell {i}: expected {count} occurrence(s) of {old!r}, found {n}"
    set_cell(d, i, src.replace(old, new))


# ---------------- 01_getting_started ----------------
d = load("01_getting_started")
replace_in_cell(
    d,
    7,
    "`get_snr(time)` returns the SNR for an exposure time in seconds. The time can be a scalar or\n"
    "an array — it broadcasts — so a single call gives you an SNR-vs-time curve.",
    '`get_snr(time)` runs the 2-D image simulation and returns a dict; `["snr"]` is the\n'
    "signal-to-noise ratio. The time can be a scalar or an array — it broadcasts — so a\n"
    "single call gives you an SNR-vs-time curve.",
)
replace_in_cell(
    d, 8, "snr_60 = simu.get_snr(time=60)", 'snr_60 = simu.get_snr(time=60)["snr"]'
)
replace_in_cell(
    d,
    8,
    "snr_curve = simu.get_snr(time=times)",
    'snr_curve = simu.get_snr(time=times)["snr"]',
)
replace_in_cell(
    d,
    10,
    "snr_before = float(simu.get_snr(60))",
    'snr_before = float(simu.get_snr(60)["snr"])',
)
replace_in_cell(
    d,
    10,
    "snr_after = float(simu.get_snr(60))",
    'snr_after = float(simu.get_snr(60)["snr"])',
)
save("01_getting_started", d)
print("patched 01")

# ---------------- 04_psf_and_image_snr ----------------
d = load("04_psf_and_image_snr")
replace_in_cell(
    d,
    2,
    "print('analytic SNR @60s, r=15:', round(float(sim.get_snr(60).value), 2))",
    "print('SNR @60s, r=15:', round(float(sim.get_snr(60)[\"snr\"]), 2))",
)
set_cell(
    d,
    17,
    "## 4. PSF-aware SNR cross-check (rendered Airy vs analytic)\n\n"
    "`get_snr` now runs the 2-D image simulation (it delegates to `get_image_snr`). With the "
    "default aperture and the diffraction-limited Airy PSF it should reproduce the legacy "
    "analytic formula — now the deprecated `get_snr_airy` (the small residual is 2-D pixelation "
    "vs the 1-D Airy curve). We assert they agree to within ~3%.",
)
set_cell(
    d,
    18,
    "import warnings\n"
    "sim.update(source__mag=15)\n"
    "with warnings.catch_warnings():\n"
    "    warnings.simplefilter('ignore', DeprecationWarning)\n"
    "    etc_snr = float(sim.get_snr_airy(60).value)   # legacy analytic path\n"
    "img_snr = sim.get_snr(60)['snr']                  # 2-D default (delegates to get_image_snr)\n"
    "diff_pct = 100.0 * (img_snr - etc_snr) / etc_snr\n"
    "\n"
    "print(f'get_snr_airy (analytic Airy) : {etc_snr:.3f}')\n"
    "print(f'get_snr (rendered Airy, 2-D) : {img_snr:.3f}')\n"
    "print(f'difference                   : {diff_pct:+.2f} %')\n"
    "\n"
    "assert abs(diff_pct) < 3.0, f'cross-check off by {diff_pct:.2f}%'\n"
    "print('OK: agree within 3%')",
)
save("04_psf_and_image_snr", d)
print("patched 04")

# ---------------- 05_n_reads_exptime ----------------
d = load("05_n_reads_exptime")
replace_in_cell(
    d,
    3,
    "then feed it back into\n`get_snr` and confirm we recover the target (within ~1%).",
    "then feed it back into\n`get_snr_airy` (the analytic forward that matches the analytic "
    "`get_exptime_for_snr`) and confirm we recover the target (within ~1%).",
)
replace_in_cell(
    d,
    4,
    "snr_back = float(sim.get_snr(t).value)",
    "import warnings\n"
    "with warnings.catch_warnings():\n"
    "    warnings.simplefilter('ignore', DeprecationWarning)\n"
    "    snr_back = float(sim.get_snr_airy(t).value)",
)
replace_in_cell(
    d,
    4,
    "print(f'get_snr(t)        = {snr_back:.4f}')",
    "print(f'get_snr_airy(t)   = {snr_back:.4f}')",
)
replace_in_cell(
    d,
    8,
    "snr_vs_reads.append(float(sim_faint.get_snr(fixed_t).value))",
    'snr_vs_reads.append(float(sim_faint.get_snr(fixed_t)["snr"]))',
)
save("05_n_reads_exptime", d)
print("patched 05")

# ---------------- 06_source_spectra ----------------
d = load("06_source_spectra")
replace_in_cell(
    d,
    11,
    "make_scene(name, m, **extra)).get_snr(60).value) for m in mags]",
    'make_scene(name, m, **extra)).get_snr(60)["snr"]) for m in mags]',
)
replace_in_cell(
    d, 13, "snr_hot = sim.get_snr(60).value", 'snr_hot = sim.get_snr(60)["snr"]'
)
replace_in_cell(
    d, 13, "snr_cool = sim.get_snr(60).value", 'snr_cool = sim.get_snr(60)["snr"]'
)
save("06_source_spectra", d)
print("patched 06")

print("DONE")
