from bokeh.plotting import figure
import scipy.interpolate
from bokeh.embed import components
import numpy as np
from wcc_etc.airy import get_airy_and_ee_curve

from flask import Flask, request, render_template
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from wcc_etc.wcc_etc import WCCETC
import astropy.units as u

app = Flask(__name__)

# List available config files in /config
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
config_files = [f for f in os.listdir(CONFIG_DIR) if f.endswith('.toml')]



@app.route('/', methods=['GET', 'POST'])
def index():
    snr = None
    photometric_precision = None
    error = None
    total_flux_e = None
    flux_e = None
    bg_flux_e = None
    bg_mag_out = None
    config_description = None
    ee_script, ee_div = '', ''
    airy_script, airy_div = '', ''
    throughput_script, throughput_div = '', ''
    source_script, source_div = '', ''
    selected_config = request.form.get('config_file') if request.method == 'POST' else config_files[0] if config_files else None
    source_files = WCCETC.get_source_files()
    selected_source = request.form.get('source_file') if request.method == 'POST' else WCCETC.get_default_source_file() if source_files else None
    print(f'Selected config file: {selected_config}')
    print(f'Selected source file: {selected_source}')
    config_path = os.path.join(CONFIG_DIR, selected_config) if selected_config else None
    source_path = os.path.join(WCCETC.SOURCE_DIR, selected_source) if selected_source and not os.path.isabs(selected_source) else selected_source
    if config_path:
        try:
            wcc = WCCETC(config_path)
            wcc.setup()
            config_description = wcc.describe()
        except Exception as e:
            config_description = f'Error loading config: {e}'

    # Default values for plots if not POST
    mag = float(request.form['mag']) if request.method == 'POST' else 26
    r_aper_mas = float(request.form['r_aper_mas']) if request.method == 'POST' else 70
    exp_time = float(request.form['exp_time']) if request.method == 'POST' else 60
    jitter = float(request.form['jitter']) if request.method == 'POST' else 0

    # Get wavelength and D from config (fallbacks if missing)
    wavelength_eff = getattr(wcc, 'wavelength')
    print(f'Wavelength: {wavelength_eff} m')
    D = getattr(wcc, 'diameter_primary')
    fnum = getattr(wcc, 'f_num')
    pixel_size = getattr(wcc, 'pixel_size')
    plate_scale = getattr(wcc, 'plate_scale') * 1000 # mas / pix
    NPIX = 15

    # Generate EE, Airy, Throughput, and Source plots using Bokeh
    try:
        r_mas, psf1d, ee, ee_aper = get_airy_and_ee_curve( wavelength=wavelength_eff, r_aper_mas=r_aper_mas, grid_size=1024,
                                                           extent_mas=500, verbose=False, jitter_sigma_mas=jitter,
                                                           plot=False, pixel_size=pixel_size, fnum=fnum, D=D)
        # EE curve plot
        p1 = figure(title="EE vs Radius. Grey vertical lines show pixel boundaries", x_axis_label="Radius [mas]", y_axis_label="Encircled Energy", width=500, height=350, x_range=(0, 300))
        p1.line(r_mas, ee, line_width=2, color="navy", legend_label="EE curve")
        p1.line([r_aper_mas, r_aper_mas], [0, ee_aper], line_dash="dashed", color="orange", legend_label="Aperture radius")
        p1.line([0, r_aper_mas], [ee_aper, ee_aper], line_dash="dashed", color="orange")
        p1.legend.location = "bottom_right"
        p1.xgrid.grid_line_color = None
        p1.ygrid.grid_line_color = None
        for i in range(NPIX):
            p1.line([i * plate_scale, i * plate_scale], [0, max(psf1d)], color="gray",alpha=0.5,line_width=0.3)# legend_label=f"{i} pixel{'s' if i!=1 else ''} ({i*plate_scale:.1f} mas)")
        ee_script, ee_div = components(p1)

            # Airy disk plot
        p2 = figure(title="Airy Disk vs Radius. Grey vertical lines show pixel boundaries", x_axis_label="Radius [mas]", y_axis_label="Normalized Flux", width=500, height=350, x_range=(0, 300))
        p2.line(r_mas, psf1d, line_width=2, color="green", legend_label="Airy disk")
        p2.line([r_aper_mas, r_aper_mas], [0, max(psf1d)], line_dash="dashed", color="orange", legend_label="Aperture radius")
        p2.legend.location = "top_right"
        p2.xgrid.grid_line_color = None
        p2.ygrid.grid_line_color = None
        for i in range(NPIX):
            p2.line([i * plate_scale, i * plate_scale], [0, max(psf1d)], color="gray",alpha=0.5,line_width=0.3)# legend_label=f"{i} pixel{'s' if i!=1 else ''} ({i*plate_scale:.1f} mas)")
        airy_script, airy_div = components(p2)


        # Final throughput plot
        wave, throughput = wcc.get_final_throughput_curve(wave_unit='nm')
        p3 = figure(title="Final Throughput Curve", x_axis_label="Wavelength [A]", y_axis_label="Throughput", width=500, height=350, x_range=(2000, 18000))
        p3.line(wave, throughput, line_width=2, color="orange", legend_label="Throughput")
        # Add vertical line at effective wavelength
        if wavelength_eff is not None:
            try:
                # Convert wavelength to Angstroms for plot (if needed)
                eff_wave_angstrom = float(wavelength_eff) * 1e10
                print(eff_wave_angstrom)
                eff_wave_y = scipy.interpolate.interp1d(wave, throughput)(eff_wave_angstrom)
                p3.line([eff_wave_angstrom, eff_wave_angstrom], [0, eff_wave_y], line_dash="dashed", color="red", legend_label="Effective Wavelength, {:.1f} A".format(eff_wave_angstrom))
                eff_wave_angstrom = float(f'{eff_wave_angstrom:.1f}')
            except Exception as e:
                print(f"Error plotting effective wavelength line: {e}")
        p3.legend.location = "top_right"
        throughput_script, throughput_div = components(p3)

    except Exception as e:
        print(f'Error generating Bokeh plots: {e}')

    if request.method == 'POST':
        print('Received POST request')
        photometric_precision = 0.
        try:
            wcc.set_source(source_path, plot=False)
            # Source spectrum plot

            wcc.set_background(
                background_file=WCCETC.get_default_background_file(),
                support_data_path=WCCETC.get_support_data_path(),
                plot=False
            )

            # Get background magnitude from form
            bg_mag = request.form.get('bg_mag', None)
            bg_flux = float(bg_mag) if bg_mag not in (None, '', 'None') else None

            snr = wcc.make_observation_and_calc_SNR(
                flux=mag,
                r_aper_mas=r_aper_mas,
                texp=exp_time,
                jitter_sigma_mas=jitter,
                bg_flux=bg_flux,
                plot=False,
                verbose=False
            )
            snr = snr.value if hasattr(snr, 'value') else snr
            if snr not in (None, 0):
                photometric_precision = 1e6/snr # in ppm
                photometric_precision = f"{photometric_precision:.1f}"
                snr = f"{snr:.2f}"
            else:
                photometric_precision = None
                snr = None
            print(f'Calculated SNR: {snr}')

            # Get total flux in electrons
            flux_e = None
            if hasattr(wcc, 'count_rate_e_per_s'):
                flux_e = wcc.count_rate_e_per_s
                if hasattr(flux_e, 'value'):
                    flux_e = flux_e.value
                flux_e = f"{flux_e:.2e} electrons/s"
            # Get background flux in electrons
            bg_flux_e = None
            if hasattr(wcc, 'sky_counts_e_per_s'):
                bg_flux_e = wcc.sky_counts_e_per_s
                if hasattr(bg_flux_e, 'value'):
                    bg_flux_e = bg_flux_e.value
                bg_flux_e = f"{bg_flux_e:.2e} electrons/s"
            # Get background magnitude
            bg_mag_out = None
            if hasattr(wcc, 'bg_magnitude'):
                bg_mag_out = f"{wcc.bg_magnitude:.2f}"

            swave, sflux = wcc.get_source_spectrum_curve()
            if swave is not None and sflux is not None:
                # Convert astropy Quantities to plain lists for Bokeh
                if hasattr(swave, 'value'):
                    swave = swave.value.tolist()
                if hasattr(sflux, 'value'):
                    sflux = sflux.value.tolist()
                p4 = figure(title="Source Spectrum", x_axis_label="Wavelength [A]", y_axis_label="Flux", width=500, height=350, x_range=(3000, 18000))
                p4.line(swave, sflux, line_width=2, color="purple", legend_label="Source Spectrum")
                p4.legend.location = "top_right"
                source_script, source_div = components(p4)
        except Exception as e:
            error = f'Error: {e}'
            print(f'Exception occurred: {error}')
            photometric_precision = None
    return render_template('index.html', snr=snr, photometric_precision = photometric_precision, error=error, config_files=config_files, selected_config=selected_config, config_description=config_description, ee_script=ee_script, ee_div=ee_div, airy_script=airy_script, airy_div=airy_div, throughput_script=throughput_script, throughput_div=throughput_div, source_script=source_script, source_div=source_div, source_files=[os.path.join(WCCETC.SOURCE_DIR, f) for f in source_files], selected_source=source_path, total_flux_e=flux_e, bg_flux_e=bg_flux_e, bg_mag_out=bg_mag_out, eff_wave_angstrom=eff_wave_angstrom)

if __name__ == '__main__':
    app.run(debug=True)
