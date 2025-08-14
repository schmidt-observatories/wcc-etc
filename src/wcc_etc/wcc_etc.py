import toml
import math
from astropy import units as u
from synphot import units, SourceSpectrum, SpectralElement, Observation, Empirical1D
from synphot.models import BlackBodyNorm1D, GaussianFlux1D, Box1D
import numpy as np
from numpy import sqrt
from scipy.interpolate import interp1d
import pandas as pd
from math import ceil, floor, log10
import os
import matplotlib.pyplot as plt

#   One line function prepends the support path to variable s2 if s1 is provided, else returns s2 as is.
prepend_if_not_none = lambda s1, s2: f"{s1}{s2}" if s1 is not None else s2

class WCCETC(object):

    def __init__(self,config_file: str):
        """
        Initialize the WCC ETC with a configuration file.

        EXAMPLE:
            WCC = wcc_etc.WCCETC("../config/config.toml")
            WCC.setup()
        """
        print('Initializing WCC ETC. Reading in Config')
        self.config = toml.load(config_file)

        # Load config
        self._load_config()

        # Perform relevant calculations

    def _load_config(self):
        """
        Load the configuration from the config file.
        """
        # Load telescope configuration
        self.tel_config = self.config['telescope']
        self.bandpass = SpectralElement(Box1D, amplitude=1, x_0=6000, width=7000)
        self.qe_curves = []
        self.filters = []
        self.num_mirrors = 0
        self.f_num = self.config['telescope']['f_num'] # f-number
        self.diameter_primary = self.config['telescope']['diameter_primary'] * u.m
        self.surf_area = np.pi * (0.5 * self.diameter_primary) ** 2
        self.focal_len = self.diameter_primary * self.f_num
        self.gain_setting = self.config['detector']['gain_setting']
        self.read_noise = self.config['detector']['read_noise']
        self.dark_current = self.config['detector']['dark_current']
        self.pixel_size = self.config['detector']['pixel_size'] * u.um/u.pix
        self.plate_scale = (self.pixel_size.value * 1e-6 /self.diameter_primary.value / self.f_num * 206265) # arcsec/pix
        self.path_qe = self.config['detector']['path_qe']
        self.path_m1_coating = self.config['telescope']['path_m1_coating']
        self.path_m2_coating = self.config['telescope']['path_m2_coating']
        self.path_m3_coating = self.config['telescope']['path_m3_coating']
        self.path_m4_coating = self.config['telescope']['path_m4_coating']
        self.path_filter = self.config['telescope']['path_filter']
        self.sensor_area = self.config['detector']['sensor_area'] * u.mm**2
        self.sensor_temp = self.config['detector']['sensor_temp'] * u.Celsius
        self.bg_surface_brightness = self.config['zodi']['zodi_mag_r']

    def setup(self,plot=True,verbose=True):
        """
        Set up the WCC ETC with the current configuration.
        """
        print('Setting up WCC ETC with current configuration')
        # Here you can add any setup code that needs to be run
        # For example, you might want to initialize some variables or load some data
        self.add_mirror(self.path_m1_coating, num_curves=1, wave_unit='nm', plot=plot, plot_title="AL")
        self.add_mirror(self.path_m1_coating, num_curves=1, wave_unit='nm', plot=plot, plot_title="AL")
        self.add_mirror(self.path_m1_coating, num_curves=1, wave_unit='nm', plot=plot, plot_title="AL")
        self.add_mirror(self.path_m1_coating, num_curves=1, wave_unit='nm', plot=plot, plot_title="AL")
        self.add_filter(self.path_filter,plot=plot)
        self.add_sensor(num_curves=1, plot=plot)
        self.calc_PSF(wavelength=None, approx_type='sq', verbose=verbose)
        if verbose:
            self.describe()

    def calc_PSF(self, wavelength=None, approx_type='sq', verbose=True):
            """
            Calculate the mean PSF based on the mean wavelength of combined Spectral Elements
    
            INPUT:
                wavelength - 
    
            # OR from a user-selected wavelength
            """
            
            if not wavelength == None:
                psf_diameter = 2*1.22*wavelength*self.f_num
            else:
                wavelength = self.bandpass.wpeak()
                self.qe_wpeak = wavelength
                psf_diameter = (2*1.22*wavelength*self.f_num)
                
            self.psf_diameter = psf_diameter.to('um')
            
            # determine total number of pixels with the psf
            n = ceil((self.psf_diameter / self.pixel_size).value)
            
            if approx_type.lower() in ['s', 'sq', 'square']:  # Gives result of nxn square
                self.num_psf_pixels = n**2 * (u.pix)
            elif approx_type.lower() in ['c', 'circ', 'circle', 'circular']:  # Gives result of all pixels with centers within a circle of radius n
                if n%2 == 0:  # n even
                    L = int(n/2)
                    sequence = [floor(0.5 + 0.5*sqrt(n**2 - (2*y-1)**2)) for y in list(range(1,L+1))]
                    self.num_psf_pixels = 4 * sum(sequence) * (u.pix)
                elif n%2 == 1:  # n odd
                    L = int((n-1)/2)
                    sequence = [floor(1 + 0.5*sqrt(n**2 - 4*(y**2))) for y in list(range(1,L+1))]
                    self.num_psf_pixels = 1 + 4 * sum(sequence) * (u.pix)
    
            #   Total PSF Area = Area of 1 pixel (arcsecond * arcsecond)/pixel * num of pixels (pixels)
            #   Gives area in square arcseconds IF self.plate_scale is in arcseconds/pixel
            self.psf_area = (self.plate_scale*self.plate_scale) * self.num_psf_pixels
    
            #   Calculate the total magnitude of the background spectrum given the PSF area (in square arcseconds)
            self.bg_magnitude = calculate_bg_normalization_magnitude(self.bg_surface_brightness,self.psf_area.value)
    
            if verbose:
                print(f"Number of PSF pixels: {self.num_psf_pixels}")
            return self.psf_diameter, self.num_psf_pixels
    
    def add_qe_curve(self, qe_fits_file, wave_unit='nm', num_curves=1, plot=False, ax = None):
        """
        Add Quantum Efficiency properties

        INPUT:
            qe_fits_file - filename of .csv file
            wave_unit - 'nm'
            num_curves - 1
            plot - plot

        NOTES:


        EXAMPLE:
            WCC.add_qe_curve("../data/support_data/sensors/ZWO_ASI6200MM/ZWO_ASI6200MM_Pro_QE_curve.csv", wave_unit='angstrom', num_curves=1, plot=True)
        """
        self.qe_curves.append(f"{qe_fits_file} x {num_curves}")
        bp = SpectralElement.from_file(qe_fits_file, wave_unit=wave_unit)
        for num in range(num_curves-1):
            bp *= SpectralElement.from_file(qe_fits_file, wave_unit=wave_unit)
        self.bandpass *= bp

        if plot==True:
            if ax is None:
                fig, ax = plt.subplots()
            w, y = bp._get_arrays(None)
            ax.plot(w,y,label='QE')
            ax.set_xlabel('Wavelength ({})'.format(wave_unit))
            ax.set_ylabel('Quantum Efficiency')
            ax.set_title('Quantum Efficiency')

    def add_sensor(self, num_curves=1
                    , gain_setting= None #100  # (0.1 dB)
                    , sensor_temp = None #0 * u.Celsius
                    , sensor_area = None
                    , sensor_pixel_size = None
                    , gain = None
                    , dark_current = None
                    , read_noise = None
                    , well_depth = None
                    , sensor_toml = None
                    , support_data_path=None
                    , plot=False
                    , plot_title="Sensor"
                    ):
            """
            Adding sensor

            INPUT:
                
            """
            if sensor_toml is None:
                sensor_toml = self.config['detector']
            if gain_setting is None:
                gain_setting = self.gain_setting
            if sensor_temp is None:
                sensor_temp = self.sensor_temp

            # Adding gain
            gain_cols = ['gain_setting', 'gain']
            self.gain = get_interpolated_value(prepend_if_not_none(support_data_path, sensor_toml['path_gain_curve']), gain_setting, gain_cols) * (u.electron / u.ct)

            dark_current_cols = ['sensor_temperature', 'dark_current']
            self.dark_current = get_interpolated_value(prepend_if_not_none(support_data_path, sensor_toml['path_dark_current']), sensor_temp, dark_current_cols) * (u.electron / (u.s * u.pix))

            read_noise_cols = ['gain_setting', 'read_noise']
            self.read_noise = get_interpolated_value(prepend_if_not_none(support_data_path, sensor_toml['path_read_noise']), gain_setting, read_noise_cols) * sqrt(1.0 * u.electron / u.pix)

            well_depth_cols = ['gain_setting', 'well_depth']
            self.well_depth = get_interpolated_value(prepend_if_not_none(support_data_path, sensor_toml['path_well_depth']), gain_setting, well_depth_cols) * (u.electron / u.pix)

            self.add_qe_curve(self.path_qe, wave_unit='nm', num_curves=1, plot=plot)


    def add_mirror(self, mirror_qe_fits_file, num_curves=1 ,wave_unit='nm', support_data_path=None, plot=False,plot_title="Mirror"):
        """
        Add mirror

        INPUT:
            mirror_qe_fits_file - filename of .fits file
            num_curves - number of curves to multiply
            wave_unit - 'nm'

        EXAMPLE:
            WCC.add_mirror("../data/support_data/coatings/NIST_1st_surface_Al.csv", num_curves=1, wave_unit='nm', plot=True, plot_title="AL")
        """
        self.num_mirrors += num_curves
        mirror_qe_fits_file = prepend_if_not_none(support_data_path,mirror_qe_fits_file)
        self.qe_curves.append(f"{mirror_qe_fits_file} x {num_curves}")
        
        bp = SpectralElement.from_file(mirror_qe_fits_file, wave_unit=wave_unit)
        for num in range(num_curves-1):
            bp *= SpectralElement.from_file(mirror_qe_fits_file, wave_unit=wave_unit)
        self.bandpass *= bp
        
        if plot==True:
            bp.plot(title=plot_title)


    def add_filter(self, filter_fits_file, wave_unit='angstrom', num_curves=1, support_data_path=None, plot=False, plot_title="Filter"):
            """
            Add filter properties

            INPUT:
                filter_fits_file - filename of .fits file
                wave_unit - 'angstrom'
                num_curves - 1
                support_data_path - path to support data files
                plot - plot the filter curve
                plot_title - title for the plot
                
            EXAMPLE:
                WCC.add_filter("../data/support_data/filters/sdss_r_005_syn.fits",plot=True)
            """
            filter_fits_file = prepend_if_not_none(support_data_path, filter_fits_file)
            self.filters.append(f"{filter_fits_file} x {num_curves}")
            
            bp = SpectralElement.from_file(filter_fits_file)

            for num in range(num_curves-1):
                bp *= SpectralElement.from_file(filter_fits_file)

            self.bandpass *= bp
            
            if plot==True:
                bp.plot(title=plot_title)

    def describe(self):
        """
        Describe summary of the system
        """
        print('###################################################')
        print('# Main')
        print('Diameter Primary: {:20.2f}'.format(self.diameter_primary))
        print('Fnum:             {:20.1f}'.format(self.f_num))
        print('Focal Length:     {:20.2f}'.format(self.focal_len))
        print('Num mirrors:      {:20.1f}'.format(self.num_mirrors))
        print('Surf Area:        {:20.2f}m2'.format(self.surf_area))
        print('')
        print('# Detector')
        print('Gain Setting:     {:20.1f}'.format(self.gain_setting))
        print('Gain:             {:20.4f}'.format(self.gain))
        print('Sensor Area:      {:20.1f}mm2'.format(self.sensor_area))
        print('Pixel Size:       {:20.3f}'.format(self.pixel_size))
        #print('Num Pixels:       {:20.1f}'.format(self.num_pixels))
        print('Read Noise:       {:20.3f}'.format(self.read_noise))
        print('Dark Current:     {:20.4f}'.format(self.dark_current))
        print('Well depth:       {:20.1f}'.format(self.well_depth))
        print('# Other')
        print('Plate Scale:      {:20.3f}'.format(self.plate_scale))
        print('Num PSF Pixels:   {:20.1f}'.format(self.num_psf_pixels))
        #print('Jitter RMS:       {}'.format(self.jitter_rms))
        print('###################################################')

    def set_source(self, source_pickles_file, source_z=0, support_data_path=None, plot=False, ax = None):
        """
        Set source from a Pickle spectrum.

        INPUT:
            source_pickles_file - filename of .fits file

        NOTES:
        Source: https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/pickles-atlas.html

        Source Spectrum: Pickles Stellar Atlas
        filename        sptype    T_eff
        --------------  --------  -------
        pickles_uk_1	O5V	      39810.7
        pickles_uk_2	O9V	      35481.4
        pickles_uk_3	B0V	      28183.8
        pickles_uk_4	B1V	      22387.2
        pickles_uk_5	B3V	      19054.6
        pickles_uk_6	B5-7V	  14125.4
        pickles_uk_7	B8V	      11749.0
        pickles_uk_9	A0V	      9549.93
        pickles_uk_10	A2V	      8912.51
        pickles_uk_11	A3V	      8790.23
        pickles_uk_12	A5V	      8491.80
        pickles_uk_14	F0V	      7211.08
        pickles_uk_15	F2V	      6776.42
        pickles_uk_16	F5V	      6531.31
        pickles_uk_20	F8V	      6039.48
        pickles_uk_23	G0V	      5807.64
        pickles_uk_26	G2V	      5636.38 ***
        pickles_uk_27	G5V	      5584.70
        pickles_uk_30	G8V	      5333.35
        pickles_uk_31	K0V	      5188.00
        pickles_uk_33	K2V	      4886.52
        pickles_uk_36	K5V	      4187.94
        pickles_uk_37	K7V	      3999.45
        pickles_uk_38	M0V	      3801.89
        pickles_uk_40	M2V	      3548.13
        pickles_uk_43	M4V	      3111.72 ***
        pickles_uk_44	M5V	      2951.21
        pickles_uk_46	B2IV	  19952.6
        pickles_uk_47	B6IV	  12589.3
        pickles_uk_48	A0IV	  9727.47
        pickles_uk_49	A4-7IV	  7943.28
        pickles_uk_50	F0-2IV	  7030.72
        pickles_uk_51	F5IV	  6561.45
        pickles_uk_52	F8IV	  6151.77
        pickles_uk_53	G0IV	  5929.25
        pickles_uk_54	G2IV	  5688.53
        pickles_uk_55	G5IV	  5597.57
        pickles_uk_56	G8IV	  5308.84
        pickles_uk_57	K0IV	  5011.87
        pickles_uk_58	K1IV	  4786.30
        pickles_uk_59	K3IV	  4570.88
        pickles_uk_60	O8III	  31622.8
        pickles_uk_61	B1-2III	  19952.6
        pickles_uk_63	B5III	  14791.1
        pickles_uk_64	B9III	  11091.8
        pickles_uk_65	A0III	  9571.94
        pickles_uk_67	A5III	  8452.79
        pickles_uk_69	F0III	  7585.78
        pickles_uk_71	F5III	  6531.31
        pickles_uk_72	G0III	  5610.48
        pickles_uk_73	G5III	  5164.16
        pickles_uk_76	G8III	  5011.87
        pickles_uk_78	K0III	  4852.89
        pickles_uk_87	K3III	  4365.16
        pickles_uk_93	K5III	  4008.67
        pickles_uk_95	M0III	  3819.44
        pickles_uk_100	M5III	  3419.79
        pickles_uk_105	M10III	  2500.35
        pickles_uk_106	B2II	  15995.6
        pickles_uk_107	B5II	  12589.3
        pickles_uk_108	F0II	  7943.28
        pickles_uk_109	F2II	  7328.25
        pickles_uk_110	G5II	  5248.07
        pickles_uk_111	K0-1II	  5011.87
        pickles_uk_112	K3-4II	  4255.98
        pickles_uk_113	M3II	  3411.93
        pickles_uk_114	B0I	      26001.6
        pickles_uk_117	B5I	      13396.8
        pickles_uk_118	B8I	      11194.4
        pickles_uk_119	A0I	      9727.47
        pickles_uk_121	F0I	      7691.30
        pickles_uk_122	F5I	      6637.43
        pickles_uk_123	F8I	      6095.37
        pickles_uk_124	G0I	      5508.08
        pickles_uk_126	G5I	      5046.61
        pickles_uk_127	G8I	      4591.98
        pickles_uk_128	K2I	      4255.98
        pickles_uk_130	K4I	      3990.25
        pickles_uk_131	M2I	      3451.44
        """
        pickles_file_path = prepend_if_not_none(support_data_path,source_pickles_file)
        self.source_spectrum = SourceSpectrum.from_file(pickles_file_path)

        self.source_spectrum.z = source_z
        self.source_name = source_pickles_file
        self.source_z = source_z

        w, y = self.source_spectrum._get_arrays(None)

        if plot==True:
            if ax is None:
                fig, ax = plt.subplots()
            basename = os.path.basename(source_pickles_file)
            ax.plot(w,y,label='{}'.format(basename))
            ax.set_xlabel('Wavelength [A]')
            ax.set_ylabel('Flux')
            #self.source_spectrum.plot()
        return w, y

    def set_background(self, background_file, support_data_path=None,  plot=False):
        """
        Set background

        INPUT:

        """

        # Create background for observation
        # info from: https://etc.stsci.edu/etcstatic/users_guide/1_ref_9_background.html
        if support_data_path is not None:
            background_file = prepend_if_not_none(support_data_path, background_file)
        self.background_spectrum = SourceSpectrum.from_file(background_file)
        self.background_name = background_file

        if plot==True:
            self.background_spectrum.plot()
    
    def make_observation(self, flux=0, flux_units=u.ABmag, bg_flux=None, bg_flux_units=u.ABmag, plot=False, verbose=False):
        """
        Create observation using source and return countrate.
        This will scale the source and the background by the magnitudes so the fluxes are correct

        INPUT:
            flux - magnitude
            flux_unit - AB Magnitude or VEGA magnitude
            bg_flux - background 
            plot - True/False

        OUTPUT:
            source_counts
            background_counts

        NOTES:
        """
        #   Calculate Background Normalization magnitude
        if bg_flux is None:
            if self.bg_magnitude is None:
                try:
                    self.calculate_bg_normalization_magnitude()
                except:
                    print("Error: Could not calculate Background magnitude.")
                    exit()
            bg_flux = self.bg_magnitude

        # Add Source
        self.source_spectrum.z = self.source_z  # make sure source spectra has proper redshift

        if flux_units in ['vega', units.VEGAMAG]:
            vega = SourceSpectrum.from_vega()  # For unit conversion
            normalization_units = flux * units.VEGAMAG
            sp_rn = self.source_spectrum.normalize(normalization_units
                                                      , self.bandpass
                                                      , vegaspec=vega
                                                      # , force='taper'
                                                      , force='extrap'
                                                     )

        elif flux_units in ['AB', 'ABmag', 'AB mag', 'AB magnitude', u.ABmag]:
            normalization_units = flux * u.ABmag
            sp_rn = self.source_spectrum.normalize(normalization_units
                                                      , self.bandpass
                                                      # , vegaspec=vega
                                                      # , force='taper'
                                                      , force='extrap'
                                                     )
        else:
            raise NotImplementedError("User-defined source flux units not currently implemented")

        self.sp_obs = Observation(sp_rn, self.bandpass, force='extrap')

        if plot==True:
            self.sp_obs.plot(title='Source')

        # Get countrate for observation
        source_counts = self.sp_obs.countrate(area=self.surf_area) * u.electron/u.ct
        self.source_counts = source_counts    #    Makes the counts in units of e/s

        # Add Background
        # Background needs to be normalized in the Johnson V band
        johnson_v_passband =  SpectralElement.from_filter('johnson_v')
        bg_rn = self.background_spectrum.normalize( bg_flux * bg_flux_units
                                                      , johnson_v_passband
                                                      #, vegaspec=vega
                                                      # , force='taper'
                                                      , force='extrap'
                                                     )

        bg_obs = Observation(bg_rn, self.bandpass, force='extrap')

        if plot==True:
            bg_obs.plot(title='Background')

        # Get countrate for observation
        background_counts = bg_obs.countrate(area=self.surf_area)* u.electron/u.ct
        self.sky_counts = background_counts

        if verbose:
            print('Source Counts: {}'.format(self.source_counts))
            print('Background Counts: {}'.format(self.sky_counts))

        return source_counts, background_counts

    def calc_saturation_time(self):
        """
        # Calculates the time for any one pixel on a sensor to completely fill it's well-depth.

        NOTES:
        # NOTE: Requires setting of source and/or background and performing 'make_observation'
        """
        return self.well_depth / ( (self.source_counts / self.num_psf_pixels) + (self.sky_counts / self.num_psf_pixels) )

    def calc_SNR(self, int_time, exp_time):
        """
        Calculate SNR for a given total integration time with set frame exposure times

        INPUT:
            int_time - 
            exp_time - ??

        NOTES:
            # NOTE: Requires setting of source and/or background and performing 'make_observation'
        """

        total_noise = sqrt(self.source_counts*int_time + self.sky_counts*int_time + (self.dark_current + self.read_noise*self.read_noise/exp_time)*int_time*self.num_psf_pixels)
        snr = self.source_counts*int_time / total_noise

        return snr.value
    
    def calc_int_time(self, snr, exp_time):
        """
        Calculate total integration time to achieve a given snr and frame exposure time

        NOTES:
            # NOTE: Requires setting of source and/or background and performing 'make_observation'
        """
        
        snr = snr * sqrt(1.0 * u.ct)  # to ensure units match
        A = ((self.source_counts/snr)**2) * exp_time
        B = self.source_counts*exp_time + self.sky_counts*exp_time + ( self.dark_current*exp_time + self.read_noise*self.read_noise)*self.num_psf_pixels
        int_time = B/A

        return int_time * u.electron / u.ct

def calculate_bg_normalization_magnitude(bg_surface_brightness, psf_area):
    """
    Convert the Background Surface Brightness into the total magnitude given the PSF area (in arcseconds squared)
    The area needs to be in square arcseconds since this the typical definition of Surface Brightness is in units
    of magnitudes per arcseconds^2
    :return: None
    """
    bg_magnitude = bg_surface_brightness - 2.5 * np.log10(psf_area)
    return bg_magnitude

def get_interpolated_value(input_file, interpolation_xval, col_headers):
    """
    Interpolate the given file columns to get the value at interpolation_xval

    INPUT:
        :param input_file: File to use x columns and y columns on
        :param interpolation_xval: Value that the interpolation function takes as argument
        :param col_headers: Names of column headers as a list

    OUTPUT:
        :return: The value of the interpolated function at interpolation_xval
    """
    df = pd.read_csv(input_file, skiprows=1, names=[col_headers[0], col_headers[1]])
    xlist = df[col_headers[0]]
    ylist = df[col_headers[1]]
    interp = interp1d(xlist, ylist)

    return interp(interpolation_xval)


if __name__ == '__main__':
    print('Main')