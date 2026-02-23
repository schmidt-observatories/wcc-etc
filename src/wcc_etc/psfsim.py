
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import AiryDisk2D
from astropy import units as u
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.stats import sigma_clipped_stats
import pandas as pd
import scipy.interpolate
import astropy.io.fits
import tifffile
from astropy.io import fits
import os
from astropy.visualization import LogStretch, SqrtStretch, AsinhStretch, HistEqStretch,ZScaleInterval
from astropy.visualization.mpl_normalize import ImageNormalize
import astropy.units as u
from scipy.ndimage import zoom, shift
from scipy.signal import fftconvolve
from scipy.interpolate import UnivariateSpline
from . import airy
from .radial_data import radial_data
import warnings

def howell_center(postage_stamp):
    """
    Howell centroiding, from Howell's Handbook of CCD astronomy

    INPUT:
     postage_stamp - A 2d numpy array to do the centroiding

    OUTPUT:
     x and y center of the numpy array

    NOTES:
    Many thanks to Thomas Beatty and the MINERVAphot.py pipeline for this method
    see here: https://github.com/TGBeatty/MINERVAphot/blob/master/MINERVAphot.py
    """
    xpixels = np.arange(postage_stamp.shape[1])
    ypixels = np.arange(postage_stamp.shape[0])
    I = np.sum(postage_stamp, axis=0)
    J = np.sum(postage_stamp, axis=1)

    Isub = I-np.sum(I)/I.size
    Isub[Isub<0] = 0
    Jsub = J-np.sum(J)/J.size
    Jsub[Jsub<0] = 0
    xc = np.sum(Isub*xpixels)/np.sum(Isub)
    yc = np.sum(Jsub*ypixels)/np.sum(Jsub)
    return xc, yc




def apply_jitter(data,jitter_mas,pixel_scale):
    """
    Apply jitter to the input data.

    INPUT:
        data: 2d data
        jitter_mas: jitter in mas
        pixel_scale: mas/pixel
    """
    total_flux = data.sum()
    sigma_pix = jitter_mas / pixel_scale
    ker = airy.gaussian_kernel_2d(sigma_pix)
    data_blur = fftconvolve(data, ker, mode='same')
    s = data_blur.sum()
    if s > 0:
        data_blur /= s
    return data_blur*total_flux

# make a function that converts a tiff file to a series of fits files
def tiff_to_fits(tiff_file, output_dir):
    """
    Convert a multi-frame TIFF file to a series of FITS files.
    """
    # Read the TIFF file
    image_data = tifffile.imread(tiff_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if the image data is 3D (multiple frames)
    if image_data.ndim == 3:
        for i in range(image_data.shape[0]):
            fits_filename = os.path.join(output_dir, f"frame_{i:03d}.fits")
            fits.writeto(fits_filename, image_data[i], overwrite=True)
            print(f"Saved {fits_filename}")
    else:
        fits_filename = os.path.join(output_dir, "image.fits")
        fits.writeto(fits_filename, image_data, overwrite=True)
        print(f"Saved {fits_filename}")

# make a function that reads in fits files from a directory and averages them to create a master flat
def create_master_flat_from_fits(directory, output_filename):
    fits_files = [f for f in os.listdir(directory) if f.endswith('.fits')]
    if not fits_files:
        raise ValueError("No FITS files found in the specified directory.")

    flat_frames = []
    for f in fits_files:
        with fits.open(os.path.join(directory, f)) as hdul:
            flat_frames.append(hdul[0].data.astype(float))

    master_flat = np.nanmedian(flat_frames, axis=0)
    master_flat = master_flat / np.nanmedian(master_flat)

    # Save the master flat
    hdu = fits.PrimaryHDU(data=master_flat.astype(np.float32))
    hdu.writeto(output_filename, overwrite=True)
    print(f'Master flat saved to {output_filename}')

def apply_nl_scaling(df_nl,data,how='makenonlinear',scale=1):
    """
    Apply non-linear scaling to the input data using the provided scaling factors.

    Parameters:
        df_nl (pd.DataFrame): DataFrame containing 'mean_value' and 'scaling_factor' columns.
        data (np.ndarray): Input data array to be scaled.
        how (str): 'makenonlinear' to apply non-linearity, 'correctnonlinear' to reverse it.

    RETURNS:
        data_scaled (np.ndarray): Scaled data array.

    EXAMPLE:

    NOTES:
        - only applies scaling to positive values; zero or negative values are unchanged.
        - Should only be applied after bias subtraction.
    """
    f_nl = scipy.interpolate.interp1d(df_nl['mean_value'].values,df_nl['scaling_factor'].values*scale,kind='linear',fill_value='extrapolate')
    data_flat = data.flatten()
    m = data_flat > 0
    data_flat_scaled = data_flat.copy()
    if how == 'makenonlinear':
        data_flat_scaled[m] = data_flat[m] / f_nl(data_flat[m])
    elif how == 'correctnonlinear':
        data_flat_scaled[m] = data_flat[m] * f_nl(data_flat[m])
    #data_flat_scaled = data_flat.copy()
    #if how == 'makenonlinear':
    #    data_flat_scaled = data_flat / f_nl(np.abs(data_flat))
    #elif how == 'correctnonlinear':
    #    data_flat_scaled = data_flat * f_nl(np.abs(data_flat))
    data_scaled = data_flat_scaled.reshape(data.shape)
    return data_scaled

class PSFSimulator(object):
    # Resolve support-data paths relative to this module so imports don't fail
    _pkg_dir = os.path.dirname(__file__)
    print(_pkg_dir)
    _path_nonlinearity = os.path.join(_pkg_dir, 'data', 'sensors', 'qCMOS', 'qCMOS_nonlinearity_scaling.csv')
    _path_gain_welldepth = os.path.join(_pkg_dir, 'data', 'sensors', 'ZWO_ASI6200MM', 'ZWO_ASI6200MM_Pro_Well_Depth_vs_Gain_Setting.csv')
    _path_master_flat = os.path.join(_pkg_dir, 'data', 'psfsim', '20251029_qCMOSflats', 'flats_1000nm', 'master_flat_1000nm.fits')
    #FIMG.plot(colorbar=True)

    # Try to read nonlinearity and gain/well-depth tables, but do not raise on import if missing.
    df_nl = pd.read_csv(_path_nonlinearity, comment='#')

    df_gain_welldepth = pd.read_csv(_path_gain_welldepth, names=['gain_setting', 'well_depth_electrons'], skiprows=1)

    def __init__(self,
                 wavelength,
                 diameter,
                 focal_ratio,
                 pixel_size,
                 total_flux,
                 exp_time,
                 dark_current_rate,
                 read_noise_rms,
                 npix,
                 verbose=True,
                 well_depth=16000,
                 flat_scale=1):
        """
        Initialize the PSF Simulator with given parameters.
        
        EXAMPLE:
            PSFSimulator(wavelength=600*u.nm,
             diameter=0.5*u.m,
             focal_ratio=8.0,
             pixel_size=15*u.micron,
             total_flux=1e6,
             exp_time=10*u.s,
             dark_current_rate=0.01*u.electron/u.s,
             read_noise_rms=5.0*u.electron,
             npix=101).generate_ideal_psf(plot=True)
        """
        self.wavelength = wavelength
        self.diameter = diameter
        self.focal_ratio = focal_ratio
        self.pixel_size = pixel_size
        self.total_flux = total_flux
        self.exp_time = exp_time
        self.dark_current_rate = dark_current_rate
        self.read_noise_rms = read_noise_rms
        self.well_depth = well_depth
        self.npix = npix
        self.data = np.zeros((self.npix, self.npix))
        self.data_nonoise = np.zeros((self.npix, self.npix))
        self._y, self._x = np.mgrid[0:self.npix, 0:self.npix]
        self.FIMG_flat = FitsImg(filename=self._path_master_flat)
        self.FIMG_flat.cropcenter(self.npix,self.npix)
        self.data_master_flat = self.FIMG_flat.data
        self.flag_noise = False
        self.flag_jitter = False
        # scale flat
        if verbose:
            print(f"Applying flat scale factor: {flat_scale}")
        self.data_master_flat = self.data_master_flat * flat_scale - np.nanmedian(self.data_master_flat * flat_scale) + 1

        # Calculate derived parameters
        self.focal_length = self.diameter * self.focal_ratio
        self.pixel_scale = airy.calc_plate_scale_from_flength(self.focal_length.value, self.pixel_size.value)
        # Calculate Airy disk radius (first null) in arcseconds
        theta_null = np.rad2deg((1.22 * self.wavelength.to(u.m).value / self.diameter.value))*3600 #.to(u.arcsec)
        # Calculate Airy disk radius in pixels
        self.radius_in_pixels = (theta_null / self.pixel_scale)
        
        if verbose:
            print(f"Pixel Scale: {self.pixel_scale:.4f} arcseconds/pixel")
            print(f"Focal Length: {self.focal_length:.4f} m")
            print(f"Airy Radius (angular): {theta_null:.4f}")
            print(f"Airy Radius (pixels): {self.radius_in_pixels:.4f}")

    def _plot_nonlinearity_curve(self):
        """
        Plot the nonlinearity curve from the loaded DataFrame.
        #PSF.df_nl['Mean Value (e-)'], PSF.df_nl['Scaling Factor']

        """
        fig, ax = plt.subplots(dpi=200)
        ax.plot(self.df_nl['mean_value'], self.df_nl['scaling_factor'], marker='o')
        ax.set_title("Sensor Nonlinearity Scaling Curve")
        ax.set_xlabel("Input Signal (electrons)")
        ax.set_ylabel("Scaling Factor")
        ax.grid(lw=0.3,alpha=0.3)
        ax.set_yscale('log')
        ax.set_xscale('log')

    def get_centroid(self,plot_cross=False,ax=None,plot_lines=False):
        """
        Find centroid using Howell centroiding.

        See phothelp for the method
        """
        self.xcenter, self.ycenter = howell_center(self.data)
        if plot_cross:
            self.plot_data(ax=ax)
            self.ax.scatter(self.xcenter,self.ycenter,marker="+",s=50,color="green")
        if plot_lines:
            self.plot(ax=ax)
            self.ax.hlines(int(self.ycenter),0,self.data.shape[1],color='#1f77b4',lw=1)
            self.ax.vlines(int(self.xcenter),0,self.data.shape[0],color='#1f77b4',lw=1)
        return self.xcenter, self.ycenter

    def apply_jitter(self,jitter_mas,data=None,verbose=True):
        """
        Apply jitter to the input data.

        INPUT:
        - jitter_mas: The amount of jitter to apply in milliarcseconds.
        - data: The data to which jitter will be applied. If None, uses self.data.
        - verbose: If True, prints information about the jitter application.
        """
        self.flag_jitter = True
        if data is None:
            data = self.data
        if verbose:
            print('Applying jitter {}mas'.format(jitter_mas))
        return apply_jitter(data,jitter_mas=jitter_mas,pixel_scale=self.pixel_scale*1000)

    def simulate_psf(self,center=None,jitter_mas=0,apply_nonlinearity=True,verbose=True,nl_scale=10,filename=None,src_micron_per_pixel=4,addnoise=True):
        """
        Generate the ideal PSF based on the current parameters.
        """
        # Center the PSF in the middle of the grid
        if center is None:
            c = (self.npix - 1) / 2.0
            center = (c,c)

        # Instantiate the model
        # The 'radius' parameter is the radius to the first null, in pixels.
        if filename is None:
            if verbose:
                print('Assuming AiryDisk2D PSF')
            self.airy_psf_model = AiryDisk2D( amplitude=1.0,x_0=center[0],y_0=center[1],radius=self.radius_in_pixels)

            # Evaluate the model on the grid
            self.psf = self.airy_psf_model(self._x, self._y)

            # Normalize the PSF so the sum of all pixels is 1
            self.psf /= np.sum(self.psf)

            # Step 6a: Create the mean signal image
            # Scale the normalized PSF by the total flux
            self.data_nonoise = self.psf * self.total_flux
        else:
            self.filename = filename
            self.cpsf = CustomPSF(self.filename, src_micron_per_pix=src_micron_per_pixel, telescope_diameter_m=self.diameter.value, 
                                  fnum=self.focal_ratio, target_pixel_size_micron=self.pixel_size.value)
            self.data_nonoise = self.cpsf.resample_to_grid(npix=self.npix, total_flux=self.total_flux, center=center)

        self.data = np.copy(self.data_nonoise)

        # Apply jitter
        if jitter_mas > 0:
            self.data = self.apply_jitter(jitter_mas=jitter_mas,data=self.data_nonoise,verbose=verbose)
            self.data_nonoise_wjitter = np.copy(self.data)

        # add noise
        if addnoise:
            self.add_noise(verbose=verbose,nl_scale=nl_scale)

        self.psf_m_saturated = self.data > self.well_depth
        self.psf_num_saturated = np.sum(self.psf_m_saturated)
        if self.psf_num_saturated > 0:
            print(f"Warning: {self.psf_num_saturated} pixels exceed the well depth of {self.well_depth} electrons.")
        self.psf_max = np.max(self.data)
        self.psf_95th = np.percentile(self.data,95)
        self.psf_radial_r, self.psf_radial_y, self.psf_hwhm = FitsImg(data=self.data).get_radial_profile(rmax=self.npix/2-1,plot=False,z=2.,return_hwzm=True,annulus_width=1,subtract_min=True)
        if np.size(self.psf_hwhm) > 1:
            print('WARNING psf_hwhm has multiple values, taking first one')
            self.psf_hwhm = self.psf_hwhm[0]
        else:
            self.psf_hwhm = float(self.psf_hwhm)
        self.psf_fwhm = self.psf_hwhm * 2
        self.psf_fwhm_mas = self.psf_fwhm * self.pixel_scale * 1000

        if verbose:
            print(f"PSF max: {self.psf_max:.2f} e-")
            print(f"PSF 95th percentile: {self.psf_95th:.2f} e-")
            print(f"PSF HWHM: {self.psf_hwhm:.2f} pixels")
            print(f"PSF FWHM: {self.psf_fwhm:.2f} pixels")
            print(f"PSF FWHM: {self.psf_fwhm_mas:.2f} mas")

    def add_noise(self,verbose=True,nl_scale=10,):
        """
        NOTE: Assumes no noise has been added
        """
        if self.flag_noise is False:
            self.flag_noise = True

            if verbose:
                print('Adding noise')
            # Step 6b: Create the mean dark current image
            dark_signal_e = self.dark_current_rate * self.exp_time

            # Step 6c: Create the total mean signal (Star + Dark)
            self.total_mean_signal = self.data + dark_signal_e.value
            #self.total_mean_signal_nl = apply_nl_scaling(self.df_nl,self.total_mean_signal,how='makenonlinear')

            #if apply_nonlinearity:
            #    # Step 6c.1: Apply non-linearity scaling
            #    total_mean_signal = apply_nl_scaling(self.df_nl,total_mean_signal,how='makenonlinear')
            #    if verbose:
            #        print("Applied non-linearity scaling to total mean signal.")

            # Step 6d: Apply Poisson noise (Shot noise from star + Dark noise)
            # np.random.poisson takes the mean (lambda) and returns a random variate.
            # This correctly simulates noise from both the star and the dark current.
            self.noisy_signal = np.random.poisson(self.total_mean_signal)
            self.noisy_signal_nl = apply_nl_scaling(self.df_nl,self.noisy_signal,how='makenonlinear',scale=1)

            # Step 6e: Apply Gaussian read noise
            # Create a map of Gaussian noise (mean=0, stddev=READ_NOISE_RMS)
            read_noise = np.random.normal(0.0, self.read_noise_rms.value, size=self.noisy_signal.shape)

            # Step 6f: Add read noise to the image
            # Convert noisy_signal to float to allow for negative values
            self.data = self.noisy_signal.astype(float) + read_noise
            self.data_nl = self.noisy_signal_nl.astype(float) + read_noise
            self.data_nlcorr = apply_nl_scaling(self.df_nl,self.data_nl,how='correctnonlinear',scale=nl_scale)

            # Flat
            self.data_flat = self.data/self.data_master_flat
            self.data_nl_flat = self.data_nl/self.data_master_flat
            self.data_nlcorr_flat = self.data_nlcorr/self.data_master_flat

        else:
            print('Noise already applied, exiting')
            sys.exit('Noise already applied')

    def plot_data(self,data=None,ax=None,title=''):
        """
        Plot the current `self.data` image.
        """
        if ax is None:
            fig, ax = plt.subplots(dpi=200)
        # --- 4. Plot the Ideal PSF --
        if data is None:
            data = self.data
        ax.imshow(data, origin='lower', interpolation='nearest', cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")
        ax.figure.colorbar(ax.images[0], ax=ax, label="Signal (electrons)")
        ax.grid(lw=0)





class FitsImgList(object):

    def __init__(self, data_list, center_list=None,**kwargs):
        """
        Initialize the FitsImgList with a list of data arrays and optional centers.
        """
        self.imglist = []
        for i, data in enumerate(data_list):
            center = center_list[i] if center_list is not None else None
            self.imglist.append(FitsImg(data=data, center=center, **kwargs))

    def aperture_photometry(self, **kwargs ):
        """
        Perform aperture photometry on all images in the list.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to pass to the `aperture_photometry` method of `FitsImg`.

        Returns
        -------
        results_list : list of dict
            List of dictionaries containing photometry results for each image.
        """
        results_list = []
        for img in self.imglist:
            result = img.aperture_photometry(**kwargs)
            results_list.append(result)
        self.df_phot = pd.DataFrame(results_list)
        self.df_phot['flux_norm'] = self.df_phot['net_flux'] / np.abs(np.median(self.df_phot['net_flux']))
        self.df_phot['flux_err_norm'] = self.df_phot['flux_err'] / np.abs(np.median(self.df_phot['net_flux']))
        return self.df_phot

    def plot_photometry(self,axes=None):
        """
        Plot the photometry results stored in self.df_phot.
        """
        if not hasattr(self, 'df_phot'):
            raise ValueError("No photometry data found. Please run aperture_photometry() first.")

        if axes is None:
            fig, axes = plt.subplots(dpi=200,nrows=3,sharex=True)
        ax, bx, cx = axes
        label = '$\sigma$={:0.0f}ppm, MedErr={:0.0f}ppm'.format(1e6*np.std(self.df_phot.flux_norm),1e6*np.median(self.df_phot.flux_err_norm))
        ax.errorbar(np.arange(len(self.df_phot)),self.df_phot.flux_norm,yerr=self.df_phot.flux_err_norm,marker='o',lw=0,mew=0.5,capsize=4,elinewidth=0.5, label=label)
        bx.plot(np.arange(len(self.df_phot)),self.df_phot['xcen'],marker='o',lw=0.5,mew=0.5)
        cx.plot(np.arange(len(self.df_phot)),self.df_phot['ycen'],marker='o',lw=0.5,mew=0.5)
        for xx in [ax,bx,cx]:
            xx.grid(lw=0.3,alpha=0.3)
            xx.minorticks_on()
        ax.set_ylabel('Normalized flux',fontsize=15)
        bx.set_ylabel('x-centroid',fontsize=15)
        cx.set_ylabel('y-centroid',fontsize=15)

        ax.legend(fontsize=8,loc='upper right')
        cx.set_xlabel('Exposure number',fontsize=15)

        




class FitsImg(object):

    def __init__(self,filename=None,data=None,header=None,center=None,dark_current_rate=0.0,exp_time=1.0,read_noise_rms=0.0,imgnumber=0):
        """
        Initialize the FitsImg with data and optional center.
        """
        if filename!=None:
            self.filename = filename
            self.hdulist = astropy.io.fits.open(self.filename)
            self.header = self.hdulist[imgnumber].header
            data = self.hdulist[imgnumber].data
            self.data = data.astype(float)
        else:
            self.filename = ""
            self.hdulist = None
            self.header = header
            self.data = data

        self.center = center
        self.dark_current_rate = dark_current_rate
        self.exp_time = exp_time
        self.read_noise_rms = read_noise_rms

    
    
    def crop(self,x,y,w,h):
        """
        Crop to a box centered at (x,y), of size w x h
        """
        x, y = int(x),int(y)
        self.data = self.data[int(y-h/2):int(y+h/2),int(x-w/2):int(x+w/2)]
        
    def cropcenter(self,w,h,points=False):
        """
        Returns an image array around the center of an image array.
        """
        shape = self.data.shape
        x, y = (int(shape[0] / 2), int(shape[1] / 2))
        self.crop(x,y,w,h)

    def cropcentroid(self,w,h):
        """
        Crop to a box centered at the centroid, of size w x h
        """
        x,y = self.get_centroid()
        self.crop(x,y,w,h)

    def get_centroid(self,plot_cross=False,ax=None,plot_lines=False):
        """
        Find centroid using Howell centroiding.

        See phothelp for the method
        """
        self.xcenter, self.ycenter = howell_center(self.data)
        if plot_cross:
            self.plot(ax=ax)
            self.ax.scatter(self.xcenter,self.ycenter,marker="+",s=50,color="green")
        if plot_lines:
            self.plot(ax=ax)
            self.ax.hlines(int(self.ycenter),0,self.data.shape[1],color='#1f77b4',lw=1)
            self.ax.vlines(int(self.xcenter),0,self.data.shape[0],color='#1f77b4',lw=1)
        return self.xcenter, self.ycenter


    def plot(self,stretch="hist",cmap="gray",origin="lower",ax=None,colorbar=False,title="",vmin=None,vmax=None,dpi=200):
        if ax == None:
            self.fig, self.ax = plt.subplots(dpi=dpi)
        else:
            self.ax = ax
        if stretch=="hist":
            print('hist stretch')
            norm = ImageNormalize(stretch=HistEqStretch(self.data))
            self.im = self.ax.imshow(self.data,cmap=cmap,origin=origin,norm=norm,vmin=vmin,vmax=vmax)
        elif stretch=='log':
            print('log stretch')
            norm = ImageNormalize(self.data,stretch=LogStretch())
            self.im = self.ax.imshow(self.data,cmap=cmap,origin=origin,norm=norm,vmin=vmin,vmax=vmax)
        else:
            print('linear stretch')
            self.im = self.ax.imshow(self.data,cmap=cmap,origin=origin,vmin=vmin,vmax=vmax)
        self.ax.set_xlim(0,self.data.shape[1]) # cols
        self.ax.set_ylim(0,self.data.shape[0]) # rows
        self.ax.set_title(title,y=1.02)
        self.ax.set_xlabel("X pixels")
        self.ax.set_ylabel("Y pixels")
        if colorbar:
            self.fig.colorbar(self.im)

    def get_radial_profile(self,rmax=None,plot=False,z=2.,return_hwzm=False,ax=None,xcen=None,ycen=None,annulus_width=1,subtract_min=False):
        """
        Plot radial profile
        """
        if not hasattr(self,"radial"):
            print("Calculating radial data")
            self.radial = radial_data(self.data,rmax=rmax,x=xcen,y=ycen,annulus_width=annulus_width)
        else:
            print("Warning: using stored radial data")
        if subtract_min:
            print("Subtracting min value from azimuthal average", np.min(self.radial.mean))
            self.radial.mean -= np.min(self.radial.mean)
        self.radial_hwhm = calc_hwhm(self.radial.r,self.radial.mean)
        self.radial_hwzm = calc_hwzm(self.radial.r,self.radial.mean,z=z)
        z = float(z)

        print("HWZM",self.radial_hwzm)
        
        if plot:
            if ax==None:
                self.fig, self.ax = plt.subplots()
            else:
                self.ax = ax
            self.ax.plot(self.radial.r,self.radial.mean)
            ymin, ymax = self.ax.get_ylim()
            self.ax.vlines(self.radial_hwhm,ymin,ymax,label="HWHM={}".format(self.radial_hwhm),color="orange",linestyle="--",lw=1)
            self.ax.vlines(self.radial_hwzm,ymin,ymax,label="HWZM(z={})={}".format(z,self.radial_hwzm),color="red",linestyle="--",lw=1)
            self.ax.legend(loc="upper right",fontsize=14)
            self.ax.grid(lw=0.5,alpha=0.3)


        if return_hwzm:
            return self.radial.r,self.radial.mean,self.radial_hwzm
        else:
            return self.radial.r,self.radial.mean


    def aperture_photometry(self, r_ap=3.0, r_in=6.0, r_out=8.0, gain=1.0, center=None, bkg_sigma_clip=3.0, bkg_maxiters=5, 
                            plot=True, ax=None, verbose=True,vmin=None,vmax=None,cmap='viridis',origin='lower',stretch='hist',colorbar=True):
        """
        Perform circular aperture photometry on the current `self.data` image using photutils.

        Parameters
        ----------
        r_ap : float
            Aperture radius in pixels.
        r_in, r_out : float
            Inner and outer radii for background annulus in pixels. If None or invalid,
            the image median is used as a background estimate.
        gain : float
            e/ADU (set to 1.0 if `self.data` is already in electrons).
        center : None or (y,x)
            If provided, use this center; otherwise the method will try `self.center`,
            then the peak pixel, then image center.
        bkg_sigma_clip : float
            Sigma for sigma-clipped background estimation in the annulus.
        bkg_maxiters : int
            Max iterations for sigma clipping.

        Returns
        -------
        dict
            Dictionary containing ap_sum, bkg_mean_per_pix, bkg_std_per_pix, bkg_sum,
            net_flux, flux_err, snr, ap_area, ann_area, ann_pixels_used, center
        """
        img = np.asarray(self.data)

        if center is None:
            cx, cy = howell_center(img)
            if verbose:
                print(f"Using centroid at (x={cx:.2f}, y={cy:.2f}) for photometry.")

        # determine center
        #if center is None:
        #    if hasattr(self, 'center') and self.center is not None:
        #        try:
        #            cy, cx = self.center if len(self.center) == 2 else (float(self.center), float(self.center))
        #        except Exception:
        #            cy = (img.shape[0] - 1) / 2.0
        #            cx = (img.shape[1] - 1) / 2.0
        #    else:
        #        try:
        #            cy, cx = np.unravel_index(np.argmax(img), img.shape)
        #            cy, cx = float(cy), float(cx)
        #        except Exception:
        #            cy = (img.shape[0] - 1) / 2.0
        #            cx = (img.shape[1] - 1) / 2.0
        #else:
        #    cy, cx = float(center[0]), float(center[1]) if len(center) == 2 else (float(center), float(center))

        pos = [(cx, cy)]  # photutils uses (x, y)

        aper = CircularAperture(pos, r=r_ap)

        use_ann = (r_in is not None) and (r_out is not None) and (r_out > r_in)
        if use_ann:
            ann = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
            phot_table = aperture_photometry(img, [aper, ann])
        else:
            phot_table = aperture_photometry(img, [aper])

        # aperture sums
        ap_sum = float(phot_table['aperture_sum_0'][0])
        ap_area = float(aper.area)
        bkg_mean = 0.0
        bkg_median = 0.0
        bkg_std = 0.0
        ann_area = 0.0
        ann_pixels_used = 0

        if use_ann:
            try:
                ann_sum = float(phot_table['aperture_sum_1'][0])
            except Exception:
                ann_sum = 0.0

            mask = ann.to_mask(method='exact')[0]
            annulus_data = mask.multiply(img)
            ann_pixels = annulus_data[mask.data > 0]
            ann_pixels_used = int(ann_pixels.size)

            if ann_pixels_used > 0:
                bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(ann_pixels, sigma=bkg_sigma_clip, maxiters=bkg_maxiters)
                ann_area = float(ann.area)
            else:
                # fallback to image statistics
                bkg_mean = float(np.median(img))
                bkg_std = float(np.std(img))
                ann_area = 0.0
        else:
            # fallback: use image median as background estimate
            bkg_mean = float(np.median(img))
            bkg_std = float(np.std(img))

        bkg_sum = bkg_mean * ap_area
        net_flux = ap_sum - bkg_sum

        # helper to extract value from astropy Quantity or bare number
        def _val(x):
            try:
                return float(x.value)
            except Exception:
                return float(x)

        dark_rate = _val(getattr(self, 'dark_current_rate'))
        exp_time_val = _val(getattr(self, 'exp_time'))
        read_noise = _val(getattr(self, 'read_noise_rms'))

        dark_per_pix = dark_rate * exp_time_val

        # noise model (electrons)
        shot_var = net_flux
        dark_var = ap_area * dark_per_pix
        read_var = ap_area * (read_noise ** 2)
        bkg_var = ap_area * (bkg_std ** 2)

        total_var = shot_var + dark_var + read_var + bkg_var
        flux_err = np.sqrt(total_var) / float(gain)
        snr = net_flux / flux_err if flux_err > 0 else np.nan

        if plot:
            if ax is None:
                fig, ax = plt.subplots(dpi=100)
            # --- 4. Plot the Ideal PSF --
            if stretch =="hist":
                norm = ImageNormalize(stretch=HistEqStretch(self.data))
                self.im = ax.imshow(self.data,cmap=cmap,origin=origin,norm=norm,vmin=vmin,vmax=vmax)
            elif stretch=='log':
                print('log stretch')
                norm = ImageNormalize(self.data,stretch=LogStretch())
                self.im = ax.imshow(self.data,cmap=cmap,origin=origin,norm=norm,vmin=vmin,vmax=vmax)
            else:
                print('linear stretch')
                self.im = ax.imshow(self.data,cmap=cmap,origin=origin,vmin=vmin,vmax=vmax)

            ax.imshow(self.data, origin='lower', interpolation='nearest', cmap='viridis')
            #ax.set_title(f"PSF with Detector Noise (Flux={self.total_flux} e-)")
            ax.set_xlabel("Pixel")
            ax.set_ylabel("Pixel")
            if colorbar:
                ax.figure.colorbar(ax.images[0], ax=ax, label="Signal (electrons)")
            ax.grid(lw=0)
            # Draw aperture and annulus outlines
            aper_patch = aper.plot(ax=ax, color='red', lw=1.6, alpha=0.9)[0]
            ann_patch = ann.plot(ax=ax, color='white', lw=1.2, alpha=0.9)[0]
            # Mark the center
            ax.plot(cx, cy, marker='+', color='yellow', markersize=10, mew=1.5)
            # Annotation: show aperture radii in pixels
            ax.text(0.02, 0.98, f"r_ap={r_ap:.1f}px, r_in={r_in:.1f}px, r_out={r_out:.1f}px",
                    transform=ax.transAxes, color='white', fontsize=9, va='top')
            ax.set_title('Centroid: (x={:.2f}, y={:.2f})'.format(cx, cy), y=1.02)

        return {
            'xcen': cx,
            'ycen': cy,
            'ap_sum': ap_sum,
            'bkg_mean_per_pix': float(bkg_mean),
            'bkg_median_per_pix': float(bkg_median),
            'bkg_std_per_pix': float(bkg_std),
            'bkg_sum': float(bkg_sum),
            'net_flux': float(net_flux),
            'flux_err': float(flux_err),
            'snr': float(snr),
            'ap_area': ap_area,
            'ann_area': ann_area,
            'ann_pixels_used': int(ann_pixels_used)
        }




def get_micron_to_mas(D_m, F):
    D = D_m * u.m
    f_mm = (D * F).to(u.mm).value             # focal length in mm (float)
    micron_to_mas = 206265.0 / f_mm          # mas per micron
    return micron_to_mas

class CustomPSF(object):
    """Updated CustomPSF: center is in resampled (output) pixel coords.

    Constructor inputs:
      - filename: path to whitespace-delimited text file
      - src_micron_per_pix: micron per source pixel
      - telescope_diameter_m, fnum -> focal_length_m = D * fnum
      - target_pixel_size_micron: desired output pixel size in microns

    Use `resample_to_grid(target_npix, center=(cx_out, cy_out))` where
    center is in output pixels. If center=None, source center maps to the
    output center.

    EXAMPLE:
        # Example usage of updated class (use this cell instead of the old definition):
        CPSF2 = CustomPSF(files[0], src_micron_per_pix=4.0, telescope_diameter_m=3.0, fnum=15.0, target_pixel_size_micron=3.76)
        # Request the source center to appear at output pixel (10,10):
        data_resampled = CPSF2.resample_to_grid(target_npix=100, center=None)
        CPSF2.plot_resampled()
        fimg = FitsImg(data=data_resampled)
        fimg.plot()
        fimg.get_radial_profile(plot=True)
    """

    def __init__(self, filename, src_micron_per_pix, telescope_diameter_m, fnum, target_pixel_size_micron, skiprows=22, encoding="utf-16", verbose=True):
        self.filename = filename
        self.src_micron_per_pix = float(src_micron_per_pix)
        self.telescope_diameter_m = telescope_diameter_m
        self.fnum = fnum
        self.focal_length_m = telescope_diameter_m * fnum
        self.micron_to_mas = get_micron_to_mas(self.telescope_diameter_m, self.fnum)
        self.target_pixel_size_micron = target_pixel_size_micron
        self.target_pixel_size_mas = float(target_pixel_size_micron) * self.micron_to_mas
        if verbose:
            print(f'CustomPSF_v2: micron_to_mas={self.micron_to_mas:.4f} mas/um; target_pixel_size_mas={self.target_pixel_size_mas:.4f} mas/pix')
        # load data
        try:
            self.data = pd.read_csv(self.filename, sep=r'\s+', skiprows=skiprows, encoding=encoding).values.astype(float)
        except Exception:
            self.data = np.loadtxt(self.filename, skiprows=skiprows)
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError(f'CustomPSF_v2: loaded data must be 2D, got shape={self.data.shape}')
        self.resampled = None

    def _center_crop_or_pad(self, img, out_shape, center=None, fill=0.0, interp_order=3):
        ny, nx = img.shape
        oy, ox = out_shape
        # center is (cx, cy)
        # Allow sub-pixel centers by integer cropping/padding then
        # applying a fractional shift to align the requested center.
        # center is provided as (cx, cy) in image coordinates (x,y).
        if center is None:
            cy = (ny - 1) / 2.0; cx = (nx - 1) / 2.0
        else:
            cx, cy = float(center[0]), float(center[1])

        # integer window start/stop
        y0 = int(np.floor(cy - (oy - 1) / 2.0))
        x0 = int(np.floor(cx - (ox - 1) / 2.0))
        y1 = y0 + oy; x1 = x0 + ox

        out = np.full((oy, ox), fill, dtype=img.dtype)
        sy0 = max(0, y0); sx0 = max(0, x0)
        sy1 = min(ny, y1); sx1 = min(nx, x1)
        if (sy1 > sy0) and (sx1 > sx0):
            oy0 = sy0 - y0; ox0 = sx0 - x0
            out[oy0:oy0 + (sy1 - sy0), ox0:ox0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]

        # Compute fractional offset of the true center within the cropped window
        center_in_out_y = cy - y0
        center_in_out_x = cx - x0
        desired_center_y = (oy - 1) / 2.0
        desired_center_x = (ox - 1) / 2.0

        shift_y = desired_center_y - center_in_out_y
        shift_x = desired_center_x - center_in_out_x

        # If there is a fractional component, apply a sub-pixel shift to align
        # the requested center to the output center. Use order=3 spline by default.
        if (abs(shift_y) > 1e-9) or (abs(shift_x) > 1e-9):
            out = shift(out, shift=(shift_y, shift_x), order=interp_order, mode='constant', cval=fill)

        return out

    def resample_to_grid(self, npix, total_flux, center=None, interp_order=1):
        """Resample the PSF onto an output grid of size npix.

        `center` is in output (resampled) pixel coordinates (cx_out, cy_out).
        The routine maps the source center to the requested output pixel.
        """
        src_pixel_scale_mas = float(self.src_micron_per_pix) * self.micron_to_mas
        zoom_factor = float(src_pixel_scale_mas) / float(self.target_pixel_size_mas)
        if zoom_factor <= 0:
            raise ValueError('Computed zoom_factor <= 0')
        img_zoomed = zoom(self.data, zoom_factor, order=interp_order, mode='constant', cval=0.0)

        # source center in zoomed coords
        cx_src = (self.data.shape[1] - 1) / 2.0
        cy_src = (self.data.shape[0] - 1) / 2.0
        cx_src_z = cx_src * zoom_factor
        cy_src_z = cy_src * zoom_factor

        # output center coords
        out_center_x = (npix - 1) / 2.0
        out_center_y = (npix - 1) / 2.0

        if center is None:
            cx_zoom = cx_src_z
            cy_zoom = cy_src_z
        else:
            cx_out = float(center[0])
            cy_out = float(center[1])
            dx_out = cx_out - out_center_x
            dy_out = cy_out - out_center_y
            # place source center in zoomed coords so it appears at requested output pixel
            cx_zoom = cx_src_z - dx_out
            cy_zoom = cy_src_z - dy_out

        out = self._center_crop_or_pad(img_zoomed, (npix, npix), center=(cx_zoom, cy_zoom), fill=0.0, interp_order=interp_order)

        m = np.sum(out < 0)
        print('Number of negative pixels after resampling: {}'.format(m))
        # normalize:
        s = out.sum()
        if s > 0:
            out = out / s
        out = out * total_flux
        self.data_resampled = np.abs(out)
        return self.data_resampled

    def plot_resampled(self, title='Resampled PSF', cmap='viridis'):
        if self.data_resampled is None:
            raise RuntimeError('No resampled image found. Call resample_to_grid(...) first')
        fig, ax = plt.subplots()
        ax.imshow(self.data, origin='lower', cmap=cmap, interpolation='nearest')
        ax.set_title('Original PSF')
        plt.colorbar(ax.images[0], ax=ax, label='Normalized flux')
        plt.show()

        fig, ax = plt.subplots()
        ax.imshow(self.data_resampled, origin='lower', cmap=cmap, interpolation='nearest')
        ax.set_title(title)
        plt.colorbar(ax.images[0], ax=ax, label='Normalized flux')
        plt.show()



def calc_hwzm(x,y,z=20):
    """
    Calculates the HWHM at the Z-th maximum for a given dataset, by finding the roots of splines.
    
    INPUTS:
        x - x input array
        y - y input array
    
    OUTPUT:
        HWZM The Half Width at Z-th Max of the data
    
    EXAMPLE:
    """
    spline = UnivariateSpline(x, y-np.max(y)/z, s=0)
    roots = spline.roots()
    return roots

def calc_hwhm(x,y):
    """
    Calculates the HWHM for a given dataset, by finding the roots of splines.
    
    INPUTS:
        x - x input array
        y - y input array
    
    OUTPUT:
        HWZM The Half Width at Z-th Max of the data
    
    EXAMPLE:
    """
    roots = calc_hwzm(x,y,z=2)
    return roots