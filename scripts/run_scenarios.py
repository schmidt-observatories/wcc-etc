# Script to run different scenarios
# These scenarios should make systematic output plots in a result directory

from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import wcc_etc
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.weight'] = 'normal'
rcParams['axes.formatter.useoffset'] = False

uasal_archive = os.environ.get("UASAL_ARCHIVE")



def calc_snr_r(TEXP=60):
    VERBOSE = True
    #TEXP = 60*2
    MAG = 26 # AB mag
    BKG_MAG = 180 # vmag / arcsec2 # make background irrelevant
    ###########################################
    #WCC = wcc_etc.WCCETC("../config/config_wcc_sdssg_from_final.toml")
    WCC = wcc_etc.WCCETC("../config/config_wcc_sdssr_from_final.toml")
    WCC.setup(verbose=VERBOSE,plot=False)
    WCC.bandpass.plot()
    source_file = '/Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/USAL_ARCHIVE/2025-06-16_uasal_archive_repo/astr_obj_models/stars/pickles_models/dat_uvk/pickles_uk_55.fits'
    bkg_file = 'ngc_2537_spec.fits'
    WCC.set_source(source_file,plot=True)
    WCC.set_background(background_file=bkg_file,support_data_path=f'{uasal_archive}/astr_obj_models/galaxies/brown/', plot=True)

    ########
    print('########################')
    snr = WCC.make_observation_and_calc_SNR(flux=MAG, r_aper_mas=70, texp=TEXP,
                                            jitter_sigma_mas=0, bg_flux=None, 
                                            plot=False, verbose=VERBOSE)
    print('SNR={:0.2f}: in {:0.3f}s'.format(snr,TEXP))

def calc_snr_g(TEXP=60):
    VERBOSE = True
    #TEXP = 60*2
    MAG = 26 # AB mag
    BKG_MAG = 180 # vmag / arcsec2 # make background irrelevant
    ###########################################
    WCC = wcc_etc.WCCETC("../config/config_wcc_sdssg_from_final.toml")
    #WCC = wcc_etc.WCCETC("../config/config_wcc_sdssr_from_final.toml")
    WCC.setup(verbose=VERBOSE,plot=False)
    WCC.bandpass.plot()
    source_file = '/Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/USAL_ARCHIVE/2025-06-16_uasal_archive_repo/astr_obj_models/stars/pickles_models/dat_uvk/pickles_uk_55.fits'
    bkg_file = 'ngc_2537_spec.fits'
    WCC.set_source(source_file,plot=True)
    WCC.set_background(background_file=bkg_file,support_data_path=f'{uasal_archive}/astr_obj_models/galaxies/brown/', plot=True)
    ###########################################

    ########
    print('########################')
    snr = WCC.make_observation_and_calc_SNR(flux=MAG, r_aper_mas=70, texp=TEXP,
                                            jitter_sigma_mas=0, bg_flux=None, 
                                            plot=False, verbose=VERBOSE)
    print('SNR={:0.2f}: in {:0.3f}s'.format(snr,TEXP))


def plot_jitter(configfile,title_extra=''):
    VERBOSE = True
    TEXP = 60
    MAG = 26
    R_APER_MAS = 70
    jitter = np.linspace(0,50,10)
    basename = os.path.basename(configfile)
    ###########################################
    WCC = wcc_etc.WCCETC(configfile)
    WCC.setup(verbose=VERBOSE,plot=False)
    WCC.bandpass.plot()
    source_file = '/Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/USAL_ARCHIVE/2025-06-16_uasal_archive_repo/astr_obj_models/stars/pickles_models/dat_uvk/pickles_uk_55.fits'
    bkg_file = 'ngc_2537_spec.fits'
    WCC.set_source(source_file,plot=True)
    WCC.set_background(background_file=bkg_file,support_data_path=f'{uasal_archive}/astr_obj_models/galaxies/brown/', plot=True)
    ###########################################
    snrs = WCC.make_observation_and_calc_SNR(flux=MAG, r_aper_mas=R_APER_MAS, texp=TEXP,
                                            jitter_sigma_mas=jitter, bg_flux=None, 
                                            plot=False, verbose=False)
    fig, ax = plt.subplots(dpi=200)
    ax.plot(jitter, snrs, marker='o', ls='-', color='C0')
    ax.set_xlabel('Jitter (mas)')
    ax.set_ylabel('SNR')
    ax.set_title(f'{basename}\nM={MAG}, texp={TEXP}s, r_aper={R_APER_MAS}mas, {title_extra}')
    ax.grid(lw=0.3,alpha=0.3)
    ax.axhline(10,ls='--',color='crimson', label='SNR=10')
    ax.legend()
    #fig.savefig('plot_snr_vs_jitter_{}.png'.format(basename), bbox_inches='tight')
    fig.savefig('plot_snr_vs_jitter_{}.png'.format(basename), bbox_inches='tight')

def plot_time(configfile,title_extra=''):
    VERBOSE = True
    TEXP = np.linspace(60,200,10)
    MAG = 25
    R_APER_MAS = 70
    basename = os.path.basename(configfile)
    ###########################################
    WCC = wcc_etc.WCCETC(configfile)
    WCC.setup(verbose=VERBOSE,plot=False)
    WCC.bandpass.plot()
    source_file = '/Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/USAL_ARCHIVE/2025-06-16_uasal_archive_repo/astr_obj_models/stars/pickles_models/dat_uvk/pickles_uk_55.fits'
    bkg_file = 'ngc_2537_spec.fits'
    WCC.set_source(source_file,plot=True)
    WCC.set_background(background_file=bkg_file,support_data_path=f'{uasal_archive}/astr_obj_models/galaxies/brown/', plot=True)
    ###########################################
    snrs = WCC.make_observation_and_calc_SNR(flux=MAG, r_aper_mas=R_APER_MAS, texp=TEXP,
                                            jitter_sigma_mas=40, bg_flux=None, 
                                            plot=False, verbose=False)
    fig, ax = plt.subplots(dpi=200)
    ax.plot(TEXP, snrs, marker='o', ls='-', color='C0',label='WCC ETC')
    ax.set_xlabel('Exposure time (s)')
    ax.set_ylabel('SNR')
    ax.set_title(f'{basename}\nM={MAG}, r_aper={R_APER_MAS}mas, {title_extra}')
    ax.grid(lw=0.3,alpha=0.3)
    #ax.plot(TEXP, snrs[0]*np.sqrt(TEXP/60), ls='--', color='C1', label='sqrt behavior')
    ax.axhline(10,ls='--',color='crimson', label='SNR=10')
    ax.legend()
    fig.savefig('plot_snr_vs_time_{}.png'.format(basename), bbox_inches='tight')
    #fig.savefig('plot_snr_vs_time_{}_2xread.png'.format(basename), bbox_inches='tight')

def plot_aper(configfile,title_extra=''):
    VERBOSE = True
    TEXP = 60
    MAG = 26
    R_APER_MAS = np.linspace(10,200,20)
    basename = os.path.basename(configfile)
    ###########################################
    WCC = wcc_etc.WCCETC(configfile)
    WCC.setup(verbose=VERBOSE,plot=False)
    WCC.bandpass.plot()
    source_file = '/Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/USAL_ARCHIVE/2025-06-16_uasal_archive_repo/astr_obj_models/stars/pickles_models/dat_uvk/pickles_uk_55.fits'
    bkg_file = 'ngc_2537_spec.fits'
    WCC.set_source(source_file,plot=True)
    WCC.set_background(background_file=bkg_file,support_data_path=f'{uasal_archive}/astr_obj_models/galaxies/brown/', plot=True)
    ###########################################
    snrs = WCC.make_observation_and_calc_SNR(flux=MAG, r_aper_mas=R_APER_MAS, texp=TEXP,
                                            jitter_sigma_mas=40, bg_flux=None, 
                                            plot=False, verbose=False)
    fig, ax = plt.subplots(dpi=200)
    ax.plot(R_APER_MAS, snrs, marker='o', ls='-', color='C0')
    ax.set_xlabel('Aperture radius (mas)')
    ax.set_ylabel('SNR')
    ax.set_title(f'{basename}\nM={MAG}, texp={TEXP}s, {title_extra}')
    ax.axhline(10,ls='--',color='crimson', label='SNR=10')
    ax.legend()
    ax.grid(lw=0.3,alpha=0.3)
    fig.savefig('plot_snr_vs_aper_{}.png'.format(basename), bbox_inches='tight')

if __name__ == "__main__":
    #calc_snr_g(TEXP=60) # 9.41 in SDSS g
    #calc_snr_r(TEXP=120) # 6.06 in SDSS r

    plot_jitter("../config/config_wcc_sdssg_from_final_eol.toml", title_extra='2x read noise')
    plot_jitter("../config/config_wcc_sdssr_from_final_eol.toml", title_extra='2x read noise')

    plot_time("../config/config_wcc_sdssg_from_final_eol.toml", title_extra='2x read noise')
    plot_time("../config/config_wcc_sdssr_from_final_eol.toml", title_extra='2x read noise')

    plot_aper("../config/config_wcc_sdssg_from_final_eol.toml", title_extra='2x read noise')
    plot_aper("../config/config_wcc_sdssr_from_final_eol.toml", title_extra='2x read noise')