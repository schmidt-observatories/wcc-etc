import warnings
import numpy as np
from synphot.models import Box1D
from synphot import SpectralElement
from astropy import units as u
from copy import deepcopy

from .meta import _MetaHolder_

class Telescope(_MetaHolder_):
    """ """
    # list of mutable parameter. This is handled by _MetaHolder_
    _mutable_parameters = ["f_num", "diameter_primary", "jitter_sigma"]
    
    def __init__(self, f_num, diameter_primary,
                 jitter_sigma=0, 
                 meta={}):
        """ 
        """
        # default hard coded. code implemented such that sensor hold the full throughput.
        self._bandpass = SpectralElement(Box1D, amplitude=1, x_0=7000, width=12000)
        meta["f_num"] = f_num
        meta["diameter_primary"] = diameter_primary
        meta["jitter_sigma"] = jitter_sigma

        # meta
        super().__init__(meta=meta)

    @classmethod
    def from_config(cls, config):
        """ """
        # make sure these key exist
        _ = [config.get(key) for key in ["f_num", "diameter_primary"]]

        # read the throughput of the system.
        return cls(**config, meta=config)
        
    # ================ #
    #  methods         #
    # ================ #
    
    # ================ #
    #  Properties      #
    # ================ #
    @property
    def f_num(self):
        """ """
        return self.meta.get("f_num")
        
    @property
    def diameter_primary(self):
        """ """
        diameter_primary = self.meta.get("diameter_primary") 
        if not isinstance(diameter_primary, u.Quantity):
            diameter_primary *= u.m
        
        return diameter_primary

    @property
    def jitter_sigma(self):
        """ """
        jitter_sigma = self.meta.get("jitter_sigma", 0)
        if not isinstance(jitter_sigma, u.Quantity):
            jitter_sigma *= u.mas
            
        return  jitter_sigma
        
    @property
    def surface(self):
        """ """
        return np.pi * (0.5 * self.diameter_primary) ** 2
        
    @property
    def focal_len(self):
        """ """
        return self.diameter_primary * self.f_num
