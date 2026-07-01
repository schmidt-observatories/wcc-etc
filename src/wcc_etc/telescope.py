from __future__ import annotations

from typing import Any, Self

import numpy as np
from synphot.models import Box1D
from synphot import SpectralElement
from astropy import units as u

from .meta import _MetaHolder_

class Telescope(_MetaHolder_):
    """
    A class representing the telescope properties.

    Attributes
    ----------
    f_num : float
        The focal ratio of the telescope.
    diameter_primary : Quantity
        The diameter of the primary mirror.
    jitter_sigma : Quantity
        The pointing jitter (sigma) in milliarcseconds.
    surface : Quantity
        The collecting area of the telescope.
    focal_len : Quantity
        The focal length of the telescope.
    """
    # list of mutable parameter. This is handled by _MetaHolder_
    _mutable_parameters = ["f_num", "diameter_primary", "jitter_sigma"]
    
    def __init__(self, f_num: float, diameter_primary: float | u.Quantity,
                 jitter_sigma: float | u.Quantity = 0,
                 meta: dict[str, Any] = {}) -> None:
        """ 
        Initialize a Telescope object.

        Parameters
        ----------
        f_num : float
            The focal ratio of the telescope.
        diameter_primary : float or Quantity
            The diameter of the primary mirror (meters if float).
        jitter_sigma : float or Quantity, optional
            The pointing jitter (sigma) (mas if float). Default is 0.
        meta : dict, optional
            Additional metadata. Default is {}.
        """
        # default hard coded. code implemented such that sensor hold the full throughput.
        self._bandpass = SpectralElement(Box1D, amplitude=1, x_0=7000, width=12000)
        meta["f_num"] = f_num
        meta["diameter_primary"] = diameter_primary
        meta["jitter_sigma"] = jitter_sigma

        # meta
        super().__init__(meta=meta)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """
        Create a Telescope instance from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing 'f_num', 'diameter_primary', etc.

        Returns
        -------
        Telescope
        """
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
    def f_num(self) -> float:
        """
        The focal ratio (f-number).
        """
        return self.meta.get("f_num")  # type: ignore[return-value]
        
    @property
    def diameter_primary(self) -> u.Quantity:
        """
        The primary mirror diameter as an astropy Quantity.
        """
        diameter_primary = self.meta.get("diameter_primary") 
        if not isinstance(diameter_primary, u.Quantity):
            diameter_primary *= u.m
        
        return diameter_primary

    @property
    def jitter_sigma(self) -> u.Quantity:
        """
        The pointing jitter (sigma) as an astropy Quantity.
        """
        jitter_sigma = self.meta.get("jitter_sigma", 0)
        if not isinstance(jitter_sigma, u.Quantity):
            jitter_sigma *= u.mas
            
        return  jitter_sigma
        
    @property
    def surface(self) -> u.Quantity:
        """
        The collecting area (surface) of the primary mirror.
        """
        return np.pi * (0.5 * self.diameter_primary) ** 2
        
    @property
    def focal_len(self) -> u.Quantity:
        """
        The focal length of the telescope.
        """
        return self.diameter_primary * self.f_num

