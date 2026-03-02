import os
from .wcc_etc import WCCETC
from .io import get_sensor_config

class WCCSensor():
    """ """
    def __init__(self, simulation):
        """ """
        self._simulation = simulation # this is a WCC_ETC

    @classmethod
    def from_name(cls, name):
        """ """
        sensor, band = name.split(":")
        return cls.from_kind_and_band(sensor, band)

    @classmethod
    def from_kind_and_band(cls, kind, band):
        """ loads the instance given the detector kind and filter

        Parameters
        ----------
        kind: str
            kind of sensor of the WCC: 
            - 'zwo' ('sony', 'imx', 'imx455' accepted)
            - 'qcmos'
        band: str
            name of the band associated to the sensor.
            e.g. bb, u, r, z etc.
        Returns
        -------
        instance
        """
        # grabs the configuration associated to this sensor
        config = get_sensor_config(kind, band)

        # loads the lower-level simulation
        simulation = WCCETC.from_config(config)
        return cls(simulation=simulation)
        
    # ================ #
    #  Properties      #
    # ================ #
    @property
    def simulation(self):
        """ core object holding most of the functionalities """
        return self._simulation
