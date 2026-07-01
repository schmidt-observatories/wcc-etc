from __future__ import annotations

from typing import TYPE_CHECKING
from numpy.typing import NDArray

import warnings
import pandas
import numpy as np
from .io import expand_path

if TYPE_CHECKING:
    from astropy.units import Quantity
    from synphot import SpectralElement


def parse_element(path_or_element: str | SpectralElement | None, wave_unit: str = 'nm') -> SpectralElement | None:
    """
    Parse a path to a bandpass file or a SpectralElement object.

    Parameters
    ----------
    path_or_element : str or SpectralElement or None
        The path to the file or the SpectralElement object.
    wave_unit : str, optional
        Wavelength unit for the bandpass file. Default is 'nm'.

    Returns
    -------
    SpectralElement or None
        The parsed SpectralElement object.

    Raises
    ------
    NotImplementedError
        If the input type is not supported.
    """
    from synphot import SpectralElement
    if path_or_element is None:
        element = None
        
    elif isinstance(path_or_element, str):
        path = expand_path(path_or_element)
        element = SpectralElement.from_file(path, wave_unit=wave_unit)
        
    elif isinstance(path_or_element, SpectralElement):
        element = path_or_element
        
    else:
        raise NotImplementedError(f"Only path or element instance accepted {type(path_or_element)=} given.")

    # build the effective througput
    return element

def parse_and_interpolate(input_file: str, xval: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """
    Interpolate values from a CSV file at a given input x-value.

    Parameters
    ----------
    input_file : str
        Path to the CSV file. The first column is assumed to be the index (x),
        and the second column is the value (y).
    xval : float or array_like
        The x-value(s) at which to interpolate.

    Returns
    -------
    float or ndarray
        The interpolated y-value(s).
    """
    input_file = expand_path(input_file)
    data = pandas.read_csv(input_file, index_col=0).iloc[:, 0]
    return np.interp(xval, data.index, data.values)

def list_of_quantity_to_array(quantities: list[Quantity]) -> Quantity | list[Quantity]:
    """
    Convert a list of astropy Quantities to a numpy array, assuming they have the same unit.

    Parameters
    ----------
    quantities : list of Quantity
        The list of astropy Quantities.

    Returns
    -------
    Quantity
        A single Quantity object containing an array of values.

    Raises
    ------
    Warning
        If the input quantities do not have the same unit.
    """
    units = [q.unit for q in quantities]
    if len(np.unique(units)) == 1:
        unit = units[0]
    else:
        warnings.warn("input quantities are not all of the same unit. Nothing can be done.")
        return quantities

    values = [q.value for q in quantities]
    return np.asarray(values, dtype="float") * unit

