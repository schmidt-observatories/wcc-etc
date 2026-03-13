import pandas
import numpy as np
from .io import expand_path


def parse_element(path_or_element, wave_unit='nm'):
    """ """
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

def parse_and_interpolate(input_file, xval):
    """
    Interpolate the given file columns to get the value at interpolation_xval

    INPUT:
        :param input_file: File to use x columns and y columns on
        :param interpolation_xval: Value that the interpolation function takes as argument
        :param col_headers: Names of column headers as a list

    OUTPUT:
        :return: The value of the interpolated function at interpolation_xval
    """
    input_file = expand_path(input_file)
    data = pandas.read_csv(input_file, index_col=0).iloc[:, 0]
    return np.interp(xval, data.index, data.values)

def list_of_quantity_to_array(quantities):
    """ """
    units = [q.unit for q in quantities]
    if len(np.unique(units)) == 1:
        unit = units[0]
    else:
        warnings.warn("input quantities are not all of the same unit. Nothing can be done.")
        return quantities

    values = [q.value for q in quantities]
    return np.asarray(values, dtype="float") * unit
