import matplotlib.pyplot as plt
import cartopy
import numpy as np

import rioxarray as rxr


def plot_file(path: str, variable_name: str):
    with rxr.open_rasterio(path) as data_obj:
        prefix = "HDFEOS_GRIDS_VIIRS_Grid_DNB_2d_Data_Fields_"
        data = data_obj[variable_name].data
        fill_value = data_obj.attrs[f"{prefix}{variable_name}__FillValue"]
        scale_factor = data_obj.attrs[f"{prefix}{variable_name}_scale_factor"]
        offset_value = data_obj.attrs[f"{prefix}{variable_name}_offset"]
