"""
Utility functions for the nightlights package.
"""
import numpy as np
import xarray as xr
import rioxarray as rxr
from typing import Dict, List, Tuple, Union, Optional
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
from shapely import wkt
import datetime
import pandas as pd

# Constants
PREFIX = "HDFEOS_GRIDS_VIIRS_Grid_DNB_2d_Data_Fields_"
DEFAULT_CMAP = "cividis"
DEFAULT_BUFFER = 0.2  # degrees

# Map styling parameters
MAP_BACKGROUND_COLOR = "white"  # Background color for cartopy maps
MAP_COASTLINE_COLOR = "black"  # Color for coastlines
MAP_BORDER_COLOR = "gray"  # Color for country borders
MAP_STATE_COLOR = "gray"  # Color for state/province borders
MAP_GRID_COLOR = "gray"  # Color for gridlines
MAP_WATER_COLOR = "lightblue"  # Color for water bodies (ocean, lakes, rivers)


def get_bounding_box(region):
    """
    Returns the bounding box of the given region in the format
    (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).

    Parameters:
    - region (Polygon, MultiPolygon, GeoDataFrame, or WKT string): The input region.

    Returns:
    - tuple: Bounding box (min_lon, min_lat, max_lon, max_lat).
    """
    if isinstance(region, (Polygon, MultiPolygon)):
        (
            minx,
            miny,
            maxx,
            maxy,
        ) = region.bounds  # Directly get bounds for shapely geometries

    elif isinstance(region, gpd.GeoDataFrame):
        (
            minx,
            miny,
            maxx,
            maxy,
        ) = region.total_bounds  # Get total bounds for GeoDataFrame

    elif isinstance(region, str):
        try:
            geom = wkt.loads(region)  # Load WKT string
            if isinstance(geom, (Polygon, MultiPolygon)):
                minx, miny, maxx, maxy = geom.bounds
            else:
                raise ValueError(
                    "WKT does not represent a valid Polygon or MultiPolygon."
                )
        except Exception as e:
            raise ValueError(f"Invalid WKT string: {e}")

    else:
        raise TypeError(
            "Unsupported region type. Must be a Polygon, MultiPolygon, GeoDataFrame, or WKT string."
        )

    return (
        minx,
        miny,
        maxx,
        maxy,
    )  # Return in (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat) format


def get_file_date(file_path: str) -> str:
    """
    Extracts the date from the file name.
    """
    julian_date = file_path.split("/")[-1].split(".")[1].replace("A", "")
    return datetime.datetime.strptime(julian_date, "%Y%j").strftime("%Y-%m-%d")


def is_in_date_range(file_date, start_date: str, end_date: str) -> bool:
    """
    Checks if the file date is within the specified date range.
    """
    return (
        pd.to_datetime(start_date)
        <= pd.to_datetime(file_date)
        <= pd.to_datetime(end_date)
    )


def load_data_from_h5(
    path: str, variable_name: str, region=None, region_crs: int = 4326
) -> Tuple[np.ndarray, dict, xr.DataArray]:
    """
    Load raw data and metadata from an h5 file.

    Args:
        path (str): Path to the h5 file
        variable_name (str): Name of the variable to extract
        region: Optional region to clip the data to
        region_crs: Coordinate reference system of the region

    Returns:
        tuple: (data, metadata, data_obj) arrays and metadata for processing
    """
    with rxr.open_rasterio(path) as data_obj:

        var_path = f"{PREFIX}{variable_name}"
        try:
            # Get the data first
            data_var = data_obj[var_path]
            fill_value = data_obj.attrs.get(f"{var_path}__FillValue")

            # Apply region clipping if provided
            if region is not None:
                # Use rio.clip to clip the data to the region
                data_var = data_var.rio.clip([region], region_crs, drop=True)

                # Extract the data array and coordinates after clipping
                data = data_var.data

                # Update the metadata to include the new coordinates
                metadata = {
                    "fill_value": fill_value,
                    "scale_factor": data_obj.attrs.get(
                        f"{var_path}_scale_factor", 1.0
                    ),
                    "offset": data_obj.attrs.get(f"{var_path}_offset", 0.0),
                    "product": data_obj.attrs.get("ShortName", ""),
                    "date": data_obj.attrs.get("RangeBeginningdate", ""),
                    "lons": data_var.x.values,  # Use clipped coordinates
                    "lats": data_var.y.values,  # Use clipped coordinates
                }

                # If we have a fill value, make sure it's properly applied
                if fill_value is not None:
                    # Create a mask for values that should be NaN
                    mask = np.isclose(data, fill_value) | np.isnan(data)
                    data = np.where(mask, np.nan, data)

                # Return early with the clipped data and updated metadata
                return data, metadata, data_var
            else:
                data = data_var.data
        except KeyError:
            raise ValueError(
                f"Variable {variable_name} not found in file {path}\n"
                f"Available variables: {list(data_obj.var())}"
            )

        # Extract metadata
        metadata = {
            "fill_value": data_obj.attrs.get(f"{var_path}__FillValue"),
            "scale_factor": data_obj.attrs.get(
                f"{var_path}_scale_factor", 1.0
            ),
            "offset": data_obj.attrs.get(f"{var_path}_offset", 0.0),
            "product": data_obj.attrs.get("ShortName", ""),
            "date": data_obj.attrs.get("RangeBeginningdate", ""),
            "lons": data_obj.x.values,
            "lats": data_obj.y.values,
        }

        return data, metadata, data_obj


def prepare_data(
    path: str,
    variable_name: str,
    log_scale: bool = True,
    region=None,
    region_crs: int = 4326,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and prepare data from an h5 file.

    Args:
        path (str): Path to the h5 file
        variable_name (str): Name of the variable to extract
        log_scale (bool): Whether to apply log scaling
        region: Optional region to clip the data to
        region_crs: Coordinate reference system of the region

    Returns:
        tuple: (data, lons, lats) arrays for plotting
    """
    data, metadata, _ = load_data_from_h5(
        path, variable_name, region, region_crs
    )

    # First replace fill values with NaN, then apply scaling and offset
    if metadata["fill_value"] is not None:
        data = np.where(
            np.isclose(data, metadata["fill_value"]) | np.isnan(data),
            np.nan,
            data,
        )
    data = data * metadata["scale_factor"] + metadata["offset"]

    # Apply log scaling if requested
    if log_scale:
        data = np.log(data + 1)  # Add 1 to avoid log(0)

    # Get the first band (assuming single band data)
    if data.ndim > 2:
        data = data[0]  # Get the first band if there are multiple bands

    return data, metadata["lons"], metadata["lats"]


def create_dataarray_from_raw(
    data: np.ndarray, lons: np.ndarray, lats: np.ndarray, attrs: dict = None
) -> xr.DataArray:
    """
    Create an xarray DataArray from raw data arrays.

    Args:
        data (np.ndarray): Data array
        lons (np.ndarray): Longitude coordinates
        lats (np.ndarray): Latitude coordinates
        attrs (dict): Attributes to add to the DataArray

    Returns:
        xr.DataArray: DataArray with the data and coordinates
    """
    # Make sure the dimensions match
    if data.shape[0] != len(lats) or data.shape[1] != len(lons):
        # If data shape doesn't match coordinates, we need to fix it
        if isinstance(data, np.ndarray) and data.ndim == 2:
            # If data is a 2D array, we can create new coordinates that match
            print(
                f"Warning: Data shape {data.shape} doesn't match coordinates ({len(lats)}, {len(lons)}). Creating new coordinates."
            )
            # Create new coordinate arrays that match the data dimensions
            if len(lats) > 1 and len(lons) > 1:
                lat_step = (lats[-1] - lats[0]) / (len(lats) - 1)
                lon_step = (lons[-1] - lons[0]) / (len(lons) - 1)
                new_lats = np.linspace(
                    lats[0],
                    lats[0] + lat_step * (data.shape[0] - 1),
                    data.shape[0],
                )
                new_lons = np.linspace(
                    lons[0],
                    lons[0] + lon_step * (data.shape[1] - 1),
                    data.shape[1],
                )
                lats, lons = new_lats, new_lons
            else:
                # If we have only one coordinate, we need to create a new array
                lats = np.linspace(
                    lats[0] if len(lats) > 0 else 0,
                    lats[0] + 1 if len(lats) > 0 else 1,
                    data.shape[0],
                )
                lons = np.linspace(
                    lons[0] if len(lons) > 0 else 0,
                    lons[0] + 1 if len(lons) > 0 else 1,
                    data.shape[1],
                )

    return xr.DataArray(
        data, dims=["y", "x"], coords={"y": lats, "x": lons}, attrs=attrs or {}
    )


def group_files_by_date(files: List[str]) -> Dict[str, List[str]]:
    """Group files by their date.

    Args:
        files (list): List of h5 file paths

    Returns:
        dict: Dictionary mapping dates to lists of files
    """
    from tqdm import tqdm
    from collections import defaultdict
    
    files_by_date = defaultdict(list)

    for file in tqdm(files, desc="Grouping files by date"):
        try:
            with rxr.open_rasterio(file) as data_obj:
                date = data_obj.attrs.get("RangeBeginningDate", "unknown")
                files_by_date[date].append(file)
        except Exception as e:
            print(f"Error reading date from {file}: {e}")

    return files_by_date
