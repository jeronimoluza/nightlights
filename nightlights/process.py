"""Module for processing nightlights data from h5 files."""

from tqdm import tqdm
import os
import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray as rxr
from shapely import Polygon, MultiPolygon
from typing import Dict, List, Tuple, Union, Optional
from collections import defaultdict

PREFIX = "HDFEOS_GRIDS_VIIRS_Grid_DNB_2d_Data_Fields_"


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
        # Extract filename information for metadata
        filename = path.split("/")[-1].replace(".h5", "")
        tile = [x for x in filename.split(".") if ("h" in x) and ("v" in x)][0]
        julian_date = filename.split(".")[1].replace("A", "")

        try:
            calendar_date = (
                datetime.datetime.strptime(julian_date, "%Y%j")
                .date()
                .strftime("%Y-%m-%d")
            )
        except ValueError:
            calendar_date = ""

        # Extract horizontal and vertical tile numbers from the filename
        try:
            h_num = int(tile.split("h")[1].split("v")[0])
            v_num = int(tile.split("v")[1])
        except (IndexError, ValueError):
            h_num = 0
            v_num = 0

        # Calculate bounds based on tile numbers
        west_bound = (10 * h_num) - 180
        north_bound = 90 - (10 * v_num)
        east_bound = west_bound + 10
        south_bound = north_bound - 10

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
                    "filename": filename,
                    "tile": tile,
                    "julian_date": julian_date,
                    "calendar_date": calendar_date,
                    "horizontal_tile_number": h_num,
                    "vertical_tile_number": v_num,
                    "west_bound": west_bound,
                    "north_bound": north_bound,
                    "east_bound": east_bound,
                    "south_bound": south_bound,
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
            "filename": filename,
            "tile": tile,
            "julian_date": julian_date,
            "calendar_date": calendar_date,
            "horizontal_tile_number": h_num,
            "vertical_tile_number": v_num,
            "west_bound": west_bound,
            "north_bound": north_bound,
            "east_bound": east_bound,
            "south_bound": south_bound,
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
    files_by_date = defaultdict(list)

    for file in tqdm(files, desc="Grouping files by date"):
        try:
            with rxr.open_rasterio(file) as data_obj:
                date = data_obj.attrs.get("RangeBeginningDate", "unknown")
                files_by_date[date].append(file)
        except Exception as e:
            print(f"Error reading date from {file}: {e}")

    return files_by_date


# The process_file function has been replaced by polygonize_file


def corners_to_geometry(df):
    """
    This function converts the latitude and longitude corners of a DataFrame into a GeoDataFrame of polygons.

    Args:
        df (pandas.DataFrame): A DataFrame containing the following columns: 'loncorner0', 'latcorner0', 'loncorner1', 'latcorner1', 'loncorner2', 'latcorner2', 'loncorner3', 'latcorner3'.

    Returns:
        pd.DataFrame: A DataFrame containing the original DataFrame columns plus a new 'geometry' column, which contains the corresponding polygons.
    """
    polygons = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        corners = [
            (row["loncorner0"], row["latcorner0"]),
            (row["loncorner1"], row["latcorner1"]),
            (row["loncorner2"], row["latcorner2"]),
            (row["loncorner3"], row["latcorner3"]),
            (row["loncorner0"], row["latcorner0"]),
        ]

        polygon = Polygon(corners)
        polygons.append(polygon.wkt)

    df["geometry"] = polygons
    corners = [col for col in df.columns if "corner" in col]
    df = df.drop(corners, axis=1)
    return df


def polygonize_file(
    path: str,
    variable_name: str,
    region: Polygon | MultiPolygon = None,
    region_crs: int = 4326,
) -> gpd.GeoDataFrame:
    """
    Process a raster file containing Black Marble data and return a GeoDataFrame with polygonized pixels.
    Uses the existing utility functions to avoid code duplication.

    Args:
        path (str): Path to the raster file
        variable_name (str): Name of the variable to extract
        region (Polygon | MultiPolygon, optional): Region to clip the data to. Defaults to None.
        region_crs (int, optional): CRS of the region. Defaults to 4326.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the polygonized pixels from the raster file.
    """
    # Use load_data_from_h5 to get the data and metadata
    _, metadata, _ = load_data_from_h5(path, variable_name, region, region_crs)

    # Extract metadata information
    Tile = metadata["tile"]
    Calendar_date = metadata["calendar_date"]

    # Get bounds from metadata
    WestBoundCoord = metadata["west_bound"]
    NorthBoundCoord = metadata["north_bound"]
    EastBoundCoord = metadata["east_bound"]
    SouthBoundCoord = metadata["south_bound"]

    # Get dimensions
    longitude_pixels = len(metadata["lons"])
    latitude_pixels = len(metadata["lats"])

    # Calculate pixel size
    long_pixel_length = (EastBoundCoord - WestBoundCoord) / longitude_pixels
    lat_pixel_length = (NorthBoundCoord - SouthBoundCoord) / latitude_pixels

    # Prepare the data (apply scaling, offset, etc.)
    data, lons, lats = prepare_data(
        path,
        variable_name,
        log_scale=False,
        region=region,
        region_crs=region_crs,
    )

    # Create a list to store the pixel information
    pixels = []

    # Iterate through each pixel in the data array
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Get the pixel value
            value = data[i, j]

            # Skip NaN values
            if np.isnan(value):
                continue

            # Get coordinates
            lon = lons[j]
            lat = lats[i]

            # Calculate corners
            loncorner0 = lon - long_pixel_length / 2
            latcorner0 = lat - lat_pixel_length / 2
            loncorner1 = lon + long_pixel_length / 2
            latcorner1 = lat - lat_pixel_length / 2
            loncorner2 = lon + long_pixel_length / 2
            latcorner2 = lat + lat_pixel_length / 2
            loncorner3 = lon - long_pixel_length / 2
            latcorner3 = lat + lat_pixel_length / 2

            # Create polygon from corners
            corners = [
                (loncorner0, latcorner0),
                (loncorner1, latcorner1),
                (loncorner2, latcorner2),
                (loncorner3, latcorner3),
                (loncorner0, latcorner0),
            ]
            polygon = Polygon(corners)

            # Add the pixel information to the list
            pixels.append(
                {
                    "tile": Tile,
                    "date": Calendar_date,
                    "variable": variable_name,
                    "value": value,
                    "geometry": polygon.wkt,
                }
            )

    # Create a GeoDataFrame from the pixel information
    if pixels:
        df = pd.DataFrame(pixels)
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]), crs=region_crs
        )
        return gdf
    else:
        # Return an empty GeoDataFrame if no pixels were processed
        return gpd.GeoDataFrame(geometry=[], crs=region_crs)


def polygonize(
    list_of_files: list,
    variable_name: str,
    region: Polygon | MultiPolygon = None,
    region_crs: int = 4326,
    batch_process: bool = False,
    extraction_dir: str = None,
    output_dir: str = None,
) -> Union[gpd.GeoDataFrame, None]:
    """
    Polygonize multiple raster files containing Black Marble data.

    Args:
        list_of_files (list): List of paths to raster files
        variable_name (str): Name of the variable to extract
        region (Polygon | MultiPolygon, optional): Region to clip the data to. Defaults to None.
        region_crs (int, optional): CRS of the region. Defaults to 4326.
        batch_process (bool, optional): Whether to process files in batch mode, saving intermediate results. Defaults to False.
        extraction_dir (str, optional): Directory to save intermediate results. Required if batch_process is True.
        output_dir (str, optional): Directory to save final output. Required if batch_process is True.

    Returns:
        Union[gpd.GeoDataFrame, None]: A GeoDataFrame containing the polygonized pixels from all files, or None if batch_process is True.
    """
    if batch_process:
        if extraction_dir is None or output_dir is None:
            raise ValueError(
                "extraction_dir and output_dir must be provided when batch_process is True"
            )

        os.makedirs(extraction_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Batch processing {len(list_of_files)} files.")
        for file in tqdm(list_of_files, desc="Processing files"):
            filename = file.split("/")[-1]
            output_file = filename.replace(".h5", ".csv")
            gdf = polygonize_file(
                file,
                variable_name=variable_name,
                region=region,
                region_crs=region_crs,
            )
            gdf.to_csv(f"{extraction_dir}/{output_file}", index=None)

        files = os.listdir(extraction_dir)
        output = []

        for file in tqdm(files, desc="Combining results"):
            if file.endswith(".csv"):
                df = pd.read_csv(f"{extraction_dir}/{file}")
                output.append(df)

        if output:
            combined_output = pd.concat(output)
            combined_output.to_csv(f"{output_dir}/output.csv", index=None)
            print("Batch processing completed")
            print(f"Extracted files are at {extraction_dir}")
            print(f"Output file is at {output_dir}/output.csv")
        else:
            print("No files were processed")

        return None
    else:
        # Process all files and return a combined GeoDataFrame
        output = []
        for file in tqdm(list_of_files, desc="Processing files"):
            gdf = polygonize_file(
                file,
                variable_name=variable_name,
                region=region,
                region_crs=region_crs,
            )
            output.append(gdf)

        if output:
            data = pd.concat(output).reset_index(drop=True)
            data = gpd.GeoDataFrame(data, geometry="geometry")
            return data
        else:
            return gpd.GeoDataFrame()


def process_files_for_date(
    files: List[str], variable_name: str, region=None, region_crs: int = 4326
) -> xr.DataArray:
    """Process files for a specific date and return combined data.

    Args:
        files (list): List of h5 file paths for a specific date
        variable_name (str): Name of the variable to extract
        region (shapely.geometry.Polygon, optional): Region to filter by
        region_crs (int): Coordinate reference system of the region

    Returns:
        xarray.DataArray: Combined data for the date
    """
    combined_data = None

    for file in files:
        try:
            # Extract data and prepare it
            data, lons, lats = prepare_data(
                file,
                variable_name,
                log_scale=True,
                region=region,
                region_crs=region_crs,
            )

            # Create xarray DataArray
            da = create_dataarray_from_raw(data, lons, lats)

            # First file - initialize the combined array
            if combined_data is None:
                combined_data = da
            else:
                # Try to align and merge with existing data
                # For simplicity, we'll take the maximum value at each pixel
                combined_data = xr.concat([combined_data, da], dim="tile")
                combined_data = combined_data.max(dim="tile", skipna=True)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    return combined_data
