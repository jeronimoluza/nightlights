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

def get_file_tile(file_path: str) -> str:
    """
    Extracts the tile from the file name.
    """
    return file_path.split("/")[-1].split(".")[2]

def is_in_date_range(file_date, start_date: str, end_date: str) -> bool:
    """
    Checks if the file date is within the specified date range.
    """
    return (
        pd.to_datetime(start_date)
        <= pd.to_datetime(file_date)
        <= pd.to_datetime(end_date)
    )


# These functions have been moved to process.py
