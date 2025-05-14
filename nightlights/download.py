import os
import earthaccess
import nightlights.utils as utils
import geopandas as gpd
import pandas as pd
import osmnx as ox

AVAILABLE_SHORT_NAMES = ["VNP46A1", "VNP46A2", "VNP46A3", "VNP46A4"]


def earthaccess_login():
    """
    Logs the user in to earthaccess
    """
    return earthaccess.login()


def filter_granules(granules: list, start_date: str, end_date: str):
    """
    Filters the granules to only include those within the specified date range.
    """
    filtered_granules = []
    for g in granules:
        file_date = utils.get_file_date(g.data_links()[0])
        if utils.is_in_date_range(file_date, start_date, end_date):
            filtered_granules.append(g)
    return filtered_granules


def download_earthaccess(
    download_dir: str,
    short_name: str,
    start_date: str,
    end_date: str,
    region,
    version: str = "1",
    count: int = -1,
) -> list:
    """
    Downloads Earth observation data from the Earth Access platform.

    This function searches for and downloads MODIS granules within a specified region and date range.
    It uses the Earth Access Python library to perform the search and download operations.

    Parameters:
    - download_dir (str): The directory where the downloaded files will be saved.
    - short_name (str): The short name of the dataset (e.g., 'MOD13Q1').
    - version (str): The version of the dataset (e.g., '061'). Default is '1'.
    - start_date (str): The start date of the search period in 'YYYY-MM-DD' format.
    - end_date (str): The end date of the search period in 'YYYY-MM-DD' format.
    - region (Polygon, MultiPolygon, GeoDataFrame, or WKT string): The region of interest.
    - count (int, optional): The maximum number of granules to download. Default is -1 (all granules).

    Returns:
    - list: A list of downloaded file paths.
    """

    if short_name not in AVAILABLE_SHORT_NAMES:
        raise ValueError(f"short_name must be one of {AVAILABLE_SHORT_NAMES}")

    # VNP46A2 is version 2, all others are version 1 FOR NOW
    if short_name == "VNP46A2":
        version = "2"
    else:
        version = "1"

    _ = earthaccess_login()

    bounding_box = utils.get_bounding_box(region=region)

    download_dir = download_dir + f"/{short_name}_{version}"
    # Define download directory
    os.makedirs(download_dir, exist_ok=True)

    # Search for granules
    granules = earthaccess.search_data(
        short_name=short_name,
        version=version,
        temporal=(start_date, end_date),
        bounding_box=bounding_box,
        count=count,
    )

    granules = filter_granules(
        granules=granules, start_date=start_date, end_date=end_date
    )

    if len(granules) == 0:
        print("No granules found.")
        return
    else:
        print(f"Downloading {len(granules)} granules.")

        # Download granules
        files = earthaccess.download(granules, local_path=download_dir)

        print(f"Downloaded {len(files)} files to {download_dir}.")
        return files


def find_region(query: str | list) -> gpd.GeoDataFrame:
    """
    Find the boundary of a region using OpenStreetMap.

    Parameters:
    - query (str | list): The query to search for the region.

    Returns:
    - gpd.GeoDataFrame: The boundary of the region.
    """
    if isinstance(query, list):
        gdf = gpd.GeoDataFrame(
            pd.concat(
                [ox.geocode_to_gdf(query=region) for region in query],
                ignore_index=True,
            ),
            geometry="geometry",
        )
        return gdf
    else:
        gdf = ox.geocode_to_gdf(query=query)
        return gdf
