import os
import earthaccess
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon
import geopandas as gpd


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


def login():
    """
    Logs the user in to earthaccess
    """
    return earthaccess.login()


def download_earthaccess(
    download_dir: str,
    short_name: str,
    version: str,
    start_date: str,
    end_date: str,
    region,
    count: int = 10,
) -> list:
    """
    Downloads Earth observation data from the Earth Access platform.

    This function searches for and downloads MODIS granules within a specified region and date range.
    It uses the Earth Access Python library to perform the search and download operations.

    Parameters:
    - download_dir (str): The directory where the downloaded files will be saved.
    - short_name (str): The short name of the dataset (e.g., 'MOD13Q1').
    - version (str): The version of the dataset (e.g., '061').
    - start_date (str): The start date of the search period in 'YYYY-MM-DD' format.
    - end_date (str): The end date of the search period in 'YYYY-MM-DD' format.
    - region (Polygon, MultiPolygon, GeoDataFrame, or WKT string): The region of interest.
    - count (int, optional): The maximum number of granules to download. Default is 10.

    Returns:
    - list: A list of downloaded file paths.
    """
    bounding_box = get_bounding_box(region=region)

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

    print(f"Found {len(granules)} granules (count parameter is {count}).")
    if len(granules) == 0:
        return
    else:
        print(f"Downloading {len(granules)} granules.")

        # Download granules
        files = earthaccess.download(granules, local_path=download_dir)

        print(f"Downloaded {len(files)} files to {download_dir}.")
        return files
