from tqdm import tqdm
import os
import datetime
import pandas as pd
import numpy as np
import rioxarray as rxr
from shapely import Polygon


def process_file(path: str, bounding_box: tuple = None) -> pd.DataFrame:
    """
    This function processes a raster file containing Black Marble data and returns a DataFrame with the extracted information.

    Args:
        path (str): The path to the raster file to be processed.
        bounding_box (tuple, optional): A tuple containing the bounding box coordinates in the format
            (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat). If provided, only data within
            this bounding box will be returned. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted information from the raster file.
    """
    with rxr.open_rasterio(path) as data_obj:
        filename = path.split("/")[-1].replace(".h5", "")
        Tile = [x for x in filename.split(".") if ("h" in x) and ("v" in x)][0]
        Julian_date = filename.split(".")[1].replace("A", "")
        Calendar_date = (
            datetime.datetime.strptime(Julian_date, "%Y%j")
            .date()
            .strftime("%Y-%m-%d")
        )

        HorizontalTileNumber = data_obj.HorizontalTileNumber
        VerticalTileNumber = data_obj.VerticalTileNumber

        WestBoundCoord = (10 * HorizontalTileNumber) - 180
        NorthBoundCoord = 90 - (10 * VerticalTileNumber)
        EastBoundCoord = WestBoundCoord + 10
        SouthBoundCoord = NorthBoundCoord - 10

        longitude_pixels = data_obj.sizes["x"]
        latitude_pixels = data_obj.sizes["y"]

        long_pixel_length = (
            EastBoundCoord - WestBoundCoord
        ) / longitude_pixels
        lat_pixel_length = (
            NorthBoundCoord - SouthBoundCoord
        ) / latitude_pixels

        def _prestep_clean(df):
            return df.drop("spatial_ref", axis=1).droplevel(level="band")

        def clean_column_name(colname):
            return colname.replace(
                "HDFEOS_GRIDS_VIIRS_Grid_DNB_2d_Data_Fields_", ""
            )

        variables = list(data_obj.var().variables)
        clean_names = [clean_column_name(col) for col in variables]

        cleaner = dict(zip(variables, clean_names))

        data = (
            data_obj.to_dataframe()
            .rename(columns=cleaner)[sorted(clean_names)]
            .reset_index()
        )

        data["latitude"] = data["y"]
        data["longitude"] = data["x"]

        # Create date variable (Julian and calendar format)
        data = data.assign(Julian_Date=Julian_date)
        data = data.assign(Date=Calendar_date)
        data = data.assign(tile=Tile)

        data["latcorner0"] = pd.to_numeric(data["latitude"] - (lat_pixel_length / 2))
        data["loncorner0"] = pd.to_numeric(data["longitude"] - (long_pixel_length / 2))

        data["latcorner1"] = pd.to_numeric(data["latitude"] - (lat_pixel_length / 2))
        data["loncorner1"] = pd.to_numeric(data["longitude"] + (long_pixel_length / 2))

        data["latcorner2"] = pd.to_numeric(data["latitude"] + (lat_pixel_length / 2))
        data["loncorner2"] = pd.to_numeric(data["longitude"] + (long_pixel_length / 2))

        data["latcorner3"] = pd.to_numeric(data["latitude"] + (lat_pixel_length / 2))
        data["loncorner3"] = pd.to_numeric(data["longitude"] - (long_pixel_length / 2))

        """ pyarrow won't write uint dtypes"""
        ints = data.select_dtypes(exclude=[float, object]).columns.tolist()
        data[ints] = data[ints].astype(np.float32)

        data = data.drop(
            [
                "x",
                "y",
                "band",
                "spatial_ref",
            ],
            axis=1,
        )
        data["filename"] = filename
        
        print(f"Total data points before filtering: {len(data)}")
        # Filter by bounding box if provided
        if bounding_box is not None:
            min_lon, min_lat, max_lon, max_lat = bounding_box
            lon_lat_filter = (
                data["longitude"].between(min_lon, max_lon) & 
                data["latitude"].between(min_lat, max_lat)
            )
            data = data[lon_lat_filter]
            
        return data

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


def process_files(list_of_files: list, output_dir: str, bounding_box: tuple = None) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(list_of_files)} files.")
    for file in list_of_files:
        filename = file.split("/")[-1]
        output_file = filename.replace(".h5", ".csv")
        df = process_file(file, bounding_box=bounding_box)
        df = corners_to_geometry(df)
        df.to_csv(f"{output_dir}/{output_file}", index=None)

    print(f"Processed files are in folder {output_dir}")
