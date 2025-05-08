from tqdm import tqdm
import os
import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import rioxarray as rxr
from shapely import Polygon, MultiPolygon


def process_file(
    path: str, variable_name: str, region: Polygon | MultiPolygon = None, region_crs: int = 4326
) -> pd.DataFrame:
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
        prefix = "HDFEOS_GRIDS_VIIRS_Grid_DNB_2d_Data_Fields_"
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

        def clean_column_name(colname):
            return colname.replace(prefix, "")

        variables = list(data_obj.var().variables)
        clean_names = [clean_column_name(col) for col in variables]

        cleaner = dict(zip(variables, clean_names))

        fill_value = data_obj.attrs[f"{prefix}{variable_name}__FillValue"]
        scale_factor = data_obj.attrs[f"{prefix}{variable_name}_scale_factor"]
        offset_value = data_obj.attrs[f"{prefix}{variable_name}_offset"]
        data_obj = data_obj[f"{prefix}{variable_name}"].squeeze()

        if region is not None:
            data_obj = data_obj.rio.clip([region], region_crs, drop=True)

        data = (
            data_obj.to_dataframe()
            .rename(columns=cleaner)
            .reset_index()
        )

        data["latitude"] = data["y"]
        data["longitude"] = data["x"]

        # Create date variable (Julian and calendar format)
        data = data.assign(Julian_date=Julian_date)
        data = data.assign(date=Calendar_date)
        data = data.assign(tile=Tile)

        data["latcorner0"] = pd.to_numeric(
            data["latitude"] - (lat_pixel_length / 2)
        )
        data["loncorner0"] = pd.to_numeric(
            data["longitude"] - (long_pixel_length / 2)
        )

        data["latcorner1"] = pd.to_numeric(
            data["latitude"] - (lat_pixel_length / 2)
        )
        data["loncorner1"] = pd.to_numeric(
            data["longitude"] + (long_pixel_length / 2)
        )

        data["latcorner2"] = pd.to_numeric(
            data["latitude"] + (lat_pixel_length / 2)
        )
        data["loncorner2"] = pd.to_numeric(
            data["longitude"] + (long_pixel_length / 2)
        )

        data["latcorner3"] = pd.to_numeric(
            data["latitude"] + (lat_pixel_length / 2)
        )
        data["loncorner3"] = pd.to_numeric(
            data["longitude"] - (long_pixel_length / 2)
        )

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

        data = data.drop(
            [
                "latitude",
                "longitude",
            ],
            axis=1,
        )

        data["variable_name"] = variable_name
        data = data.rename(columns={variable_name: "value"})
        data["value"] = data["value"].replace(fill_value, np.nan)
        data["value"] = data["value"] * scale_factor + offset_value

        data = data[
            [
                "filename",
                "date",
                "loncorner0",
                "latcorner0",
                "loncorner1",
                "latcorner1",
                "loncorner2",
                "latcorner2",
                "loncorner3",
                "latcorner3",
                "value",
                "variable_name",
            ]
        ]

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


def process_files(
    list_of_files: list, variable_name: str, region: Polygon | MultiPolygon = None, region_crs: int = 4326
) -> gpd.GeoDataFrame:
    output = []
    for file in list_of_files:
        df = process_file(
            file, variable_name=variable_name, region=region, region_crs=region_crs
        )
        output.append(df)

    output = pd.concat(output).reset_index(drop=True)
    gdf = corners_to_geometry(output)
    return gdf


def batch_process_files(
    list_of_files: list,
    variable_name: str,
    extraction_dir: str,
    output_dir: str,
    region: Polygon | MultiPolygon = None, region_crs: int = 4326
) -> None:
    os.makedirs(extraction_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Batch processing {len(list_of_files)} files.")
    for file in list_of_files:
        filename = file.split("/")[-1]
        output_file = filename.replace(".h5", ".csv")
        df = process_file(
            file, variable_name=variable_name, region=region, region_crs=region_crs
        )
        df = corners_to_geometry(df)
        df.to_csv(f"{extraction_dir}/{output_file}", index=None)

    files = os.listdir(extraction_dir)
    output = []

    for file in files:
        df = pd.read_csv(f"{extraction_dir}/{file}")
        output.append(df)

    output = pd.concat(output)
    output.to_csv(f"{output_dir}/output.csv", index=None)
    print("Batch processing completed")
    print(f"Extracted files are at {extraction_dir}")
    print(f"Output file is at {output_dir}")

