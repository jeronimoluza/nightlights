import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import rioxarray as rxr
from pathlib import Path
import xarray as xr
from shapely.geometry import box
from tqdm import tqdm
from collections import defaultdict
import imageio.v2 as imageio
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional


# Constants
PREFIX = "HDFEOS_GRIDS_VIIRS_Grid_DNB_2d_Data_Fields_"
DEFAULT_CMAP = "cividis"
DEFAULT_BUFFER = 0.2  # degrees

# Map styling parameters
MAP_BACKGROUND_COLOR = "white"  # Background color for cartopy maps
MAP_COASTLINE_COLOR = "black"   # Color for coastlines
MAP_BORDER_COLOR = "gray"       # Color for country borders
MAP_STATE_COLOR = "gray"        # Color for state/province borders
MAP_GRID_COLOR = "gray"         # Color for gridlines
MAP_WATER_COLOR = "lightblue"   # Color for water bodies (ocean, lakes, rivers)


def load_data_from_h5(
    path: str, variable_name: str, region=None, region_crs: int = 4326
) -> Tuple[np.ndarray, dict, xr.DataArray]:
    """
    Load raw data and metadata from an h5 file.

    Args:
        path (str): Path to the h5 file
        variable_name (str): Name of the variable to extract

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
                    "scale_factor": data_obj.attrs.get(f"{var_path}_scale_factor", 1.0),
                    "offset": data_obj.attrs.get(f"{var_path}_offset", 0.0),
                    "product": data_obj.attrs.get("ShortName", ""),
                    "date": data_obj.attrs.get("RangeBeginningdate", ""),
                    "lons": data_var.x.values,  # Use clipped coordinates
                    "lats": data_var.y.values,   # Use clipped coordinates
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
    path: str, variable_name: str, log_scale: bool = True, region=None, region_crs: int = 4326
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and prepare data from an h5 file.

    Args:
        path (str): Path to the h5 file
        variable_name (str): Name of the variable to extract
        log_scale (bool): Whether to apply log scaling

    Returns:
        tuple: (data, lons, lats) arrays for plotting
    """
    data, metadata, _ = load_data_from_h5(path, variable_name, region, region_crs)

    # First replace fill values with NaN, then apply scaling and offset
    if metadata["fill_value"] is not None:
        data = np.where(np.isclose(data, metadata["fill_value"]) | np.isnan(data), np.nan, data)
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
            print(f"Warning: Data shape {data.shape} doesn't match coordinates ({len(lats)}, {len(lons)}). Creating new coordinates.")
            # Create new coordinate arrays that match the data dimensions
            if len(lats) > 1 and len(lons) > 1:
                lat_step = (lats[-1] - lats[0]) / (len(lats) - 1)
                lon_step = (lons[-1] - lons[0]) / (len(lons) - 1)
                new_lats = np.linspace(lats[0], lats[0] + lat_step * (data.shape[0] - 1), data.shape[0])
                new_lons = np.linspace(lons[0], lons[0] + lon_step * (data.shape[1] - 1), data.shape[1])
                lats, lons = new_lats, new_lons
            else:
                # If we have only one coordinate, we need to create a new array
                lats = np.linspace(lats[0] if len(lats) > 0 else 0, lats[0] + 1 if len(lats) > 0 else 1, data.shape[0])
                lons = np.linspace(lons[0] if len(lons) > 0 else 0, lons[0] + 1 if len(lons) > 0 else 1, data.shape[1])
    
    return xr.DataArray(
        data, dims=["y", "x"], coords={"y": lats, "x": lons}, attrs=attrs or {}
    )


def setup_map_figure(
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure with a map projection and common features.

    Args:
        figsize (tuple): Figure size

    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set background color
    ax.set_facecolor(MAP_BACKGROUND_COLOR)

    # Add water bodies
    ax.add_feature(cfeature.OCEAN, facecolor=MAP_WATER_COLOR)
    ax.add_feature(cfeature.LAKES, facecolor=MAP_WATER_COLOR)
    ax.add_feature(cfeature.RIVERS, edgecolor=MAP_WATER_COLOR, linewidth=0.5)

    # Add coastlines, borders, and other features
    ax.coastlines(resolution="10m", color=MAP_COASTLINE_COLOR, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor=MAP_BORDER_COLOR)
    ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor=MAP_STATE_COLOR)

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color=MAP_GRID_COLOR,
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    return fig, ax


def set_map_extent(
    ax: plt.Axes,
    bounds: Union[Tuple, xr.DataArray, np.ndarray],
    buffer: float = DEFAULT_BUFFER,
):
    """
    Set the extent of a map axis based on data bounds or region bounds.

    Args:
        ax (plt.Axes): Matplotlib axis
        bounds (tuple or DataArray): Either (min_lon, min_lat, max_lon, max_lat) or a DataArray with x and y coordinates
        buffer (float): Buffer to add around the extent in degrees
    """
    if isinstance(bounds, tuple) and len(bounds) == 4:
        # Region bounds
        min_lon, min_lat, max_lon, max_lat = bounds
    elif hasattr(bounds, "x") and hasattr(bounds, "y"):
        # DataArray with coordinates
        min_lon, max_lon = bounds.x.min().item(), bounds.x.max().item()
        min_lat, max_lat = bounds.y.min().item(), bounds.y.max().item()
    else:
        # Arrays of lons and lats
        min_lon, max_lon = np.min(bounds[0]), np.max(bounds[0])
        min_lat, max_lat = np.min(bounds[1]), np.max(bounds[1])

    ax.set_extent(
        [
            min_lon - buffer,
            max_lon + buffer,
            min_lat - buffer,
            max_lat + buffer,
        ],
        crs=ccrs.PlateCarree(),
    )


def add_colorbar(
    ax: plt.Axes, mesh, variable_name: str, log_scale: bool = True
):
    """
    Add a colorbar to a plot.

    Args:
        ax (plt.Axes): Matplotlib axis
        mesh: Mesh object from pcolormesh
        variable_name (str): Name of the variable for the label
        log_scale (bool): Whether the data is log-scaled

    Returns:
        colorbar: The created colorbar
    """
    cbar = plt.colorbar(mesh, ax=ax, pad=0.01, shrink=0.8)
    if log_scale:
        cbar.set_label(f"Log radiance (nW·cm$^{-2}$·sr$^{-1}$)")
    else:
        cbar.set_label(f"Radiance (nW·cm$^{-2}$·sr$^{-1}$)")
    return cbar


def plot_nightlights(
    file_path: str,
    variable_name: str,
    output_dir: str = None,
    log_scale: bool = True,
    cmap: str = DEFAULT_CMAP,
    vmin: float = None,
    vmax: float = None,
    region=None,
    region_crs: int = 4326,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot nightlights data from an h5 file using cartopy and matplotlib.

    Args:
        file_path (str): Path to the h5 file
        variable_name (str): Name of the variable to plot
        output_dir (str, optional): Directory to save the plot. If None, the plot will be displayed.
        log_scale (bool): Whether to apply log scaling
        cmap (str): Colormap to use for the plot
        vmin (float, optional): Minimum value for color scaling
        vmax (float, optional): Maximum value for color scaling

    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    # Extract data from the h5 file
    data, lons, lats = prepare_data(file_path, variable_name, log_scale, region, region_crs)

    # Create a figure with a map projection and common features
    fig, ax = setup_map_figure()

    # Create a mesh grid for the longitudes and latitudes
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    # Plot the data using pcolormesh
    mesh = ax.pcolormesh(
        lon_mesh,
        lat_mesh,
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )

    # Add a colorbar
    add_colorbar(ax, mesh, variable_name, log_scale)

    # Set the extent to the data bounds
    set_map_extent(ax, (lons, lats))

    # Add title with filename
    filename = os.path.basename(file_path)
    title = f"Nightlights: {filename}\nVariable: {variable_name}"
    if log_scale:
        title += "\nLog Scale"
    plt.title(title)

    # Save the plot if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"{os.path.splitext(filename)[0]}_{variable_name}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
        plt.close(fig)

    return fig, ax


def plot_file(path: str, variable_name: str, output_dir: str, region=None, region_crs: int = 4326) -> bool:
    """Run the plotting function and save the result.

    Args:
        path (str): Path to the h5 file
        variable_name (str): Name of the variable to plot
        output_dir (str): Directory to save the plot

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract the filename for the plot title
        filename = os.path.basename(path)

        # Plot the file
        fig, ax = plot_nightlights(
            file_path=path, variable_name=variable_name, output_dir=output_dir,
            region=region, region_crs=region_crs
        )

        # Close the figure to free up memory
        plt.close(fig)

        print(f"Successfully plotted {filename}")
        return True
    except Exception as e:
        print(f"Error plotting {path}: {e}")
        return False


def interpret_timeliness(product: str) -> str:
    """Interprets the timeliness of the product.

    Args:
        product (str): Product name/code

    Returns:
        str: Human-readable product timeliness
    """
    product_types = {
        "VNP46A2": "Daily",
        "VNP46A3": "Monthly",
        "VNP46A4": "Annual",
    }

    for code, description in product_types.items():
        if code in product:
            return description

    return product


def plot_all_files(
    files: List[str],
    variable_name: str,
    title: str,
    output_dir: str,
    region=None,
) -> None:
    """Plot all individual files and also create combined plots and timelapse if applicable.

    Args:
        files (list): List of h5 file paths
        variable_name (str): Name of the variable to plot
        title (str): Title to display at the top of the plot
        output_dir (str): Directory to save the plots
        region (shapely.geometry.Polygon, optional): Region to filter by
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Group files by date for processing
    files_by_date = group_files_by_date(files)
    print(f"Found {len(files_by_date)} different dates in the data.")

    # Create directory for individual tile plots
    single_tiles_dir = os.path.join(output_dir, "single_tiles")
    os.makedirs(single_tiles_dir, exist_ok=True)

    # Create directory for combined plots
    combined_dir = os.path.join(output_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    # 1. Plot individual tiles (all tiles, all dates)
    print(f"Plotting {len(files)} files individually...")
    for file in tqdm(files, desc="Plotting individual files"):
        plot_file(file, variable_name, single_tiles_dir, region, region_crs=4326)

    # 2. Create combined plots for each date
    print("Creating combined plots for each date...")
    for date, date_files in tqdm(
        files_by_date.items(), desc="Processing dates for combined plots"
    ):
        # Create combined plot for this date (all tiles)
        date_output_path = os.path.join(
            combined_dir, f"combined_{date.replace('-','')}_all_tiles.png"
        )
        combined_data = process_and_plot_date(
            date_files,
            variable_name,
            f"{title}\ndate: {date}",
            date_output_path,
            None,  # No region filtering
        )

        # If region is provided, create filtered version for this date
        if region is not None and combined_data is not None:
            filtered_output_path = os.path.join(
                combined_dir,
                f"combined_{date.replace('-','')}_region_filtered.png",
            )
            filter_and_plot_region(
                combined_data,
                variable_name,
                f"{title}\ndate: {date}",
                filtered_output_path,
                region,
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


def create_timelapse_gif(
    files: List[str],
    variable_name: str,
    title: str,
    output_dir: str,
    region=None,
    region_crs: int = 4326,
    fps: float = 5.0,
    plot_series: bool = False,
) -> None:
    """Create a timelapse GIF from multiple dates of data, using only region-filtered data.

    Args:
        files_by_date (dict): Dictionary mapping dates to lists of files
        variable_name (str): Name of the variable to extract and plot
        title (str): Title to display at the top of the plot
        output_dir (str): Directory to save the plots and GIF
        region (shapely.geometry.Polygon): Region to filter by (required)
        fps (float): Frames per second for the GIF
        plot_series (bool): If True, show lineplots of data values for all dates below the map
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Group files by date for processing
    files_by_date = group_files_by_date(files)
    # 3. Create timelapse GIF if we have multiple dates
    if len(files_by_date) > 1 and region is not None:
        print(f"Creating timelapse GIF from {len(files_by_date)} dates...")
        print(f"Found {len(files_by_date)} different dates in the data.")
        if region is None:
            print("Error: Region is required for timelapse GIF creation.")
            return

        # Create timelapse directory
        timelapse_dir = os.path.join(output_dir, "timelapse")
        os.makedirs(timelapse_dir, exist_ok=True)

        # First, find global min and max values across all dates for consistent color scaling
        print("Finding global min/max values for consistent color scaling...")
        global_min = float("inf")
        global_max = float("-inf")

        # Process each date to get combined data and find global min/max
        date_data_dict = {}
        for date, date_files in tqdm(
            files_by_date.items(), desc="Processing dates for min/max"
        ):
            # Process files for this date
            combined_data = process_files_for_date(date_files, variable_name, region, region_crs)
            if combined_data is None:
                print(f"Warning: No valid data for date {date}, skipping")
                continue

            # We've already applied region clipping in process_files_for_date
            # Just use the data as is
            filtered_data = combined_data
            
            # Make sure any remaining fill values are properly masked
            if not np.isnan(filtered_data.min()):
                # Check for suspiciously high values that might be fill values
                # This is a fallback in case the fill value wasn't properly handled earlier
                suspicious_values = filtered_data > filtered_data.quantile(0.99) * 10
                if suspicious_values.any():
                    filtered_data = filtered_data.where(~suspicious_values)

            date_data_dict[date] = filtered_data

            # Update global min/max from filtered data
            if not np.isnan(filtered_data.min()):
                global_min = min(global_min, filtered_data.min().item())
            if not np.isnan(filtered_data.max()):
                global_max = max(global_max, filtered_data.max().item())

        if not date_data_dict:
            print("No valid data found for any date. Cannot create timelapse.")
            return

        # Create a frame for each date with consistent color scaling
        frame_paths = []
        sorted_dates = sorted(date_data_dict.keys())

        for date in tqdm(sorted_dates, desc="Creating timelapse frames"):
            filtered_data = date_data_dict[date]
            frame_path = create_frame(
                filtered_data,
                date,
                variable_name,
                title,
                timelapse_dir,
                region,
                region_crs,
                global_min,
                global_max,
                plot_series=plot_series,
                all_dates=sorted_dates,
                all_data=date_data_dict,
            )
            frame_paths.append(frame_path)

        if not frame_paths:
            print("No frames were created. Cannot create timelapse.")
            return

        # Create the GIF
        gif_path = os.path.join(output_dir, f"timelapse_{variable_name}.gif")
        create_gif(frame_paths, gif_path, fps)
        print(f"Timelapse GIF created at: {gif_path}")


def process_files_for_date(
    files: List[str], variable_name: str, region=None, region_crs: int = 4326
) -> xr.DataArray:
    """Process files for a specific date and return combined data.

    Args:
        files (list): List of h5 file paths for a specific date
        variable_name (str): Name of the variable to extract
        region (shapely.geometry.Polygon, optional): Region to filter by

    Returns:
        xarray.DataArray: Combined data for the date
    """
    combined_data = None

    for file in files:
        try:
            # Extract data and prepare it
            data, lons, lats = prepare_data(
                file, variable_name, log_scale=True, region=region, region_crs=region_crs
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


def process_and_plot_date(
    files: List[str],
    variable_name: str,
    title: str,
    output_path: str,
    region=None,
    region_crs: int = 4326,
) -> xr.DataArray:
    """Process files for a specific date and create a plot.

    Args:
        files (list): List of h5 file paths for a specific date
        variable_name (str): Name of the variable to extract and plot
        title (str): Title for the plot
        output_path (str): Path to save the plot
        region (shapely.geometry.Polygon, optional): Region to filter by

    Returns:
        xarray.DataArray: Combined data for the date
    """
    # Process and combine the files
    combined_data = process_files_for_date(files, variable_name, region, region_crs)

    if combined_data is None:
        print(f"Warning: No valid data found for plot {output_path}")
        return None

    # Create the plot
    fig, ax = setup_map_figure()

    # Plot the data (with auto scaling for this specific plot)
    mesh = ax.pcolormesh(
        combined_data.x,
        combined_data.y,
        combined_data,
        cmap=DEFAULT_CMAP,
        transform=ccrs.PlateCarree(),
    )

    # Add a colorbar
    add_colorbar(ax, mesh, variable_name, log_scale=True)

    # Set the extent based on region or data bounds
    if region is not None:
        set_map_extent(ax, region.bounds)
    else:
        set_map_extent(ax, combined_data)

    # Add title
    plt.title(title)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close(fig)

    return combined_data


def filter_and_plot_region(
    data: xr.DataArray,
    variable_name: str,
    title: str,
    output_path: str,
    region,
) -> xr.DataArray:
    """Filter data by region and create a plot.

    Args:
        data (xarray.DataArray): Data to filter and plot
        variable_name (str): Name of the variable being plotted
        title (str): Title for the plot
        output_path (str): Path to save the plot
        region (shapely.geometry.Polygon): Region to filter by

    Returns:
        xarray.DataArray: Filtered data
    """
    # Apply region filtering
    min_lon, min_lat, max_lon, max_lat = region.bounds
    mask = (
        (data.x < min_lon)
        | (data.x > max_lon)
        | (data.y < min_lat)
        | (data.y > max_lat)
    )
    filtered_data = data.where(~mask)

    # Create the plot
    fig, ax = setup_map_figure()

    # Plot the filtered data (with auto scaling for this specific plot)
    mesh = ax.pcolormesh(
        filtered_data.x,
        filtered_data.y,
        filtered_data,
        cmap=DEFAULT_CMAP,
        transform=ccrs.PlateCarree(),
    )

    # Add a colorbar
    add_colorbar(ax, mesh, variable_name, log_scale=True)

    # Set the extent to the region bounds
    set_map_extent(ax, region.bounds)

    # Add title
    plt.title(title)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close(fig)

    return filtered_data


def create_frame(
    data: xr.DataArray,
    date: str,
    variable_name: str,
    title: str,
    output_dir: str,
    region=None,
    region_crs: int = 4326,
    vmin: float = None,
    vmax: float = None,
    plot_series: bool = False,
    all_dates: List[str] = None,
    all_data: Dict[str, xr.DataArray] = None,
) -> str:
    """Create a single frame for the timelapse.

    Args:
        data (xarray.DataArray): Data to plot
        date (str): date string for the frame
        variable_name (str): Name of the variable being plotted
        title (str): Title to display at the top of the plot
        output_dir (str): Directory to save the frame
        region (shapely.geometry.Polygon, optional): Region used for filtering
        vmin (float): Minimum value for color scaling
        vmax (float): Maximum value for color scaling
        plot_series (bool): If True, show lineplots of data values for all dates below the map
        all_dates (List[str]): List of all dates in the timelapse (required if plot_series is True)
        all_data (Dict[str, xr.DataArray]): Dictionary of all data by date (required if plot_series is True)

    Returns:
        str: Path to the saved frame image
    """
    # Format date for display
    date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")

    if plot_series and all_dates and all_data:
        # Create figure with two subplots - map on top, line plot on bottom
        fig = plt.figure(figsize=(12, 12))
        
        # Create a simple 2-row grid with height ratios 3:1
        gs = plt.GridSpec(2, 1, figure=fig, height_ratios=[5, 1])
        
        # Map subplot takes up 75% of the height
        ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        
        # Line plot subplot takes up 25% of the height
        ax_box = fig.add_subplot(gs[1])
        
        # Set up map features on the map subplot
        ax_map.set_facecolor(MAP_BACKGROUND_COLOR)
        
        # Add water bodies
        ax_map.add_feature(cfeature.OCEAN, facecolor=MAP_WATER_COLOR)
        ax_map.add_feature(cfeature.LAKES, facecolor=MAP_WATER_COLOR)
        ax_map.add_feature(cfeature.RIVERS, edgecolor=MAP_WATER_COLOR, linewidth=0.5)
        
        # Add coastlines, borders, and other features
        ax_map.coastlines(resolution="10m", color=MAP_COASTLINE_COLOR, linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor=MAP_BORDER_COLOR)
        ax_map.add_feature(cfeature.STATES, linewidth=0.2, edgecolor=MAP_STATE_COLOR)
        
        # Add gridlines
        gl = ax_map.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color=MAP_GRID_COLOR,
            alpha=0.5,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        
        # Use ax_map for the map
        ax = ax_map
    else:
        # Create the plot with common features (single plot)
        fig, ax = setup_map_figure()

    # Plot the data with consistent color scaling
    mesh = ax.pcolormesh(
        data.x,
        data.y,
        data,
        cmap=DEFAULT_CMAP,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )

    # Add a colorbar
    add_colorbar(ax, mesh, variable_name, log_scale=True)

    # Set the extent based on region or data bounds
    if region is not None:
        set_map_extent(ax, region.bounds)
    else:
        set_map_extent(ax, data)

    # Add title to the map
    if plot_series and all_dates and all_data:
        ax.set_title(f"{title}\ndate: {date}\nVariable: {variable_name}")
        
        # Create line plot for mean and median values across all dates
        date_objects = []
        mean_values = []
        median_values = []
        
        # Get data values for each date
        for d in all_dates:
            # Get non-NaN values for this date
            values = all_data[d].values.flatten()
            values = values[~np.isnan(values)]
            if len(values) > 0:
                # Apply log transformation to match the map data
                # Add 1 to avoid log(0) just like in prepare_data function
                #values = np.log(values + 1)
                
                # Convert date string to datetime object for proper plotting
                date_obj = datetime.strptime(d, "%Y-%m-%d")
                date_objects.append(date_obj)
                
                # Calculate mean and median
                mean_values.append(np.mean(values))
                median_values.append(np.median(values))
        
        # Create line plot if we have data
        if date_objects and mean_values and median_values:
            # Plot mean values in red
            ax_box.plot(date_objects, mean_values, 'r-', linewidth=2, label='Mean')
            
            # Plot median values in blue
            ax_box.plot(date_objects, median_values, 'b-', linewidth=2, label='Median')
            
            # Add vertical line for current date
            current_date_obj = datetime.strptime(date, "%Y-%m-%d")
            ax_box.axvline(x=current_date_obj, color='gray', linestyle='--', linewidth=1.5, 
                          label='Current Date')
            
            # Format x-axis to show dates nicely
            ax_box.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax_box.get_xticklabels(), rotation=45, ha='right')
            
            # Add legend, title and labels
            ax_box.legend(loc='best')
            ax_box.set_title('Mean and Median Values Over Time')
            ax_box.set_ylabel(f'Radiance (nW·cm$^{-2}$·sr$^{-1}$)')
            ax_box.set_xlabel('Date')
            
            # Add grid for better readability
            ax_box.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust layout
            plt.tight_layout()
    else:
        # Add title for single plot
        plt.title(f"{title}\ndate: {date}\nVariable: {variable_name}")

    # Save the frame
    frame_path = os.path.join(output_dir, f"frame_{date.replace('-','')}.png")
    plt.savefig(frame_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return frame_path


def create_gif(
    frame_paths: List[str], output_path: str, fps: float = 1.0
) -> None:
    """Create a GIF from a list of image paths.

    Args:
        frame_paths (list): List of paths to frame images
        output_path (str): Path to save the GIF
        fps (float): Frames per second for the GIF
    """
    try:
        images = [imageio.imread(frame) for frame in frame_paths]
        imageio.mimsave(output_path, images, fps=fps, loop=0)
    except Exception as e:
        print(f"Error creating GIF: {e}")


def combine_and_plot_tiles(
    files: List[str],
    variable_name: str,
    title: str,
    output_dir: str,
    region=None,
    region_crs: int = 4326,
) -> xr.DataArray:
    """
    Combine data from all tiles, filter by region if provided, and create plots.

    Args:
        files (list): List of h5 file paths to combine
        variable_name (str): Name of the variable to extract and plot
        title (str): Title to display at the top of the plot
        output_dir (str): Directory to save the plots
        region (shapely.geometry.Polygon, optional): Region to filter by

    Returns:
        xarray.DataArray: Combined data (filtered by region if provided)
    """
    # Create output directory if it doesn't exist
    combined_dir = os.path.join(output_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    # Process and combine all files
    print("Combining tiles...")
    combined_data = process_files_for_date(files, variable_name, region, region_crs)

    if combined_data is None:
        print("Error: Failed to combine tiles. No valid data found.")
        return None

    # Get metadata from the first file for title
    try:
        with rxr.open_rasterio(files[0]) as data_obj:
            product = data_obj.attrs.get("ShortName", "")
            date = data_obj.attrs.get("RangeBeginningdate", "")
    except Exception as e:
        print(f"Warning: Could not extract metadata from files: {e}")
        product = ""
        date = ""

    # Create the unfiltered combined plot (all tiles)
    print("Creating combined plot (all tiles)...")
    fig, ax = setup_map_figure()

    # Plot the combined data
    mesh = ax.pcolormesh(
        combined_data.x,
        combined_data.y,
        combined_data,
        cmap=DEFAULT_CMAP,
        transform=ccrs.PlateCarree(),
    )

    # Add a colorbar
    add_colorbar(ax, mesh, variable_name, log_scale=True)

    # Set the extent to the data bounds
    set_map_extent(ax, combined_data)

    # Add title
    product_desc = interpret_timeliness(product)
    plt.title(
        f"{title}\ndate: {date}\nProduct: {product_desc}\nVariable: {variable_name}"
    )

    # Save the plot
    output_path = os.path.join(
        combined_dir, f"combined_{variable_name}_all_tiles.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Combined plot saved to: {output_path}")
    plt.close(fig)

    # If region is provided, create the filtered version
    if region is not None:
        print("Creating region-filtered plot...")

        # Apply region filtering
        min_lon, min_lat, max_lon, max_lat = region.bounds
        mask = (
            (combined_data.x < min_lon)
            | (combined_data.x > max_lon)
            | (combined_data.y < min_lat)
            | (combined_data.y > max_lat)
        )
        combined_data_filtered = combined_data.where(~mask)

        # Plot the filtered data
        fig, ax = setup_map_figure()

        # Plot the combined filtered data
        mesh = ax.pcolormesh(
            combined_data_filtered.x,
            combined_data_filtered.y,
            combined_data_filtered,
            cmap=DEFAULT_CMAP,
            transform=ccrs.PlateCarree(),
        )

        # Add a colorbar
        add_colorbar(ax, mesh, variable_name, log_scale=True)

        # Set the extent to the region bounds
        set_map_extent(ax, region.bounds)

        # Add title
        plt.title(
            f"{title}\ndate: {date}\nProduct: {product_desc}\nVariable: {variable_name}"
        )

        # Save the plot
        output_path = os.path.join(
            combined_dir, f"combined_{variable_name}_region_filtered.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Combined filtered plot saved to: {output_path}")
        plt.close(fig)

        return combined_data_filtered
    else:
        return combined_data
