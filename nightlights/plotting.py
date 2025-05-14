"""Module for visualizing nightlights data."""

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import xarray as xr
from tqdm import tqdm
import imageio.v2 as imageio
from datetime import datetime
from typing import List, Tuple


# Constants
DEFAULT_CMAP = "cividis"
DEFAULT_BUFFER = 0.2  # degrees

# Map styling parameters
MAP_BACKGROUND_COLOR = "white"  # Background color for cartopy maps
MAP_COASTLINE_COLOR = "black"  # Color for coastlines
MAP_BORDER_COLOR = "gray"  # Color for country borders
MAP_STATE_COLOR = "gray"  # Color for state/province borders
MAP_GRID_COLOR = "gray"  # Color for gridlines
MAP_WATER_COLOR = "lightblue"  # Color for water bodies (ocean, lakes, rivers)

from nightlights.process import (
    process_files_for_date,
    prepare_data,
    create_dataarray_from_raw,
    group_files_by_date
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
    ax: plt.Axes, mesh, log_scale: bool = True
):
    """
    Add a colorbar to a plot.

    Args:
        ax (plt.Axes): Matplotlib axis
        mesh: The mesh object to colorbar
        log_scale (bool): Whether the data is log-scaled

    Returns:
        colorbar: The created colorbar
    """
    fig = ax.get_figure()
    cbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.8)
    if log_scale:
        cbar.set_label(f"Log radiance (nW·cm$^{-2}$·sr$^{-1}$)")
    else:
        cbar.set_label(f"Radiance (nW·cm$^{-2}$·sr$^{-1}$)")
    return cbar


def plot_nightlights(
    files: list,
    variable_name: str,
    date: str,
    output_dir: str = None,
    log_scale: bool = True,
    cmap: str = DEFAULT_CMAP,
    region=None,
    region_crs: int = 4326,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot nightlights data from a list of h5 files for a specific date using cartopy and matplotlib.

    Args:
        files (list): List of paths to h5 files
        variable_name (str): Name of the variable to plot
        date (str): Date string to filter files by (format: YYYY-MM-DD)
        output_dir (str, optional): Directory to save the plot. If None, the plot will be displayed.
        log_scale (bool): Whether to apply log scaling
        cmap (str): Colormap to use for the plot
        region (shapely.geometry.Polygon, optional): Region to filter by
        region_crs (int): Coordinate reference system of the region
        bins (int): Number of bins for color scaling
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    # Group files by date
    files_by_date = group_files_by_date(files)
    combined_data = None
    
    # Find files for the requested date
    if date in files_by_date:
        # Process files for this date
        combined_data = process_files_for_date(
            files_by_date[date], variable_name, log_scale=log_scale, region=region, region_crs=region_crs
        )
    else:
        print(f"No files found for date: {date}")
        print(f"Available dates: {list(files_by_date.keys())}")
        return None, None
    
    if combined_data is None:
        print(f"Failed to process data for date: {date}")
        return None, None
        
    # Extract data arrays from the combined data
    data = combined_data.values
    lons = combined_data.x.values
    lats = combined_data.y.values

    # Create a figure with a map projection and common features
    fig, ax = setup_map_figure()

    # Create a mesh grid for the longitudes and latitudes
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    # Prepare to create frames for each date with consistent color scaling
    levels = MaxNLocator(nbins=bins).tick_values(global_min, global_max)
    norm = BoundaryNorm(levels, ncolors=plt.colormaps[DEFAULT_CMAP].N, clip=True)

    # Plot the data using pcolormesh
    mesh = ax.pcolormesh(
        lon_mesh,
        lat_mesh,
        data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )

    # Add a colorbar
    add_colorbar(ax, mesh, log_scale=log_scale)

    # Set the extent to the data bounds
    set_map_extent(ax, (lons, lats))

    # Add title with date and variable information
    title = f"Nightlights: {date}\nDate: {date}\nVariable: {variable_name}"
    if log_scale:
        title += "\nLog Scale"
    plt.title(title)

    # Save the plot if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"nightlights_{date}_{variable_name}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
        plt.close(fig)

    return fig, ax



def create_timelapse_gif(
    files: List[str],
    variable_name: str,
    title: str,
    output_dir: str,
    region=None,
    region_crs: int = 4326,
    fps: float = 5.0,
    bins: int = 15,
) -> None:
    """Create a timelapse GIF from multiple dates of data, using only region-filtered data.

    Args:
        files (List[str]): List of file paths to process
        variable_name (str): Name of the variable to extract and plot
        title (str): Title to display at the top of the plot
        output_dir (str): Directory to save the plots and GIF
        region (shapely.geometry.Polygon): Region to filter by (required)
        region_crs (int): Coordinate reference system of the region
        fps (float): Frames per second for the GIF
        bins (int): Number of bins for color scaling
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
        for date, date_files in files_by_date.items():
            # Process files for this date
            combined_data = process_files_for_date(
                files=date_files, variable_name=variable_name, log_scale=True, region=region, region_crs=region_crs
            )
            if combined_data is None:
                print(f"Warning: No valid data for date {date}, skipping")
                continue

            # We've already applied region clipping in process_files_for_date
            # Just use the data as is
            filtered_data = combined_data

            date_data_dict[date] = filtered_data

            # Update global min/max from filtered data
            if not np.isnan(filtered_data.min()):
                global_min = min(global_min, filtered_data.min().item())
            if not np.isnan(filtered_data.max()):
                global_max = max(global_max, filtered_data.max().item())

        if not date_data_dict:
            print("No valid data found for any date. Cannot create timelapse.")
            return

        # Prepare to create frames for each date with consistent color scaling
        levels = MaxNLocator(nbins=bins).tick_values(global_min, global_max)
        norm = BoundaryNorm(levels, ncolors=plt.colormaps[DEFAULT_CMAP].N, clip=True)
        
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
                norm,
            )
            frame_paths.append(frame_path)

        if not frame_paths:
            print("No frames were created. Cannot create timelapse.")
            return

        # Create the GIF
        gif_path = os.path.join(output_dir, f"timelapse_{variable_name}.gif")
        create_gif(frame_paths, gif_path, fps)
        print(f"Timelapse GIF created at: {gif_path}")


def create_frame(
    data: xr.DataArray,
    date: str,
    variable_name: str,
    title: str,
    output_dir: str,
    region=None,
    norm: BoundaryNorm = None,
    cmap: str = DEFAULT_CMAP,
) -> str:
    """Create a single frame for the timelapse.

    Args:
        data (xarray.DataArray): Data to plot
        date (str): date string for the frame
        variable_name (str): Name of the variable being plotted
        title (str): Title to display at the top of the plot
        output_dir (str): Directory to save the frame
        region (shapely.geometry.Polygon, optional): Region used for filtering
        norm (BoundaryNorm): BoundaryNorm object for color scaling
        cmap (str): Colormap to use for the plot
    Returns:
        str: Path to the saved frame image
    """
    # Format date for display
    date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")

    fig, ax = setup_map_figure()

    # Plot the data with consistent color scaling
    mesh = ax.pcolormesh(
        data.x,
        data.y,
        data,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        norm=norm,
    )

    # Add a colorbar
    add_colorbar(ax, mesh, log_scale=True)

    # Set the extent based on region or data bounds
    if region is not None:
        set_map_extent(ax, region.bounds)
    else:
        set_map_extent(ax, data)

    # Add title for single plot
    ax.set_title(f"{title}\ndate: {date}\nVariable: {variable_name}")

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
