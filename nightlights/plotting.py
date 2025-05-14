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
from typing import List, Tuple, Union

from nightlights.process import (
    process_files_for_date,
    group_files_by_date,
)

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

LOG_RADIANCE_LABEL = f"Log radiance (nW·cm$^{-2}$·sr$^{-1}$)"
RADIANCE_LABEL = f"Radiance (nW·cm$^{-2}$·sr$^{-1}$)"


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


def add_colorbar(ax: plt.Axes, mesh, log_scale: bool = True):
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
        cbar.set_label(LOG_RADIANCE_LABEL)
    else:
        cbar.set_label(RADIANCE_LABEL)
    return cbar


def plot_nightlights(
    files: list,
    title: str,
    variable_name: str,
    date: str,
    output_dir: str = None,
    log_scale: bool = True,
    cmap: str = DEFAULT_CMAP,
    region=None,
    region_crs: int = 4326,
    bins: int = 15,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot nightlights data from a list of h5 files for a specific date using cartopy and matplotlib.

    Args:
        files (list): List of paths to h5 files
        title (str): Title to display at the top of the plot
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
            files_by_date[date],
            variable_name,
            log_scale=log_scale,
            region=region,
            region_crs=region_crs,
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

    global_min = np.nanmin(data)
    global_max = np.nanmax(data)

    # Prepare to create frames for each date with consistent color scaling
    levels = MaxNLocator(nbins=bins).tick_values(global_min, global_max)
    norm = BoundaryNorm(
        levels, ncolors=plt.colormaps[DEFAULT_CMAP].N, clip=True
    )

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
    title = f"{title}\nDate: {date}"
    plt.title(title)

    # Save the plot if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"nightlights_{date.replace('-','')}_{variable_name}.png"
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
        if region is None:
            print("Error: Region is required for timelapse GIF creation.")
            return

        # Create timelapse directory
        timelapse_dir = os.path.join(output_dir, "timelapse")
        os.makedirs(timelapse_dir, exist_ok=True)

        # First, find global min and max values across all dates for consistent color scaling
        global_min = float("inf")
        global_max = float("-inf")

        # Process each date to get combined data and find global min/max
        date_data_dict = {}
        for date, date_files in files_by_date.items():
            # Process files for this date
            combined_data = process_files_for_date(
                files=date_files,
                variable_name=variable_name,
                log_scale=True,
                region=region,
                region_crs=region_crs,
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
        norm = BoundaryNorm(
            levels, ncolors=plt.colormaps[DEFAULT_CMAP].N, clip=True
        )

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
    ax.set_title(f"{title}\nDate: {date}")

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


def create_lineplot(
    files: List[str],
    variable_name: str,
    title: str,
    output_dir: str,
    cut_off: float = 0.0,
    log_scale: bool = False,
    region=None,
    region_crs: int = 4326,
    functions: List[dict] = None,
    events: List[Tuple[str, str]] = None,
) -> str:
    """Create a lineplot showing the values of a variable over time, applying different functions to the data.

    Args:
        files (List[str]): List of file paths to process
        variable_name (str): Name of the variable to extract and plot
        title (str): Title for the plot
        output_dir (str): Directory to save the lineplot
        cut_off (float): Value to cut off the data at
        log_scale (bool): Whether to apply log scaling
        region: Optional region to filter data by (Polygon or MultiPolygon)
        region_crs (int): Coordinate reference system of the region
        functions (List[dict]): List of dictionaries with format {"label": function},
                               where function is a callable that processes the data
        events (List[Tuple[str, str]], optional): List of tuples containing (event_name, date)
                               to mark on the plot with vertical lines and annotations.
                               Date should be in format "YYYY-MM-DD".

    Returns:
        str: Path to the saved lineplot image
    """
    if not functions:
        functions = [{"Mean": np.mean}]  # Default function if none provided

    # Group files by date
    files_by_date = group_files_by_date(files)

    if not files_by_date:
        print("No valid files found for plotting.")
        return None

    # Create a dictionary to store processed data for each date and function
    date_values = {}

    # Process each date's data
    for date, date_files in sorted(files_by_date.items()):
        try:
            # Process files for this date
            combined_data = process_files_for_date(
                date_files,
                variable_name,
                log_scale=log_scale,
                region=region,
                region_crs=region_crs,
            )

            if combined_data is None or np.all(np.isnan(combined_data)):
                print(f"Warning: No valid data for date {date}, skipping")
                continue

            # Store the processed data for this date
            date_values[date] = combined_data
        except Exception as e:
            print(f"Error processing data for date {date}: {e}")

    if not date_values:
        print("No valid data found for any date. Cannot create lineplot.")
        return None

    # Calculate function values for each date
    results = {}
    for func_dict in functions:
        for label, func in func_dict.items():
            results[label] = []

    dates = sorted(date_values.keys())
    x_values = [datetime.strptime(date, "%Y-%m-%d") for date in dates]

    # Apply each function to each date's data
    for date in dates:
        data = date_values[date]
        # Remove NaN values for calculations
        valid_data = data.values[~np.isnan(data.values)]
        valid_data = valid_data[valid_data > cut_off]

        if len(valid_data) > 0:
            for func_dict in functions:
                for label, func in func_dict.items():
                    try:
                        value = func(valid_data)
                        results[label].append(value)
                    except Exception as e:
                        print(
                            f"Error applying function {label} to data for date {date}: {e}"
                        )
                        results[label].append(np.nan)
        else:
            # If no valid data, add NaN for all functions
            for func_dict in functions:
                for label in func_dict.keys():
                    results[label].append(np.nan)

    # Create the lineplot
    fig, ax = plt.subplots(figsize=(12, 8))

    for func_dict in functions:
        for label, _ in func_dict.items():
            ax.plot(
                x_values,
                results[label],
                marker="o",
                linestyle="-",
                label=label,
            )

    # Format the plot
    title = f"{title}"
    if cut_off > 0:
        title += f"\nCut-off Value: {cut_off}"
    ax.set_title(
        title
    )
    ax.set_xlabel("Date")
    if log_scale:
        ax.set_ylabel(LOG_RADIANCE_LABEL)
    else:
        ax.set_ylabel(RADIANCE_LABEL)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    # Add event markers if provided
    if events:
        # Get y-axis limits for positioning annotations
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        for event_name, event_date in events:
            try:
                # Convert event date string to datetime
                event_datetime = datetime.strptime(event_date, "%Y-%m-%d")

                # Plot vertical line for the event
                ax.axvline(
                    x=event_datetime, color="red", linestyle="--", alpha=0.7
                )

                # Add annotation at the top of the plot, to the right of the line
                ax.annotate(
                    event_name,
                    xy=(event_datetime, y_max - 0.05 * y_range),
                    xytext=(
                        event_datetime + (x_values[-1] - x_values[0]) * 0.02,
                        y_max - 0.05 * y_range,
                    ),
                    ha="left",
                    va="top",
                    color="red",
                    fontweight="bold",
                )
            except ValueError as e:
                print(f"Error plotting event {event_name}: {e}")

    # Format x-axis dates
    fig.autofmt_xdate()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"lineplot_{variable_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Lineplot created at: {output_path}")
    return output_path


def side_by_side(
    files: List[str],
    variable_name: str,
    title: str,
    date1: str,
    date2: str,
    region=None,
    region_crs: int = 4326,
    output_dir: str = None,
    bins: int = 15,
    log_scale: bool = True,
) -> str:
    """Create a side-by-side comparison of nightlights data for two dates.

    Args:
        files (List[str]): List of file paths to process
        variable_name (str): Name of the variable to extract and plot
        title (str): Title for the plot
        date1 (str): First date to compare (format: YYYY-MM-DD)
        date2 (str): Second date to compare (format: YYYY-MM-DD)
        region: Optional region to filter data by (Polygon or MultiPolygon)
        region_crs (int): Coordinate reference system of the region
        output_dir (str): Directory to save the output image
        bins (int): Number of bins for color scaling
        log_scale (bool): Whether to apply log scaling

    Returns:
        str: Path to the saved comparison image
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Group files by date
    files_by_date = group_files_by_date(files)

    if not files_by_date:
        print("No valid files found for plotting.")
        return None

    # Check if both dates exist in the data
    if date1 not in files_by_date:
        print(f"Date {date1} not found in the data.")
        return None
    if date2 not in files_by_date:
        print(f"Date {date2} not found in the data.")
        return None

    # Process data for both dates
    data1 = process_files_for_date(
        files_by_date[date1],
        variable_name,
        log_scale=log_scale,
        region=region,
        region_crs=region_crs,
    )
    data2 = process_files_for_date(
        files_by_date[date2],
        variable_name,
        log_scale=log_scale,
        region=region,
        region_crs=region_crs,
    )

    if data1 is None or np.all(np.isnan(data1)):
        print(f"No valid data for date {date1}.")
        return None
    if data2 is None or np.all(np.isnan(data2)):
        print(f"No valid data for date {date2}.")
        return None

    # Find global min and max for consistent color scaling
    global_min = min(data1.min().item(), data2.min().item())
    global_max = max(data1.max().item(), data2.max().item())

    # Create levels and norm for consistent color scaling
    levels = MaxNLocator(nbins=bins).tick_values(global_min, global_max)
    norm = BoundaryNorm(
        levels, ncolors=plt.colormaps[DEFAULT_CMAP].N, clip=True
    )

    # Create a figure with two subplots side by side, with minimal spacing
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    plt.subplots_adjust(wspace=0.05)  # Reduce space between subplots

    # Format date strings for display
    date1_display = datetime.strptime(date1, "%Y-%m-%d").strftime("%Y-%m-%d")
    date2_display = datetime.strptime(date2, "%Y-%m-%d").strftime("%Y-%m-%d")

    # Set up both maps with common features
    for ax in [ax1, ax2]:
        # Turn off the axes to remove blank space
        # ax.axis('off')

        # Set background color
        ax.set_facecolor(MAP_BACKGROUND_COLOR)

        # Add water bodies
        ax.add_feature(cfeature.OCEAN, facecolor=MAP_WATER_COLOR)
        ax.add_feature(cfeature.LAKES, facecolor=MAP_WATER_COLOR)
        ax.add_feature(
            cfeature.RIVERS, edgecolor=MAP_WATER_COLOR, linewidth=0.5
        )

        # Add coastlines, borders, and other features
        ax.coastlines(
            resolution="10m", color=MAP_COASTLINE_COLOR, linewidth=0.5
        )
        ax.add_feature(
            cfeature.BORDERS, linewidth=0.3, edgecolor=MAP_BORDER_COLOR
        )
        ax.add_feature(
            cfeature.STATES, linewidth=0.2, edgecolor=MAP_STATE_COLOR
        )

    # Plot data on the left subplot (date1)
    mesh1 = ax1.pcolormesh(
        data1.x,
        data1.y,
        data1,
        cmap=DEFAULT_CMAP,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )
    ax1.set_title(f"Date: {date1_display}")

    # Plot data on the right subplot (date2)
    mesh2 = ax2.pcolormesh(
        data2.x,
        data2.y,
        data2,
        cmap=DEFAULT_CMAP,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )
    ax2.set_title(f"Date: {date2_display}")

    # Set the extent for both maps
    if region is not None:
        for ax in [ax1, ax2]:
            set_map_extent(ax, region.bounds)
    else:
        # Use the combined extent of both datasets
        min_x = min(data1.x.min().item(), data2.x.min().item())
        max_x = max(data1.x.max().item(), data2.x.max().item())
        min_y = min(data1.y.min().item(), data2.y.min().item())
        max_y = max(data1.y.max().item(), data2.y.max().item())
        for ax in [ax1, ax2]:
            set_map_extent(ax, (min_x, min_y, max_x, max_y))

    # Add a single horizontal colorbar below the two maps
    cbar_ax = fig.add_axes(
        [0.15, 0.05, 0.7, 0.05]
    )  # [left, bottom, width, height]
    cbar = fig.colorbar(mesh2, cax=cbar_ax, orientation="horizontal")
    if log_scale:
        cbar.set_label(LOG_RADIANCE_LABEL)
    else:
        cbar.set_label(RADIANCE_LABEL)

    # Add overall title
    fig.suptitle(f"{title}", fontsize=16, y=0.95)

    # Save the figure
    if output_dir:
        output_path = os.path.join(
            output_dir,
            f"comparison_{variable_name}_{date1_display.replace('-','')}_{date2_display.replace('-','')}.png",
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Side-by-side comparison created at: {output_path}")
        return output_path
    else:
        return None
