"""Module for visualizing nightlights data."""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import imageio.v2 as imageio
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional
import seaborn as sns
import pandas as pd

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
        mesh: Mesh object from pcolormesh
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
    files: list,
    variable_name: str,
    date: str,
    output_dir: str = None,
    log_scale: bool = True,
    cmap: str = DEFAULT_CMAP,
    vmin: float = None,
    vmax: float = None,
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
        vmin (float, optional): Minimum value for color scaling
        vmax (float, optional): Maximum value for color scaling
        region (shapely.geometry.Polygon, optional): Region to filter by
        region_crs (int): Coordinate reference system of the region

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
    plot_series: bool = False,
    use_confidence_interval: bool = False,
    confidence_level: float = 0.95,
    sample_size: int = None,
    cut_off: float = 0,
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
        plot_series (bool): If True, show lineplots of data values for all dates below the map
        use_confidence_interval (bool): If True, use confidence interval plot instead of mean/median
        confidence_level (float): Confidence level for the interval (0-1), default is 0.95 (95%)
        sample_size (int): Number of values to sample for each date
        cut_off (float): Minimum value to include in the series plot
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
                global_min,
                global_max,
                plot_series=plot_series,
                all_dates=sorted_dates,
                all_data=date_data_dict,
                use_confidence_interval=use_confidence_interval,
                confidence_level=confidence_level,
                sample_size=sample_size,
                cut_off=cut_off,
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
    vmin: float = None,
    vmax: float = None,
    plot_series: bool = False,
    all_dates: List[str] = None,
    all_data: Dict[str, xr.DataArray] = None,
    use_confidence_interval: bool = False,
    confidence_level: float = 0.95,
    sample_size: int = None,
    cut_off: float = 0,
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
        use_confidence_interval (bool): If True, use confidence interval plot instead of mean/median
        confidence_level (float): Confidence level for the interval (0-1), default is 0.95 (95%)
        sample_size (int): Number of values to sample for each date
        cut_off (float): Minimum value to include in the series plot

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
        ax_map.add_feature(
            cfeature.RIVERS, edgecolor=MAP_WATER_COLOR, linewidth=0.5
        )

        # Add coastlines, borders, and other features
        ax_map.coastlines(
            resolution="10m", color=MAP_COASTLINE_COLOR, linewidth=0.5
        )
        ax_map.add_feature(
            cfeature.BORDERS, linewidth=0.3, edgecolor=MAP_BORDER_COLOR
        )
        ax_map.add_feature(
            cfeature.STATES, linewidth=0.2, edgecolor=MAP_STATE_COLOR
        )

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
    add_colorbar(ax, mesh, log_scale=True)

    # Set the extent based on region or data bounds
    if region is not None:
        set_map_extent(ax, region.bounds)
    else:
        set_map_extent(ax, data)

    # Add title to the map
    if plot_series and all_dates and all_data:
        ax.set_title(f"{title}\nDate: {date}\nVariable: {variable_name}")
        if cut_off > 0:
            ax.set_title(f"{ax.get_title()}\nCut-off: {cut_off}")

        date_objects = []

        # Get data values for each date
        for d in all_dates:
            # Get non-NaN values for this date
            values = all_data[d].values.flatten()
            values = values[~np.isnan(values)]
            values = values[values >= cut_off]
            if len(values) > 0:
                # Apply log transformation to match the map data
                # Add 1 to avoid log(0) just like in prepare_data function
                # values = np.log(values + 1)

                # Convert date string to datetime object for proper plotting
                date_obj = datetime.strptime(d, "%Y-%m-%d")
                date_objects.append(date_obj)

        # Create line plot if we have data
        if use_confidence_interval:
            # Use the seaborn confidence interval plot
            plot_confidence_interval(
                ax_box,
                all_dates,
                all_data,
                date,
                confidence_level,
                sample_size=sample_size,
                cut_off=cut_off,
            )
        elif date_objects:
            # Add vertical line for current date
            current_date_obj = datetime.strptime(date, "%Y-%m-%d")
            ax_box.axvline(
                x=current_date_obj,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                label="Current Date",
            )

            # Format x-axis to show dates nicely
            ax_box.xaxis.set_major_formatter(
                plt.matplotlib.dates.DateFormatter("%Y-%m-%d")
            )
            plt.setp(ax_box.get_xticklabels(), rotation=45, ha="right")

            # Add legend, title and labels
            ax_box.legend(loc="best")
            ax_box.set_title("Mean and Median of Non-Zero Values Over Time")
            if cut_off > 0:
                ax_box.set_title(f"{ax_box.get_title()}\nCut-off: {cut_off}")
            ax_box.set_ylabel(f"Radiance (nW·cm$^{-2}$·sr$^{-1}$)")
            ax_box.set_xlabel("Date")

            # Add grid for better readability
            ax_box.grid(True, linestyle="--", alpha=0.7)

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


def plot_confidence_interval(
    ax, 
    all_dates: List[str], 
    all_data: Dict[str, xr.DataArray], 
    current_date: str,
    confidence_level: float = 0.95,
    fixed_y_limits: bool = True,
    sample_size: int = None,
    cut_off: float = 0,
) -> None:
    """Create a seaborn lineplot with confidence intervals for each date.
    
    Args:
        ax (matplotlib.axes.Axes): The axis to plot on
        all_dates (List[str]): List of all dates in the timelapse
        all_data (Dict[str, xr.DataArray]): Dictionary of all data by date
        current_date (str): The current date being displayed in the main plot
        confidence_level (float): Confidence level for the interval (0-1)
        fixed_y_limits (bool): Whether to use fixed y-axis limits across all frames
        sample_size (int): Number of values to sample for each date
        """
    # Prepare data for seaborn lineplot
    data_for_plot = []
    all_values = []
    
    # Process data for each date
    for date_str in all_dates:
        if date_str not in all_data or all_data[date_str] is None:
            print(f"Missing or invalid data for date: {date_str}")
            continue
            
        # Get non-NaN values for this date
        try:
            values = all_data[date_str].values.flatten()
            values = values[~np.isnan(values)]
            values = np.exp(values) - 1
 
            # Apply cut-off filter (use greater than or equal to)
            values = values[values >= cut_off]
            
            if len(values) > 0:
                # Store all values for y-axis limit calculation
                all_values.extend(values)
                
                # Create multiple entries for each date to allow for proper CI calculation
                # Sample the data if there are too many points to keep performance reasonable
                if sample_size is not None and len(values) > sample_size:
                    # Random sampling to reduce data size while preserving distribution
                    indices = np.random.choice(len(values), size=sample_size, replace=False)
                    values = values[indices]
                    
                # Create a dataframe entry for each value
                for val in values:
                    data_for_plot.append({
                        'date': date_str,
                        'value': val
                    })
            else:
                print(f"No valid values above cut-off for date: {date_str}")
        except Exception as e:
            print(f"Error processing data for date {date_str}: {e}")
            continue
    
    # Convert to pandas DataFrame
    if not data_for_plot:
        print("No valid data found for confidence interval plot")
        return
        
    df = pd.DataFrame(data_for_plot)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create the seaborn lineplot with confidence interval
    sns.lineplot(
        data=df,
        x='date',
        y='value',
        errorbar=("ci", confidence_level*100),
        # estimator='mean',
        color='blue',
        ax=ax
    )
    
    # Add vertical line for current date
    current_date_obj = pd.to_datetime(current_date)
    ax.axvline(
        x=current_date_obj,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label='Current Date'
    )
    
    # Calculate and set fixed y-axis limits if requested
    if fixed_y_limits and all_values:
        # Calculate percentiles to exclude extreme outliers
        y_min = np.mean(all_values) - np.std(all_values)
        y_max = np.mean(all_values) + np.std(all_values)
        
        # Add some padding
        y_range = y_max - y_min
        y_min = max(0, y_min - 0.05 * y_range)  # Ensure it's not negative
        y_max = y_max + 0.05 * y_range
        
        # Set the limits
        ax.set_ylim(y_min, y_max)
    
    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter('%Y-%m-%d')
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add legend, title and labels
    ax.legend(loc='best')
    ax.set_title(f'Nightlight Values with {confidence_level*100:.0f}% Confidence Interval')
    if cut_off > 0:
        ax.set_title(f"{ax.get_title()}\nCut-off: {cut_off}")
    ax.set_ylabel(f'Radiance (nW·cm$^{{-2}}$·sr$^{{-1}}$)')
    ax.set_xlabel('Date')
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)


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
