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
DEFAULT_CMAP = 'cividis'
DEFAULT_BUFFER = 0.5  # degrees


def load_data_from_h5(path: str, variable_name: str) -> Tuple[np.ndarray, dict, xr.DataArray]:
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
        data = data_obj[var_path].data
        
        # Extract metadata
        metadata = {
            'fill_value': data_obj.attrs.get(f"{var_path}__FillValue"),
            'scale_factor': data_obj.attrs.get(f"{var_path}_scale_factor", 1.0),
            'offset': data_obj.attrs.get(f"{var_path}_offset", 0.0),
            'product': data_obj.attrs.get("ShortName", ""),
            'date': data_obj.attrs.get("RangeBeginningDate", ""),
            'lons': data_obj.x.values,
            'lats': data_obj.y.values
        }
        
        return data, metadata, data_obj


def prepare_data(path: str, variable_name: str, log_scale: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and prepare data from an h5 file.
    
    Args:
        path (str): Path to the h5 file
        variable_name (str): Name of the variable to extract
        log_scale (bool): Whether to apply log scaling
        
    Returns:
        tuple: (data, lons, lats) arrays for plotting
    """
    data, metadata, _ = load_data_from_h5(path, variable_name)
    
    # First replace fill values with NaN, then apply scaling and offset
    data = np.where(data == metadata['fill_value'], np.nan, data)
    data = data * metadata['scale_factor'] + metadata['offset']
    
    # Apply log scaling if requested
    if log_scale:
        data = np.log(data + 1)  # Add 1 to avoid log(0)
    
    # Get the first band (assuming single band data)
    if data.ndim > 2:
        data = data[0]  # Get the first band if there are multiple bands
    
    return data, metadata['lons'], metadata['lats']


def create_dataarray_from_raw(data: np.ndarray, lons: np.ndarray, lats: np.ndarray, attrs: dict = None) -> xr.DataArray:
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
    return xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={
            'y': lats,
            'x': lons
        },
        attrs=attrs or {}
    )


def setup_map_figure(figsize: Tuple[int, int] = (12, 8)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure with a map projection and common features.
    
    Args:
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add coastlines, borders, and other features
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')
    ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor='gray')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    return fig, ax


def set_map_extent(ax: plt.Axes, bounds: Union[Tuple, xr.DataArray, np.ndarray], buffer: float = DEFAULT_BUFFER):
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
    elif hasattr(bounds, 'x') and hasattr(bounds, 'y'):
        # DataArray with coordinates
        min_lon, max_lon = bounds.x.min().item(), bounds.x.max().item()
        min_lat, max_lat = bounds.y.min().item(), bounds.y.max().item()
    else:
        # Arrays of lons and lats
        min_lon, max_lon = np.min(bounds[0]), np.max(bounds[0])
        min_lat, max_lat = np.min(bounds[1]), np.max(bounds[1])
    
    ax.set_extent([min_lon-buffer, max_lon+buffer, min_lat-buffer, max_lat+buffer], crs=ccrs.PlateCarree())


def add_colorbar(ax: plt.Axes, mesh, variable_name: str, log_scale: bool = True):
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
        cbar.set_label(f'{variable_name} Value (log scale)')
    else:
        cbar.set_label(f'{variable_name} Value')
    return cbar


def plot_nightlights(file_path: str, variable_name: str, output_dir: str = None, log_scale: bool = True, 
                 cmap: str = DEFAULT_CMAP, vmin: float = None, vmax: float = None) -> Tuple[plt.Figure, plt.Axes]:
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
    data, lons, lats = prepare_data(file_path, variable_name, log_scale)
    
    # Create a figure with a map projection and common features
    fig, ax = setup_map_figure()
    
    # Create a mesh grid for the longitudes and latitudes
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    # Plot the data using pcolormesh
    mesh = ax.pcolormesh(lon_mesh, lat_mesh, data, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    
    # Add a colorbar
    add_colorbar(ax, mesh, variable_name, log_scale)
    
    # Set the extent to the data bounds
    set_map_extent(ax, (lons, lats))
    
    # Add title with filename
    filename = os.path.basename(file_path)
    title = f'Nightlights: {filename}\nVariable: {variable_name}'
    if log_scale:
        title += '\nLog Scale'
    plt.title(title)
    
    # Save the plot if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{variable_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        plt.close(fig)
    
    return fig, ax


def plot_file(path: str, variable_name: str, output_dir: str) -> bool:
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
            file_path=path,
            variable_name=variable_name,
            output_dir=output_dir
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
        "VNP46A1": "Daily",
        "VNP46A2": "Daily Gap-Filled",
        "VNP46A3": "Monthly",
        "VNP46A4": "Annual"
    }
    
    for code, description in product_types.items():
        if code in product:
            return description
    
    return product


def plot_all_files(files: List[str], variable_name: str, upper_title: str, output_dir: str, region=None) -> None:
    """Plot all individual files and also create combined plots and timelapse if applicable.
    
    Args:
        files (list): List of h5 file paths
        variable_name (str): Name of the variable to plot
        upper_title (str): Title to display at the top of the plot
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
        plot_file(file, variable_name, single_tiles_dir)
    
    # 2. Create combined plots for each date
    print("Creating combined plots for each date...")
    for date, date_files in tqdm(files_by_date.items(), desc="Processing dates for combined plots"):
        # Create combined plot for this date (all tiles)
        date_output_path = os.path.join(combined_dir, f"combined_{date}_all_tiles.png")
        combined_data = process_and_plot_date(
            date_files, 
            variable_name, 
            f"{upper_title}\nDate: {date}", 
            date_output_path,
            None  # No region filtering
        )
        
        # If region is provided, create filtered version for this date
        if region is not None and combined_data is not None:
            filtered_output_path = os.path.join(combined_dir, f"combined_{date}_region_filtered.png")
            filter_and_plot_region(
                combined_data,
                variable_name,
                f"{upper_title}\nDate: {date}",
                filtered_output_path,
                region
            )
    
    # 3. Create timelapse GIF if we have multiple dates
    if len(files_by_date) > 1 and region is not None:
        print(f"Creating timelapse GIF from {len(files_by_date)} dates...")
        create_timelapse_gif(files_by_date, variable_name, upper_title, output_dir, region)


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


def create_timelapse_gif(files_by_date: Dict[str, List[str]], variable_name: str, upper_title: str, 
                       output_dir: str, region=None, duration: float = 1.0) -> None:
    """Create a timelapse GIF from multiple dates of data, using only region-filtered data.
    
    Args:
        files_by_date (dict): Dictionary mapping dates to lists of files
        variable_name (str): Name of the variable to extract and plot
        upper_title (str): Title to display at the top of the plot
        output_dir (str): Directory to save the plots and GIF
        region (shapely.geometry.Polygon): Region to filter by (required)
        duration (float): Duration of each frame in seconds
    """
    if region is None:
        print("Error: Region is required for timelapse GIF creation.")
        return
        
    # Create timelapse directory
    timelapse_dir = os.path.join(output_dir, "timelapse")
    os.makedirs(timelapse_dir, exist_ok=True)
    
    # First, find global min and max values across all dates for consistent color scaling
    print("Finding global min/max values for consistent color scaling...")
    global_min = float('inf')
    global_max = float('-inf')
    
    # Process each date to get combined data and find global min/max
    date_data_dict = {}
    for date, date_files in tqdm(files_by_date.items(), desc="Processing dates for min/max"):
        # Process files for this date
        combined_data = process_files_for_date(date_files, variable_name)
        if combined_data is None:
            print(f"Warning: No valid data for date {date}, skipping")
            continue
        
        # Apply region filtering
        min_lon, min_lat, max_lon, max_lat = region.bounds
        mask = ((combined_data.x < min_lon) | (combined_data.x > max_lon) | 
                (combined_data.y < min_lat) | (combined_data.y > max_lat))
        filtered_data = combined_data.where(~mask)
        
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
        frame_path = create_frame(filtered_data, date, variable_name, upper_title, 
                                 timelapse_dir, region, global_min, global_max)
        frame_paths.append(frame_path)
    
    if not frame_paths:
        print("No frames were created. Cannot create timelapse.")
        return
        
    # Create the GIF
    gif_path = os.path.join(output_dir, f"timelapse_{variable_name}.gif")
    create_gif(frame_paths, gif_path, duration)
    print(f"Timelapse GIF created at: {gif_path}")


def process_files_for_date(files: List[str], variable_name: str, region=None) -> xr.DataArray:
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
            data, lons, lats = prepare_data(file, variable_name, log_scale=True)
            
            # Create xarray DataArray
            da = create_dataarray_from_raw(data, lons, lats)
            
            # First file - initialize the combined array
            if combined_data is None:
                combined_data = da
            else:
                # Try to align and merge with existing data
                # For simplicity, we'll take the maximum value at each pixel
                combined_data = xr.concat([combined_data, da], dim='tile')
                combined_data = combined_data.max(dim='tile', skipna=True)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Apply region filtering if provided
    if region is not None and combined_data is not None:
        min_lon, min_lat, max_lon, max_lat = region.bounds
        mask = ((combined_data.x < min_lon) | (combined_data.x > max_lon) | 
                (combined_data.y < min_lat) | (combined_data.y > max_lat))
        combined_data = combined_data.where(~mask)
    
    return combined_data


def process_and_plot_date(files: List[str], variable_name: str, title: str, output_path: str, region=None) -> xr.DataArray:
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
    combined_data = process_files_for_date(files, variable_name)
    
    if combined_data is None:
        print(f"Warning: No valid data found for plot {output_path}")
        return None
    
    # Create the plot
    fig, ax = setup_map_figure()
    
    # Plot the data (with auto scaling for this specific plot)
    mesh = ax.pcolormesh(combined_data.x, combined_data.y, combined_data, 
                       cmap=DEFAULT_CMAP, transform=ccrs.PlateCarree())
    
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close(fig)
    
    return combined_data


def filter_and_plot_region(data: xr.DataArray, variable_name: str, title: str, output_path: str, region) -> xr.DataArray:
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
    mask = ((data.x < min_lon) | (data.x > max_lon) | 
            (data.y < min_lat) | (data.y > max_lat))
    filtered_data = data.where(~mask)
    
    # Create the plot
    fig, ax = setup_map_figure()
    
    # Plot the filtered data (with auto scaling for this specific plot)
    mesh = ax.pcolormesh(filtered_data.x, filtered_data.y, filtered_data, 
                       cmap=DEFAULT_CMAP, transform=ccrs.PlateCarree())
    
    # Add a colorbar
    add_colorbar(ax, mesh, variable_name, log_scale=True)
    
    # Set the extent to the region bounds
    set_map_extent(ax, region.bounds)
    
    # Add title
    plt.title(title)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close(fig)
    
    return filtered_data


def create_frame(data: xr.DataArray, date: str, variable_name: str, upper_title: str, 
               output_dir: str, region=None, vmin: float = None, vmax: float = None) -> str:
    """Create a single frame for the timelapse.
    
    Args:
        data (xarray.DataArray): Data to plot
        date (str): Date string for the frame
        variable_name (str): Name of the variable being plotted
        upper_title (str): Title to display at the top of the plot
        output_dir (str): Directory to save the frame
        region (shapely.geometry.Polygon, optional): Region used for filtering
        vmin (float): Minimum value for color scaling
        vmax (float): Maximum value for color scaling
        
    Returns:
        str: Path to the saved frame image
    """
    # Format date for display
    date = datetime.strptime(date, "%Y-%m-%d")
    
    # Create the plot with common features
    fig, ax = setup_map_figure()
    
    # Plot the data with consistent color scaling
    mesh = ax.pcolormesh(data.x, data.y, data, cmap=DEFAULT_CMAP, 
                       transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
    
    # Add a colorbar
    add_colorbar(ax, mesh, variable_name, log_scale=True)
    
    # Set the extent based on region or data bounds
    if region is not None:
        set_map_extent(ax, region.bounds)
    else:
        set_map_extent(ax, data)
    
    # Add title
    plt.title(f'{upper_title}\nDate: {date}\nVariable: {variable_name}')
    
    # Save the frame
    frame_path = os.path.join(output_dir, f"frame_{date}.png")
    plt.savefig(frame_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return frame_path


def create_gif(frame_paths: List[str], output_path: str, duration: float = 1.0) -> None:
    """Create a GIF from a list of image paths.
    
    Args:
        frame_paths (list): List of paths to frame images
        output_path (str): Path to save the GIF
        duration (float): Duration of each frame in seconds
    """
    try:
        images = [imageio.imread(frame) for frame in frame_paths]
        imageio.mimsave(output_path, images, duration=duration, loop=0)
    except Exception as e:
        print(f"Error creating GIF: {e}")


def combine_and_plot_tiles(files: List[str], variable_name: str, upper_title: str, output_dir: str, region=None) -> xr.DataArray:
    """
    Combine data from all tiles, filter by region if provided, and create plots.
    
    Args:
        files (list): List of h5 file paths to combine
        variable_name (str): Name of the variable to extract and plot
        upper_title (str): Title to display at the top of the plot
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
    combined_data = process_files_for_date(files, variable_name)
    
    if combined_data is None:
        print("Error: Failed to combine tiles. No valid data found.")
        return None
    
    # Get metadata from the first file for title
    try:
        with rxr.open_rasterio(files[0]) as data_obj:
            product = data_obj.attrs.get("ShortName", "")
            date = data_obj.attrs.get("RangeBeginningDate", "")
    except Exception as e:
        print(f"Warning: Could not extract metadata from files: {e}")
        product = ""
        date = ""
    
    # Create the unfiltered combined plot (all tiles)
    print("Creating combined plot (all tiles)...")
    fig, ax = setup_map_figure()
    
    # Plot the combined data
    mesh = ax.pcolormesh(combined_data.x, combined_data.y, combined_data, 
                       cmap=DEFAULT_CMAP, transform=ccrs.PlateCarree())
    
    # Add a colorbar
    add_colorbar(ax, mesh, variable_name, log_scale=True)
    
    # Set the extent to the data bounds
    set_map_extent(ax, combined_data)
    
    # Add title
    product_desc = interpret_timeliness(product)
    plt.title(f'{upper_title}\nDate: {date}\nProduct: {product_desc}\nVariable: {variable_name}')
    
    # Save the plot
    output_path = os.path.join(combined_dir, f"combined_{variable_name}_all_tiles.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {output_path}")
    plt.close(fig)
    
    # If region is provided, create the filtered version
    if region is not None:
        print("Creating region-filtered plot...")
        
        # Apply region filtering
        min_lon, min_lat, max_lon, max_lat = region.bounds
        mask = ((combined_data.x < min_lon) | (combined_data.x > max_lon) | 
                (combined_data.y < min_lat) | (combined_data.y > max_lat))
        combined_data_filtered = combined_data.where(~mask)
        
        # Plot the filtered data
        fig, ax = setup_map_figure()
        
        # Plot the combined filtered data
        mesh = ax.pcolormesh(combined_data_filtered.x, combined_data_filtered.y, 
                          combined_data_filtered, cmap=DEFAULT_CMAP, transform=ccrs.PlateCarree())
        
        # Add a colorbar
        add_colorbar(ax, mesh, variable_name, log_scale=True)
        
        # Set the extent to the region bounds
        set_map_extent(ax, region.bounds)
        
        # Add title
        plt.title(f'{upper_title}\nDate: {date}\nProduct: {product_desc}\nVariable: {variable_name}')
        
        # Save the plot
        output_path = os.path.join(combined_dir, f"combined_{variable_name}_region_filtered.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined filtered plot saved to: {output_path}")
        plt.close(fig)
        
        return combined_data_filtered
    else:
        return combined_data