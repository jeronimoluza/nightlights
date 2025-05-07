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


def prepare_data(path: str, variable_name: str, log_scale: bool = True):
    """
    Extract and prepare data from an h5 file.
    
    Args:
        path (str): Path to the h5 file
        variable_name (str): Name of the variable to extract
        
    Returns:
        tuple: (data, lons, lats) arrays for plotting
    """
    with rxr.open_rasterio(path) as data_obj:
        prefix = "HDFEOS_GRIDS_VIIRS_Grid_DNB_2d_Data_Fields_"
        data = data_obj[f"{prefix}{variable_name}"].data
        fill_value = data_obj.attrs[f"{prefix}{variable_name}__FillValue"]
        scale_factor = data_obj.attrs[f"{prefix}{variable_name}_scale_factor"]
        offset_value = data_obj.attrs[f"{prefix}{variable_name}_offset"]
        
        # First replace fill values with NaN, then apply scaling and offset
        data = np.where(data == fill_value, np.nan, data)
        data = data * scale_factor + offset_value
        
        # Apply log scaling if requested
        if log_scale:
            data = np.log(data + 1)  # Add 1 to avoid log(0)
        
        lons = data_obj.x.values
        lats = data_obj.y.values
        return data, lons, lats


def plot_nightlights(file_path: str, variable_name: str, output_dir: str = None, log_scale: bool = True, cmap: str = 'cividis', vmin: float = None, vmax: float = None):
    """
    Plot nightlights data from an h5 file using cartopy and matplotlib.
    
    Args:
        file_path (str): Path to the h5 file
        variable_name (str): Name of the variable to plot
        output_dir (str, optional): Directory to save the plot. If None, the plot will be displayed.
        cmap (str, optional): Colormap to use for the plot. Defaults to 'viridis'.
        vmin (float, optional): Minimum value for color scaling. Defaults to None (auto-determined).
        vmax (float, optional): Maximum value for color scaling. Defaults to None (auto-determined).
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    # Extract data from the h5 file
    data, lons, lats = prepare_data(file_path, variable_name, log_scale)
    
    # Get the first band (assuming single band data)
    if data.ndim > 2:
        data = data[0]  # Get the first band if there are multiple bands
    
    # Create a figure with a map projection
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add coastlines, borders, and other features
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')
    ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor='gray')
    
    # Create a mesh grid for the longitudes and latitudes
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    # Plot the data using pcolormesh
    mesh = ax.pcolormesh(lon_mesh, lat_mesh, data, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    
    # Add a colorbar
    cbar = plt.colorbar(mesh, ax=ax, pad=0.01, shrink=0.8)
    if log_scale:
        cbar.set_label(f'{variable_name} Value (log scale)')
    cbar.set_label(f'{variable_name} Value')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Set the extent to the data bounds with a small buffer
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    buffer = 0.5  # degrees
    ax.set_extent([lon_min-buffer, lon_max+buffer, lat_min-buffer, lat_max+buffer], crs=ccrs.PlateCarree())
    
    # Add title with filename
    filename = os.path.basename(file_path)
    if log_scale:
        plt.title(f'Nightlights: {filename}\nVariable: {variable_name}\nLog Scale')
    plt.title(f'Nightlights: {filename}\nVariable: {variable_name}')
    
    # Save or display the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if log_scale:
            output_path = os.path.join(output_dir, f"{Path(filename).stem}_{variable_name}_log.png")
        output_path = os.path.join(output_dir, f"{Path(filename).stem}_{variable_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    return fig, ax


def plot_file(path:str, variable_name:str, output_dir:str):
    """Run the plotting function and display the result."""
    # Ensure the file exists
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return
    
    print(f"Plotting {variable_name} from {path}...")
    
    output_dir += "/single_tiles/"
    
    # Plot the data
    fig, ax = plot_nightlights(
        file_path=path,
        variable_name=variable_name,
        output_dir=output_dir,
        # cmap="cividis", # You can try other colormaps like 'plasma', 'inferno', 'magma', etc.
    )
    
    print("Done plotting!")

def interpret_timeliness(product):
    """Interprets the timeliness of the product."""
    if product == "VNP46A2":
        product += " (Daily Nightlights)"
    elif product == "VNP46A3":
        product += " (Monthly Nightlights)"
    elif product == "VNP46A4":
        product += " (Yearly Nightlights)"
    return product

def plot_all_files(files: list, variable_name: str, upper_title: str, output_dir: str, region=None):
    """Plot all individual files and also create combined plots."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Plotting {len(files)} files individually...")
    
    # Plot each file individually
    for file in files:
        plot_file(file, variable_name, output_dir)
    
    # Now create the combined plots
    print("Creating combined plots...")
    
    # Use the dedicated function for combined plots
    combined_data = combine_and_plot_tiles(files, variable_name, upper_title, output_dir, region)
    
    # Check if we have multiple dates for timelapse
    files_by_date = group_files_by_date(files)
    if len(files_by_date) > 1:
        print(f"Found {len(files_by_date)} different dates. Creating timelapse GIF...")
        create_timelapse_gif(files_by_date, variable_name, upper_title, output_dir, region)


def group_files_by_date(files: list):
    """Group files by their date.
    
    Args:
        files (list): List of h5 file paths
        
    Returns:
        dict: Dictionary mapping dates to lists of files
    """
    files_by_date = defaultdict(list)
    
    for file in files:
        with rxr.open_rasterio(file) as data_obj:
            date = data_obj.attrs["RangeBeginningDate"]
            files_by_date[date].append(file)
    
    return files_by_date


def create_timelapse_gif(files_by_date, variable_name, upper_title, output_dir, region=None):
    """Create a timelapse GIF from multiple dates of data.
    
    Args:
        files_by_date (dict): Dictionary mapping dates to lists of files
        variable_name (str): Name of the variable to extract and plot
        upper_title (str): Title to display at the top of the plot
        output_dir (str): Directory to save the plots and GIF
        region (shapely.geometry.Polygon, optional): Region to filter by. Defaults to None.
    """
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
        combined_data = process_files_for_date(date_files, variable_name, region)
        date_data_dict[date] = combined_data
        
        # Update global min/max
        if not np.isnan(combined_data.min()):
            global_min = min(global_min, combined_data.min().item())
        if not np.isnan(combined_data.max()):
            global_max = max(global_max, combined_data.max().item())
    
    # Create a frame for each date with consistent color scaling
    frame_paths = []
    sorted_dates = sorted(date_data_dict.keys())
    
    for date in tqdm(sorted_dates, desc="Creating frames"):
        combined_data = date_data_dict[date]
        frame_path = create_frame(combined_data, date, variable_name, upper_title, 
                                 timelapse_dir, region, global_min, global_max)
        frame_paths.append(frame_path)
    
    # Create the GIF
    gif_path = os.path.join(output_dir, f"timelapse_{variable_name}.gif")
    create_gif(frame_paths, gif_path)
    print(f"Timelapse GIF created at: {gif_path}")


def process_files_for_date(files, variable_name, region=None):
    """Process files for a specific date and return combined data.
    
    Args:
        files (list): List of h5 file paths for a specific date
        variable_name (str): Name of the variable to extract
        region (shapely.geometry.Polygon, optional): Region to filter by. Defaults to None.
        
    Returns:
        xarray.DataArray: Combined data for the date
    """
    combined_data = None
    
    for file in files:
        # Load the data
        with rxr.open_rasterio(file) as data_obj:
            prefix = "HDFEOS_GRIDS_VIIRS_Grid_DNB_2d_Data_Fields_"
            data = data_obj[f"{prefix}{variable_name}"].data
            fill_value = data_obj.attrs[f"{prefix}{variable_name}__FillValue"]
            scale_factor = data_obj.attrs[f"{prefix}{variable_name}_scale_factor"]
            offset_value = data_obj.attrs[f"{prefix}{variable_name}_offset"]
            
            # First replace fill values with NaN, then apply scaling and offset
            data = np.where(data == fill_value, np.nan, data)
            data = data * scale_factor + offset_value
            
            # Apply log scaling
            data = np.log(data + 1)  # Add 1 to avoid log(0)
            
            # Get the first band (assuming single band data)
            if data.ndim > 2:
                data = data[0]  # Get the first band if there are multiple bands
            
            lons = data_obj.x.values
            lats = data_obj.y.values
            
            # Create xarray DataArray for this file
            da = xr.DataArray(
                data,
                dims=['y', 'x'],
                coords={
                    'y': lats,
                    'x': lons
                },
                attrs=data_obj[f"{prefix}{variable_name}"].attrs
            )
        
        # First file - initialize the combined array
        if combined_data is None:
            combined_data = da
        else:
            # Try to align and merge with existing data
            # For simplicity, we'll take the maximum value at each pixel
            combined_data = xr.concat([combined_data, da], dim='tile')
            combined_data = combined_data.max(dim='tile', skipna=True)
    
    # Apply region filtering if provided
    if region is not None:
        min_lon, min_lat, max_lon, max_lat = region.bounds
        mask = ((combined_data.x < min_lon) | (combined_data.x > max_lon) | 
                (combined_data.y < min_lat) | (combined_data.y > max_lat))
        combined_data = combined_data.where(~mask)
    
    return combined_data


def create_frame(data, date, variable_name, upper_title, output_dir, region, vmin, vmax):
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
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%B %d, %Y")
    except:
        formatted_date = date
    
    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add coastlines, borders, and other features
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')
    ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor='gray')
    
    # Plot the data with consistent color scaling
    mesh = ax.pcolormesh(data.x, data.y, data, cmap='cividis', 
                       transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
    
    # Add a colorbar
    cbar = plt.colorbar(mesh, ax=ax, pad=0.01, shrink=0.8)
    cbar.set_label(f'{variable_name} Value (log scale)')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Set the extent based on region or data bounds
    if region is not None:
        min_lon, min_lat, max_lon, max_lat = region.bounds
        buffer = 0.5  # degrees
        ax.set_extent([min_lon-buffer, max_lon+buffer, min_lat-buffer, max_lat+buffer], crs=ccrs.PlateCarree())
    else:
        lon_min, lon_max = data.x.min().item(), data.x.max().item()
        lat_min, lat_max = data.y.min().item(), data.y.max().item()
        buffer = 0.5  # degrees
        ax.set_extent([lon_min-buffer, lon_max+buffer, lat_min-buffer, lat_max+buffer], crs=ccrs.PlateCarree())
    
    # Add title
    plt.title(f'{upper_title}\nDate: {formatted_date}\nVariable: {variable_name}')
    
    # Save the frame
    frame_path = os.path.join(output_dir, f"frame_{date}.png")
    plt.savefig(frame_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return frame_path


def create_gif(frame_paths, output_path, duration=1.0):
    """Create a GIF from a list of image paths.
    
    Args:
        frame_paths (list): List of paths to frame images
        output_path (str): Path to save the GIF
        duration (float, optional): Duration of each frame in seconds. Defaults to 1.0.
    """
    images = [imageio.imread(frame) for frame in frame_paths]
    imageio.mimsave(output_path, images, duration=duration, loop=0)


def combine_and_plot_tiles(files: list, variable_name: str, upper_title: str, output_dir: str, region=None):
    """
    Combine data from all tiles, filter by region if provided, and create plots.
    
    Args:
        files (list): List of h5 file paths to combine
        variable_name (str): Name of the variable to extract and plot
        output_dir (str): Directory to save the plots
        region (shapely.geometry.Polygon, optional): Region to filter by. Defaults to None.
    """
    # Create output directory if it doesn't exist
    output_dir += "/combined/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all data and combine
    print("Combining tiles...")
    combined_data = None
    
    for file in tqdm(files, desc="Processing files"):
        # Load the data
        with rxr.open_rasterio(file) as data_obj:
            prefix = "HDFEOS_GRIDS_VIIRS_Grid_DNB_2d_Data_Fields_"
            data = data_obj[f"{prefix}{variable_name}"].data
            fill_value = data_obj.attrs[f"{prefix}{variable_name}__FillValue"]
            scale_factor = data_obj.attrs[f"{prefix}{variable_name}_scale_factor"]
            offset_value = data_obj.attrs[f"{prefix}{variable_name}_offset"]
            product = data_obj.attrs["ShortName"]
            date = data_obj.attrs["RangeBeginningDate"]
            
            # First replace fill values with NaN, then apply scaling and offset
            data = np.where(data == fill_value, np.nan, data)
            data = data * scale_factor + offset_value
            
            # Apply log scaling
            data = np.log(data + 1)  # Add 1 to avoid log(0)
            
            # Get the first band (assuming single band data)
            if data.ndim > 2:
                data = data[0]  # Get the first band if there are multiple bands
            
            lons = data_obj.x.values
            lats = data_obj.y.values
            
            # Create xarray DataArray for this file
            da = xr.DataArray(
                data,
                dims=['y', 'x'],
                coords={
                    'y': lats,
                    'x': lons
                },
                attrs=data_obj[f"{prefix}{variable_name}"].attrs
            )
        
        # First file - initialize the combined array
        if combined_data is None:
            combined_data = da
        else:
            # Try to align and merge with existing data
            # For simplicity, we'll take the maximum value at each pixel
            combined_data = xr.concat([combined_data, da], dim='tile')
            combined_data = combined_data.max(dim='tile', skipna=True)
    
    # Create the unfiltered combined plot (all tiles)
    print("Creating combined plot (all tiles)...")
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add coastlines, borders, and other features
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')
    ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor='gray')
    
    # Plot the combined data
    mesh = ax.pcolormesh(combined_data.x, combined_data.y, combined_data, 
                        cmap='cividis', transform=ccrs.PlateCarree())
    
    # Add a colorbar
    cbar = plt.colorbar(mesh, ax=ax, pad=0.01, shrink=0.8)
    cbar.set_label(f'{variable_name} Value (log scale)')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Set the extent to the data bounds with a small buffer
    lon_min, lon_max = combined_data.x.min().item(), combined_data.x.max().item()
    lat_min, lat_max = combined_data.y.min().item(), combined_data.y.max().item()
    buffer = 0.5  # degrees
    ax.set_extent([lon_min-buffer, lon_max+buffer, lat_min-buffer, lat_max+buffer], crs=ccrs.PlateCarree())
    
    # Add title
    product = interpret_timeliness(product)
    plt.title(f'{upper_title}\nDate: {date}\nProduct: {product}\nVariable: {variable_name}')
    
    # Save the plot
    output_path = os.path.join(output_dir, f"combined_{variable_name}_all_tiles.png")
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
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add coastlines, borders, and other features
        ax.coastlines(resolution='10m', color='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')
        ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor='gray')
        
        # Plot the combined filtered data
        mesh = ax.pcolormesh(combined_data_filtered.x, combined_data_filtered.y, 
                           combined_data_filtered, cmap='cividis', transform=ccrs.PlateCarree())
        
        # Add a colorbar
        cbar = plt.colorbar(mesh, ax=ax, pad=0.01, shrink=0.8)
        cbar.set_label(f'{variable_name} Value (log scale)')
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Set the extent to the region bounds with a small buffer
        buffer = 0.5  # degrees
        ax.set_extent([min_lon-buffer, max_lon+buffer, min_lat-buffer, max_lat+buffer], crs=ccrs.PlateCarree())
        
        # Add title
        product = interpret_timeliness(product)
        plt.title(f'{upper_title}\nDate: {date}\nProduct: {product}\nVariable: {variable_name}\n')
        
        # Save the plot
        output_path = os.path.join(output_dir, f"combined_{variable_name}_region_filtered.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined filtered plot saved to: {output_path}")
        plt.close(fig)
        
        return combined_data_filtered
    else:
        return combined_data