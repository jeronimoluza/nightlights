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


def plot_all_files(files: list, variable_name: str, output_dir: str, region=None):
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
    combine_and_plot_tiles(files, variable_name, output_dir, region)


def combine_and_plot_tiles(files: list, variable_name: str, output_dir: str, region=None):
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
    plt.title(f'Combined Nightlights\nVariable: {variable_name}\nAll Tiles')
    
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
        plt.title(f'Combined Nightlights\nVariable: {variable_name}\nRegion Filtered')
        
        # Save the plot
        output_path = os.path.join(output_dir, f"combined_{variable_name}_region_filtered.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined filtered plot saved to: {output_path}")
        plt.close(fig)
    
    return combined_data