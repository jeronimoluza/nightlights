import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import rioxarray as rxr
from pathlib import Path


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
        output_path = os.path.join(output_dir, f"{Path(filename).stem}_{variable_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    return fig, ax
