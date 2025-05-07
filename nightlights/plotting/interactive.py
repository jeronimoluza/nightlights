import folium
import geopandas as gpd
import pandas as pd
import numpy as np
from branca.colormap import LinearColormap
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex


def create_folium_map(gdf, variable_name, output_file, colormap='viridis'):
    """
    Creates a folium map from a geodataframe and saves it as an HTML file.
    
    Args:
        gdf (geopandas.GeoDataFrame): A geodataframe with 'geometry' and 'value' columns
        variable_name (str): The name of the variable being plotted
        output_file (str): Path to save the HTML output file
        colormap (str, optional): The colormap to use. Defaults to 'viridis'.
        
    Returns:
        str: Path to the saved HTML file
    """
    # Make a copy to avoid modifying the original
    gdf = gdf.copy()
    
    # Convert to GeoDataFrame if it's not already
    if not isinstance(gdf, gpd.GeoDataFrame):
        try:
            gdf = gpd.GeoDataFrame(gdf, geometry=gpd.GeoSeries.from_wkt(gdf['geometry']))
        except Exception as e:
            print(f"Error converting to GeoDataFrame: {e}")
            # Try another approach if the first one fails
            if 'geometry' in gdf.columns and isinstance(gdf['geometry'].iloc[0], str):
                gdf = gpd.GeoDataFrame(gdf, geometry=gpd.GeoSeries.from_wkt(gdf['geometry']))
    
    # Create a base map centered on the mean coordinates of the geodataframe
    try:
        bounds = gdf.geometry.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Calculate zoom level based on the bounding box size
        # This uses a simple formula that works well for most geographic data
        # The formula is based on the fact that each zoom level shows roughly 2x the area
        
        # Get the width and height in degrees
        width = bounds[2] - bounds[0]  # longitude extent
        height = bounds[3] - bounds[1]  # latitude extent
        
        # Use the larger dimension to determine zoom
        max_dimension = max(width, height)
        
        # Calculate zoom using a logarithmic scale
        # 360 degrees is the entire globe (zoom 0)
        # We use log base 2 because each zoom level doubles the detail
        zoom_level = int(np.log2(360 / max_dimension)) + 1
        
        # Clamp zoom level to reasonable values
        zoom_level = max(5, min(zoom_level, 15))
        
    except Exception as e:
        print(f"Error calculating map center and zoom: {e}")
        # Default to a reasonable center if there's an error
        center_lat, center_lon = -34.65, -59.43  # Default to region center
        zoom_level = 10
    
    # Create the map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level)
    
    # Ensure we have a 'value' column
    if 'value' not in gdf.columns:
        print("Warning: 'value' column not found in geodataframe. Using a default value.")
        gdf['value'] = 1.0
    
    # Get min and max values for the colormap
    min_val = gdf['value'].min()
    max_val = gdf['value'].max()
    
    # Generate a colormap
    cmap = plt.cm.get_cmap(colormap)
    colors = [to_hex(cmap(i)) for i in np.linspace(0, 1, 8)]
    
    # Create a colormap for the legend
    color_map = LinearColormap(
        colors=colors,
        vmin=min_val,
        vmax=max_val,
        caption=f'{variable_name} values'
    )
    
    # Add the colormap to the map
    color_map.add_to(m)
    
    # Add each polygon to the map with its own styling
    for idx, row in gdf.iterrows():
        # Calculate color based on value
        norm_value = (row['value'] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        fill_color = to_hex(cmap(norm_value))
        
        # Create a GeoJSON feature for this polygon
        folium.GeoJson(
            data=gpd.GeoDataFrame([row], geometry='geometry').__geo_interface__,
            style_function=lambda x, color=fill_color: {
                'fillColor': color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            },
            tooltip=folium.Tooltip(f"{variable_name}: {row['value']:.2f}")
        ).add_to(m)

    # Add layer control and other map elements
    folium.LayerControl().add_to(m)
    
    # Add a title to the map
    title_html = f'''
    <h3 align="center" style="font-size:16px"><b>{variable_name}</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save to HTML file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(output_file)
    
    print(f"Map saved to {output_file}")
    return output_file
