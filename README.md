# Nightlights

A Python toolkit for downloading, processing, and visualizing NASA's Black Marble Nightlights Suite (VIIRS) satellite data. This project provides tools to analyze nighttime light patterns over time, which can be used to study urbanization, economic activity, power outages, recovery from natural disasters, and more.

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Poetry (for dependency management)
- Earth Access account

### Setup

```bash
# Clone the repository
git clone https://github.com/jeronimoluza/nightlights.git
cd nightlights

# Install dependencies using Poetry
poetry install
```

## Usage

### 1. Set up directories and parameters

```python
from nightlights import download, plotting, process
import pandas as pd
import numpy as np

# Define directories
download_dir = "./data/raw"
plot_dir = "./data/plots"

# Define search parameters
short_name = "VNP46A3"  # VIIRS Black Marble product
start_date = "2023-04-15"
end_date = "2024-05-05"

# Define region of interest
regions = ["Andalucía, Spain"]
region_gdf = download.find_region(query=regions)
region_crs = region_gdf.crs.to_epsg()
region = region_gdf.union_all()
```

### 2. Download data

```python
files = download.download_earthaccess(
    download_dir=download_dir,
    short_name=short_name,
    start_date=start_date,
    end_date=end_date,
    region=region,
)

variable_name = "AllAngle_Composite_Snow_Free"  # for VNP46A3 and VNP46A4 or "DNB_BRDF-Corrected_NTL" for VNP46A2
```

### 3. Create visualizations

#### Time Series Lineplot with Events

```python
# Define functions to apply to the data
functions = [
    {"Mean": np.mean},
    {"Median": np.median},
]

# Define important events to mark on the plot
events = [("Event 1", "2023-10-01"), ("Event 2", "2024-02-01")]

# Create the lineplot
plotting.create_lineplot(
    files=files,
    variable_name=variable_name,
    title="Nightlights Trends",
    output_dir=plot_dir,
    region=region,
    region_crs=region_crs,
    functions=functions,
    events=events
)
```

#### Side-by-Side Comparison

```python
plotting.side_by_side(
    files=files,
    variable_name=variable_name,
    title="Nightlights Comparison",
    date1="2023-05-01",
    date2="2024-05-01",
    region=region,
    region_crs=region_crs,
    output_dir=plot_dir,
    bins=15,
    log_scale=True
)
```

#### Timelapse Animation

```python
plotting.create_timelapse_gif(
    files=files,
    variable_name=variable_name,
    title="Andalucía Nightlights",
    output_dir=plot_dir,
    region=region,
    region_crs=region_crs,
    fps=2.0,
)
```

#### Single Date Map

```python
plotting.plot_nightlights(
    files,
    variable_name=variable_name,
    date="2023-05-01",
    output_dir=plot_dir,
    region=region,
)
```

### 4. Advanced Processing: Polygonize Data

```python
# Convert raster data to vector polygons for GIS analysis
gdf = process.polygonize(
    files, 
    variable_name=variable_name, 
    region=region, 
    region_crs=region_crs, 
    optimize_geometry=True
)

# Save the results
gdf.to_csv('data/polygonized.csv')
```

## License

MIT License
