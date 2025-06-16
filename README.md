# Nightlights

A Python toolkit for downloading, processing, and visualizing NASA's Black Marble Nightlights Suite (VIIRS) satellite data. This project provides tools to analyze nighttime light patterns over time, which can be used to study urbanization, economic activity, power outages, recovery from natural disasters, and more.

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Poetry (for dependency management)
- Access to an [Earthdata Account](https://urs.earthdata.nasa.gov/)

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

# Define directories
download_dir = "./data/raw"
plot_dir = "./assets/"

# Define search parameters
short_name = "VNP46A3" # VIIRS Black Marble products: VNP46A2, VNP46A3, VNP46A4
start_date = "2021-11-01"
end_date = "2022-05-01"

# Define region of interest
regions = [
    "Kyiv, Ukraine",
    "Kyiv Oblast, Ukraine",
]

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

# Variable names:
# "DNB_BRDF-Corrected_NTL" for VNP46A2 (daily data)
# "AllAngle_Composite_Snow_Free" for VNP46A3 (monthly data) and VNP46A4 (annual data)
variable_name = "AllAngle_Composite_Snow_Free"  
```

### 3. Create visualizations

#### Single Date Map

```python
plotting.plot_nightlights(
    files,
    title="Nightlight Intensity\nKyiv City and Oblast",
    variable_name=variable_name,
    date="2021-11-01",
    output_dir=plot_dir,
    region=region,
)
```
![Single Date Map](./assets/nightlights_20211101_AllAngle_Composite_Snow_Free.png)

#### Time Series Lineplot with Events

```python
import numpy as np

# Define functions to apply to the data
functions = [
    {"Mean": np.mean},
    {"Median": np.median},
]

# Define important events to mark on the plot
events = [
    ("Start of the conflict\nAttacks on Kyiv's Energy Infrastructure", "2022-02-01"),
]

plotting.create_lineplot(
    files=files,
    variable_name=variable_name,
    title="Nightlight Intensity\nKyiv City and Oblast",
    output_dir=plot_dir,
    region=region,
    region_crs=region_crs,
    functions=functions,
    events=events,
    cut_off=1, # Will only consider values higher than the cut_off parameter for the lineplot
)
```
![Time Series Lineplot with Events](./assets/lineplot_AllAngle_Composite_Snow_Free.png)

#### Side-by-Side Comparison

```python
plotting.side_by_side(
    files=files,
    variable_name=variable_name,
    title="Nightlight Intensity\nKyiv City and Oblast: Pre-War vs Post-War",
    date1="2021-11-01",
    date2="2022-07-01",
    region=region,
    region_crs=region_crs,
    output_dir=plot_dir,
    bins=15,
    log_scale=True
)
```
![Side-by-Side Comparison](./assets/comparison_AllAngle_Composite_Snow_Free_20211101_20220501.png)

#### Timelapse Animation

```python
plotting.create_timelapse_gif(
    files=files,
    variable_name=variable_name,
    title="Nightlight Intensity\nKyiv City and Oblast",
    output_dir=plot_dir,
    region=region,
    region_crs=region_crs,
    fps=2.0,
)
```
![Timelapse Animation](./assets/timelapse_AllAngle_Composite_Snow_Free.gif)


### 4. Advanced Processing: Polygonize Data

The extraction of pixel boundaries from each file involves a calculation of geographical coordinates based on the Black Marble tile's limits and file matrix dimensions. The process begins by determining the size of each tile in degrees using its west, north, east, and south bounds. Then, utilizing the matrix dimensions (rows and columns), the distances between pixel boundaries are calculated. Specifically, the distance from the west-east bounds of the tile is divided by the number of rows (X) to determine the longitude increment per pixel, while the distance from the north-south bounds is divided by the number of columns (Y) to calculate the latitude increment per pixel.

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
gdf.head()
```

```markdown
|    | pixel_id                         | tile   | date       | variable                     |   value | geometry                                                                                                                                                                   |
|---:|:---------------------------------|:-------|:-----------|:-----------------------------|--------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | 832d7e3f2e1e1b4c082450dc2c55695b | h21v04 | 2021-11-01 | AllAngle_Composite_Snow_Free |       0 | POLYGON ((30 49.99583333333334, 30.00416666666667 49.99583333333334, 30.00416666666667 50, 30 50, 30 49.99583333333334)) |
|  1 | cc3f405d6588e83e068cf95139226b4e | h21v04 | 2021-11-01 | AllAngle_Composite_Snow_Free |       0 | POLYGON ((30.004166666666666 49.99583333333334, 30.008333333333336 49.99583333333334, 30.008333333333336 50, 30.004166666666666 50, 30.004166666666666 49.99583333333334)) |
|  2 | fbcda96016cddc5701decea68cfb2b0d | h21v04 | 2021-11-01 | AllAngle_Composite_Snow_Free |       0 | POLYGON ((30.008333333333333 49.99583333333334, 30.012500000000003 49.99583333333334, 30.012500000000003 50, 30.008333333333333 50, 30.008333333333333 49.99583333333334)) |
|  3 | ef914453eb14b7c27022450f6966e5a9 | h21v04 | 2021-11-01 | AllAngle_Composite_Snow_Free |       0 | POLYGON ((30.0125 49.99583333333334, 30.01666666666667 49.99583333333334, 30.01666666666667 50, 30.0125 50, 30.0125 49.99583333333334)) |
|  4 | a7281d42070a22a0b69de920afea3681 | h21v04 | 2021-11-01 | AllAngle_Composite_Snow_Free |       0 | POLYGON ((30.016666666666666 49.99583333333334, 30.020833333333336 49.99583333333334, 30.020833333333336 50, 30.016666666666666 50, 30.016666666666666 49.99583333333334)) |
```

## License

[MIT License](LICENSE)
