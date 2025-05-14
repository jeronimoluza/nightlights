
def do_main():
    import geopandas as gpd
    import pandas as pd
    import numpy as np

    from nightlights import download, plotting, process

    download_dir = "./data/raw"
    extraction_dir = "./data/extracted"
    plot_dir = "./data/plots"
    final_output_dir = "./data/output"

    # Define search parameters
    short_name = "VNP46A3"
    start_date = "2023-04-15"
    end_date = "2024-05-05"

    regions = ["Andalucía, Spain"]
    region_gdf = gpd.GeoDataFrame(
        pd.concat(
            [download.find_region(query=region) for region in regions], ignore_index=True
        ), geometry="geometry"
    )

    region_crs = region_gdf.crs.to_epsg()
    region = region_gdf.union_all()

    files = download.download_earthaccess(
        download_dir=download_dir,
        short_name=short_name,
        start_date=start_date,
        end_date=end_date,
        region=region,
    )

    variable_name = "AllAngle_Composite_Snow_Free"
    # variable_name = "DNB_BRDF-Corrected_NTL"

    # plotting.plot_nightlights(
    #     files,
    #     variable_name=variable_name,
    #     date="2021-01-01",
    #     output_dir=plot_dir,
    #     region=region,
    # )

    # plotting.create_timelapse_gif(
    #     files,
    #     variable_name=variable_name,
    #     title="Andalucía Nightlights",
    #     output_dir=plot_dir,
    #     region=region,
    #     region_crs=region_crs,
    #     fps=2.0,
    # )

    # # Define functions to apply to the data
    functions = [
        {"Mean": np.mean},
        {"Median": np.median},
    ]

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

#     plotting.side_by_side(
#     files=files,
#     variable_name=variable_name,
#     title="Nightlights Comparison",
#     date1="2023-05-01",
#     date2="2024-05-01",
#     region=region,
#     region_crs=region_crs,
#     output_dir=plot_dir,
#     bins=15,
#     log_scale=True
# )
    raise
    # Use the optimized polygonization approach (default behavior)
    gdf = process.polygonize(
        files, variable_name=variable_name, region=region, region_crs=region_crs, optimize_geometry=True
    )
    print(gdf.head())
    gdf[gdf.date=='2021-01-01'].to_csv('data/polygonized.csv')


if __name__ == "__main__":
    do_main()