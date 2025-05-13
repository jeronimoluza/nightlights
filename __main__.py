
def do_main():
    from shapely import wkt
    import geopandas as gpd
    import pandas as pd

    from nightlights import download, plotting, process

    download_dir = "./data/raw"
    extraction_dir = "./data/extracted"
    plot_dir = "./data/plots"
    final_output_dir = "./data/output"

    # Define search parameters
    short_name = "VNP46A2"
    start_date = "2019-03-01"
    end_date = "2019-03-17"

    regions = ["Venezuela"]
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

    # variable_name = "AllAngle_Composite_Snow_Free"
    variable_name = "DNB_BRDF-Corrected_NTL"

    # plotting.plot_nightlights(
    #     files,
    #     variable_name=variable_name,
    #     date="2021-01-01",
    #     output_dir=plot_dir,
    #     region=region,
    # )

    plotting.create_timelapse_gif(
        files,
        variable_name=variable_name,
        title="Venezuela Blackout",
        output_dir=plot_dir,
        region=region,
        region_crs=region_crs,
        fps=2.0,
        plot_series=True,
        sample_size=1000000,
        cut_off=10,
        events=("Venezuela Blackout", "2019-03-07"),
    )


    raise
    # Use the optimized polygonization approach (default behavior)
    gdf = process.polygonize(
        files, variable_name=variable_name, region=region, region_crs=region_crs, optimize_geometry=True
    )
    print(gdf.head())
    gdf[gdf.date=='2021-01-01'].to_csv('data/polygonized.csv')


if __name__ == "__main__":
    do_main()