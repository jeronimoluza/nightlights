from shapely import wkt
import geopandas as gpd

if __name__ == "__main__":

    from nightlights import download, plotting, process

    download_dir = "./data/raw"
    extraction_dir = "./data/extracted"
    plot_dir = "./data/plots"
    final_output_dir = "./data/output"

    # Define search parameters
    short_name = "VNP46A2"
    version = "2"
    start_date = "2019-03-01"
    end_date = "2019-03-15"

    # region = wkt.loads(
    #     # "POLYGON((33.0 14.89, 47.98 14.89, 47.98 3.4, 33.0 3.4, 33.0 14.89))"
    #     # "POLYGON((34.8826 34.6924, 36.625 34.6924, 36.625 33.055, 34.8826 33.055, 34.8826 34.6924))"
    #     "POLYGON((-4.71 41.04, -2.63 41.04, -2.63 39.71, -4.71 39.71, -4.71 41.04))"
    # "POLYGON((-4.215952 40.772594, -3.148765 40.772594, -3.148765 40.099842, -4.215952 40.099842, -4.215952 40.772594))"
    # "POLYGON((35.922737 36.423821, 36.400167 36.423821, 36.400167 36.025398, 35.922737 36.025398, 35.922737 36.423821))"
    # "POLYGON((35.9227 36.785, 37.8 36.785, 37.8 36.0254, 35.9227 36.0254, 35.9227 36.785))"
    # "POLYGON((92.67 26.59, 98.58 26.59, 98.58 19.49, 92.67 19.49, 92.67 26.59))"
    # "POLYGON((70.287 34.5157, 74.5148 34.5157, 74.5148 31.3723, 70.287 31.3723, 70.287 34.5157))"
    # )

    country="Uruguay"
    # country="Venezuela"
    gdf = gpd.read_file(
        "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/refs/heads/main/countries.geojson"
    )
    region_crs = gdf.crs.to_epsg()
    region = gdf[gdf.name == country].geometry.values[0]

    files = download.download_earthaccess(
        download_dir=download_dir,
        short_name=short_name,
        version=version,
        start_date=start_date,
        end_date=end_date,
        region=region,
    )

    # variable_name = "AllAngle_Composite_Snow_Free"
    variable_name = "DNB_BRDF-Corrected_NTL"

    # process.process_files(files, variable_name=variable_name, output_dir=extraction_dir, bounding_box=region.bounds)

    # process.produce_output(extraction_dir, final_output_dir)

    # plotting.plot_nightlights(
    #     files,
    #     variable_name=variable_name,
    #     date="2022-01-01",
    #     output_dir=plot_dir,
    #     region=region,
    # )

    plotting.create_timelapse_gif(
        files,
        variable_name=variable_name,
        title="Venezuela",
        output_dir=plot_dir,
        region=region,
        fps=2.0,
        plot_series=True,
        use_confidence_interval=True,
        confidence_level=0.95,
    )

    # gdf = process.process_files(
    #     files, variable_name=variable_name, region=region, region_crs=region_crs
    # )
