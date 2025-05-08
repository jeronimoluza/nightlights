from shapely import wkt
import geopandas as gpd

if __name__ == "__main__":

    from nightlights import download, plotting, process

    download_dir = "./data/raw"
    extraction_dir = "./data/extracted"
    plot_dir = "./data/plots"
    final_output_dir = "./data/output"

    # Define search parameters
    short_name = "VNP46A4"
    version = "1"
    start_date = "2022-01-01"
    end_date = "2023-01-31"

    # region = wkt.loads(
    #     # "POLYGON((33.0 14.89, 47.98 14.89, 47.98 3.4, 33.0 3.4, 33.0 14.89))"
    #     # "POLYGON((34.8826 34.6924, 36.625 34.6924, 36.625 33.055, 34.8826 33.055, 34.8826 34.6924))"
    #     "POLYGON((-4.71 41.04, -2.63 41.04, -2.63 39.71, -4.71 39.71, -4.71 41.04))"
    # )

    gdf = gpd.read_file(
        "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/refs/heads/main/countries.geojson"
    )
    region_crs = gdf.crs.to_epsg()
    region = gdf[gdf.region_wb == "Latin America & Caribbean"].geometry.unary_union

    files = download.download_earthaccess(
        download_dir=download_dir,
        short_name=short_name,
        version=version,
        start_date=start_date,
        end_date=end_date,
        region=region,
    )

    variable_name = "AllAngle_Composite_Snow_Free"
    # variable_name = "DNB_BRDF-Corrected_NTL"

    # process.process_files(files, variable_name=variable_name, output_dir=extraction_dir, bounding_box=region.bounds)

    # process.produce_output(extraction_dir, final_output_dir)

    # plotting.plot_all_files(
    #     files,
    #     variable_name=variable_name,
    #     title="Title",
    #     output_dir=plot_dir,
    #     region=region,
    # )

    plotting.create_timelapse_gif(
        files,
        variable_name=variable_name,
        title="Latin America & Caribbean",
        output_dir=plot_dir,
        region=region,
        region_crs=region_crs,
        fps=2.0,
        plot_series=True,
    )

    # gdf = process.process_files(
    #     files, variable_name=variable_name, region=region, region_crs=region_crs
    # )
