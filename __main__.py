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
    start_date = "2012-01-01"
    end_date = "2024-12-15"

    region = download.find_region(
        query="Provincia de Buenos Aires",
    )
    region_crs = region.crs.to_epsg()
    region = region.geometry.values[0]

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

    # plotting.plot_nightlights(
    #     files,
    #     variable_name=variable_name,
    #     date="2022-01-01",
    #     output_dir=plot_dir,
    #     region=region,
    # )

    gdf = process.polygonize(
        files, variable_name=variable_name, region=region, region_crs=region_crs
    )
    gdf['datetimestr'] = gdf.date.dt.strftime('%Y-%m-%d 00:00:00')
    gdf.to_csv('test.csv')
    plotting.create_timelapse_gif(
        files,
        variable_name=variable_name,
        title="Ethiopia Through the Years",
        output_dir=plot_dir,
        region=region,
        fps=2.0,
        plot_series=True,
        use_confidence_interval=True,
        confidence_level=0.95,
        #cut_off=0.5,
    )

