from shapely import wkt

if __name__ == "__main__":

    from nightlights import download, plotting, process, postprocess

    session_slug = "dc"
    download_dir = f"./data/{session_slug}/raw"
    extraction_dir = f"./data/{session_slug}/extracted"
    plot_dir = f"./data/{session_slug}/plots"
    final_output_dir = f"./data/{session_slug}/output"

    # Define search parameters
    short_name = "VNP46A3"
    version = "1"
    start_date = "2024-12-12"
    end_date = "2024-12-15"
    count = 10

    region = wkt.loads(
        # "POLYGON((-66.45 -28.66, -55.14 -28.66, -55.14 -38.67, -66.45 -38.67, -66.45 -28.66))"
        # "POLYGON((-64.98 -33.97, -57.16 -33.97, -57.16 -35.26, -64.98 -35.26, -64.98 -33.97))"
        "POLYGON((-77.9205 41.7018, -72.8854 41.7018, -72.8854 38.0936, -77.9205 38.0936, -77.9205 41.7018))"
    )
    auth = download.login()

    files = download.download_earthaccess(
        download_dir=download_dir,
        short_name=short_name,
        version=version,
        start_date=start_date,
        end_date=end_date,
        region=region,
        count=count,
    )

    variable_name="AllAngle_Composite_Snow_Free"

    # process.process_files(files, variable_name=variable_name, output_dir=extraction_dir, bounding_box=region.bounds)

    # postprocess.produce_output(extraction_dir, final_output_dir)
    
    #plotting.plot_file()
    plotting.plot_all_files(files, variable_name=variable_name, output_dir=plot_dir, region=region)