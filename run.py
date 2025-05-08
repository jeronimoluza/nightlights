from shapely import wkt

if __name__ == "__main__":

    from nightlights import download, plotting, process

    session_slug = "main"
    download_dir = f"./data/{session_slug}/raw"
    extraction_dir = f"./data/{session_slug}/extracted"
    plot_dir = f"./data/{session_slug}/plots"
    final_output_dir = f"./data/{session_slug}/output"

    # Define search parameters
    short_name = "VNP46A2"
    version = "2"
    start_date = "2025-04-20"
    end_date = "2025-05-5"

    region = wkt.loads(
        # "POLYGON((33.0 14.89, 47.98 14.89, 47.98 3.4, 33.0 3.4, 33.0 14.89))"
        # "POLYGON((34.8826 34.6924, 36.625 34.6924, 36.625 33.055, 34.8826 33.055, 34.8826 34.6924))"
        "POLYGON((-4.71 41.04, -2.63 41.04, -2.63 39.71, -4.71 39.71, -4.71 41.04))"
    )

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

    # plotting.plot_all_files(
    #     files,
    #     variable_name=variable_name,
    #     title="Title",
    #     output_dir=plot_dir,
    #     region=region,
    # )

    plotting.static.create_timelapse_gif(
        files,
        variable_name=variable_name,
        title="Lebanon Blackouts\nMonthly Nightlights",
        output_dir=plot_dir,
        region=region,
        fps=2.0,
        plot_series=True,
    )

    # gdf = process.process_files(
    #     files, variable_name=variable_name, bounding_box=region.bounds
    # )
    # plotting.interactive.create_folium_map(
    #     gdf, variable_name=variable_name, output_file="test.html"
    # )
    #
