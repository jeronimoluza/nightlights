from shapely import wkt

if __name__ == "__main__":

    from nightlights import download, plotting, process

    session_slug = "main"
    download_dir = f"./data/{session_slug}/raw"
    extraction_dir = f"./data/{session_slug}/extracted"
    plot_dir = f"./data/{session_slug}/plots"
    final_output_dir = f"./data/{session_slug}/output"

    # Define search parameters
    short_name = "VNP46A3"
    version = "1"
    start_date = "2024-01-01"
    end_date = "2025-03-30"

    region = wkt.loads(
        # "POLYGON((33.0 14.89, 47.98 14.89, 47.98 3.4, 33.0 3.4, 33.0 14.89))"
        "POLYGON((34.8826 34.6924, 36.625 34.6924, 36.625 33.055, 34.8826 33.055, 34.8826 34.6924))"
    )

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
    #     upper_title="Title",
    #     output_dir=plot_dir,
    #     region=region,
    # )

    plotting.static.create_timelapse_gif(
        files,
        variable_name=variable_name,
        upper_title="Lebanon Blackouts\nMonthly Nightlights",
        output_dir=plot_dir,
        region=region,
        fps=2.0,
        # plot_series=True,
    )

    # gdf = process.process_files(
    #     files, variable_name=variable_name, bounding_box=region.bounds
    # )
    # plotting.interactive.create_folium_map(
    #     gdf, variable_name=variable_name, output_file="test.html"
    # )
    #
