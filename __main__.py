
def do_main():
    import numpy as np

    from nightlights import download, plotting, process

    download_dir = "./data/raw"
    plot_dir = "./assets/"

    # Define search parameters
    short_name = "VNP46A3"
    start_date = "2021-11-01"
    end_date = "2022-05-01"

    regions = [
        "Kyiv, Ukraine",
        "Kyiv Oblast, Ukraine",
        ]
    
    region_gdf = download.find_region(query=regions)

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

    plotting.plot_nightlights(
        files,
        title="Nightlight Intensity\nKyiv City and Oblast",
        variable_name=variable_name,
        date="2021-11-01",
        output_dir=plot_dir,
        region=region,
    )

    plotting.create_timelapse_gif(
        files,
        variable_name=variable_name,
        title="Nightlight Intensity\nKyiv City and Oblast",
        output_dir=plot_dir,
        region=region,
        region_crs=region_crs,
        fps=2.0,
    )

    events = [
        ("Start of the conflict\nAttacks on Kyiv's Energy Infrastructure", "2022-02-01"),
    ]
    # Define functions to apply to the data
    functions = [
        {"Mean": np.mean},
        {"Median": np.median},
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
        cut_off=1,
    )

    plotting.side_by_side(
        files=files,
        variable_name=variable_name,
        title="Nightlight Intensity\nKyiv City and Oblast: Pre-War vs Post-War",
        date1="2021-11-01",
        date2="2022-05-01",
        region=region,
        region_crs=region_crs,
        output_dir=plot_dir,
        bins=15,
        log_scale=True,
    )
    gdf = process.polygonize(
        files, variable_name=variable_name, region=region, region_crs=region_crs, optimize_geometry=True
    )
    print(gdf.head().to_markdown())


if __name__ == "__main__":
    do_main()