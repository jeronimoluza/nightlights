from shapely import wkt

if __name__ == "__main__":

    from nightlights import download, process, postprocess

    download_dir = "./data/raw"
    extraction_dir = "./data/extracted"
    output_dir = "./data/output"

    # Define search parameters
    short_name = "VNP46A3"
    version = "1"
    start_date = "2024-12-12"
    end_date = "2024-12-15"
    count = 10

    region = wkt.loads(
        "POLYGON((-62.32 -33.01, -56.6 -33.01, -56.6 -37.34, -62.32 -37.34, -62.32 -33.01))"
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

    process.process_files(files, variable_name="AllAngle_Composite_Snow_Free", output_dir=extraction_dir, bounding_box=region.bounds)

    postprocess.produce_output(extraction_dir, output_dir)
