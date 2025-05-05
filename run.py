from shapely import wkt

if __name__ == "__main__":

    from nightlights import download, process

    download_dir = "./data/raw"
    process_dir = "./data/processed"

    # Define search parameters
    short_name = "VNP46A3"
    version = "1"  # Latest version of MOD13Q1
    start_date = "2024-12-12"
    end_date = "2024-12-15"
    count = 10

    region = wkt.loads(
        "POLYGON((-63.58 -34.32, -58.34 -34.32, -58.34 -34.71, -63.58 -34.71, -63.58 -34.32))"
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

    process.process_files(files, process_dir)
