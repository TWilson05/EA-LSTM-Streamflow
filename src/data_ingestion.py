import math
import cdsapi
import calendar
import requests
import rasterio
import urllib.parse
import pandas as pd
from rasterio.merge import merge
from src.config import PROCESSED_DATA_DIR, ERA5_PRECIP_DIR, ERA5_TEMP_DIR, ERA5_RAD_DIR, SPATIAL_BOUNDS, ELEVATION_DIR, LAKE_COVER

bounds = pd.read_csv(SPATIAL_BOUNDS)

# Area bounds lat/lons: [North, West, South, East]
ERA5_BOUNDS = [
    float(bounds.loc[0, "north"]),
    float(bounds.loc[0, "west"]),
    float(bounds.loc[0, "south"]),
    float(bounds.loc[0, "east"]),
]

def build_wateroffice_url(stations, start_date : int, end_date : int, parameter="flow"):
    base = "https://wateroffice.ec.gc.ca/services/daily_data/csv/inline?"
    station_params = "&".join([f"stations[]={urllib.parse.quote(s)}" for s in stations])
    param_part = f"parameters[]={urllib.parse.quote(parameter)}"
    date_part = f"start_date={start_date}&end_date={end_date}"
    return base + station_params + "&" + param_part + "&" + date_part

def fetch_streamflow_batch(stations, start_year : int, end_year : int, output_filename = None, batch_size : int = 50):
    """
    Downloads and pivots streamflow data for a list of stations.
    """
    all_data = []
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    print(f"Downloading data for {len(stations)} stations...")
    for i in range(0, len(stations), batch_size):
        batch = stations[i:i + batch_size]
        url = build_wateroffice_url(batch, start_date, end_date)
        
        # Add error handling in case a batch fails
        try:
            df_batch = pd.read_csv(url)
            all_data.append(df_batch[[" ID", "Date", "Value/Valeur"]])
        except Exception as e:
            print(f"Error downloading batch starting at index {i}: {e}")

    df_long = pd.concat(all_data, ignore_index=True)
    df_long["Date"] = pd.to_datetime(df_long["Date"])
    
    # Pivot to wide format
    requested_stations = sorted(list(set([s.strip() for s in stations])))
    df_wide = df_long.pivot(index="Date", columns=" ID", values="Value/Valeur")
    df_wide = df_wide.reindex(columns=requested_stations)

    # save to csv
    if output_filename:
        df_wide.to_csv(PROCESSED_DATA_DIR / output_filename)
        print("Data downloaded and saved to combined_streamflow.csv")

    print(f"{df_wide.shape[0]} days of data saved for {df_wide.shape[1]} stations")
    return df_wide.sort_index().sort_index(axis=1)

def get_cds_client():
    return cdsapi.Client()

def download_era5_land(variable, file_name, directory, years, months=range(1, 13)):
    """Downloads ERA5 land data in monthly files."""
    client = get_cds_client()
    dataset = "reanalysis-era5-land"

    for year in years:
        for month in months:
            out_file = directory / f"{file_name}_{year}_{month:02d}.nc"

            if out_file.exists():
                print(f"✔ Skipping {file_name} {year}-{month:02d} (already exists)")
                continue

            days_in_month = calendar.monthrange(year, month)[1]
            day_list = [f"{d:02d}" for d in range(1, days_in_month + 1)]

            request = {
                "variable": [variable],
                "area": ERA5_BOUNDS,
                "year": str(year),
                "month": f"{month:02d}",
                "day": day_list,
                "time": [f"{h:02d}:00" for h in range(24)],
                "data_format": "netcdf",
                "download_format": "unarchived"
            }

            try:
                print(f"⏳ Downloading {file_name} {year}-{month:02d} ...")
                client.retrieve(dataset, request, str(out_file))
            except Exception as e:
                print(f"❌ Failed {year}-{month:02d}: {e}")

def download_era5_temperature(years):
    download_era5_land("2m_temperature", "era5_temp", ERA5_TEMP_DIR, years)

def download_era5_precipitation(years):
    download_era5_land("total_precipitation", "era5_precip", ERA5_PRECIP_DIR, years)

def download_era5_solar_rad(years):
    download_era5_land("surface_solar_radiation_downwards", "era5_rad", ERA5_RAD_DIR, years)

def download_era5_lakes():
    """Downloads a static lake cover mask"""
    client = get_cds_client()
    dataset = "reanalysis-era5-land"

    if LAKE_COVER.exists():
        print("✔ Skipping lake cover download (already exists)")
    else:
        request = {
            "variable": ["lake_cover"],
            "data_format": "netcdf",
            "download_format": "unarchived"
        }

        try:
            print("⏳ Downloading lake cover ...")
            client.retrieve(dataset, request, str(LAKE_COVER))
        except Exception as e:
            print(f"❌ Failed to download lake cover: {e}")

def latlon_to_tile(lat, lon, zoom):
    """Converts Lat/Lon to Web Mercator XYZ tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def download_aws_dem(bounds, output_filename="western_canada_dem.tif", zoom=10):
    """
    Downloads AWS Terrain Tiles for the given bounds and merges them.
    Zoom 10 ~= 90m resolution (SRTM equivalent).
    """
    output_path = ELEVATION_DIR / output_filename
    
    # If already exists, skip
    if output_path.exists():
        print(f"ℹ️ DEM already exists at {output_path}. Skipping download.")
        return output_path

    print(f"⏳ Downloading DEM tiles for bounds: {bounds}...")
    
    # Parse bounds dictionary
    min_lat, max_lat = bounds['south'], bounds['north']
    min_lon, max_lon = bounds['west'], bounds['east']
    
    # Get Tile Ranges
    # Note: Y-tile origin is North, so max_lat -> min_y_tile
    x_min, y_min = latlon_to_tile(max_lat, min_lon, zoom)
    x_max, y_max = latlon_to_tile(min_lat, max_lon, zoom)
    
    total_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
    print(f"   Grid: X[{x_min}-{x_max}] Y[{y_min}-{y_max}] ({total_tiles} tiles)")
    
    src_files = []
    
    # Temp dir for individual tiles
    tile_dir = ELEVATION_DIR / "tiles"
    tile_dir.mkdir(exist_ok=True)
    
    count = 0
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            count += 1
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{zoom}/{x}/{y}.tif"
            tile_path = tile_dir / f"tile_z{zoom}_x{x}_y{y}.tif"
            
            # Download
            if not tile_path.exists():
                try:
                    r = requests.get(url, stream=True, timeout=30)
                    if r.status_code == 200:
                        with open(tile_path, 'wb') as f:
                            for chunk in r.iter_content(8192): f.write(chunk)
                    else:
                        print(f"   ❌ Failed: {url} ({r.status_code})")
                        continue
                except Exception as e:
                    print(f"   ❌ Error downloading tile {x},{y}: {e}")
                    continue
            
            # Open
            try:
                src = rasterio.open(tile_path)
                src_files.append(src)
            except:
                print(f"   ⚠️ Warning: Corrupt tile {x},{y}")
                
            if count % 10 == 0:
                print(f"   Progress: {count}/{total_tiles}...", end="\r")
                
    print("\n🧩 Merging tiles...")
    
    if not src_files:
        raise RuntimeError("❌ No tiles were downloaded successfully.")

    mosaic, out_trans = merge(src_files)
    
    # Update Meta (EPSG:3857 for AWS Tiles)
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": "EPSG:3857"
    })
    
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
        
    # Cleanup handles
    for src in src_files: src.close()
    
    print(f"✅ Saved merged DEM to {output_path}")
    return output_path
