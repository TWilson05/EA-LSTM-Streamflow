import cdsapi
import calendar
import urllib.parse
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR, ERA5_PRECIP_DIR, ERA5_TEMP_DIR, ERA5_BOUNDS

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
        df_wide.to_csv(RAW_DATA_DIR / output_filename)
        print("Data downloaded and saved to combined_streamflow.csv")

    print(f"{df_wide.shape[0]} days of data saved for {df_wide.shape[1]} stations")
    return df_wide.sort_index().sort_index(axis=1)

def get_cds_client():
    return cdsapi.Client()

def download_era5_precipitation(years, months=range(1, 13)):
    """Downloads ERA5 daily sum precipitation in monthly files."""
    client = get_cds_client()
    dataset = "derived-era5-single-levels-daily-statistics"

    for year in years:
        for month in months:
            out_file = ERA5_PRECIP_DIR / f"era5_precip_{year}_{month:02d}.nc"

            if out_file.exists():
                print(f"✔ Skipping Precip {year}-{month:02d} (already exists)")
                continue

            days_in_month = calendar.monthrange(year, month)[1]
            day_list = [f"{d:02d}" for d in range(1, days_in_month + 1)]

            request = {
                "product_type": "reanalysis",
                "variable": ["total_precipitation"],
                "daily_statistic": "daily_sum",
                "time_zone": "utc-07:00",
                "frequency": "1_hourly",
                "area": ERA5_BOUNDS,
                "year": str(year),
                "month": f"{month:02d}",
                "day": day_list
            }

            try:
                print(f"⏳ Downloading Precip: {year}-{month:02d} ...")
                client.retrieve(dataset, request, str(out_file))
            except Exception as e:
                print(f"❌ Failed {year}-{month:02d}: {e}")

def download_era5_temperature(years):
    """Downloads ERA5 2m temperature in yearly files."""
    client = get_cds_client()
    dataset = "reanalysis-era5-single-levels"

    for year in years:
        out_file = ERA5_TEMP_DIR / f"era5_temp_{year}.grib"

        if out_file.exists():
            print(f"✔ Skipping Temp {year} (already exists)")
            continue

        request = {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature"],
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "grib",
            "download_format": "unarchived",
            "area": ERA5_BOUNDS,
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)]
        }

        try:
            print(f"⏳ Downloading Temp: {year} ...")
            client.retrieve(dataset, request, str(out_file))
        except Exception as e:
            print(f"❌ Failed Temp {year}: {e}")
