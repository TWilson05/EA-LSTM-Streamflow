import urllib.parse
import pandas as pd
from src.config import RAW_DATA_DIR

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
    df_wide = df_long.pivot(index="Date", columns=" ID", values="Value/Valeur")

    # save to csv
    if output_filename:
        df_wide.to_csv(RAW_DATA_DIR / output_filename)
        print("Data downloaded and saved to combined_streamflow.csv")

    print(f"{df_wide.shape[0]} days of data saved for {df_wide.shape[1]} stations")
    return df_wide.sort_index().sort_index(axis=1)
