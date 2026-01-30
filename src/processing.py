import pandas as pd

def filter_stations_by_annual_completeness(df, max_missing_pct=40.0):
    """
    Filters out stations that have more than max_missing_pct missing data 
    in ANY of the years present in the dataset.
    """
    df.index = pd.to_datetime(df.index)
    
    # 1. Group by year and calculate the percentage of NaNs per year per station
    # This creates a DataFrame where index=Year and columns=Station IDs
    annual_nan_pct = df.groupby(df.index.year).apply(lambda x: x.isna().mean() * 100)
    
    # 2. Identify stations where the maximum annual missingness is below the threshold
    # .max() checks across all years for each station
    stations_to_keep = annual_nan_pct.columns[annual_nan_pct.max() <= max_missing_pct]
    stations_to_drop = annual_nan_pct.columns[annual_nan_pct.max() > max_missing_pct]
    
    print(f"Filtering at {max_missing_pct}% annual threshold:")
    print(f" - Keeping {len(stations_to_keep)} stations.")
    print(f" - Dropping {len(stations_to_drop)} stations due to incomplete years.")

    return df[stations_to_keep]

def filter_stations_by_mda(df, mda_codes=["05", "07"]):
    """
    Filters columns (stations) to only include those starting with specific
    MDA (Major Drainage Area) codes.
    
    Args:
        df: DataFrame with station IDs as columns.
        mda_codes: List of strings (e.g. ["05", "07"]).
    """
    # Create a tuple for startswith (it requires a tuple, not a list)
    prefixes = tuple(mda_codes)
    
    # Identify matching columns
    valid_stations = [col for col in df.columns if col.startswith(prefixes)]
    dropped_stations = set(df.columns) - set(valid_stations)
    
    print(f"Filtering by MDA codes {mda_codes}:")
    print(f" - Keeping {len(valid_stations)} stations.")
    print(f" - Dropping {len(dropped_stations)} stations (wrong region).")
    
    return df[valid_stations]
