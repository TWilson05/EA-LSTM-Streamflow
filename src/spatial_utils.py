import geopandas as gpd
import pandas as pd
import warnings
import os
import requests
import zipfile
import rasterio
from rasterio.merge import merge
from rasterstats import zonal_stats
from src.config import OUTPUT_STATIC_ATTR, OUTPUT_GLACIER_VOL, RAW_DATA_DIR, ELEVATION_DIR

# Standard Equal Area projection for Western Canada
CANADA_ALBERS_CRS = "+proj=aea +lat_1=50 +lat_2=70 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"

def download_srtm_90m_tiles(bounds, cache_dir=ELEVATION_DIR):
    """
    Downloads SRTM 90m (v4.1) tiles from CGIAR-CSI server.
    Grid is 5x5 degrees. 
    """
    
    minx, miny, maxx, maxy = bounds
    
    # SRTM v4.1 Grid Calculation
    # x: 1 (-180) to 72 (180)
    # y: 1 (60N) to 24 (60S)
    
    x_start = int((minx + 180) / 5) + 1
    x_end = int((maxx + 180) / 5) + 1
    
    y_start = int((60 - maxy) / 5) + 1
    y_end = int((60 - miny) / 5) + 1
    
    y_start = max(1, y_start)
    y_end = min(24, y_end)
    
    tif_paths = []
    
    print(f"   ⏳ Identified SRTM tiles from X[{x_start}-{x_end}] Y[{y_start}-{y_end}]...")
    
    for x in range(x_start, x_end + 1):
        for y in range(y_start, y_end + 1):
            filename = f"srtm_{x:02d}_{y:02d}.zip"
            local_zip = os.path.join(cache_dir, filename)
            local_tif = local_zip.replace(".zip", ".tif")
            
            # 1. Check if we already have the TIF
            if os.path.exists(local_tif):
                tif_paths.append(local_tif)
                continue
                
            # 2. Download ZIP
            url = f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/{filename}"
            
            if not os.path.exists(local_zip):
                print(f"      Downloading tile {x},{y}...")
                try:
                    response = requests.get(url, stream=True)
                    if response.status_code != 200:
                        print(f"      ⚠️ Tile {filename} not found (ocean or >60N). Skipping.")
                        continue
                        
                    with open(local_zip, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk: f.write(chunk)
                except Exception as e:
                    print(f"      ❌ Download error for {filename}: {e}")
                    continue
            
            # 3. Extract TIF
            try:
                with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                    tif_name = next((n for n in zip_ref.namelist() if n.lower().endswith('.tif')), None)
                    if tif_name:
                        with open(local_tif, 'wb') as target:
                            target.write(zip_ref.read(tif_name))
                        tif_paths.append(local_tif)
            except zipfile.BadZipFile:
                print(f"      ❌ Corrupt zip file: {filename}")
                # Optional: os.remove(local_zip)

    return tif_paths

def get_elevation_mosaic(bounds, output_path):
    """
    Downloads tiles and merges them into a single mosaic.
    """
    if os.path.exists(output_path):
        print("   ℹ️ Using cached DEM mosaic.")
        return output_path
        
    tiles = download_srtm_90m_tiles(bounds)
    
    if not tiles:
        # Fallback logic if NO tiles are found (e.g. all >60N)
        print("   ⚠️ No SRTM tiles found. Returning None.")
        return None
    
    print(f"   ⏳ Merging {len(tiles)} tiles into mosaic...")
    
    src_files_to_mosaic = []
    for fp in tiles:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
        
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })
    
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
        
    for src in src_files_to_mosaic: src.close()
    
    print(f"   ✅ Saved DEM mosaic to {output_path}")
    return output_path

def process_spatial_attributes(basin_gpkg_paths, glacier_shp_path, mass_balance_path, stations_list):
    """
    Main workflow: Basins + Glaciers + Elevation.
    """
    # 1. LOAD BASINS
    print("⏳ Loading and merging basin files...")
    basins_list = []
    for path in basin_gpkg_paths:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                gdf = gpd.read_file(path, layer='DrainageBasin_BassinDeDrainage')
                
            id_col = next((c for c in gdf.columns if c.lower() in ['stationnum', 'id', 'station_id']), None)
            if id_col:
                gdf = gdf[[id_col, 'geometry']].rename(columns={id_col: 'station_id'})
                basins_list.append(gdf)
        except Exception as e:
            print(f"❌ Error loading {path.name}: {e}")

    gdf_basins = pd.concat(basins_list, ignore_index=True)
    
    gdf_basins['station_id'] = gdf_basins['station_id'].astype(str).str.strip()
    requested_stations = set([s.strip() for s in stations_list])
    gdf_basins = gdf_basins[gdf_basins['station_id'].isin(requested_stations)].copy()
    print(f"✅ Processing {len(gdf_basins)} basins.")

    # 2. ELEVATION (SRTM 90m)
    print("⏳ Processing Elevation Data...")
    
    # Use Config Path for Mosaic
    dem_path = RAW_DATA_DIR / "srtm_90m_mosaic.tif"
    
    basins_wgs84 = gdf_basins.to_crs("EPSG:4326")
    total_bounds = basins_wgs84.total_bounds
    
    try:
        mosaic_result = get_elevation_mosaic(total_bounds, str(dem_path))
        
        if mosaic_result:
            print("   Computing mean elevation per basin...")
            stats = zonal_stats(basins_wgs84, str(dem_path), stats="mean")
            gdf_basins['mean_elev'] = [s['mean'] for s in stats]
            
            # Fill NaNs with median
            missing = gdf_basins['mean_elev'].isna().sum()
            if missing > 0:
                print(f"   ⚠️ Warning: {missing} basins missing elevation. Filling with median.")
                gdf_basins['mean_elev'] = gdf_basins['mean_elev'].fillna(gdf_basins['mean_elev'].median())
        else:
            print("   ⚠️ Mosaic generation failed. Using 0 elevation.")
            gdf_basins['mean_elev'] = 0.0
            
    except Exception as e:
        print(f"   ❌ Elevation processing failed: {e}")
        gdf_basins['mean_elev'] = 0.0

    # 3. REPROJECT & AREAS
    gdf_basins = gdf_basins.to_crs(CANADA_ALBERS_CRS)
    gdf_basins['basin_area_km2'] = gdf_basins.geometry.area / 1e6

    # 4. GLACIER INTERSECTION
    print("⏳ Loading glaciers...")
    gdf_glaciers = gpd.read_file(glacier_shp_path)
    rgi_col = next((c for c in gdf_glaciers.columns if 'rgiid' in c.lower()), 'RGIId')
    gdf_glaciers = gdf_glaciers.rename(columns={rgi_col: 'RGIId'})
    gdf_glaciers = gdf_glaciers.to_crs(CANADA_ALBERS_CRS)

    intersection = gpd.overlay(
        gdf_glaciers[['RGIId', 'geometry']], 
        gdf_basins[['station_id', 'geometry']], 
        how='intersection'
    )
    intersection['glacier_area_km2'] = intersection.geometry.area / 1e6

    # 5. SAVE STATIC
    glacier_sums = intersection.groupby('station_id')['glacier_area_km2'].sum()
    
    static_df = gdf_basins[['station_id', 'basin_area_km2', 'mean_elev']].set_index('station_id')
    static_df['glacier_area_km2'] = glacier_sums
    static_df['glacier_area_km2'] = static_df['glacier_area_km2'].fillna(0)
    static_df['glacier_pct'] = (static_df['glacier_area_km2'] / static_df['basin_area_km2']) * 100
    
    static_df.to_csv(OUTPUT_STATIC_ATTR)
    print(f"✅ Static attributes saved to {OUTPUT_STATIC_ATTR}")

    # 6. VOLUME CHANGE
    print("⏳ Calculating volume changes...")
    area_matrix = intersection.pivot_table(
        index='RGIId', columns='station_id', values='glacier_area_km2', 
        aggfunc='sum', fill_value=0
    )

    mb_df = pd.read_csv(mass_balance_path, index_col=0)
    common_glaciers = area_matrix.index.intersection(mb_df.index)
    
    if len(common_glaciers) > 0:
        vol_change = mb_df.loc[common_glaciers].T.dot(area_matrix.loc[common_glaciers])
        vol_change.index = pd.to_datetime(vol_change.index)
        vol_change.to_csv(OUTPUT_GLACIER_VOL)
        print(f"✅ Volume changes saved to {OUTPUT_GLACIER_VOL}")
        return static_df, vol_change
    else:
        return static_df, None
