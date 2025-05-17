# app/data_utils.py
import pandas as pd
import geopandas as gpd
import os
import time
import traceback

# --- Configuration ---
PARCEL_GEOJSON_PATH = os.path.join("data", "King_County_Parcels___parcel_area.geojson")
ASSESSMENT_FILE_PATH = os.path.join("data", "kc_assessment_data.csv")

PIN_COLUMN_PARCELS = "PIN"
PIN_COLUMN_ASSESSMENT = "PIN"
ASSESSMENT_COLUMNS_TO_KEEP = ['ADDRESS', 'ASSESSED_VALUE', 'BUILDING_VALUE', 'ACREAGE', 'USE_CODE', 'CITY_CODE'] # PIN will be merge key

BELLEVUE_BOUNDS = (-122.24, 47.56, -122.10, 47.65)

_PRIMARY_DATA_CACHE = None # Changed cache name

# --- load_king_county_data (Revised to prioritize GeoJSON parcels) ---
def load_king_county_data(force_reload=False):
    global _PRIMARY_DATA_CACHE
    print(f"DEBUG load_king_county_data: Called. force_reload={force_reload}. Cache is {'None' if _PRIMARY_DATA_CACHE is None else 'Populated'}")
    if _PRIMARY_DATA_CACHE is not None and not force_reload:
        print(f"DEBUG load_king_county_data: Returning CACHED _PRIMARY_DATA_CACHE with {len(_PRIMARY_DATA_CACHE)} rows.")
        return _PRIMARY_DATA_CACHE
    
    print("DEBUG load_king_county_data: Cache is empty or force_reload=True. Proceeding with full data load.")
    start_time = time.time()
    parcels_gdf = None
    assessment_df = None

    # Step 1: Load Parcels from GeoJSON (This is our primary geometry source)
    print(f"Loading parcel data from GeoJSON: {PARCEL_GEOJSON_PATH}")
    try:
        parcels_gdf = gpd.read_file(PARCEL_GEOJSON_PATH)
        print(f"Loaded {len(parcels_gdf)} initial parcels from GeoJSON.")
        if parcels_gdf.crs: # Should be EPSG:4326 or CRS84
            if str(parcels_gdf.crs).upper() != "EPSG:4326" and "CRS84" not in str(parcels_gdf.crs).upper():
                print(f"Warning: Parcel CRS is {parcels_gdf.crs}. Forcing to EPSG:4326.")
                parcels_gdf = parcels_gdf.set_crs("EPSG:4326", allow_override=True)
        else:
            print("WARNING: Parcel GeoJSON CRS not detected. Assuming EPSG:4326 (WGS84).")
            parcels_gdf.set_crs(epsg=4326, inplace=True, allow_override=True)

        if PIN_COLUMN_PARCELS not in parcels_gdf.columns:
             print(f"ERROR: PIN column '{PIN_COLUMN_PARCELS}' not found in GeoJSON properties! Available: {parcels_gdf.columns.tolist()}")
             return None
        parcels_gdf[PIN_COLUMN_PARCELS] = parcels_gdf[PIN_COLUMN_PARCELS].astype(str)
        
        # Clean parcel geometries early
        initial_parcel_count = len(parcels_gdf)
        parcels_gdf = parcels_gdf[parcels_gdf.geometry.is_valid & ~parcels_gdf.geometry.isna()]
        print(f"Parcels after geometry cleaning: {len(parcels_gdf)} ({initial_parcel_count - len(parcels_gdf)} removed)")

    except FileNotFoundError: # ... (error handling) ...
        print(f"ERROR: Parcel GeoJSON not found: {PARCEL_GEOJSON_PATH}")
        return None
    except Exception as e: # ... (error handling) ...
        print(f"ERROR loading GeoJSON: {e}")
        traceback.print_exc()
        return None
    
    if parcels_gdf.empty:
        print("ERROR: No valid parcels loaded from GeoJSON. Cannot proceed.")
        _PRIMARY_DATA_CACHE = parcels_gdf # Cache empty gdf
        return _PRIMARY_DATA_CACHE


    # Step 2: Load Assessment Data (This will be 'right' table in merge)
    print(f"Loading assessment data from CSV: {ASSESSMENT_FILE_PATH}")
    try:
        assessment_df = pd.read_csv(ASSESSMENT_FILE_PATH, low_memory=False)
        if PIN_COLUMN_ASSESSMENT not in assessment_df.columns:
            print(f"ERROR: PIN column '{PIN_COLUMN_ASSESSMENT}' not in Assessment CSV. Available: {assessment_df.columns.tolist()}")
            # Continue without assessment data if it's missing PIN, but log warning
            assessment_df = pd.DataFrame(columns=[PIN_COLUMN_ASSESSMENT] + ASSESSMENT_COLUMNS_TO_KEEP) # empty df with PIN
        else:
            assessment_df[PIN_COLUMN_ASSESSMENT] = assessment_df[PIN_COLUMN_ASSESSMENT].astype(str)
            # Select only PIN + columns to keep (avoiding duplicate non-geometry attributes from parcels_gdf if any)
            cols_to_keep_from_assessment = [PIN_COLUMN_ASSESSMENT] + ASSESSMENT_COLUMNS_TO_KEEP
            cols_to_keep_from_assessment = [col for col in cols_to_keep_from_assessment if col in assessment_df.columns]
            assessment_df = assessment_df[cols_to_keep_from_assessment]
        print(f"Loaded {len(assessment_df)} assessment records.")
    except FileNotFoundError: # ... (error handling) ...
        print(f"WARNING: Assessment file not found: '{ASSESSMENT_FILE_PATH}'. Proceeding with parcel data only.")
        assessment_df = pd.DataFrame(columns=[PIN_COLUMN_ASSESSMENT] + ASSESSMENT_COLUMNS_TO_KEEP) # empty df with PIN
    except Exception as e: # ... (error handling) ...
        print(f"WARNING: Error reading assessment file: {e}. Proceeding with parcel data only.")
        assessment_df = pd.DataFrame(columns=[PIN_COLUMN_ASSESSMENT] + ASSESSMENT_COLUMNS_TO_KEEP) # empty df with PIN

    # --- Merging: parcels_gdf is LEFT, assessment_df is RIGHT ---
    print("Merging parcel geometries with assessment data (parcels are primary)...")
    try:
        # Ensure PIN column names match for the merge key
        # (parcels_gdf uses PIN_COLUMN_PARCELS, assessment_df uses PIN_COLUMN_ASSESSMENT)
        if PIN_COLUMN_PARCELS != PIN_COLUMN_ASSESSMENT and PIN_COLUMN_PARCELS in parcels_gdf.columns and PIN_COLUMN_ASSESSMENT in assessment_df.columns:
             # This should ideally be handled by making sure both are just 'PIN'
             # For now, let's assume both are 'PIN' as defined in config. If not, this merge will be tricky.
             print(f"Using '{PIN_COLUMN_PARCELS}' from parcels and '{PIN_COLUMN_ASSESSMENT}' from assessments for merge.")

        # Perform a left merge to keep all parcels, add assessment data if PIN matches
        # We need to ensure the PIN column used for merging exists in both and has the same name temporarily if needed
        # Or specify left_on and right_on
        merged_gdf = parcels_gdf.merge(
            assessment_df, 
            left_on=PIN_COLUMN_PARCELS, 
            right_on=PIN_COLUMN_ASSESSMENT, 
            how='left'
        )
        print(f"Merged data shape: {merged_gdf.shape}")

        # If the merge created duplicate PIN columns (e.g. PIN_x, PIN_y), clean up
        if f"{PIN_COLUMN_PARCELS}_x" in merged_gdf.columns: # Check if pandas added suffixes
            merged_gdf[PIN_COLUMN_PARCELS] = merged_gdf[f"{PIN_COLUMN_PARCELS}_x"]
            cols_to_drop_after_merge = [col for col in merged_gdf.columns if col.endswith("_x") or col.endswith("_y")]
            merged_gdf.drop(columns=cols_to_drop_after_merge, inplace=True, errors='ignore')


        # Geometry and CRS should be preserved from parcels_gdf (the left GeoDataFrame)
        # Fill NaNs for attribute columns that came from assessment_df
        for col in ASSESSMENT_COLUMNS_TO_KEEP: # Exclude PIN which should be there
            if col in merged_gdf.columns: # Check if column exists after merge
                merged_gdf[col] = merged_gdf[col].fillna("N/A")
            else: # If an assessment column wasn't even in assessment_df, add it as N/A
                merged_gdf[col] = "N/A"
        
        # Ensure PIN is clean
        if PIN_COLUMN_PARCELS in merged_gdf.columns:
            merged_gdf[PIN_COLUMN_PARCELS] = merged_gdf[PIN_COLUMN_PARCELS].fillna("N/A")
        elif PIN_COLUMN_ASSESSMENT in merged_gdf.columns and PIN_COLUMN_ASSESSMENT != PIN_COLUMN_PARCELS:
             merged_gdf[PIN_COLUMN_ASSESSMENT] = merged_gdf[PIN_COLUMN_ASSESSMENT].fillna("N/A")


        # Park distance temporarily disabled
        merged_gdf['distance_to_park_meters'] = "N/A (disabled)"

        _PRIMARY_DATA_CACHE = merged_gdf
        end_time = time.time()
        print(f"Data loading and processing complete in {time.time() - start_time:.2f} seconds. Final shape: {merged_gdf.shape}")
        return _PRIMARY_DATA_CACHE
    except Exception as e:
        print(f"ERROR during data merging or final processing: {e}")
        traceback.print_exc()
        return None

# --- get_parcels_geojson_subset (No change from previous version where filter was removed) ---
def get_parcels_geojson_subset(bounds=None):
    print("DEBUG get_parcels_geojson_subset: Called.")
    gdf = load_king_county_data() # This now uses the GeoJSON for parcels as base
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty or gdf.geometry.isnull().all():
        print("DEBUG get_parcels_geojson_subset: gdf is None, empty, or invalid. Returning empty GeoJSON.")
        return '{"type": "FeatureCollection", "features": []}'
    print(f"DEBUG get_parcels_geojson_subset: Received gdf with {len(gdf)} rows before spatial filtering.")
    filtered_gdf = gdf
    if bounds:
        print(f"DEBUG get_parcels_geojson_subset: Filtering {len(gdf)} parcels to bounds: {bounds}")
        try:
            if filtered_gdf.geometry.isnull().all():
                print("DEBUG get_parcels_geojson_subset: All geometries are null before spatial filter.")
                filtered_gdf = gpd.GeoDataFrame(columns=gdf.columns, geometry=[], crs=gdf.crs)
            else:
                filtered_gdf = gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
            print(f"DEBUG get_parcels_geojson_subset: Found {len(filtered_gdf)} parcels within bounds.")
        except Exception as e:
             print(f"ERROR during spatial filtering: {e}. Proceeding with all {len(gdf)} parcels.")
             filtered_gdf = gdf
    if filtered_gdf.empty:
        print("DEBUG get_parcels_geojson_subset: filtered_gdf is empty. Returning empty GeoJSON.")
        return '{"type": "FeatureCollection", "features": []}'
    print(f"DEBUG get_parcels_geojson_subset: Converting {len(filtered_gdf)} final parcels to GeoJSON...")
    try:
        # Select only essential columns for GeoJSON to keep it smaller
        # Properties from GeoJSON + key assessment data. PIN is essential for click.
        # The geometry column is handled by to_json() automatically.
        cols_for_geojson = [PIN_COLUMN_PARCELS] # Start with PIN (or the main PIN column after merge)
        # Add other properties from original GeoJSON if they exist and are desired
        # e.g. if parcels_gdf had 'MAJOR', 'MINOR' and they are in merged_gdf
        for col in ['MAJOR', 'MINOR', 'OBJECTID']: # Example props from GeoJSON
            if col in filtered_gdf.columns and col != PIN_COLUMN_PARCELS: # Avoid duplicating PIN
                cols_for_geojson.append(col)
        
        # Add key assessment columns to show in popup *if they are needed by onEachFeature directly*
        # Currently, onEachFeature only uses PIN, and details are fetched via API.
        # So, for the GeoJSON sent to the map, we only strictly need PIN and geometry.
        # However, if you wanted to display something directly from GeoJSON props on hover, add them here.
        # For now, let's just send PIN and a few basic original parcel props.
        
        # Ensure the geometry column is also included if not implicitly handled
        geo_col_name = filtered_gdf.geometry.name
        if geo_col_name not in cols_for_geojson:
             final_gdf_for_geojson = filtered_gdf[cols_for_geojson + [geo_col_name]]
        else:
             final_gdf_for_geojson = filtered_gdf[cols_for_geojson]


        geojson_data = final_gdf_for_geojson.to_json()
        print("DEBUG get_parcels_geojson_subset: GeoJSON conversion successful.")
        return geojson_data
    except Exception as e:
        print(f"ERROR converting to GeoJSON in get_parcels_geojson_subset: {e}")
        traceback.print_exc()
        return '{"type": "FeatureCollection", "features": []}'

# --- REVISED get_info_for_pin (PCA temporarily disabled) ---
def get_info_for_pin(pin):
    merged_gdf = load_king_county_data() # This now uses the GeoJSON as base
    if merged_gdf is None: return {"error": "Data not loaded"}
    if merged_gdf.empty: return {"error": f"PIN {str(pin)} not found (data empty)."}
    try:
        pin_str = str(pin)
        # Use the primary PIN column (should be PIN_COLUMN_PARCELS as defined in config if merge was correct)
        # If merge resulted in PIN_x, PIN_y, ensure this uses the correct one.
        # Assuming the main PIN column in merged_gdf is PIN_COLUMN_PARCELS after merge cleanup
        property_data = merged_gdf.loc[merged_gdf[PIN_COLUMN_PARCELS] == pin_str]
        
        if property_data.empty: return {"error": f"PIN {pin_str} not found"}
        
        property_series = property_data.iloc[0]
        geo_col = merged_gdf.geometry.name
        
        # Create info dict excluding geometry first
        info_cols_to_drop = [geo_col]
        # If PIN_COLUMN_ASSESSMENT was different and got merged in as a separate column, consider dropping it too if PIN_COLUMN_PARCELS is primary
        if PIN_COLUMN_ASSESSMENT in property_series.index and PIN_COLUMN_ASSESSMENT != PIN_COLUMN_PARCELS:
            info_cols_to_drop.append(PIN_COLUMN_ASSESSMENT)

        info = property_series.drop(labels=info_cols_to_drop, errors='ignore').to_dict()

        # Add centroid if geometry is valid
        if geo_col in property_series and property_series[geo_col] is not None and hasattr(property_series[geo_col], 'is_valid') and property_series[geo_col].is_valid :
            centroid = property_series[geo_col].centroid
            info['latitude'] = centroid.y
            info['longitude'] = centroid.x
        
        if 'ASSESSED_VALUE' in info and info['ASSESSED_VALUE'] != "N/A":
            try: info['AssessedValueFormatted'] = f"${float(info['ASSESSED_VALUE']):,.0f}"
            except (ValueError, TypeError): info['AssessedValueFormatted'] = info['ASSESSED_VALUE']
        
        info['pca_top_factors'] = ["PCA temporarily disabled"] # Placeholder
        return info
    except Exception as e:
        traceback.print_exc(); return {"error": f"Error fetching details for PIN {pin_str}"}

# --- Keep get_pca_factors_for_pin function but ensure it's not called if disabled ---
def get_pca_factors_for_pin(pin_to_analyze): # This function is not called by get_info_for_pin for now
    print(f"INFO: PCA calculation for PIN {pin_to_analyze} requested but currently bypassed in get_info_for_pin.")
    return {"message": "PCA processing is currently bypassed for performance."}