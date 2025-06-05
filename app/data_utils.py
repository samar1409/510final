# app/data_utils.py
import pandas as pd
import geopandas as gpd
import os
import time
import traceback
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from google.cloud import storage

# --- Configuration ---
# Add GCS configs
BUCKET_NAME = "samar-kingcounty-data"
GCS_PARCEL_PATH = "King_County_Parcels___parcel_area.geojson"
GCS_ASSESSMENT_PATH = "kc_assessment_data.csv"

# Keep local paths as fallback
PARCEL_GEOJSON_PATH = os.path.join("data", "King_County_Parcels___parcel_area.geojson")
ASSESSMENT_FILE_PATH = os.path.join("data", "kc_assessment_data.csv")
# PARKS_SHAPEFILE_PATH = os.path.join("data", "kc_parks.shp") # For future park integration

PIN_COLUMN_PARCELS = "PIN"
PIN_COLUMN_ASSESSMENT = "PIN"
ASSESSMENT_COLUMNS_TO_KEEP = [PIN_COLUMN_ASSESSMENT, 'ADDRESS', 'ASSESSED_VALUE', 'BUILDING_VALUE', 'ACREAGE', 'USE_CODE', 'CITY_CODE']
GEOJSON_PROPS_TO_KEEP = [PIN_COLUMN_PARCELS, 'MAJOR', 'MINOR', 'OBJECTID', 'Shape_Area', 'Shape_Length']
BELLEVUE_BOUNDS = (-122.24, 47.56, -122.10, 47.65)

_PRIMARY_DATA_CACHE = None
# _PARKS_UNION_GEOM_CACHE = None

# def load_parks_data(): ... # Keep for future

def get_gcs_file(blob_path):
    """Get file from Google Cloud Storage."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_path)
        return blob.download_as_string()
    except Exception as e:
        print(f"GCS Error: {e}. Falling back to local file.")
        return None

def load_king_county_data(force_reload=False):
    global _PRIMARY_DATA_CACHE
    # ... (cache check logic) ...
    print(f"DEBUG load_king_county_data: Called. force_reload={force_reload}. Cache is {'None' if _PRIMARY_DATA_CACHE is None else 'Populated'}")
    if _PRIMARY_DATA_CACHE is not None and not force_reload:
        print(f"DEBUG load_king_county_data: Returning CACHED _PRIMARY_DATA_CACHE with {len(_PRIMARY_DATA_CACHE)} rows.")
        return _PRIMARY_DATA_CACHE
    
    print("DEBUG load_king_county_data: Cache is empty or force_reload=True. Proceeding with full data load.")
    start_time = time.time()

    # Step 1: Load Parcels from GeoJSON
    try:
        # Try GCS first
        print("Attempting to load data from Google Cloud Storage...")
        parcel_data = get_gcs_file(GCS_PARCEL_PATH)
        if parcel_data:
            print("Successfully loaded from GCS!")
            parcels_gdf = gpd.read_file(parcel_data)
        else:
            print("Falling back to local files...")
            parcels_gdf = gpd.read_file(PARCEL_GEOJSON_PATH)
            
        print(f"Loaded {len(parcels_gdf)} initial parcels from GeoJSON.")
        # CRS Handling (ensure EPSG:4326)
        if parcels_gdf.crs:
            if str(parcels_gdf.crs).upper() != "EPSG:4326" and "CRS84" not in str(parcels_gdf.crs).upper():
                parcels_gdf = parcels_gdf.set_crs("EPSG:4326", allow_override=True)
        else: parcels_gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
        
        if PIN_COLUMN_PARCELS not in parcels_gdf.columns: raise ValueError(f"PIN column '{PIN_COLUMN_PARCELS}' not found in GeoJSON")
        parcels_gdf[PIN_COLUMN_PARCELS] = parcels_gdf[PIN_COLUMN_PARCELS].astype(str)
        
        geojson_shape_area_col = 'Shape_Area_geojson' # Standardized name
        if 'Shape_Area' in parcels_gdf.columns:
            parcels_gdf[geojson_shape_area_col] = pd.to_numeric(parcels_gdf['Shape_Area'], errors='coerce')
        elif geojson_shape_area_col in parcels_gdf.columns: # If already named this
             parcels_gdf[geojson_shape_area_col] = pd.to_numeric(parcels_gdf[geojson_shape_area_col], errors='coerce')
        else: parcels_gdf[geojson_shape_area_col] = np.nan
            
        parcels_gdf = parcels_gdf[parcels_gdf.geometry.is_valid & ~parcels_gdf.geometry.isna()]
        print(f"Parcels after geometry cleaning: {len(parcels_gdf)}")
    except Exception as e: print(f"ERROR loading GeoJSON: {e}"); traceback.print_exc(); return None
    if parcels_gdf.empty: print("ERROR: No valid parcels from GeoJSON."); _PRIMARY_DATA_CACHE = parcels_gdf; return _PRIMARY_DATA_CACHE

    # Step 2: Load Assessment Data
    try:
        assessment_df = pd.read_csv(ASSESSMENT_FILE_PATH, low_memory=False)
        if PIN_COLUMN_ASSESSMENT not in assessment_df.columns:
            assessment_df = pd.DataFrame(columns=[PIN_COLUMN_ASSESSMENT]) 
        else:
            assessment_df[PIN_COLUMN_ASSESSMENT] = assessment_df[PIN_COLUMN_ASSESSMENT].astype(str)
            actual_assessment_cols = [col for col in ASSESSMENT_COLUMNS_TO_KEEP if col in assessment_df.columns]
            if PIN_COLUMN_ASSESSMENT not in actual_assessment_cols and PIN_COLUMN_ASSESSMENT in assessment_df.columns:
                actual_assessment_cols = [PIN_COLUMN_ASSESSMENT] + [c for c in actual_assessment_cols if c != PIN_COLUMN_ASSESSMENT]
            unique_assessment_cols = list(dict.fromkeys(actual_assessment_cols))
            assessment_df = assessment_df[unique_assessment_cols]
        print(f"Loaded {len(assessment_df)} assessment records. Columns: {assessment_df.columns.tolist()}")
    except Exception as e: print(f"WARNING: Error reading assessment: {e}."); assessment_df = pd.DataFrame(columns=[PIN_COLUMN_ASSESSMENT])

    # --- Merging ---
    geo_col = parcels_gdf.geometry.name
    props_from_geojson_to_keep = [PIN_COLUMN_PARCELS, geo_col, geojson_shape_area_col] + \
                                 [col for col in GEOJSON_PROPS_TO_KEEP if col in parcels_gdf.columns and col not in [PIN_COLUMN_PARCELS, geo_col, 'Shape_Area', geojson_shape_area_col]]
    parcels_subset_gdf = parcels_gdf[list(dict.fromkeys(props_from_geojson_to_keep))].copy()

    merged_gdf = parcels_subset_gdf.merge(
        assessment_df.drop_duplicates(subset=[PIN_COLUMN_ASSESSMENT]),
        on=PIN_COLUMN_PARCELS, how='left', suffixes=('_geojson', '')
    )
    print(f"Merged data shape: {merged_gdf.shape}")
    
    # If PIN from assessment_df (right side) was suffixed, remove it as PIN_geojson is primary
    if PIN_COLUMN_ASSESSMENT + '' in merged_gdf.columns and PIN_COLUMN_ASSESSMENT != PIN_COLUMN_PARCELS : # Check if suffixed PIN exists and is different from main PIN
        merged_gdf.drop(columns=[PIN_COLUMN_ASSESSMENT + ''], inplace=True, errors='ignore')


    # --- Generate/Calculate Missing Data & Convert Types ---
    print("Generating/Calculating placeholder data for missing assessment fields...")
    
    # Initialize columns if they don't exist after merge (e.g., if assessment_df was empty)
    for col in ['ADDRESS', 'ASSESSED_VALUE', 'BUILDING_VALUE', 'ACREAGE', 'USE_CODE']:
        if col not in merged_gdf.columns:
            merged_gdf[col] = np.nan # Use NaN for numeric types initially

    # ADDRESS
    addr_col = 'ADDRESS'
    missing_addr_mask = merged_gdf[addr_col].isna() | (merged_gdf[addr_col] == "N/A")
    objid_col_for_addr = next((c for c in ['OBJECTID_geojson', 'OBJECTID'] if c in merged_gdf.columns), None)
    if objid_col_for_addr:
        merged_gdf.loc[missing_addr_mask, addr_col] = "GenAddr-" + pd.to_numeric(merged_gdf.loc[missing_addr_mask, objid_col_for_addr], errors='coerce').fillna(0).astype(int).astype(str)
    else: merged_gdf.loc[missing_addr_mask, addr_col] = "GenAddr-PIN-" + merged_gdf.loc[missing_addr_mask, PIN_COLUMN_PARCELS]

    # ASSESSED_VALUE & BUILDING_VALUE (Numeric, then fill missing)
    for col in ['ASSESSED_VALUE', 'BUILDING_VALUE']:
        merged_gdf[col] = pd.to_numeric(merged_gdf[col], errors='coerce')
    
    missing_assessed_mask = merged_gdf['ASSESSED_VALUE'].isna()
    num_missing_assessed = missing_assessed_mask.sum()
    if num_missing_assessed > 0:
        merged_gdf.loc[missing_assessed_mask, 'ASSESSED_VALUE'] = np.random.randint(200000, 3000001, size=num_missing_assessed)

    missing_bldg_mask = merged_gdf['BUILDING_VALUE'].isna()
    target_for_bldg_gen = missing_bldg_mask & merged_gdf['ASSESSED_VALUE'].notna()
    num_missing_bldg = target_for_bldg_gen.sum()
    if num_missing_bldg > 0:
        random_ratios = np.random.uniform(0.5, 0.9, size=num_missing_bldg)
        merged_gdf.loc[target_for_bldg_gen, 'BUILDING_VALUE'] = \
            (merged_gdf.loc[target_for_bldg_gen, 'ASSESSED_VALUE'] * random_ratios).astype(int)

    # USE_CODE
    use_code_col = 'USE_CODE'
    common_use_codes = ["SFR-Gen", "MultiFam-Gen", "Comm-Gen", "Indust-Gen", "Vacant-Gen"]
    missing_use_mask = merged_gdf[use_code_col].isna() | (merged_gdf[use_code_col] == "N/A")
    num_missing_use = missing_use_mask.sum()
    if num_missing_use > 0:
        merged_gdf.loc[missing_use_mask, use_code_col] = np.random.choice(common_use_codes, size=num_missing_use)

    # ACREAGE
    acre_col = 'ACREAGE' # Already numeric from earlier to_numeric or initialized as NaN
    missing_acreage_mask = merged_gdf[acre_col].isna()
    
    if geojson_shape_area_col in merged_gdf.columns: # geojson_shape_area_col = 'Shape_Area_geojson'
        rows_calc_geojson_area = missing_acreage_mask & merged_gdf[geojson_shape_area_col].notna() & (merged_gdf[geojson_shape_area_col] > 0)
        if rows_calc_geojson_area.any():
            print(f"DEBUG: Calculating acreage from GeoJSON '{geojson_shape_area_col}' for {rows_calc_geojson_area.sum()} parcels.")
            avg_lat_rad = np.deg2rad((BELLEVUE_BOUNDS[1] + BELLEVUE_BOUNDS[3]) / 2)
            sq_deg_to_acres_factor = (111133.0 * 111320.0 * np.cos(avg_lat_rad)) * 0.000247105
            calculated_acres = merged_gdf.loc[rows_calc_geojson_area, geojson_shape_area_col] * sq_deg_to_acres_factor
            merged_gdf.loc[rows_calc_geojson_area, acre_col] = calculated_acres
    
    missing_acreage_mask = merged_gdf[acre_col].isna() # Re-evaluate
    rows_calc_geom_area = missing_acreage_mask & merged_gdf.geometry.is_valid & ~merged_gdf.geometry.is_empty
    if rows_calc_geom_area.any():
        print(f"DEBUG: Calculating acreage from geometry for {rows_calc_geom_area.sum()} parcels.")
        try:
            area_sq_ft = merged_gdf.loc[rows_calc_geom_area].geometry.to_crs(epsg=2926).area
            merged_gdf.loc[rows_calc_geom_area, acre_col] = area_sq_ft / 43560.0
        except Exception as e_area: print(f"ERROR calculating area from geometry: {e_area}")
    
    # Final numeric conversion for acreage
    merged_gdf[acre_col] = pd.to_numeric(merged_gdf[acre_col], errors='coerce')


    # Park distance (still disabled for now to keep focus)
    merged_gdf['distance_to_park_meters'] = np.nan 

    # Fill any remaining NaNs in object columns with "N/A" for display
    for col in merged_gdf.select_dtypes(include='object').columns:
        merged_gdf[col].fillna("N/A", inplace=True)

    _PRIMARY_DATA_CACHE = merged_gdf
    end_time = time.time()
    print(f"Data loading, generation, and processing complete in {time.time() - start_time:.2f} seconds. Final shape: {merged_gdf.shape}")
    return _PRIMARY_DATA_CACHE

# --- RE-ENABLE and REFINE PCA function ---
def get_pca_analysis(pin_to_analyze): # Renamed for clarity
    """
    Performs PCA on selected features for the Bellevue subset of data.
    Returns top influencing factors and a simple predictive hint.
    """
    print(f"DEBUG: Performing PCA analysis (context of PIN {pin_to_analyze})")
    merged_gdf = load_king_county_data()
    if merged_gdf is None or merged_gdf.empty:
        return {"error": "Data not loaded or empty for PCA."}

    pca_subset_gdf = merged_gdf.cx[BELLEVUE_BOUNDS[0]:BELLEVUE_BOUNDS[2], BELLEVUE_BOUNDS[1]:BELLEVUE_BOUNDS[3]]
    if pca_subset_gdf.empty: return {"error": "No data in Bellevue subset for PCA."}

    features_for_pca = ['ACREAGE', 'ASSESSED_VALUE', 'BUILDING_VALUE'] # Add 'distance_to_park_meters' when parks are re-enabled
    
    # Ensure features are numeric and select only existing ones
    numerical_data_for_pca = pd.DataFrame()
    actual_features_used = []
    for col in features_for_pca:
        if col in pca_subset_gdf.columns:
            numerical_data_for_pca[col] = pd.to_numeric(pca_subset_gdf[col], errors='coerce')
            actual_features_used.append(col)
        else: print(f"PCA: Feature '{col}' not found in subset, skipping.")
    
    if numerical_data_for_pca.empty or len(actual_features_used) < 2:
        return {"error": "Not enough valid numerical features for PCA."}

    # Handle missing values (NaNs from coercion or original data)
    imputer = SimpleImputer(strategy='mean') # Use mean, median, or constant
    data_imputed = imputer.fit_transform(numerical_data_for_pca)
    data_imputed_df = pd.DataFrame(data_imputed, columns=actual_features_used, index=numerical_data_for_pca.index)

    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed_df)

    if data_scaled.shape[0] < max(2, data_scaled.shape[1]): # Need more samples than features typically
        return {"error": "Not enough data points after preprocessing for PCA."}

    # Apply PCA
    n_components = min(len(actual_features_used), 2) # Aim for 2 components
    pca = PCA(n_components=n_components)
    try:
        pca.fit(data_scaled)
    except ValueError as ve:
        return {"error": f"PCA fitting error (likely due to NaNs/Infs remaining): {ve}"}


    # Interpret components for the specific PIN if found, or general for Bellevue
    pin_data_scaled = None
    if pin_to_analyze in pca_subset_gdf[PIN_COLUMN_PARCELS].values:
        pin_row_original = data_imputed_df[pca_subset_gdf[PIN_COLUMN_PARCELS] == pin_to_analyze]
        if not pin_row_original.empty:
            pin_data_scaled = scaler.transform(pin_row_original) # Transform the specific PIN's data
            # pin_pca_transformed = pca.transform(pin_data_scaled) # Not directly used for hint yet

    # Get feature loadings on PC1
    pc1_loadings_abs = np.abs(pca.components_[0])
    pc1_loadings_signed = pca.components_[0]

    # Create a list of (feature_name, absolute_loading, signed_loading)
    feature_contributions = []
    for i, feature_name in enumerate(actual_features_used):
        feature_contributions.append((feature_name, pc1_loadings_abs[i], pc1_loadings_signed[i]))
    
    # Sort by absolute loading to find most influential
    sorted_contributions = sorted(feature_contributions, key=lambda x: x[1], reverse=True)
    
    top_factors_pc1 = [f"{feat} ({loading_signed:.2f})" for feat, _, loading_signed in sorted_contributions[:3]] # Top 3

    # Simple Predictive Hint based on the strongest factor in PC1
    predictive_hint = "Market dynamics are complex."
    if sorted_contributions:
        top_factor_name = sorted_contributions[0][0]
        top_factor_signed_loading = sorted_contributions[0][2] # Signed loading

        if top_factor_name == 'ASSESSED_VALUE':
            if top_factor_signed_loading > 0:
                predictive_hint = "Higher assessed values are a key differentiator. Market may favor premium properties if demand holds."
            else:
                predictive_hint = "Relative value (lower assessed value for size/type) might be a key search factor."
        elif top_factor_name == 'ACREAGE':
            if top_factor_signed_loading > 0:
                predictive_hint = "Larger property sizes (acreage) are a significant positive factor in this area's valuation."
            else:
                predictive_hint = "Smaller, denser property use may be influencing value variations more than large acreage."
        elif top_factor_name == 'BUILDING_VALUE':
             if top_factor_signed_loading > 0:
                predictive_hint = "Substantial building values contribute positively. Quality of structures is likely key."
             else:
                predictive_hint = "Land value or other factors might be overshadowing building value in variations."
        # elif top_factor_name == 'distance_to_park_meters': # When parks re-enabled
        #     if top_factor_signed_loading < 0: # Negative loading means smaller distance (closer) is positive for PC1
        #         predictive_hint = "Proximity to parks appears to positively influence property appeal."
        #     else:
        #         predictive_hint = "Distance to parks shows less correlation with the primary market variations."


    pca_results = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "top_factors_pc1": top_factors_pc1,
        "predictive_hint": predictive_hint,
        # "components_matrix": pca.components_.tolist(), # Optional: For more detailed display
        # "feature_names": actual_features_used # Optional
    }
    print(f"PCA results (context {pin_to_analyze}): {pca_results}")
    return pca_results


# --- REVISED get_info_for_pin to include PCA and use new predictive hint ---
def get_info_for_pin(pin):
    merged_gdf = load_king_county_data()
    if merged_gdf is None: return {"error": "Data not loaded"}
    if merged_gdf.empty: return {"error": f"PIN {str(pin)} not found (data empty)."}
    try:
        pin_str = str(pin)
        property_data = merged_gdf.loc[merged_gdf[PIN_COLUMN_PARCELS] == pin_str]
        
        if property_data.empty: return {"error": f"PIN {pin_str} not found"}
        
        property_series = property_data.iloc[0]
        geo_col = merged_gdf.geometry.name
        
        # Start building the info dictionary
        info_cols_to_drop = [geo_col] 
        # Add suffixed columns to drop if they exist from merge conflicts and are not primary
        for col_name in property_series.index:
            if (col_name.endswith("_geojson") or col_name.endswith("_assess")) and \
               col_name.replace("_geojson","").replace("_assess","") in property_series.index:
                info_cols_to_drop.append(col_name)
        info = property_series.drop(labels=list(set(info_cols_to_drop)), errors='ignore').to_dict()


        # Add centroid if geometry is valid
        info['latitude'] = None 
        info['longitude'] = None
        if geo_col in property_series and property_series[geo_col] is not None and hasattr(property_series[geo_col], 'is_valid') and property_series[geo_col].is_valid :
            centroid = property_series[geo_col].centroid
            info['latitude'] = centroid.y
            info['longitude'] = centroid.x
        
        # Format Assessed Value
        assessed_val_key = 'ASSESSED_VALUE'
        if assessed_val_key in info and pd.notna(info[assessed_val_key]) and info[assessed_val_key] != "N/A": # Check for string "N/A" too
            try: info['AssessedValueFormatted'] = f"${float(info[assessed_val_key]):,.0f}"
            except (ValueError, TypeError): info['AssessedValueFormatted'] = str(info[assessed_val_key]) 
        else: info['AssessedValueFormatted'] = "N/A"

        # Format Acreage
        acre_key = 'ACREAGE'
        if acre_key in info and pd.notna(info[acre_key]) and info[acre_key] != "N/A": # Check for string "N/A"
            try:
                acreage_val = float(info[acre_key])
                info['AcreageFormatted'] = f"{acreage_val:.2f} acres"
                # Removed ACREAGE_SRC display from here as per user request
            except (ValueError, TypeError): info['AcreageFormatted'] = str(info[acre_key]) 
        else: info['AcreageFormatted'] = "N/A"
            
        # Ensure other expected text fields default to "N/A" string if missing or pd.NA
        for key_to_check in ['ADDRESS', 'BUILDING_VALUE', 'USE_CODE', 'CITY_CODE']:
             if key_to_check not in info or pd.isna(info[key_to_check]):
                 info[key_to_check] = "N/A" # Ensure string "N/A" for these
        
        # Handle ParkDistance specifically (as it might be set to "N/A (disabled)")
        if 'distance_to_park_meters' in info and info['distance_to_park_meters'] == "N/A (disabled)":
            info['ParkDistance'] = "N/A (disabled)"
        elif 'ParkDistance' not in info: # If the key wasn't even created
             info['ParkDistance'] = "N/A"


        # --- Add PCA Factors & Predictive Hint ---
        # Ensure this call is made only AFTER info dict is mostly populated with values needed for PCA (if any were direct)
        # Or ensure get_pca_analysis fetches its own data for the PIN/area.
        pca_results = get_pca_analysis(pin_str) 
        if pca_results and "error" not in pca_results:
            info['pca_top_factors'] = pca_results.get('top_factors_pc1', []) # This is a list
            info['predictive_hint'] = pca_results.get('predictive_hint', "Outlook not available.")
        else:
            info['pca_top_factors'] = ["PCA not available"] # This is a list
            info['predictive_hint'] = pca_results.get("error", "Outlook not available (PCA error).") if pca_results else "Outlook not available."

        # --- CRUCIAL FIX: Convert NaN/pd.NA to None for JSON compatibility ---
        cleaned_info = {}
        for key, value in info.items():
            # Directly pass through lists (like pca_top_factors)
            if isinstance(value, list):
                cleaned_info[key] = value
            elif pd.isna(value): # Handles np.nan, pd.NA, NaT for scalars
                cleaned_info[key] = None
            # No need for the extra float nan check if pd.isna handles it.
            # else if isinstance(value, (float, np.floating)) and np.isnan(value):
            #      cleaned_info[key] = None
            else:
                cleaned_info[key] = value
        
        print(f"DEBUG get_info_for_pin: Returning cleaned_info for PIN {pin_str}")
        return cleaned_info
        
    except Exception as e:
        print(f"ERROR retrieving details for PIN {pin_str} in get_info_for_pin: {e}")
        traceback.print_exc()
        return {"error": f"Error fetching full details for PIN {pin_str}"}


# --- get_parcels_geojson_subset (No changes from last version that removed value filter) ---
def get_parcels_geojson_subset(bounds=None):
    # ... (Paste the full working function from the previous successful response) ...
    print("DEBUG get_parcels_geojson_subset: Called.")
    gdf = load_king_county_data() 
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty or gdf.geometry.isnull().all():
        return '{"type": "FeatureCollection", "features": []}'
    print(f"DEBUG get_parcels_geojson_subset: Received gdf with {len(gdf)} rows before spatial filtering.")
    filtered_gdf = gdf
    if bounds:
        try:
            if not filtered_gdf.geometry.isnull().all(): 
                filtered_gdf = gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
            print(f"DEBUG get_parcels_geojson_subset: Found {len(filtered_gdf)} parcels within bounds.")
        except Exception as e: filtered_gdf = gdf
    if filtered_gdf.empty: return '{"type": "FeatureCollection", "features": []}'
    
    print(f"DEBUG get_parcels_geojson_subset: Converting {len(filtered_gdf)} final parcels to GeoJSON...")
    try:
        geo_col_name = filtered_gdf.geometry.name
        # Include PIN for clicks. DATA_SOURCE_NOTES is removed.
        props_to_include_in_geojson = [PIN_COLUMN_PARCELS] 
        if 'MAJOR' in filtered_gdf.columns: props_to_include_in_geojson.append('MAJOR')
        if 'MINOR' in filtered_gdf.columns: props_to_include_in_geojson.append('MINOR')
        
        final_cols_for_geojson = list(dict.fromkeys(props_to_include_in_geojson)) 
        if geo_col_name not in final_cols_for_geojson : final_cols_for_geojson.append(geo_col_name)
        
        geojson_export_gdf = filtered_gdf[final_cols_for_geojson].copy()
        geojson_data = geojson_export_gdf.to_json()
        print("DEBUG get_parcels_geojson_subset: GeoJSON conversion successful.")
        return geojson_data
    except Exception as e:
        traceback.print_exc(); return '{"type": "FeatureCollection", "features": []}'


def test_gcs_connection():
    """Test Google Cloud Storage connection"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs(max_results=1))
        print("✓ Connection successful!")
        print(f"Found files: {[blob.name for blob in blobs]}")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

# You can test it by adding:
if __name__ == "__main__":
    test_gcs_connection()