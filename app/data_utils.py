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

# --- Configuration ---
PARCEL_GEOJSON_PATH = os.path.join("data", "King_County_Parcels___parcel_area.geojson")
ASSESSMENT_FILE_PATH = os.path.join("data", "kc_assessment_data.csv")
# PARKS_SHAPEFILE_PATH = os.path.join("data", "kc_parks.shp") # For future park integration

PIN_COLUMN_PARCELS = "PIN"
PIN_COLUMN_ASSESSMENT = "PIN"
ASSESSMENT_COLUMNS_TO_KEEP = [
    PIN_COLUMN_ASSESSMENT, 'ADDRESS', 'ASSESSED_VALUE', 'BUILDING_VALUE',
    'ACREAGE', 'USE_CODE', 'CITY_CODE'
]
GEOJSON_PROPS_TO_KEEP = [PIN_COLUMN_PARCELS, 'MAJOR', 'MINOR', 'OBJECTID', 'Shape_Area', 'Shape_Length']
BELLEVUE_BOUNDS = (-122.24, 47.56, -122.10, 47.65)

_PRIMARY_DATA_CACHE = None
# _PARKS_UNION_GEOM_CACHE = None # For future

# def load_parks_data(): # Keep for future, ensure it returns None if file not found or error
# global _PARKS_UNION_GEOM_CACHE
# if _PARKS_UNION_GEOM_CACHE is not None: return _PARKS_UNION_GEOM_CACHE
# if not os.path.exists(PARKS_SHAPEFILE_PATH):
# print(f"INFO: Parks Shapefile not found. Park features disabled.")
#         _PARKS_UNION_GEOM_CACHE = False 
# return None
#     # ... (rest of park loading logic) ...
# _PARKS_UNION_GEOM_CACHE = False # Default if other errors
# return None


def load_king_county_data(force_reload=False):
    global _PRIMARY_DATA_CACHE
    print(f"DEBUG load_king_county_data: Called. force_reload={force_reload}. Cache is {'None' if _PRIMARY_DATA_CACHE is None else 'Populated'}")
    if _PRIMARY_DATA_CACHE is not None and not force_reload:
        print(f"DEBUG load_king_county_data: Returning CACHED _PRIMARY_DATA_CACHE with {len(_PRIMARY_DATA_CACHE)} rows.")
        return _PRIMARY_DATA_CACHE
    
    print("DEBUG load_king_county_data: Cache is empty or force_reload=True. Proceeding with full data load.")
    start_time = time.time()

    # Step 1: Load Parcels from GeoJSON
    try:
        parcels_gdf = gpd.read_file(PARCEL_GEOJSON_PATH)
        print(f"Loaded {len(parcels_gdf)} initial parcels from GeoJSON.")
        if parcels_gdf.crs:
            if str(parcels_gdf.crs).upper() != "EPSG:4326" and "CRS84" not in str(parcels_gdf.crs).upper():
                print(f"Warning: Parcel CRS is {parcels_gdf.crs}. Setting to EPSG:4326.")
                parcels_gdf = parcels_gdf.set_crs("EPSG:4326", allow_override=True)
            else:
                print(f"Parcel GeoJSON CRS is {parcels_gdf.crs} (compatible with EPSG:4326).")
        else:
            print("WARNING: Parcel GeoJSON CRS not detected. Assuming EPSG:4326 (WGS84).")
            parcels_gdf.set_crs(epsg=4326, inplace=True, allow_override=True)

        if PIN_COLUMN_PARCELS not in parcels_gdf.columns: raise ValueError(f"PIN column '{PIN_COLUMN_PARCELS}' not found in GeoJSON")
        parcels_gdf[PIN_COLUMN_PARCELS] = parcels_gdf[PIN_COLUMN_PARCELS].astype(str)
        
        geojson_shape_area_col = 'Shape_Area_geojson' 
        if 'Shape_Area' in parcels_gdf.columns:
            parcels_gdf[geojson_shape_area_col] = pd.to_numeric(parcels_gdf['Shape_Area'], errors='coerce')
        elif geojson_shape_area_col in parcels_gdf.columns:
             parcels_gdf[geojson_shape_area_col] = pd.to_numeric(parcels_gdf[geojson_shape_area_col], errors='coerce')
        else: 
            print(f"WARNING: '{geojson_shape_area_col}' or 'Shape_Area' not in GeoJSON. Will try to calculate area from geometry.")
            parcels_gdf[geojson_shape_area_col] = np.nan
            
        parcels_gdf = parcels_gdf[parcels_gdf.geometry.is_valid & ~parcels_gdf.geometry.isna()]
        print(f"Parcels after geometry cleaning: {len(parcels_gdf)}")
    except Exception as e: print(f"ERROR loading GeoJSON: {e}"); traceback.print_exc(); return None
    if parcels_gdf.empty: print("ERROR: No valid parcels from GeoJSON."); _PRIMARY_DATA_CACHE = parcels_gdf; return _PRIMARY_DATA_CACHE

    # Step 2: Load Assessment Data
    try:
        assessment_df = pd.read_csv(ASSESSMENT_FILE_PATH, low_memory=False)
        if PIN_COLUMN_ASSESSMENT not in assessment_df.columns:
            print(f"WARNING: PIN column '{PIN_COLUMN_ASSESSMENT}' not in Assessment CSV. Will generate all assessment fields.")
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
        on=PIN_COLUMN_PARCELS, how='left', suffixes=('_geojson', '') # Suffix original GeoJSON columns if overlap, prefer assessment names
    )
    print(f"Merged data shape: {merged_gdf.shape}")
    
    # Clean up suffixed PIN from assessment if it occurred
    assessment_pin_suffixed = PIN_COLUMN_ASSESSMENT + '' # Empty suffix means it's the main one
    if assessment_pin_suffixed in merged_gdf.columns and assessment_pin_suffixed != PIN_COLUMN_PARCELS:
        merged_gdf.drop(columns=[assessment_pin_suffixed], inplace=True, errors='ignore')

    # --- Generate/Calculate Missing Data & Convert Types ---
    print("Generating/Calculating placeholder data for missing assessment fields...")
    
    for col in ['ADDRESS', 'ASSESSED_VALUE', 'BUILDING_VALUE', 'ACREAGE', 'USE_CODE']:
        if col not in merged_gdf.columns: merged_gdf[col] = np.nan

    addr_col = 'ADDRESS'
    missing_addr_mask = merged_gdf[addr_col].isna() | (merged_gdf[addr_col] == "N/A")
    objid_col_for_addr = next((c for c in ['OBJECTID_geojson', 'OBJECTID'] if c in merged_gdf.columns), PIN_COLUMN_PARCELS) # Fallback to PIN for address
    merged_gdf.loc[missing_addr_mask, addr_col] = "GenAddr-" + merged_gdf.loc[missing_addr_mask, objid_col_for_addr].astype(str)

    for col in ['ASSESSED_VALUE', 'BUILDING_VALUE']:
        merged_gdf[col] = pd.to_numeric(merged_gdf[col], errors='coerce')
    
    missing_assessed_mask = merged_gdf['ASSESSED_VALUE'].isna()
    if missing_assessed_mask.any():
        merged_gdf.loc[missing_assessed_mask, 'ASSESSED_VALUE'] = np.random.randint(200000, 3000001, size=missing_assessed_mask.sum())

    missing_bldg_mask = merged_gdf['BUILDING_VALUE'].isna()
    target_for_bldg_gen = missing_bldg_mask & merged_gdf['ASSESSED_VALUE'].notna()
    if target_for_bldg_gen.any():
        random_ratios = np.random.uniform(0.5, 0.9, size=target_for_bldg_gen.sum())
        merged_gdf.loc[target_for_bldg_gen, 'BUILDING_VALUE'] = (merged_gdf.loc[target_for_bldg_gen, 'ASSESSED_VALUE'] * random_ratios).astype(int)

    use_code_col = 'USE_CODE'
    common_use_codes = ["SFR-Gen", "MultiFam-Gen", "Comm-Gen", "Indust-Gen", "Vacant-Gen"]
    missing_use_mask = merged_gdf[use_code_col].isna() | (merged_gdf[use_code_col] == "N/A")
    if missing_use_mask.any():
        merged_gdf.loc[missing_use_mask, use_code_col] = np.random.choice(common_use_codes, size=missing_use_mask.sum())

    acre_col = 'ACREAGE'
    merged_gdf[acre_col] = pd.to_numeric(merged_gdf[acre_col], errors='coerce')
    missing_acreage_mask = merged_gdf[acre_col].isna()
    
    if geojson_shape_area_col in merged_gdf.columns:
        rows_calc_geojson_area = missing_acreage_mask & merged_gdf[geojson_shape_area_col].notna() & (merged_gdf[geojson_shape_area_col] > 0)
        if rows_calc_geojson_area.any():
            avg_lat_rad = np.deg2rad((BELLEVUE_BOUNDS[1] + BELLEVUE_BOUNDS[3]) / 2)
            sq_deg_to_acres_factor = (111133.0 * 111320.0 * np.cos(avg_lat_rad)) * 0.000247105
            calculated_acres = merged_gdf.loc[rows_calc_geojson_area, geojson_shape_area_col] * sq_deg_to_acres_factor
            merged_gdf.loc[rows_calc_geojson_area, acre_col] = calculated_acres
    
    missing_acreage_mask = merged_gdf[acre_col].isna()
    rows_calc_geom_area = missing_acreage_mask & merged_gdf.geometry.is_valid & ~merged_gdf.geometry.is_empty
    if rows_calc_geom_area.any():
        try:
            area_sq_ft = merged_gdf.loc[rows_calc_geom_area].geometry.to_crs(epsg=2926).area
            merged_gdf.loc[rows_calc_geom_area, acre_col] = area_sq_ft / 43560.0
        except Exception as e_area: print(f"ERROR calculating area from geometry: {e_area}")
    merged_gdf[acre_col] = pd.to_numeric(merged_gdf[acre_col], errors='coerce') # Final numeric conversion

    merged_gdf['distance_to_park_meters'] = np.nan # Park distance not active

    for col in merged_gdf.select_dtypes(include='object').columns:
        merged_gdf[col].fillna("N/A", inplace=True)

    _PRIMARY_DATA_CACHE = merged_gdf
    end_time = time.time()
    print(f"Data loading, generation, and processing complete in {time.time() - start_time:.2f} seconds. Final shape: {merged_gdf.shape}")
    return _PRIMARY_DATA_CACHE

# --- PCA Function ---
def get_pca_analysis(pin_to_analyze):
    print(f"DEBUG: Performing PCA analysis (context of PIN {pin_to_analyze})")
    merged_gdf = load_king_county_data()
    if merged_gdf is None or merged_gdf.empty: return {"error": "Data not loaded for PCA."}

    # Use Bellevue subset for PCA context for now
    pca_subset_gdf = merged_gdf.cx[BELLEVUE_BOUNDS[0]:BELLEVUE_BOUNDS[2], BELLEVUE_BOUNDS[1]:BELLEVUE_BOUNDS[3]]
    if pca_subset_gdf.empty: return {"error": "No data in Bellevue subset for PCA."}

    features_for_pca = ['ACREAGE', 'ASSESSED_VALUE', 'BUILDING_VALUE'] # Add 'distance_to_park_meters' when parks re-enabled
    
    numerical_data_for_pca = pd.DataFrame()
    actual_features_used = []
    for col in features_for_pca:
        if col in pca_subset_gdf.columns:
            # Ensure data is numeric before PCA
            series_numeric = pd.to_numeric(pca_subset_gdf[col], errors='coerce')
            if not series_numeric.isna().all(): # Only add if some values are numeric
                 numerical_data_for_pca[col] = series_numeric
                 actual_features_used.append(col)
            else: print(f"PCA: Feature '{col}' is all NaN after to_numeric, skipping.")
        else: print(f"PCA: Feature '{col}' not found in subset, skipping.")
    
    if numerical_data_for_pca.empty or len(actual_features_used) < 2:
        return {"error": "Not enough valid numerical features for PCA after attempting conversion."}

    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(numerical_data_for_pca)
    
    if data_imputed.shape[0] < 2 : # Check after imputation
         return {"error": "Not enough samples for PCA after imputation."}

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    if data_scaled.shape[0] < max(2, data_scaled.shape[1]):
        return {"error": "Not enough data points vs features after preprocessing for PCA."}

    n_components = min(len(actual_features_used), 2)
    if n_components < 1: return {"error": "Too few features for PCA components."} # Should be caught by len(actual_features_used) < 2
    
    pca = PCA(n_components=n_components)
    try:
        pca.fit(data_scaled)
    except ValueError as ve: return {"error": f"PCA fitting error: {ve}"}

    pc1_loadings_signed = pca.components_[0]
    feature_contributions = sorted(zip(actual_features_used, pc1_loadings_signed), key=lambda x: abs(x[1]), reverse=True)
    top_factors_pc1 = [f"{feat} ({loading_signed:.2f})" for feat, loading_signed in feature_contributions[:3]]

    predictive_hint = "Market dynamics are influenced by a combination of factors." # Default
    if feature_contributions:
        top_factor_name = feature_contributions[0][0]
        # ... (predictive hint logic based on top_factor_name, as before) ...
        if top_factor_name == 'ASSESSED_VALUE': predictive_hint = "Higher assessed values are key. Market may favor premium if demand holds."
        elif top_factor_name == 'ACREAGE': predictive_hint = "Larger property sizes are significant in this area's valuation."
        elif top_factor_name == 'BUILDING_VALUE': predictive_hint = "Building values contribute strongly. Structure quality is likely key."

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "top_factors_pc1": top_factors_pc1,
        "predictive_hint": predictive_hint,
    }

# --- get_info_for_pin (includes PCA and Predictive Hint) ---
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
        info_cols_to_drop = [geo_col, 'Shape_Area_geojson'] # Drop helper geojson shape area
        # Drop any _geojson or _assess suffixed columns if base name exists
        for col_name in property_series.index:
            if (col_name.endswith("_geojson") or col_name.endswith("_assess")):
                base_name = col_name.replace("_geojson","").replace("_assess","")
                if base_name in property_series.index and base_name != col_name:
                     info_cols_to_drop.append(col_name)
        info = property_series.drop(labels=list(set(info_cols_to_drop)), errors='ignore').to_dict()

        info['latitude'] = None; info['longitude'] = None
        if geo_col in property_series and property_series[geo_col] is not None and hasattr(property_series[geo_col], 'is_valid') and property_series[geo_col].is_valid :
            centroid = property_series[geo_col].centroid
            info['latitude'] = centroid.y; info['longitude'] = centroid.x
        
        if 'ASSESSED_VALUE' in info and pd.notna(info['ASSESSED_VALUE']):
            try: info['AssessedValueFormatted'] = f"${float(info['ASSESSED_VALUE']):,.0f}"
            except: info['AssessedValueFormatted'] = str(info['ASSESSED_VALUE']) 
        else: info['AssessedValueFormatted'] = "N/A"

        if 'ACREAGE' in info and pd.notna(info['ACREAGE']):
            try: info['AcreageFormatted'] = f"{float(info['ACREAGE']):.2f} acres"
            except: info['AcreageFormatted'] = str(info['ACREAGE']) 
        else: info['AcreageFormatted'] = "N/A"
            
        for key_to_check in ['ADDRESS', 'BUILDING_VALUE', 'USE_CODE', 'CITY_CODE', 'ParkDistance']:
             if key_to_check not in info or pd.isna(info[key_to_check]): info[key_to_check] = "N/A"
        
        if 'distance_to_park_meters' in info and info['distance_to_park_meters'] == "N/A (disabled)":
            info['ParkDistance'] = "N/A (disabled)"
        
        pca_results = get_pca_analysis(pin_str) 
        if pca_results and "error" not in pca_results:
            info['pca_top_factors'] = pca_results.get('top_factors_pc1', [])
            info['predictive_hint'] = pca_results.get('predictive_hint', "Outlook not available.")
        else:
            info['pca_top_factors'] = ["PCA not available"]
            info['predictive_hint'] = pca_results.get("error", "Outlook not available (PCA error).") if pca_results else "Outlook not available."

        cleaned_info = {}
        for key, value in info.items():
            if isinstance(value, list): cleaned_info[key] = value
            elif pd.isna(value): cleaned_info[key] = None
            else: cleaned_info[key] = value
        
        return cleaned_info
    except Exception as e:
        print(f"ERROR retrieving details for PIN {pin_str} in get_info_for_pin: {e}")
        traceback.print_exc(); return {"error": f"Error fetching full details for PIN {pin_str}"}

# --- get_parcels_geojson_subset ---
def get_parcels_geojson_subset(bounds=None):
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
        props_to_include_in_geojson = [PIN_COLUMN_PARCELS] 
        if 'MAJOR' in filtered_gdf.columns: props_to_include_in_geojson.append('MAJOR')
        if 'MINOR' in filtered_gdf.columns: props_to_include_in_geojson.append('MINOR')
        # DATA_SOURCE_NOTES was removed, so don't include it.
        
        final_cols_for_geojson = list(dict.fromkeys(props_to_include_in_geojson)) 
        if geo_col_name not in final_cols_for_geojson : final_cols_for_geojson.append(geo_col_name)
        
        geojson_export_gdf = filtered_gdf[final_cols_for_geojson].copy()
        geojson_data = geojson_export_gdf.to_json()
        print("DEBUG get_parcels_geojson_subset: GeoJSON conversion successful.")
        return geojson_data
    except Exception as e:
        traceback.print_exc(); return '{"type": "FeatureCollection", "features": []}'