# app/routes.py
from flask import render_template, jsonify # No need for request here unless keeping form
from app import app # Import the app instance
# Import necessary functions
from .data_utils import get_parcels_geojson_subset, get_info_for_pin, BELLEVUE_BOUNDS, load_king_county_data

# Optional: Pre-load data when the app starts to avoid delay on first request
# print("Pre-loading King County data...")
# load_king_county_data()
# print("Pre-loading complete.")


@app.route('/')
@app.route('/index')
def home():
    """Renders the homepage with parcel data overlay."""
    map_center_lat = 47.61 # Centered more on Bellevue
    map_center_lon = -122.17
    map_zoom = 13 # Zoom in closer

    # Get parcel data for the Bellevue subset as GeoJSON string
    parcels_geojson_data = get_parcels_geojson_subset(bounds=BELLEVUE_BOUNDS)

    if parcels_geojson_data == '{"type": "FeatureCollection", "features": []}':
        print("WARNING: No parcel data loaded or found in bounds. Map parcel layer may be empty.")
        # Optionally pass a flag/message to the template

    return render_template(
        'index.html',
        title='King County Real Estate Dashboard (Bellevue View)',
        map_center_lat=map_center_lat,
        map_center_lon=map_center_lon,
        map_zoom=map_zoom,
        parcels_geojson_data=parcels_geojson_data # Pass GeoJSON string
    )

# --- API Endpoint (remains the same, uses updated data_utils) ---
@app.route('/api/property_info/<pin>')
def property_info_api(pin):
    """API endpoint to get property details by PIN."""
    print(f"API call received for PIN: {pin}")
    info = get_info_for_pin(pin)
    if "error" in info:
        return jsonify(info), 404
    else:
        return jsonify(info)

# Remove the '/get_info' route if you remove the form from index.html