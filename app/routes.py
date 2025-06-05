# app/routes.py
from flask import render_template, jsonify, request
from app import app # Import the app instance

# Import necessary functions from data_utils
from .data_utils import get_parcels_geojson_subset, get_info_for_pin, BELLEVUE_BOUNDS

# You could pre-load data when the app starts to avoid delay on first request.
# This can sometimes be tricky with Flask's development server reloader.
# If you enable this, the first "flask run" command will take a while.
# print("Pre-loading King County data on startup...")
# load_king_county_data()
# print("Pre-loading complete.")


@app.route('/')
@app.route('/index')
def home():
    """Renders the homepage with parcel data overlay."""
    map_center_lat = 47.61 # Centered on Bellevue
    map_center_lon = -122.17
    map_zoom = 13 # A good starting zoom for the Bellevue subset

    # Get parcel data for the Bellevue subset as a GeoJSON string
    # This function now calls load_king_county_data internally
    parcels_geojson_data = get_parcels_geojson_subset(bounds=BELLEVUE_BOUNDS)

    if parcels_geojson_data == '{"type": "FeatureCollection", "features": []}':
        print("WARNING in route/home: No parcel data was returned to be displayed on the map.")
        # An optional message could be passed to the template to inform the user
        # e.g., messages=["No parcels found for the current view."]

    return render_template(
        'index.html',
        title='King County Real Estate Dashboard',
        map_center_lat=map_center_lat,
        map_center_lon=map_center_lon,
        map_zoom=map_zoom,
        parcels_geojson_data=parcels_geojson_data # Pass the GeoJSON string to the template
    )


@app.route('/api/property_info/<pin>')
def property_info_api(pin):
    """
    API endpoint to get property details by PIN.
    This is called by JavaScript when a user clicks a parcel or searches.
    """
    print(f"API call received for property_info, PIN: {pin}")
    
    # This function now includes PCA results and the predictive hint from data_utils
    info = get_info_for_pin(pin) 
    
    if "error" in info:
        return jsonify(info), 404 # Return "Not Found" status if PIN lookup fails
    else:
        return jsonify(info) # Return the complete info dictionary as JSON