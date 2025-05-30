# app/routes.py
from flask import render_template, jsonify, request # Added request for potential future use
from app import app # Import the app instance

# Import necessary functions from data_utils
from .data_utils import get_parcels_geojson_subset, get_info_for_pin, BELLEVUE_BOUNDS
# load_king_county_data is called internally by the functions above, so not strictly needed here unless pre-loading

# --- Vertex AI Imports and Initialization ---
import os
import traceback # For detailed error logging
import vertexai
from vertexai.generative_models import GenerativeModel #, Part # Part might not be needed for simple text prompts

# Initialize Vertex AI (Ideally done once when app starts)
# Ensure GOOGLE_APPLICATION_CREDENTIALS env var is set,
# or you've run `gcloud auth application-default login`
# and GOOGLE_CLOUD_PROJECT is set.
try:
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
    LOCATION = "us-west1" # Or your preferred Vertex AI region that supports Gemini

    if PROJECT_ID:
        print(f"Attempting to initialize Vertex AI with Project ID: {PROJECT_ID}, Location: {LOCATION}")
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        # Choose a suitable Gemini model for text generation
        # "gemini-1.0-pro" or "gemini-1.5-flash" are good options.
        # "gemini-pro" is an alias that often points to a stable version of gemini-1.0-pro.
        # Using "gemini-1.5-flash-001" as it's recent and efficient
        ai_model_name = "gemini-1.0-pro"
        model = GenerativeModel(ai_model_name)
        print(f"Vertex AI initialized successfully. Using model: {ai_model_name}")
    else:
        model = None
        print("WARNING: GOOGLE_CLOUD_PROJECT environment variable not set. AI Outlook feature will be disabled.")
except ImportError:
    model = None
    print("ERROR: google-cloud-aiplatform library not found. pip install google-cloud-aiplatform. AI Outlook disabled.")
except Exception as e_vertex:
    model = None
    print(f"ERROR: Failed to initialize Vertex AI: {e_vertex}. AI Outlook feature will be disabled.")
    traceback.print_exc()
# --- End Vertex AI Initialization ---


# Optional: Pre-load data when the app starts to avoid delay on first request
# print("Pre-loading King County data...")
# load_king_county_data() # Call this if you want to pre-cache
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
        print("WARNING in route/home: No parcel data loaded or found in bounds. Map parcel layer may be empty.")

    return render_template(
        'index.html',
        title='King County Real Estate Dashboard', # Updated title slightly
        map_center_lat=map_center_lat,
        map_center_lon=map_center_lon,
        map_zoom=map_zoom,
        parcels_geojson_data=parcels_geojson_data
    )

# API endpoint to get property details by PIN
@app.route('/api/property_info/<pin>')
def property_info_api(pin):
    """API endpoint to get property details by PIN."""
    print(f"API call received for property_info, PIN: {pin}")
    info = get_info_for_pin(pin) # This function now includes PCA results
    if "error" in info:
        return jsonify(info), 404 # Not Found
    else:
        return jsonify(info)

# --- NEW API Endpoint for AI Outlook ---
@app.route('/api/outlook/<factor_name>')
def get_ai_outlook(factor_name):
    """API endpoint to get an AI-generated outlook for a given real estate factor."""
    print(f"AI Outlook API call received for factor: {factor_name}")

    if not model:
        print("ERROR in /api/outlook: AI model not initialized.")
        return jsonify({"error": "AI model not initialized. Check server logs and GCP setup."}), 503 # Service Unavailable

    if not factor_name or len(factor_name.strip()) == 0:
        return jsonify({"error": "Factor name cannot be empty."}), 400 # Bad Request

    # Clean up the factor name for a better prompt
    # Example: "ASSESSED_VALUE (0.71)" becomes "Assessed Value"
    clean_factor_name = factor_name.split(' (')[0].replace("_", " ").title()

    prompt = f"""
    Provide a concise, neutral-toned, near-future outlook (next 1-2 years)
    for the impact of '{clean_factor_name}' on residential property values
    in the King County, Washington area. Limit response to 2-3 sentences.
    Focus on general trends. Do not use markdown formatting.
    """
    print(f"Generated AI Prompt for factor '{clean_factor_name}':\n{prompt}")

    try:
        # For Gemini, generate_content can take a list of Parts or just a string for simple prompts
        response = model.generate_content(prompt)
        
        # Accessing the text from the response
        # The structure can vary slightly based on model and response type.
        # For simple text generation, response.text should work.
        # If it's more complex, you might need to inspect response.candidates[0].content.parts[0].text
        outlook_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            outlook_text = response.candidates[0].content.parts[0].text
        elif hasattr(response, 'text'): # Fallback for simpler response structures
            outlook_text = response.text
        else:
            print("ERROR: AI response structure not as expected. Full response:", response)
            return jsonify({"error": "AI response format unexpected."}), 500


        print(f"AI Outlook for '{clean_factor_name}': {outlook_text}")
        return jsonify({"outlook": outlook_text.strip()})
    except Exception as e_ai:
        print(f"ERROR calling AI API for factor '{clean_factor_name}': {e_ai}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to get AI outlook: {str(e_ai)}"}), 500 # Internal Server Error