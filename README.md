# Real Estate Dashboard Project (King County Focus)

This project aims to create a dashboard for visualizing local real estate market factors, their impact on home values, and potential future outlooks, initially focusing on King County, WA.

*(Current as of: 2025-05-30)*

## Project Objectives

* Visualize market factors affecting home values using map overlays and parcel data.
* Integrate public data sources (King County GIS, Assessment data).
* Provide tools for basic property lookup and data exploration with analytical insights.
* Demonstrate capabilities for Principal Component Analysis (PCA) and AI-driven outlooks.
* Target users: Developers, investors, brokers, consumers.

## Tech Stack

* Python
* Flask (Web Framework & Backend API)
* Pandas / GeoPandas (Data Manipulation & Geospatial Handling)
* Scikit-learn (for PCA)
* Google Cloud Vertex AI SDK (`google-cloud-aiplatform` for AI Outlooks)
* HTML / CSS / JavaScript (Frontend)
* Leaflet.js (Interactive Mapping Library)
* Pytest (Unit Testing)

## Current Features

* **Interactive Map Display:** Shows a Leaflet map centered on a subset of King County (currently Bellevue, WA).
* **Parcel Data Integration:**
    * Loads King County parcel boundaries from a local GeoJSON file (`King_County_Parcels___parcel_area.geojson`).
    * Loads King County assessment data from a local CSV file (`kc_assessment_data.csv`).
    * Merges these datasets, linking parcel geometries with assessment attributes. Acreage is calculated from GeoJSON `Shape_Area` if missing from assessment.
* **Parcel Visualization:** Overlays parcel boundaries for the filtered area (Bellevue) onto the map with refined styling.
* **Property Details Panel:** Clicking a parcel or searching by PIN displays comprehensive details in a side panel.
* **Search by PIN:** Allows users to find a property by its 10-digit PIN; zooms/flies the map to the found parcel and displays its info.
* **PCA-derived Market Insights:** For a selected property (within the context of the Bellevue data subset), displays key factors (e.g., Assessed Value, Acreage) influencing property characteristics, derived from Principal Component Analysis.
* **Predictive Hint:** Provides a simple, qualitative outlook hint based on the primary PCA factor.
* **AI Outlook Integration (In Progress):** Includes a button to fetch a near-future outlook for key market factors using Google Vertex AI (Gemini model). *Full functionality depends on user's Vertex AI setup and model access in their GCP project (currently debugging model/region availability).*
* **Cleaned UI:** Features an improved layout with a side panel for controls and information, and enhanced styling.
* **Unit Testing:** Basic unit test implemented for the homepage route.
* **(Future/Optional):** Park distance calculation (augmenting parcels with distance to nearest park) can be re-enabled.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd your-project-name
    ```

2.  **Create and activate Python virtual environment:**
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows (Command Prompt):
    # .\venv\Scripts\activate
    # On Windows (PowerShell):
    # .\venv\Scripts\Activate.ps1
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: GeoPandas installation can sometimes be complex. Refer to GeoPandas documentation. Ensure `scikit-learn` and `google-cloud-aiplatform` are also installed via requirements.txt).*

4.  **Obtain Data (Crucial - Not included in Git Repository):**
    * **Required:** Create a `data/` folder in the project root. Place the following files directly inside this `data/` folder:
        * **King County Parcel GeoJSON file:** Must contain parcel geometries and a `PIN` property.
            * **Ensure the file is named `King_County_Parcels___parcel_area.geojson`** or update `PARCEL_GEOJSON_PATH` in `app/data_utils.py`.
            * Expected CRS is WGS84 (EPSG:4326), like `urn:ogc:def:crs:OGC:1.3:CRS84`.
        * **King County Assessment Data CSV file:** Should contain assessment details and a `PIN` column.
            * **Ensure it is named `kc_assessment_data.csv`** or update `ASSESSMENT_FILE_PATH` in `app/data_utils.py`.
    * **(Optional for Park Distance Feature):**
        * King County Parks Shapefile set (e.g., `kc_parks.shp` and its companions). If re-enabling this feature, update `PARKS_SHAPEFILE_PATH` in `app/data_utils.py`.
    * *(Data Source Hint: King County GIS open data portals).*

5.  **Set up Environment Variables:**
    * **For Flask (Recommended for Dev):**
        ```bash
        # On macOS/Linux:
        export FLASK_APP=app
        export FLASK_ENV=development
        # On Windows:
        # set FLASK_APP=app
        # set FLASK_ENV=development
        ```
    * **For Google Vertex AI (Required for AI Outlook Feature):**
        1.  Complete Google Cloud Project setup, enable Vertex AI API, and run `gcloud auth application-default login`.
        2.  Set the `GOOGLE_CLOUD_PROJECT` environment variable to your GCP Project ID in your terminal *before* running Flask:
            ```bash
            # On macOS/Linux:
            export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
            # On Windows:
            # set GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
            ```

6.  **Run the application:**
    * Ensure you are in the project root and your virtual environment is active.
    * The first data load may take 20-60 seconds. Subsequent loads (within the same Flask session) use cached data.
    ```bash
    flask run
    ```

7.  Open your web browser to `http://127.0.0.1:5000`.


## Testing

This project uses `pytest`.

1.  Activate the virtual environment.
2.  Ensure development dependencies are installed.
3.  From the **project root directory**, run:
    ```bash
    python -m pytest tests/
    ```

## Project Progress & Sprint Status

* **Sprint 1: Map Integration:** Completed.
* **Sprint 2: King County Integration:** Completed (using GeoJSON for parcels, merged with CSV assessment data, property details display).
* **Sprint 3: Property Augmentation & PCA:**
    * PCA implemented using core attributes (Acreage, Assessed/Building Value) for the Bellevue subset. Top influencing factors displayed.
    * Park data augmentation and its inclusion in PCA is a future enhancement.
* **Sprint 4: Research Assistance (AI Outlooks):** In Progress.
    * Backend API route (`/api/outlook/<factor>`) using Vertex AI (Gemini) is implemented.
    * Frontend button triggers API call and displays results.
    * *Current Status: User needs to finalize Vertex AI model access/region configuration in their GCP project to resolve "model not found" errors.*
* **Sprint 5: Clean-up (EDC: 2025-05-30):**
    * Significant UI cleanup and layout improvements completed.
    * Dummy data generation implemented for a more complete UI demonstration.
    * Predictive hint derived from PCA results is displayed.
    * Final debugging and testing in progress.