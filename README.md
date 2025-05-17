# Real Estate Dashboard Project (King County Focus)

This project aims to create a dashboard for visualizing local real estate market factors, their impact on home values, and potential future outlooks, initially focusing on King County, WA.

*(Current as of: 2025-05-01)*

## Project Objectives

* Visualize market factors affecting home values using map overlays and parcel data.
* Integrate public data sources (King County GIS, Assessment data).
* Provide tools for basic property lookup and filtering.
* Lay the groundwork for future analysis (PCA, AI outlooks).
* Target users: Developers, investors, brokers, consumers.

## Tech Stack

* Python
* Flask (Web Framework & Backend API)
* Pandas / GeoPandas (Data Manipulation & Geospatial Handling)
* HTML / CSS / JavaScript (Frontend)
* Leaflet.js (Interactive Mapping Library)
* Pytest (Unit Testing)

## Current Features

* Displays an interactive Leaflet map centered on a filtered area (currently Bellevue, WA).
* Loads King County parcel boundaries (from local Shapefile) and assessment data (from local CSV).
* Overlays parcel boundaries for the filtered area onto the map.
* Allows users to click on a displayed parcel to view its details (PIN, Address, Assessed Value, Acreage, etc.) in a popup via an API call.
* Provides a search box to find a property by its 10-digit PIN; zooms/flies the map to the found parcel's location and displays its info in a marker popup.
* Includes controls to filter displayed parcels based on a minimum and maximum Assessed Value range (reloads the page with filtered data).
* Basic unit test implemented for the homepage route.

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
    # On Windows:
    # .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: GeoPandas installation can sometimes be complex due to C library dependencies like GEOS, PROJ. Refer to GeoPandas documentation if installation fails.)*

4.  **Obtain Data:**
    * **Required:** Place the following files **directly inside the `data/` folder** in the project root:
        * King County Parcel Shapefile set (must include at least `parcel.shp`, `parcel.shx`, `parcel.dbf`, `parcel.prj`). **Ensure the `.shp` file is named `parcel.shp`** or update `PARCEL_SHAPEFILE_PATH` in `app/data_utils.py`. Finding a Shapefile with valid geometry and CRS is crucial.
        * King County Assessment Data CSV file. **Ensure it is named `kc_assessment_data.csv`** or update `ASSESSMENT_FILE_PATH` in `app/data_utils.py`. Ensure it contains a `PIN` column matching the Shapefile.
    * *(Data Source Hint: Look for Parcel and Assessor data on King County's GIS open data portals.)*

5.  **Set up environment variables (Optional but Recommended for Dev):**
    ```bash
    # On macOS/Linux:
    export FLASK_APP=app
    export FLASK_ENV=development
    # On Windows:
    # set FLASK_APP=app
    # set FLASK_ENV=development
    ```

6.  **Run the application:**
    * Make sure you are in the project root directory (`your-project-name/`).
    * Make sure your virtual environment is active.
    ```bash
    flask run
    ```

7.  Open your web browser and navigate to `http://127.0.0.1:5000` (or the address provided).

## Testing

This project uses `pytest`.

1.  Ensure you have activated the virtual environment (`source venv/bin/activate`).
2.  Make sure development dependencies are installed (`pip install -r requirements.txt` - ensure `pytest` is listed).
3.  From the **project root directory**, run tests using:
    ```bash
    python -m pytest tests/
    ```

## Project Progress & Sprint Status

* **Sprint 1: Map Integration:** Completed. (Basic app structure, Leaflet map displayed).
* **Sprint 2: King County Integration:** Completed. (Parcel Shapefile loaded & displayed, assessment data loaded & merged, click-to-view info implemented).
* **Additional Features Implemented:**
    * Search by PIN with map zoom/highlight.
    * Filter displayed parcels by Assessed Value range.
* **Sprint 3: Property Augmentation (PCA Prep):** Starting/Next Steps. Focus will be on integrating additional relevant datasets (e.g., schools, parks) via spatial analysis and preparing data for PCA.
* **Sprint 4: Research Assistance (AI Outlooks):** Not Started.
* **Sprint 5: Clean-up:** Not Started.