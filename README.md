# Real Estate Dashboard Project (King County Focus)

This project aims to create a dashboard for visualizing local real estate market factors, their impact on home values, and potential future outlooks, initially focusing on King County, WA.

*(Current as of: 2025-05-17)*

## Project Objectives

* Visualize market factors affecting home values using map overlays and parcel data.
* Integrate public data sources (King County GIS, Assessment data).
* Provide tools for basic property lookup and data exploration.
* Lay the groundwork for future analysis like PCA and AI-driven insights.
* Target users: Developers, investors, brokers, consumers.

## Tech Stack

* Python
* Flask (Web Framework & Backend API)
* Pandas / GeoPandas (Data Manipulation & Geospatial Handling)
* HTML / CSS / JavaScript (Frontend)
* Leaflet.js (Interactive Mapping Library)
* Pytest (Unit Testing)

## Current Features

* **Interactive Map Display:** Shows a Leaflet map centered on a subset of King County (currently Bellevue, WA).
* **Parcel Data Integration:**
    * Loads King County parcel boundaries from a local GeoJSON file (`King_County_Parcels___parcel_area.geojson`).
    * Loads King County assessment data from a local CSV file (`kc_assessment_data.csv`).
    * Merges these datasets to link parcel geometries with their assessment attributes.
* **Parcel Visualization:** Overlays parcel boundaries for the filtered area (Bellevue) onto the map.
* **Click Interaction:** Allows users to click on a displayed parcel to view its details (PIN, Address, Assessed Value, Acreage, etc.) in a popup via an API call.
* **Search by PIN:** Provides a search box to find a property by its 10-digit PIN; zooms/flies the map to the found parcel's location and displays its info in a marker popup.
* **Cleaned UI:** Features an improved layout and styling for better user experience.
* **Unit Testing:** Basic unit test implemented for the homepage route.
* **(Temporarily Disabled for Performance Tuning):**
    * Park distance calculation (augmenting parcels with distance to nearest park).
    * PCA factor display in popups.

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
    *(Note: GeoPandas installation can sometimes be complex due to C library dependencies like GEOS, PROJ. Refer to GeoPandas documentation if installation fails.)*

4.  **Obtain Data (Crucial - Not included in Git Repository):**
    * **Required:** Create a `data/` folder in the project root. Place the following files directly inside this `data/` folder:
        * **King County Parcel GeoJSON file:** This file should contain parcel geometries and at least a `PIN` property.
            * **Ensure the file is named `King_County_Parcels___parcel_area.geojson`** or update `PARCEL_GEOJSON_PATH` in `app/data_utils.py`.
            * The data should ideally have its CRS as WGS84 (EPSG:4326), like `urn:ogc:def:crs:OGC:1.3:CRS84`.
        * **King County Assessment Data CSV file:** This file should contain assessment details like `ADDRESS`, `ASSESSED_VALUE`, etc., and a `PIN` column for merging.
            * **Ensure it is named `kc_assessment_data.csv`** or update `ASSESSMENT_FILE_PATH` in `app/data_utils.py`.
    * **(Optional for Park Distance Feature - Currently Disabled):**
        * King County Parks Shapefile set (e.g., `kc_parks.shp` and its companions `.shx`, `.dbf`, `.prj`). If you enable this feature, update `PARKS_SHAPEFILE_PATH` in `app/data_utils.py` if your filename differs.
    * *(Data Source Hint: Look for "Parcels" (GeoJSON or Shapefile) and "Assessor" data on King County's GIS open data portals like `https://gis-kingcounty.opendata.arcgis.com/`.)*

5.  **Set up environment variables (Optional but Recommended for Dev):**
    ```bash
    # On macOS/Linux:
    export FLASK_APP=app
    export FLASK_ENV=development
    # On Windows (Command Prompt):
    # set FLASK_APP=app
    # set FLASK_ENV=development
    # On Windows (PowerShell):
    # $env:FLASK_APP = "app"
    # $env:FLASK_ENV = "development"
    ```
    *(Note: Setting `FLASK_ENV=development` enables debug mode and auto-reloading.)*

6.  **Run the application:**
    * Make sure you are in the project root directory (`your-project-name/`).
    * Make sure your virtual environment is active.
    * The first time you run it, data processing might take some time (e.g., 20-60 seconds depending on your machine and the GeoJSON size). Subsequent loads should be faster due to caching in memory.
    ```bash
    flask run
    ```

7.  Open your web browser and navigate to `http://127.0.0.1:5000` (or the address provided by Flask).

## Project Structure