import json
import re
import os
import sys

def main():
    src_file = 'flood_risk_pipeline.py' if os.path.exists('flood_risk_pipeline.py') else 'recovered_notebook.py'
    with open(src_file, 'r', encoding='utf-8') as f:
        code = f.read()

    # 1. Update HydroRIVERS URL
    code = code.replace(
        'https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_af.zip',
        'https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_af_shp.zip'
    )

    # 2. Update ESA WorldCover URL (regional bucket deprecated -> global S3 endpoint)
    code = code.replace(
        'https://esa-worldcover-s3.s3.eu-central-1.amazonaws.com',
        'https://esa-worldcover.s3.amazonaws.com'
    )

    # 3. Update UNOSAT URL
    code = re.sub(
        r'UNOSAT_URL\s*=\s*\([^)]+\)',
        'UNOSAT_URL = "https://unosat.org/static/unosat_filesystem/3427/FL20221019NGA_SHP.zip"',
        code
    )

    # 4. Save cleaned flood_risk_pipeline.py
    with open('flood_risk_pipeline.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print("flood_risk_pipeline.py successfully written!")

    # 5. Convert to Jupyter Notebook (.ipynb)
    # Split by stage markers
    stage_pattern = re.compile(r'(# ─+\s*\n# STAGE \d+ [^\n]+\n# ─+)', re.MULTILINE)
    parts = stage_pattern.split(code)

    cells = []
    # Header / intro cell
    header_text = """# AI-Powered Flood Risk Prediction & Geospatial Intelligence System
### Study Area: Lokoja Confluence & Lower Niger Basin, Kogi State, Nigeria
### Target Reference Event: October 2022 Nigeria Flood (Peak Inundation Composite, 1-25 Oct 2022)
### Flood Sensor Ground Truth: VIIRS (Suomi NPP & NOAA-20), UNOSAT Activation FL20221019NGA (Product 3427)
### Capstone Project — TechCrush C8

---
This notebook implements an end-to-end, reproducible, scientifically defensible flood susceptibility modeling pipeline:
1. **Authoritative Geospatial Data Ingestion**: SRTM 30m DEM, CHIRPS rainfall, HydroRIVERS hydrography, ESA WorldCover, UNOSAT satellite flood extent (VIIRS sensor, FL20221019NGA)
2. **Spatial Preprocessing**: Alignment to a common 90m UTM Zone 32N metric grid (EPSG:32632)
3. **Feature Engineering**: Elevation, slope, rainfall accumulation, Euclidean distance to river, soil permeability index
4. **Data Leakage Prevention**: 2x2 Spatial block split (quadrant-based evaluation with domain-shift audit)
5. **Baseline Benchmark & Modelling**: Naive Majority Baseline, Balanced Logistic Regression, and Balanced Random Forest
6. **Model Evaluation & Geospatial Intelligence**: ROC-AUC, PR-AUC (Precision-Recall curves), Confusion Matrices, Continuous Flood Susceptibility Index (FSI), 5-Tier Susceptibility Map, Interactive Folium Map
7. **Decision Log & Limitations**: Defensible, transparent reporting of all assumptions, domain shift findings, and trade-offs.
"""
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in header_text.split("\n")]
    })

    # First code cell before stage 1 (pip install & title docstring)
    first_chunk = parts[0].strip()
    if first_chunk:
        # Prepend %pip install for Colab
        colab_install_cell = """# Install required geospatial & ML libraries (Google Colab / Jupyter)
!pip install -q rasterio geopandas pyproj pyogrio folium scikit-learn matplotlib seaborn requests shapely scipy
"""
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [colab_install_cell]
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in first_chunk.split("\n")]
        })

    # Iterate through stage pairs
    for i in range(1, len(parts), 2):
        stage_header = parts[i].strip()
        stage_code = parts[i+1].strip() if i+1 < len(parts) else ""

        # Extract stage title
        match = re.search(r'STAGE (\d+) [—–\-] (.+)', stage_header)
        if match:
            stage_num = match.group(1)
            stage_name = match.group(2).strip()
            md_title = f"## Stage {stage_num}: {stage_name}\n"
        else:
            md_title = f"## {stage_header}\n"

        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [md_title]
        })

        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in stage_code.split("\n")]
        })

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "None",
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open('flood_risk_pipeline.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print("flood_risk_pipeline.ipynb successfully generated!")

if __name__ == '__main__':
    main()
