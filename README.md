# AI-Powered Flood Risk Prediction & Geospatial Intelligence System

**Capstone Project — TechCrush C8**  
**Study Area:** Lokoja Confluence & Lower Niger Basin, Kogi State, Nigeria  
**Target Reference Event:** October 2022 Nigeria Extreme Flood (Peak Inundation Composite, 1–25 October 2022)  
**Ground-Truth Flood Sensor:** VIIRS (Suomi NPP & NOAA-20), UNOSAT Activation `FL20221019NGA` (Product ID 3427)  
**Primary Deliverables:**
- Jupyter Notebook: [`flood_risk_pipeline.ipynb`](file:///c:/Users/GMatts/Desktop/TechCrush-C8/Flood%20Risk/flood_risk_pipeline.ipynb)
- Master Pipeline Script: [`flood_risk_pipeline.py`](file:///c:/Users/GMatts/Desktop/TechCrush-C8/Flood%20Risk/flood_risk_pipeline.py)
- Pipeline Generator: [`build_pipeline.py`](file:///c:/Users/GMatts/Desktop/TechCrush-C8/Flood%20Risk/build_pipeline.py)
- Audit & Correction Log: [`CORRECTION_LOG.md`](file:///c:/Users/GMatts/Desktop/TechCrush-C8/Flood%20Risk/CORRECTION_LOG.md)
- Experiment Manifest: [`EXPERIMENT_MANIFEST.md`](file:///c:/Users/GMatts/Desktop/TechCrush-C8/Flood%20Risk/EXPERIMENT_MANIFEST.md)

---

## 1. Project Overview

Flooding in the Lokoja confluence region is driven by the confluence of Nigeria's two largest rivers (the Niger and Benue) combined with upstream reservoir releases (Lagdo, Kainji, and Jebba dams) and heavy monsoon precipitation. During the catastrophic October 2022 disaster, vast sections of Kogi State were submerged.

This project delivers a **technically defensible, transparent, and reproducible** Machine Learning + GIS workflow:
- **Strict Data Provenance**: Ground truth is derived from the official UNITAR-UNOSAT activation `FL20221019NGA` based on the Visible Infrared Imaging Radiometer Suite (**VIIRS**) aboard Suomi NPP and NOAA-20 satellites (375m native resolution, 25-day maximum composite), rather than unverified SAR claims.
- **No Data Fabrication**: All datasets are acquired from authoritative global providers (NASA, USGS, ESA, HydroSHEDS, UNITAR-UNOSAT).
- **Spatial Independence**: Utilizes a **2×2 spatial block split** (SE quadrant withheld for testing) rather than a random train/test split, preventing spatial autocorrelation data leakage.
- **Mandatory Benchmark Baseline**: Evaluated against a naive majority-class classifier to contextualize headline accuracy under extreme class imbalance (98.4% dry holdout).
- **Imbalance-Aware Metrics**: Implementation and visualization of **PR-AUC (Precision-Recall curves)** alongside ROC-AUC and confusion matrices.
- **Dual Deliverables**: Binary ground truth mapped to a continuous **Flood Susceptibility Index (FSI)** and a **5-tier hazard zonation map** (Very Low to Very High).

---

## 2. Quickstart: Running Locally or in Google Colab

### Option A: Local Execution
The pipeline automatically detects local data directories:
```powershell
python flood_risk_pipeline.py
```

### Option B: Interactive Streamlit Web Application (Section 6.3 Demo)
To launch the self-contained interactive intelligence dashboard:
```powershell
pip install -r requirements.txt
streamlit run app.py
```
This opens the browser UI featuring the interactive Leaflet/Folium map, real-time risk simulator, model diagnostics, and architectural decision log.

### Option C: Google Colab (Recommended for Pipeline)
Google Colab provides a pre-configured Linux environment with GDAL and PROJ support:
1. Open [colab.research.google.com](https://colab.research.google.com).
2. Upload [`flood_risk_pipeline.ipynb`](file:///c:/Users/GMatts/Desktop/TechCrush-C8/flood-risk/flood_risk_pipeline.ipynb).
3. Select **Runtime** > **Run all** (`Ctrl + F9`).
4. Stage 1 automatically installs required libraries (`rasterio`, `geopandas`, `pyproj`, `folium`, `scikit-learn`).
5. All figures, maps, and CSV logs are generated in `outputs/`.

---

## 3. The 18-Stage Implementation Architecture

| Stage | Name | Description & Methodology |
|:---|:---|:---|
| **Stage 1** | **Environment Setup** | Automated package verification and import of geospatial (`rasterio`, `geopandas`, `pyproj`) and ML stack (`scikit-learn`). Configures `RANDOM_STATE = 42`. |
| **Stage 2** | **Project Configuration** | Spatial extent (6.40°E–7.20°E, 7.40°N–8.20°N; ~88.8 × 88.2 km), metric projection (`EPSG:32632` UTM Zone 32N), 90m master analysis grid. |
| **Stage 3** | **Data Acquisition** | Automated downloads: NASA SRTM 30m DEM, CHIRPS Sep/Oct 2022 rainfall, HydroRIVERS hydrography, ESA WorldCover 2021 (10m), UNOSAT `FL20221019NGA` VIIRS flood vectors. |
| **Stage 4** | **Dataset Understanding** | Coordinate reference system checking, boundary inspection, and raw layer visualization (`stage4_raw_datasets.png`). |
| **Stage 5** | **Spatial Preprocessing** | Resampling all rasters to the 90m UTM 32N master grid (bilinear for continuous variables, nearest-neighbor for categorical land cover). |
| **Stage 6** | **Data Integration** | Stacking harmonized raster bands and masking out permanent water bodies (ESA WorldCover Class 80). |
| **Stage 7** | **Data Cleaning** | Screening NoData values (`-9999` to `NaN`), checking physical bounds, and logging clean sample counts (946,183 valid pixels). |
| **Stage 8** | **Exploratory Data Analysis** | Distribution profiles, bivariate class histograms, and correlation matrix heatmap (`stage8_correlation.png`). |
| **Stage 9** | **Feature Engineering** | Derivation of 5 hydrologic features: `elevation` (m), `slope` (degrees), `rainfall` (mm), `dist_river` (m via Euclidean distance transform), `permeability` (soil/land cover runoff index). |
| **Stage 10** | **Target Construction** | Rasterization of UNOSAT satellite flood footprints into binary labels ($y=1$ flooded, $y=0$ dry). |
| **Stage 11** | **Spatial Block Split** | Division into 4 geographic quadrants; SE quadrant withheld as test holdout to test real spatial generalization (domain shift documented). |
| **Stage 12** | **Modelling** | Training **Naive Majority Baseline**, **Balanced Logistic Regression**, and **Balanced Random Forest** (`class_weight='balanced'`). |
| **Stage 13** | **Evaluation** | Imbalance-aware evaluation: ROC-AUC, PR-AUC, Precision-Recall curves, ROC curves, and confusion matrices (`stage13_evaluation.png`). |
| **Stage 14** | **Model Comparison** | Tabular and graphical comparison of baseline and candidate models (`outputs/model_comparison.csv`). |
| **Stage 15** | **Final Model Selection** | Selection of Balanced Logistic Regression based on superior flood recall (69.91% vs 59.85%) and PR-AUC (0.7615 vs 0.6985). |
| **Stage 16** | **Spatial Prediction** | Grid inference across all 946,183 pixels to produce the continuous Flood Susceptibility Index ($0.0 \le \text{FSI} \le 1.0$). |
| **Stage 17** | **Flood Susceptibility Map** | Export of static publication maps (`flood_susceptibility_map.png`), float32 GeoTIFF (`flood_susceptibility.tif`), and interactive Folium web map. |
| **Stage 18** | **Findings & Decision Log** | Structured export of 14 architecture decisions (`decision_log.csv`) and 8 documented limitations (`limitations.csv`). |

---

## 4. Authoritative Datasets Used

1. **Digital Elevation Model (DEM)**: NASA SRTM 1 Arc-Second GL1 (~30m) via OpenTopography / CGIAR-CSI.
2. **Precipitation**: CHIRPS v2.0 Monthly (UC Santa Barbara / USGS) for September & October 2022 (5.3km downscaled to 90m).
3. **River Network**: HydroSHEDS / WWF HydroRIVERS Africa vector reach dataset.
4. **Land Use / Land Cover**: ESA WorldCover 10m 2021 v200 (AWS S3 Tile `N06E006`).
5. **Observed Flood Ground Truth**: UNITAR-UNOSAT activation `FL20221019NGA` (Product ID 3427), Visible Infrared Imaging Radiometer Suite (**VIIRS**) 375m maximum flood extent composite (1–25 October 2022).

---

## 5. Model Performance Comparison (Test Holdout)

The Southeast quadrant holdout represents a real-world **domain shift challenge**: it is an upland plateau region (+100.7m higher mean elevation than training quadrants) with a flood prevalence of only **1.57%** (3,808 flooded cells vs 238,252 dry cells).

| Model | Accuracy | Precision (Flood) | Recall (Flood) | F1-Score | ROC-AUC | PR-AUC | Gain over Baseline PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Majority Class)** | 0.9843 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0157 | 1.0× |
| **Logistic Regression** | **0.9867** | **0.5624** | **0.6991** | **0.6233** | **0.9924** | **0.7615** | **48.5×** |
| **Random Forest** | **0.9906** | **0.7531** | **0.5985** | **0.6670** | **0.9263** | **0.6985** | **44.5×** |

### Key Evaluation Takeaways:
1. **Accuracy is Deceptive**: The naive majority-class classifier achieves 98.43% accuracy simply by predicting all cells as dry, yet it identifies zero actual flooded cells.
2. **PR-AUC Proves Real Predictive Power**: Both models achieve massive gains over the baseline prevalence line ($0.0157$), with Logistic Regression achieving **0.7615** (48.5× baseline) and Random Forest achieving **0.6985** (44.5× baseline).
3. **Why Logistic Regression is Selected**: In disaster risk management, **minimizing false negatives (maximizing recall)** is life-critical. Logistic Regression successfully detects **69.91%** of actual inundated cells in the holdout area (capturing 383 more flooded cells than Random Forest).

---

## 6. Scientific Interpretation & Coefficient Analysis

### Why do some Logistic Regression coefficients have counterintuitive signs?
- Multivariable coefficients:
  - `elevation`: $-12.3419$ (dominant negative effect)
  - `slope`: $+1.6583$ (positive)
  - `dist_river`: $+0.4370$ (positive)
  - `rainfall`: $-0.5803$ (negative)
- **Root Cause 1: Simpson's Paradox / Elevation Collinearity**:
  In bivariate analysis, all features have a negative correlation with flood ($r_{elev} = -0.50$, $r_{slope} = -0.22$, $r_{rain} = -0.18$, $r_{dist} = -0.12$). However, elevation is collinear with slope ($r = +0.40$) and distance to river ($r = +0.43$). Because elevation is assigned a massive negative weight ($-12.34$), the linear model uses slope and distance as positive suppressor terms to adjust for elevation over-prediction.
- **Root Cause 2: Fluvial Hydrology vs Pluvial Ponding**:
  Flooding at the Lokoja confluence is driven by **fluvial inundation** from upstream Niger and Benue river discharge, not local pluvial ponding. High-elevation plateaus receive higher orographic rainfall (mean 468mm) than the valley confluence (mean 362mm), yet water sheds into the valley. Thus, local rainfall is negatively associated with flood extent in this topography.
- **Random Forest Contrast**:
  Random Forest avoids linear sign inversion by non-linear recursive partitioning: `elevation` is the dominant split variable (importance 0.755), followed by `slope` (0.155) and `rainfall` (0.070).

---

## 7. Project Artifacts & File Structure

```text
├── app.py                          # Streamlit interactive intelligence dashboard (Section 6.3 Demo)
├── requirements.txt                # Unified Python dependency specification
├── flood_risk_pipeline.py          # Master executable pipeline (Stages 1-18)
├── flood_risk_pipeline.ipynb       # Jupyter Notebook for Google Colab
├── build_pipeline.py               # Notebook generator script
├── CORRECTION_LOG.md               # Audit findings, root causes, and verification log
├── EXPERIMENT_MANIFEST.md          # Full project experiment specification
├── README.md                       # Synchronized documentation
├── data/                           # Downloaded rasters and shapefiles (auto-created)
└── outputs/                        # Generated deliverables
    ├── figures/
    │   ├── stage4_raw_datasets.png
    │   ├── stage8_distributions.png
    │   ├── stage8_correlation.png
    │   ├── stage8_spatial_eda.png
    │   └── stage13_evaluation.png
    ├── maps/
    │   ├── flood_susceptibility.tif
    │   ├── flood_susceptibility_map.png
    │   └── flood_susceptibility_interactive.html
    ├── models/
    │   ├── logistic_regression.joblib
    │   └── random_forest.joblib
    ├── decision_log.csv
    ├── limitations.csv
    ├── model_comparison.csv
    └── project_metadata.json
```
