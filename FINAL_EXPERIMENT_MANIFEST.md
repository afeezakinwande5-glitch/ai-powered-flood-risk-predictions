# Final Experiment Manifest

## AI-Powered Flood Risk Prediction & Geospatial Intelligence System

**Project Developer:** Afeez Akinwande  
**Capstone Track:** TechCrush C8 — AI/ML  
**Project Group:** Group 16  
**Study Area:** Lokoja Confluence & Lower Niger Basin, Kogi State, Nigeria  
**Specification Version:** 2.0 (Post-Correction Audited)

---

## 1. Project Metadata

- **Project Name:** AI-Powered Flood Risk Prediction & Geospatial Intelligence System
- **Project Developer:** Afeez Akinwande
- **Programme:** TechCrush C8 Capstone Project
- **Project Group:** Group 16
- **Study Area Extent:** Bounding box [6.40°E – 7.20°E, 7.40°N – 8.20°N], approximately 88.8 km × 88.2 km (~7,832 km²).
- **Administrative Location:** Lokoja Confluence, Kogi State, North-Central Nigeria.
- **Geographic Context:** Confluence of the Niger and Benue Rivers, a major drainage convergence point for the Upper Niger and Benue basins.
- **Reference Disaster Event:** October 2022 Nigeria Extreme Flood.
- **Project Focus:** Application of Artificial Intelligence, Machine Learning, and Geographic Information Systems to flood susceptibility modelling and geospatial risk assessment.

## 2. Ground-Truth Data & Provenance

- **Dataset Name:** UNOSAT Satellite-Derived Maximum Flood Water Extent — Nigeria
- **Dataset Provider:** UNITAR-UNOSAT (United Nations Satellite Centre / United Nations Institute for Training and Research)
- **Sensor / Platform:** VIIRS (Visible Infrared Imaging Radiometer Suite) aboard the Suomi NPP and NOAA-20 polar-orbiting satellites.
- **Flood Product Name:** `VIIRS_20221001_20221025_MaximumFloodWaterExtent_Nigeria.shp`
- **Activation / Event ID:** `FL20221019NGA`
- **UNOSAT Product ID:** 3427
- **Preliminary Alert Code:** `FL20221014NGA`
- **Flood Observation Period:** 1 October 2022 to 25 October 2022.
- **Source Distribution URL:** `https://unosat.org/static/unosat_filesystem/3427/FL20221019NGA_SHP.zip`
- **License:** Creative Commons Attribution (CC BY-IGO 3.0).

The UNOSAT flood extent dataset provided the reference labels used to construct the binary flood target for the machine-learning workflow.

## 3. Prediction Task & Target Construction

- **Prediction Task:** Binary spatial classification of flood susceptibility, producing a continuous Flood Susceptibility Index (FSI) between 0.0 and 1.0.
- **Target Formulation:** Predict whether a terrestrial pixel is associated with observed inundation using terrain, drainage proximity, land-cover characteristics, and precipitation.
- **Target Definition:** Binary raster indicator, where `1 = Flooded` and `0 = Dry/Uninundated`.
- **Positive-Label Definition:** A pixel is labelled flooded when its centroid falls within an observed UNOSAT VIIRS maximum flood extent polygon.
- **Negative-Label Definition:** Terrestrial pixels outside the observed flood extent polygons after permanent-water masking.
- **Permanent Water Masking:** ESA WorldCover 2021 Class 80 was used to identify permanent open water. The manifest reports 21,077 pixels excluded from training, testing, and susceptibility modelling.
- **Duplicate Screening:** Grid-cell centroids were indexed by `(row, col)` to identify duplicate spatial samples.

The target represents observed inundation associated with the October 2022 reference event. It should not be interpreted as a direct observation of future flood probability.

## 4. Input Features & Data Sources

| Feature Name | Description | Source Dataset | Native Resolution | Spatial Operation | Temporal Reference |
|---|---|---|---|---|---|
| `elevation` | Surface elevation above sea level (m) | NASA SRTM 1 Arc-Second GL1 | ~30 m | Reprojected and bilinear resampling to 90 m UTM 32N | Static |
| `slope` | Terrain gradient (degrees) | Derived from SRTM DEM | 90 m analysis grid | Derived from elevation gradients | Static |
| `rainfall` | Cumulative precipitation (mm) | CHIRPS v2.0 Monthly | 0.05° (~5.3 km) | Bilinear resampling to 90 m UTM 32N | September and October 2022 |
| `dist_river` | Euclidean distance to nearest stream (m) | HydroSHEDS / WWF HydroRIVERS Africa | Vector hydrography | Rasterization and Euclidean distance transform | Static hydrography |
| `permeability` | Ordinal runoff resistance / infiltration index (1–9) | ESA WorldCover 2021 v200 | 10 m | Nearest-neighbour resampling and ordinal land-cover mapping | 2021 baseline |

### Feature Engineering Notes

- Elevation and slope represent terrain conditions associated with water accumulation and drainage.
- Rainfall represents accumulated precipitation during the selected reference period.
- River distance represents proximity to mapped drainage networks.
- The permeability feature represents an ordinal hydrological index derived from land-cover classes; it should not be described as a direct field measurement of soil permeability.

## 5. Spatial Reference & Grid Parameters

- **Master Analysis CRS:** UTM Zone 32N (`EPSG:32632`).
- **Geographic CRS:** WGS84 (`EPSG:4326`).
- **Master Grid Dimensions:** 987 columns × 980 rows.
- **Total Grid Pixels:** 967,260.
- **Spatial Resolution:** 90 m × 90 m.
- **Nominal Area per Pixel:** 0.81 hectares.
- **Valid Terrestrial Pixels:** 946,183.
- **Permanent Water Pixels Excluded:** 21,077.

The projected coordinate system supports metric distance calculations and terrain analysis.

## 6. Validation Strategy & Data Split

### Spatial Block Split

The workflow used a 2×2 spatial block split to separate training and held-out test regions.

| Quadrant | Spatial Region | Pixels | Flood Prevalence | Assignment |
|---|---|---:|---:|---|
| Quadrant 0 | Northwest | 235,285 | 10.98% | Training |
| Quadrant 1 | Northeast | 234,906 | 24.26% | Training |
| Quadrant 2 | Southwest | 233,932 | 5.45% | Training |
| Quadrant 3 | Southeast | 242,060 | 1.57% | Testing |

### Training Partition

- **Samples:** 704,123
- **Share of Valid Domain:** 74.4%
- **Flooded Samples:** 95,568
- **Flood Prevalence:** 13.57%
- **Approximate Dry-to-Flooded Ratio:** 6.4:1

### Test Partition

- **Samples:** 242,060
- **Share of Valid Domain:** 25.6%
- **Flooded Samples:** 3,808
- **Flood Prevalence:** 1.57%
- **Approximate Dry-to-Flooded Ratio:** 62.6:1

### Leakage Prevention

- Spatially separate training quadrants and the held-out Southeast quadrant.
- Fit `StandardScaler` on training data only and apply the fitted transformation to the test data.
- Keep the held-out quadrant separate during model evaluation.

### Domain Shift

The Southeast holdout quadrant has a reported mean elevation of 259.5 m, compared with 158.8 m for the training quadrants. Its lower observed flood prevalence provides a challenging test of spatial generalization.

## 7. Evaluated Models & Preprocessing

### 7.1 Naive Majority-Class Baseline

- **Strategy:** `DummyClassifier(strategy='most_frequent')`
- **Prediction:** Always predicts the majority class, `y = 0`.
- **Purpose:** Establishes a baseline for evaluating model performance under substantial class imbalance.

### 7.2 Balanced Logistic Regression

- **Solver:** L-BFGS.
- **Maximum Iterations:** `max_iter=1000`.
- **Random State:** `random_state=42`.
- **Class Weighting:** `class_weight='balanced'`.
- **Preprocessing:** `StandardScaler` fitted on the training partition through a scikit-learn pipeline.

### 7.3 Random Forest Classifier

- **Number of Estimators:** 200.
- **Maximum Depth:** `max_depth=12`.
- **Minimum Samples per Leaf:** `min_samples_leaf=50`.
- **Random State:** `random_state=42`.
- **Parallel Processing:** `n_jobs=-1`.
- **Class Weighting:** `class_weight='balanced'`.

Random Forest provides a non-linear comparison against Logistic Regression, allowing the experiment to evaluate whether a tree-based model improves the identification of flooded areas.

## 8. Final Performance Metrics

The following metrics are reproduced from the group's experiment manifest and should be checked against the final exported evaluation results before publication.

| Model | Accuracy | Precision (Flood) | Recall (Flood) | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (Majority) | 0.9843 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0157 |
| Logistic Regression | 0.9867 | 0.5624 | 0.6991 | 0.6233 | 0.9924 | 0.7615 |
| Random Forest | 0.9906 | 0.7531 | 0.5985 | 0.6670 | 0.9263 | 0.6985 |

- **Held-Out Test Samples:** 242,060.
- **Flooded Test Samples:** 3,808.
- **Dry Test Samples:** 238,252.

### Metric Interpretation

- **Accuracy** measures the proportion of correctly classified samples but can be misleading when the dataset is heavily imbalanced.
- **Precision** measures the proportion of predicted flooded samples that are actually flooded.
- **Recall** measures the proportion of observed flooded samples correctly identified.
- **F1-score** balances precision and recall.
- **ROC-AUC** measures the model's ability to rank positive samples above negative samples across classification thresholds.
- **PR-AUC** summarises precision-recall performance and is particularly useful for evaluating rare positive classes.

## 9. Model Selection & Decision Rationale

**Selected Model:** Balanced Logistic Regression.

The group's reported evaluation favoured Logistic Regression because it achieved higher flood recall and PR-AUC on the held-out test region.

### Reasons for Selection

1. **Higher Flood Recall:** Logistic Regression achieved 69.91% recall, compared with 59.85% for Random Forest.
2. **Higher PR-AUC:** Logistic Regression achieved a reported PR-AUC of 0.7615, compared with 0.6985 for Random Forest.
3. **Continuous Output:** Logistic Regression produces probability estimates that can be used to construct a continuous susceptibility surface.
4. **Alternative Model Strength:** Random Forest achieved higher precision in the reported results, which may be useful in applications where limiting false-positive alerts is important.

The model choice reflects a trade-off between identifying more observed flooded cells and limiting false-positive predictions. The results support susceptibility screening, not a guarantee of future flooding.

## 10. Risk Mapping & Susceptibility Classification

### Continuous Flood Susceptibility Index

The selected model generates an FSI value between 0.0 and 1.0 for each modelled terrestrial pixel.

### Five-Tier Classification

| Susceptibility Class | FSI Range |
|---|---|
| Very Low | `[0.0, 0.2)` |
| Low | `[0.2, 0.4)` |
| Moderate | `[0.4, 0.6)` |
| High | `[0.6, 0.8)` |
| Very High | `[0.8, 1.0]` |

### Expected Cartographic Outputs

- Static susceptibility map: `outputs/maps/flood_susceptibility_map.png`
- GeoTIFF susceptibility raster: `outputs/maps/flood_susceptibility.tif`
- Interactive Folium map: `outputs/maps/flood_susceptibility_interactive.html`

Confirm that each listed output exists in the repository before treating it as a delivered file.

## 11. Known Technical Limitations

1. **Single Reference Event:** The model uses the October 2022 flood event. Generalisation to other flood events requires additional observations.
2. **Rainfall Resolution:** CHIRPS has a native resolution of approximately 5.3 km. Resampling it to 90 m does not create new rainfall observations at that finer resolution.
3. **Fluvial and Pluvial Processes:** Flooding around the Niger–Benue confluence is influenced by catchment discharge and river dynamics, not just local rainfall.
4. **VIIRS Resolution:** The reference flood dataset originates from VIIRS observations with a native spatial resolution of approximately 375 m. Small channels and local inundation features may not be represented.
5. **Antecedent Soil Moisture:** Pre-event soil saturation data, such as SMAP observations, was not included in the project workflow.
6. **Spatial Domain Shift:** The held-out Southeast region has a substantially lower flood prevalence than the training regions, which affects model generalisation and interpretation of the reported metrics.
7. **Historical Reference Labels:** The mapped susceptibility is associated with the selected historical flood event and available predictor data; it is not a real-time flood forecast.

## 12. Reproducibility & Software Environment

- **Random Seed:** `42`.
- **Python Version:** `3.12.10`.

### Core Dependencies

| Library | Reported Version |
|---|---|
| `scikit-learn` | `1.9.0` |
| `rasterio` | `1.5.2` |
| `geopandas` | `1.2.0` |
| `numpy` | `2.5.2` |
| `pandas` | `3.0.5` |
| `shapely` | `2.1.2` |
| `pyproj` | `3.7.2` |
| `folium` | `0.20.0` |

Reproducibility requires the same input datasets, preprocessing steps, spatial split, model parameters, and software environment. The dependency versions above are reproduced from the supplied manifest and should be verified against the actual project environment.

## 13. Personal Contribution & Project Context

This project was completed as part of the TechCrush C8 AI/ML capstone in Group 16.

My project portfolio documents the application of machine learning and geospatial analysis to a flood susceptibility problem using satellite-derived flood extent, terrain data, rainfall data, drainage information, and land-cover characteristics.

The experiment demonstrates a workflow for:

- Preparing and integrating geospatial datasets.
- Constructing a spatially aligned binary target.
- Engineering terrain and hydrological predictor variables.
- Training and evaluating classification models.
- Comparing model performance under class imbalance.
- Generating continuous susceptibility estimates and mapped outputs.
- Documenting data provenance, validation strategy, and limitations.

**Important:** This section describes the shared group project. Any claim about specific individual tasks should reflect the work personally completed.

---

**Project:** AI-Powered Flood Risk Prediction & Geospatial Intelligence System  
**Programme:** TechCrush C8 AI/ML Capstone  
**Group:** Group 16  
**Study Area:** Lokoja Confluence & Lower Niger Basin, Kogi State, Nigeria
