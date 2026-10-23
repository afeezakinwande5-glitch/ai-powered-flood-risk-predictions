# Scientific Correction and Validation Log

## AI-Powered Flood Risk Prediction & Geospatial Intelligence System
**Capstone Project — TechCrush C8**  
**Study Area:** Lokoja Confluence & Lower Niger Basin, Kogi State, Nigeria  
**Audit Status:** POST-CORRECTION AUDIT READY  

---

### CRITICAL-01: Flood-Label Sensor Provenance Contradiction
- **Issue ID:** CRITICAL-01
- **Severity:** Critical
- **Original finding:** Documentation, decision log (DEC-007), and limitations (LIM-07) claimed the ground-truth flood footprint was derived from Sentinel-1 Synthetic Aperture Radar (SAR) imagery. However, the downloaded shapefile was `VIIRS_20221001_20221025_MaximumFloodWaterExtent_Nigeria.shp`.
- **Root cause:** The initial project design anticipated using Sentinel-1 SAR imagery from UNOSAT, but the published UNOSAT activation dataset covering the entire Nigerian flood disaster extent for October 2022 utilized the Visible Infrared Imaging Radiometer Suite (VIIRS). The code downloaded the official VIIRS composite, but legacy documentation was not synchronized with the operational download.
- **Evidence:**
  - Primary shapefile metadata: `VIIRS_20221001_20221025_MaximumFloodWaterExtent_Nigeria.shp`
  - Sensor: VIIRS aboard Suomi NPP and NOAA-20 polar-orbiting satellites
  - Provider: UNITAR-UNOSAT (Product ID: 3427)
  - Processing: Multi-day maximum flood water extent composite (1–25 October 2022) at 375m native resolution
- **Correction:**
  1. Updated all references across `flood_risk_pipeline.py`, `flood_risk_pipeline.ipynb`, `build_pipeline.py`, and `README.md` to state **VIIRS optical/infrared radiometry** as the sensor.
  2. Removed all erroneous claims of "Sentinel-1 SAR".
  3. Documented sensor-specific implications: optical radiometry requires multi-day compositing to mitigate cloud cover, and native 375m pixels were vector-polygonized by UNOSAT and rasterized to the 90m analysis grid.
- **Files changed:** `flood_risk_pipeline.py`, `build_pipeline.py`, `flood_risk_pipeline.ipynb`, `README.md`
- **Experiment rerun:** Full pipeline rerun Stages 1–18.
- **Validation performed:** Verified rasterization of VIIRS polygons into `flood_mask_90m_utm32n.tif`.
- **Result:** Complete factual alignment between source data, code, and documentation.
- **Status:** Resolved

---

### CRITICAL-02: UNOSAT Activation / Event Identifier Discrepancy
- **Issue ID:** CRITICAL-02
- **Severity:** Critical
- **Original finding:** Decision log and text cited UNOSAT activation `FL20221014NGA`, whereas code downloaded `FL20221019NGA_SHP.zip`.
- **Root cause:** On 14 October 2022, UNOSAT issued an initial emergency alert code (`FL20221014NGA`) following the initial flood surge in Kogi State. The comprehensive national composite covering peak inundation was formally archived, published, and distributed under activation `FL20221019NGA` (Product ID 3427).
- **Evidence:**
  - UNOSAT static repository URL: `https://unosat.org/static/unosat_filesystem/3427/FL20221019NGA_SHP.zip`.
- **Correction:**
  1. Unified the activation identifier to `FL20221019NGA` throughout all scripts, notebook cells, and documentation, explicitly noting that `FL20221014NGA` was the preliminary activation code.
- **Files changed:** `flood_risk_pipeline.py`, `build_pipeline.py`, `README.md`
- **Experiment rerun:** Verified URL download and extraction logic.
- **Validation performed:** Download URL integrity test and shapefile load test.
- **Result:** Zero ambiguity in activation provenance.
- **Status:** Resolved

---

### CRITICAL-03: Erroneous SAR Attribution in Decision Log & Limitations
- **Issue ID:** CRITICAL-03
- **Severity:** Critical
- **Original finding:** Decision DEC-007 ("Sentinel-1 SAR verified") and Limitation LIM-07 ("UNOSAT SAR flood delineation accuracy") explicitly referenced SAR.
- **Root cause:** Documentation was drafted based on pre-project scoping templates before verifying the downloaded shapefile attributes.
- **Evidence:** `decision_log.csv` line 7 and `limitations.csv` line 7 contained literal strings mentioning "SAR".
- **Correction:**
  - Updated DEC-007 to: "UNOSAT FL20221019NGA VIIRS flood extent (Product 3427), 375m VIIRS multi-day composite (1-25 Oct 2022)".
  - Updated LIM-07 to: "UNOSAT VIIRS optical/radiometer flood delineation accuracy; native 375m resolution, cloud mitigation via temporal compositing".
- **Files changed:** `flood_risk_pipeline.py`, `decision_log.csv`, `limitations.csv`
- **Experiment rerun:** Regenerated CSV exports in Stage 18.
- **Validation performed:** Inspected exported CSVs to verify removal of SAR claims.
- **Result:** Complete consistency across all decision records and limitation catalogs.
- **Status:** Resolved

---

### CRITICAL-04: PR-AUC Claimed in Documentation but Omitted in Implementation
- **Issue ID:** CRITICAL-04
- **Severity:** Critical
- **Original finding:** The project claimed PR-AUC evaluation in the README and Stage 13 documentation, but `average_precision_score` was never imported, never computed, and omitted from `model_comparison.csv`.
- **Root cause:** The initial pipeline code implemented `roc_auc_score` and `roc_curve` but failed to call `average_precision_score` or compute `precision_recall_curve`.
- **Evidence:** `grep "average_precision_score" flood_risk_pipeline.py` returned 0 occurrences.
- **Correction:**
  1. Imported `average_precision_score` and `precision_recall_curve` from `sklearn.metrics`.
  2. Implemented PR-AUC computation for all evaluated models.
  3. Added a dedicated Precision-Recall Curve subplot to `stage13_evaluation.png` showing the horizontal baseline prevalence line ($y=0.0157$).
  4. Added `PR-AUC` as a mandatory column in `model_comparison.csv`.
- **Files changed:** `flood_risk_pipeline.py`, `build_pipeline.py`
- **Experiment rerun:** Full test set inference and metric computation.
- **Validation performed:** Verified exact mathematical computation: Baseline PR-AUC = 0.0157, Logistic Regression PR-AUC = 0.7615, Random Forest PR-AUC = 0.6985.
- **Result:** PR-AUC fully integrated and visualized.
- **Status:** Resolved

---

### HIGH-01: Missing Baseline Benchmark for Severely Imbalanced Test Set
- **Issue ID:** HIGH-01
- **Severity:** High
- **Original finding:** The spatial test set is 98.4% dry ($y=0$), meaning a naive majority-class classifier achieves 98.43% accuracy without predicting a single flooded cell. Without a baseline, the 98.67% and 99.06% model accuracies appeared misleadingly strong.
- **Root cause:** No naive baseline (majority class or random guess) was implemented in Stage 12 or 13.
- **Correction:** Implemented `DummyClassifier(strategy='most_frequent')` in Stage 12 and included it as `Baseline (Majority Class)` in Stages 13 and 14. Documented that accuracy is an uninformative metric under high imbalance, and that PR-AUC and Recall are the primary discriminating metrics.
- **Files changed:** `flood_risk_pipeline.py`, `build_pipeline.py`
- **Experiment rerun:** Full evaluation run.
- **Validation performed:**
  - Baseline: Accuracy 0.9843, Precision 0.0000, Recall 0.0000, F1 0.0000, ROC-AUC 0.5000, PR-AUC 0.0157.
  - Logistic Regression outperforms baseline by 48.5x in PR-AUC (0.7615 vs 0.0157) and achieves 0.6991 recall.
- **Result:** Rigorous benchmark established, contextualizing headline accuracy.
- **Status:** Resolved

---

### HIGH-02: Counterintuitive Logistic Regression Coefficients
- **Issue ID:** HIGH-02
- **Severity:** High
- **Original finding:** Logistic Regression coefficients appeared counterintuitive:
  - `dist_river`: +0.4370 (positive -> farther from river increases flood odds?)
  - `slope`: +1.6583 (positive -> steeper terrain increases flood odds?)
  - `rainfall`: -0.5803 (negative -> higher rainfall decreases flood odds?)
- **Root cause:**
  1. **Simpson's Paradox / Collinearity with Elevation:** Elevation is the overwhelming primary predictor ($\beta = -12.3419$). In bivariate analysis, every feature has a negative correlation with flood ($r_{elev} = -0.50$, $r_{slope} = -0.22$, $r_{rain} = -0.18$, $r_{dist} = -0.12$). However, elevation is strongly collinear with slope ($r = +0.40$) and distance to river ($r = +0.43$). Once the model fixes a massive negative coefficient on elevation, the residual variation forces slope and distance to act as suppressor terms.
  2. **Fluvial Hydrology vs. Pluvial Ponding:** Flooding in Lokoja is regional riverine (fluvial) inundation caused by upstream Niger and Benue river discharge, not local pluvial ponding. The steep, high-elevation plateaus in the south receive higher orographic rainfall (mean 468mm) than the valley confluence (mean 362mm), yet water sheds into the valley. Thus, local rainfall is negatively associated with flood extent in this regional topography.
- **Evidence:** Bivariate correlation matrix and Random Forest contrast (elevation importance 0.755, slope 0.155), where tree-based non-linear partitioning avoids linear sign inversion.
- **Correction:** Documented the exact mathematical and physical root causes in the pipeline comments, stdout logging, and project documentation. Emphasized that multivariable coefficients must never be interpreted as causal bivariate drivers.
- **Files changed:** `flood_risk_pipeline.py`, `README.md`
- **Validation performed:** Correlation analysis, feature distribution by class, and comparison with tree-based importance.
- **Result:** Scientifically defensible interpretation replacing superficial sign inspection.
- **Status:** Resolved

---

### HIGH-03: Spatial Domain Shift in Geographic Holdout Split
- **Issue ID:** HIGH-03
- **Severity:** High
- **Original finding:** Training flood prevalence was 13.57%, whereas the test set flood prevalence was 1.57% (a 9:1 discrepancy), creating an extreme domain shift between train and test.
- **Root cause:** The 2x2 quadrant split allocated the Southeast (SE) quadrant to the test holdout. Topographically, the SE quadrant is an upland plateau region with a mean elevation of 259.5m (median 265.4m), compared to the training quadrants' mean elevation of 158.8m (median 151.6m). The upland terrain sheds water rapidly, resulting in very little observed inundation despite high precipitation.
- **Evidence:**
  - Train: 704,123 pixels (13.57% flooded)
  - Test (SE): 242,060 pixels (1.57% flooded)
  - Mean elevation difference: +100.7 meters in holdout quadrant.
- **Correction:** Documented this domain shift explicitly as a core finding on spatial model generalizability. Added `LIM-08` in `limitations.csv` and detailed domain shift metrics in Stage 11. Refused to artificially rebalance the test set, preserving the integrity of the spatial holdout test.
- **Files changed:** `flood_risk_pipeline.py`, `build_pipeline.py`, `README.md`
- **Result:** Transparent presentation of spatial generalization challenges.
- **Status:** Resolved

---

### HIGH-04: Model Selection Rationale
- **Issue ID:** HIGH-04
- **Severity:** High
- **Original finding:** Previous documentation selected Logistic Regression based solely on ROC-AUC (0.9924 vs 0.9411) without evaluating flood recall or PR-AUC trade-offs against Random Forest.
- **Root cause:** Mechanistic selection rule without domain-specific disaster risk trade-off justification.
- **Correction:** Provided a comprehensive disaster risk evaluation:
  - For disaster early warning, **false negatives** (unpredicted floods) are catastrophic, making **recall** critical.
  - Logistic Regression achieved **69.91% recall** and **PR-AUC 0.7615**, compared to Random Forest's **59.85% recall** and **PR-AUC 0.6985**.
  - Random Forest achieved higher precision (75.31% vs 56.24%), making it suitable for conservative insurance/zoning applications, while Logistic Regression is preferred for regional susceptibility screening.
- **Files changed:** `flood_risk_pipeline.py`, `build_pipeline.py`, `README.md`
- **Result:** Multi-criteria, decision-theoretic justification for final model selection.
- **Status:** Resolved
