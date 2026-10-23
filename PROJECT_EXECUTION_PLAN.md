# PROJECT EXECUTION PLAN
## AI-Powered Flood Risk Prediction & Geospatial Intelligence System

**Programme:** TechCrush C8 Capstone Project  
**Group:** Group 16  
**Project Developer:** Afeez Akinwande  
**Study Area:** Lokoja Confluence and Lower Niger Basin, Kogi State, Nigeria  
**Reference Flood Event:** October 2022 Nigeria Flood  
**Domain:** Climate Resilience, Geographic Information Systems (GIS), Artificial Intelligence and Machine Learning (AI/ML)

---

# PROJECT OVERVIEW

## Problem Statement

Flooding can cause significant damage to communities, infrastructure, property, and livelihoods. Understanding areas that may be susceptible to flooding requires the analysis of environmental and geographic factors, including elevation, slope, rainfall, drainage proximity, and land characteristics.

This project demonstrates how geospatial data and machine-learning techniques can be integrated to analyse flood susceptibility and produce geographic information that can support flood-risk assessment, environmental planning, and disaster preparedness.

## Project Goal

To develop an AI/ML-based geospatial workflow that analyses environmental and geographic variables to estimate flood susceptibility and present the resulting information through interpretable maps and analytical outputs.

## Project Objectives

The project aims to:

1. Acquire and validate relevant real-world environmental and geospatial datasets.
2. Understand the study area's geographic and environmental characteristics.
3. Prepare and clean the data for analysis.
4. Conduct exploratory data analysis (EDA).
5. Develop meaningful environmental and geographic features.
6. Establish a suitable machine-learning problem formulation.
7. Train and compare at least two appropriate machine-learning models.
8. Evaluate model performance using appropriate metrics.
9. Select a final model based on evaluation evidence.
10. Generate flood-susceptibility predictions.
11. Visualize the results geographically.
12. Document the workflow, findings, assumptions, and limitations.
13. Produce a reproducible notebook and supporting project report.

## Core Machine-Learning Workflow

Data Acquisition → Data Understanding → Data Cleaning → Exploratory Data Analysis → Data Preprocessing → Feature Engineering → Train/Test Separation → Modelling → Evaluation → Model Comparison → Final Predictions → Geospatial Visualization → Documentation.

---

# STAGE 1 — PROJECT DEFINITION

## 1.1 Project Scope

The project focuses on flood susceptibility in Lokoja and the Lower Niger Basin in Kogi State, Nigeria.

The geographic extent is approximately:

- Western longitude: 6.40°E
- Eastern longitude: 7.20°E
- Southern latitude: 7.40°N
- Northern latitude: 8.20°N

The study covers an area of approximately 7,832 km².

The reference event is the October 2022 Nigeria flood.

## 1.2 Intended Prediction

The project uses environmental and geographic information to estimate flood susceptibility across the study area.

The final prediction product is a Flood Susceptibility Index (FSI), represented by continuous scores and five susceptibility classes.

The results are intended to support flood-risk assessment and geographic interpretation. They are not presented as guaranteed predictions of future flood events.

## 1.3 Expected Inputs

The project considers environmental and geographic information such as:

- Elevation.
- Slope.
- Rainfall.
- Distance to rivers.
- Land use and land cover.
- Soil-related or surface-permeability characteristics.
- Historical flood-extent information.
- Geographic coordinates and spatial reference information.

The actual features must be supported by the available datasets. Missing features must not be fabricated simply to satisfy the original project concept.

## 1.4 Expected Outputs

The project is expected to produce:

- A prepared environmental and geospatial dataset.
- Exploratory data analysis results.
- Engineered predictor variables.
- At least two trained machine-learning models.
- Model evaluation metrics.
- A model comparison.
- A justified final model.
- Flood-susceptibility predictions.
- Geographic visualizations.
- Documented findings and limitations.
- A reproducible notebook.
- A project report.

## 1.5 Project Boundaries

The project does not attempt to:

- Guarantee real-time flood warnings.
- Predict every future flood event with certainty.
- Replace official emergency-management systems.
- Claim universal applicability outside the validated study area.
- Treat susceptibility predictions as measured flood depth or duration.
- Build a complete commercial flood-management platform before completing the core analysis.

## 1.6 Stage 1 Completion Criteria

Stage 1 is complete when the problem, study area, objective, intended prediction, scope, expected inputs, deliverables, and project limitations are clearly defined.

---

# STAGE 2 — DATASET SELECTION AND VALIDATION

## 2.1 Objective

Identify and validate credible real-world datasets that support the project's flood-susceptibility analysis.

## 2.2 Dataset Selection Criteria

Each dataset should be assessed using the following criteria:

- Relevance to flood susceptibility.
- Geographic coverage.
- Spatial resolution.
- Temporal coverage.
- Availability of relevant environmental variables.
- Source credibility.
- Data quality.
- Compatibility with GIS and machine-learning workflows.
- Known limitations.
- Suitability for the project timeframe.

## 2.3 Selected Data Sources

The project integrates the following data sources:

### A. UNOSAT Flood-Extent Data

The reference flood extent is derived from UNOSAT flood mapping associated with the October 2022 Nigeria flood.

The project uses the UNOSAT product identified as Product 3427, associated with `FL20221019NGA`, covering the October 2022 flood period.

The relevant source product is:

`VIIRS_20221001_20221025_MaximumFloodWaterExtent_Nigeria.shp`

Data source:

https://unosat.org/static/unosat_filesystem/3427/FL20221019NGA_SHP.zip

This dataset supplies reference flood-extent information for constructing the flood-related target used in the analysis.

### B. SRTM Digital Elevation Model

The Shuttle Radar Topography Mission (SRTM) elevation data supplies elevation information.

Elevation is important for understanding terrain characteristics and identifying areas whose topography may be associated with flood susceptibility.

Slope is derived from elevation data.

### C. CHIRPS Rainfall Data

Climate Hazards Group InfraRed Precipitation with Station data (CHIRPS) supplies rainfall information.

September and October 2022 rainfall information is used to represent rainfall conditions associated with the reference flood period.

The native CHIRPS resolution is approximately 5.3 km. Resampling this rainfall information to the analysis grid does not create new rainfall observations at 90 m resolution.

### D. HydroRIVERS

HydroRIVERS supplies river-network information.

It is used to derive distance-to-river information for the study area.

### E. ESA WorldCover 2021

ESA WorldCover provides land-cover information.

The project uses this dataset to derive a surface-permeability index and identify permanent-water areas.

The permeability index is an ordinal proxy based on land-cover classes. It should not be interpreted as direct field measurement of soil permeability.

## 2.4 Dataset Validation

The datasets must be checked for:

- Source credibility.
- Geographic coverage.
- Coordinate reference systems.
- Spatial resolution.
- Missing or invalid values.
- Geometry validity.
- Duplicate records.
- Compatibility between spatial layers.
- Temporal relevance.
- Target suitability.
- Known limitations.

## 2.5 Confirmed Analysis Grid

The project uses the following principal spatial settings:

- Projected CRS: EPSG:32632.
- Analysis grid resolution: 90 m.
- Grid columns: 987.
- Grid rows: 980.
- Total grid cells: 967,260.
- Permanent-water cells excluded from the terrestrial prediction domain: 21,077.
- Terrestrial prediction cells: 946,183.

The spatial grid provides a common framework for combining the environmental predictor layers.

## 2.6 Target Construction

The flood-related target is constructed using the reference flood-extent information.

The target identifies flood-extent observations and is used to train the machine-learning models.

Permanent-water areas are excluded from the terrestrial modelling domain.

The target represents a flood-related reference label for the selected event. It is not a complete record of every flood that has occurred or will occur in the study area.

## 2.7 Stage 2 Completion Criteria

Stage 2 is complete when:

- Relevant data sources have been identified.
- Dataset sources have been documented.
- Geographic coverage has been confirmed.
- Spatial reference systems and resolutions have been examined.
- The flood-related target has been established.
- Relevant predictor variables have been identified.
- Dataset limitations have been documented.
- The data is suitable for the subsequent processing workflow.

---

# STAGE 3 — COLAB NOTEBOOK AND DATA PIPELINE

## 3.1 Objective

Build a structured, reproducible notebook that integrates the selected datasets and prepares them for modelling.

The pipeline covers data acquisition, data inspection, cleaning, exploratory analysis, preprocessing, feature engineering, and preparation of the modelling data.

## 3.2 Notebook Structure

The notebook should contain the following sections:

1. Project Introduction.
2. Study Area.
3. Data Sources.
4. Data Acquisition.
5. Dataset Understanding.
6. Data Cleaning and Validation.
7. Exploratory Data Analysis.
8. Feature Engineering.
9. Data Preprocessing.
10. Spatial Data Integration.
11. Train/Test Separation.
12. Modelling.
13. Evaluation.
14. Model Comparison.
15. Final Predictions.
16. Geospatial Visualization.
17. Findings and Limitations.
18. Conclusion.
19. References.

## 3.3 Data Cleaning

The processing workflow should address applicable issues involving:

- Missing values.
- Duplicate records.
- Incorrect data types.
- Invalid values.
- Invalid geometries.
- Irrelevant fields.
- Inconsistent coordinate reference systems.
- Misaligned raster layers.
- Inconsistent spatial resolution.

Each important cleaning decision must be justified.

Outliers should not be removed automatically. Their validity must be assessed in the context of the data.

## 3.4 Exploratory Data Analysis

EDA should investigate:

- Elevation distributions.
- Slope distributions.
- Rainfall patterns.
- Distance-to-river distributions.
- Land-cover composition.
- Flood-label distribution.
- Relationships between predictor variables.
- Relationships between environmental factors and the flood-related target.
- Geographic patterns in the study area.

Visualizations should include appropriate titles, labels, legends, and concise interpretations.

Correlation indicates association and does not, by itself, establish causation.

## 3.5 Feature Engineering

The project uses the following principal predictor variables:

- `elevation`
- `slope`
- `rainfall`
- `dist_river`
- `permeability`

These features represent terrain, rainfall, drainage proximity, and land-surface characteristics.

Feature definitions must remain consistent throughout the notebook and prediction workflow.

## 3.6 Data Preprocessing

The workflow should ensure that:

- Spatial layers use compatible coordinate reference systems.
- Raster layers align to the common analysis grid.
- Feature values are valid.
- The target is separated from predictor variables.
- Preprocessing is applied consistently.
- Training and testing information remain appropriately separated.
- No target leakage is introduced.

Where scaling is used, its parameters must be learned from training data only and then applied to testing data.

## 3.7 Spatial Separation

The project uses a 2 × 2 spatial block strategy to assess model performance across different geographic sections of the study area.

The reported split is:

- Training quadrants: Q0, Q1, and Q2.
- Testing quadrant: Q3, the southeast block.

Reported training size: 704,123 terrestrial cells.

Reported testing size: 242,060 terrestrial cells.

The training and testing blocks have different flood-label proportions. The southeast test block also differs in elevation from the training domain.

These differences must be considered when interpreting the evaluation results.

## 3.8 Stage 3 Completion Criteria

Stage 3 is complete when:

- The datasets have been loaded and inspected.
- Data-quality issues have been assessed.
- Required cleaning has been performed.
- EDA has been completed.
- Predictor features have been prepared.
- Spatial alignment has been checked.
- The training and testing separation is established.
- The notebook documents important preprocessing decisions.
- The prepared data is ready for modelling.

---

# STAGE 4 — MODELLING

## 4.1 Objective

Train and document at least two appropriate machine-learning models for the flood-susceptibility problem.

## 4.2 Confirmed Modelling Approach

The project treats flood-extent identification as a binary classification problem for model training and evaluation.

The trained models estimate whether a terrestrial grid cell belongs to the flood-related reference class.

The resulting predictions are subsequently used to create a continuous Flood Susceptibility Index.

The susceptibility index should not be confused with a directly observed probability of a future flood event.

## 4.3 Model 1 — Dummy Baseline

A majority-class DummyClassifier provides a baseline against which the other models can be compared.

The baseline demonstrates how a simple strategy performs without learning meaningful relationships between the predictor variables and target.

It helps establish whether the more sophisticated models provide useful predictive improvement.

## 4.4 Model 2 — Balanced Logistic Regression

The project uses Logistic Regression with class balancing.

The documented configuration includes:

- Solver: `lbfgs`
- Maximum iterations: `1000`
- Random state: `42`
- Class weighting: `balanced`
- StandardScaler applied where required.

Scaling parameters must be fitted using training data only.

Logistic Regression provides a relatively interpretable approach to modelling relationships between environmental predictors and the binary flood-related target.

## 4.5 Model 3 — Balanced Random Forest

The project also uses a Random Forest classifier.

The documented configuration includes:

- Number of trees: `200`
- Maximum depth: `12`
- Minimum samples per leaf: `50`
- Random state: `42`
- Number of jobs: `-1`
- Class weighting: `balanced`

Random Forest can represent nonlinear relationships and interactions between predictors.

## 4.6 Training Standards

For all models:

- Use the prepared training data.
- Preserve the spatial train/test separation.
- Avoid training on test observations.
- Record important parameters.
- Maintain reproducibility where applicable.
- Generate the appropriate predictions or prediction scores.
- Document errors and implementation issues.
- Avoid introducing unsupported predictors.

## 4.7 Stage 4 Completion Criteria

Stage 4 is complete when:

- The modelling problem is clearly defined.
- The baseline and machine-learning models are trained.
- The relevant model configurations are documented.
- Predictions are generated.
- The training/testing separation is maintained.
- The models are ready for formal evaluation.

---

# STAGE 5 — EVALUATION AND MODEL COMPARISON

## 5.1 Objective

Evaluate the models using appropriate metrics, compare their performance, examine important limitations, and select a final model based on evidence.

## 5.2 Evaluation Metrics

For binary classification, relevant metrics include:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Precision-Recall AUC.
- Confusion matrix.

Accuracy alone is insufficient when the positive class is relatively uncommon.

Recall is important because it measures the proportion of actual positive flood-related observations identified by the model.

Precision helps show how many positive predictions correspond to positive reference labels.

ROC-AUC measures ranking performance across thresholds, while Precision-Recall AUC provides additional insight into positive-class performance.

## 5.3 Spatial Evaluation

The held-out southeast spatial block provides a geographic test region that was not used for model training.

This helps assess whether the model generalizes to a different spatial area.

However, the test block differs from the training region in its flood-label proportion and elevation characteristics.

Consequently, the evaluation must be interpreted with attention to geographic distribution shift.

## 5.4 Reported Model Results

The following metrics are included as the project's reported evaluation results. Before publication, they should be checked against the final notebook outputs to ensure consistency.

| Metric | Balanced Logistic Regression | Balanced Random Forest |
|---|---:|---:|
| Accuracy | 0.9867 | 0.9906 |
| Precision | 0.5624 | 0.7531 |
| Recall | 0.6991 | 0.5985 |
| F1-score | 0.6233 | 0.6670 |
| ROC-AUC | 0.9924 | 0.9263 |
| Precision-Recall AUC | 0.7615 | 0.6985 |

**Important:** These are the figures from the project manifest used to prepare this document. The final repository should use the exact values produced by the authoritative evaluation outputs. If the notebook reports different figures, reconcile the discrepancy rather than silently choosing one set.

## 5.5 Model Comparison

The reported results show different trade-offs.

### Logistic Regression

Logistic Regression reports higher recall, ROC-AUC, and Precision-Recall AUC in the comparison above.

Its higher recall means it identifies a larger proportion of positive reference observations in the evaluated test block.

### Random Forest

Random Forest reports higher accuracy, precision, and F1-score in the comparison above.

Its higher precision means a larger proportion of its positive predictions correspond to positive reference observations.

### Baseline

The majority-class baseline is used to demonstrate the limitations of predicting the dominant class without learning meaningful predictor relationships.

## 5.6 Final Model Selection

The project selects Balanced Logistic Regression as the final model based on its reported recall and discrimination metrics.

This choice reflects the project's emphasis on identifying flood-related positive observations.

The choice does not establish that Logistic Regression is universally superior to Random Forest. The selection applies to this dataset, spatial test design, target construction, and evaluation.

## 5.7 Error Analysis and Limitations

The evaluation should consider:

- False positives.
- False negatives.
- Differences between training and testing regions.
- Target imbalance.
- The quality of the reference flood-extent data.
- Coarse rainfall resolution.
- Potential geographic distribution shift.
- Missing environmental predictors.
- The distinction between susceptibility and future flood forecasting.

## 5.8 Stage 5 Completion Criteria

Stage 5 is complete when:

- Both machine-learning models have been evaluated.
- Appropriate metrics have been calculated.
- Results have been compared.
- Important trade-offs have been explained.
- The final model has been selected using evaluation evidence.
- Prediction limitations have been documented.
- The final prediction outputs are ready for presentation.

---

# STAGE 6 — FINAL RESULTS AND PRESENTATION

## 6.1 Objective

Consolidate the completed analysis into a clean, reproducible, presentation-ready notebook and set of results.

This stage organizes the evidence produced by the earlier stages rather than inventing new findings.

## 6.2 Final Prediction Grid

The project generates predictions across 946,183 terrestrial grid cells.

The reported Flood Susceptibility Index values range from approximately 0.000 to 0.999, with a mean of approximately 0.142.

These values represent model-derived susceptibility scores for the study domain. They should not automatically be interpreted as calibrated probabilities of future flood occurrence.

## 6.3 Susceptibility Classification

The continuous FSI is divided into five classes:

| Class | FSI Range |
|---|---|
| Very Low | 0.0 ≤ FSI < 0.2 |
| Low | 0.2 ≤ FSI < 0.4 |
| Moderate | 0.4 ≤ FSI < 0.6 |
| High | 0.6 ≤ FSI < 0.8 |
| Very High | 0.8 ≤ FSI ≤ 1.0 |

The reported distribution is:

| Susceptibility Class | Number of Cells | Approximate Share |
|---|---:|---:|
| Very Low | 784,176 | 82.9% |
| Low | 18,827 | 2.0% |
| Moderate | 14,973 | 1.6% |
| High | 17,820 | 1.9% |
| Very High | 110,387 | 11.7% |

The reported class counts and percentages should be verified against the final raster or summary table before publication.

## 6.4 Final Visualizations

The final notebook should retain the most informative outputs, including:

- Study-area map.
- Environmental predictor visualizations.
- Target distribution.
- Model evaluation tables.
- Confusion matrices.
- ROC and Precision-Recall curves where available.
- Model comparison.
- Flood-susceptibility map.
- Supporting prediction summaries.

Every important visualization should have an appropriate title, legend, labels, and interpretation.

## 6.5 Final Findings

The final presentation should explain:

- What the data revealed about the study area.
- How the environmental variables were combined.
- How the models performed.
- Why Logistic Regression was selected.
- What the susceptibility map shows.
- Which limitations affect interpretation.
- How the outputs may support geographic assessment and planning.

The findings must distinguish model-derived results from independently verified real-world flood conditions.

## 6.6 Expected Map Outputs

The project is expected to organize its principal map outputs under `outputs/maps/`, including:

- `flood_susceptibility_map.png`
- `flood_susceptibility.tif`
- `flood_susceptibility_interactive.html`

These filenames should be retained if they match the actual files generated by the project. If the final repository uses different filenames, the documentation should be updated to match the repository.

## 6.7 Final Notebook Quality Check

Before final submission:

- Run the notebook from beginning to end.
- Preserve important outputs.
- Remove unnecessary development code.
- Confirm that metric values match the evaluation outputs.
- Ensure that plots are labelled.
- Document dataset sources.
- Explain significant decisions.
- Document findings and limitations.
- Include appropriate references.
- Check that the notebook and report agree.

## 6.8 Stage 6 Completion Criteria

Stage 6 is complete when the notebook, tables, metrics, predictions, maps, findings, and limitations are organized into a reproducible and presentation-ready form.

---

# STAGE 7 — DECISION DOCUMENTATION

## 7.1 Objective

Maintain a clear record of important project decisions, the evidence supporting them, alternatives considered, and limitations that remain.

Decision documentation should be continuous across the project rather than postponed until the end.

## 7.2 Decisions to Document

### Project Definition

Record why the flood-susceptibility problem, study area, scope, and intended outputs were selected.

### Dataset Selection

Record:

- Why each dataset was selected.
- Why its geographic coverage is appropriate.
- How it supports the project objective.
- Which alternatives were considered.
- Which limitations remain.

### Target Construction

Explain:

- How the flood-related target was constructed.
- What the target represents.
- Why it is suitable for the selected analysis.
- What it does not represent.

### Feature Selection

Explain why elevation, slope, rainfall, river distance, and the permeability index were used.

Document any unavailable, excluded, redundant, or unreliable features.

### Data Cleaning

Explain important decisions concerning missing values, duplicates, invalid data, geometry, raster alignment, and irrelevant fields.

### Preprocessing

Document coordinate transformations, grid alignment, feature preparation, scaling where applicable, and safeguards against target leakage.

### Spatial Split

Explain why spatial separation was used and acknowledge the geographic differences between the training and testing regions.

### Model Selection

Document why Logistic Regression and Random Forest were considered suitable and what the baseline contributes to the comparison.

### Evaluation

Explain why the chosen metrics are appropriate and why accuracy alone is insufficient.

### Final Model Selection

Record the actual evidence supporting the final model choice, including trade-offs between recall, precision, F1-score, ROC-AUC, and Precision-Recall AUC.

## 7.3 Decision Record Format

Use a consistent format such as:

**Decision:** What was selected or changed?

**Reason:** Why was the decision necessary?

**Evidence:** What data, evaluation, or technical information supported it?

**Alternatives:** What reasonable alternatives were considered?

**Limitations:** What uncertainty remains?

## 7.4 Documentation Standards

Decision records must be:

- Evidence-based.
- Relevant to the project objective.
- Consistent with the actual notebook.
- Clear and reproducible.
- Honest about uncertainty.

Do not claim that an alternative was tested if it was not actually tested.

Do not invent retrospective explanations that misrepresent the decisions made during development.

## 7.5 Stage 7 Output

A continuous decision record that demonstrates how the project developed, why major choices were made, and what limitations affect the conclusions.

---

# STAGE 8 — FINAL PROJECT REPORT

## 8.1 Objective

Produce the final project report documenting the completed flood-susceptibility workflow, results, challenges, lessons, and limitations.

The target length is approximately 3,000 words, subject to the course requirements.

## 8.2 Required Report Structure

The report must use the required structure:

1. Introduction
2. Dataset
3. Data Cleaning
4. EDA
5. Preprocessing
6. Modelling
7. Results/Evaluation
8. Challenges/Lessons
9. Conclusion
10. References

## 8.3 Introduction

Explain:

- The flood-related problem.
- The relevance of flood susceptibility analysis.
- The project goal and objectives.
- The Lokoja and Lower Niger Basin study area.
- The role of GIS and machine learning.
- The workflow used.

## 8.4 Dataset

Document:

- Dataset names and sources.
- Geographic coverage.
- Important environmental variables.
- Spatial reference system.
- Spatial resolution.
- Target construction.
- Dataset suitability.
- Relevant data limitations.

## 8.5 Data Cleaning

Describe the cleaning and validation actually performed.

Explain important decisions and their effects on the data.

Do not describe procedures that were not carried out.

## 8.6 EDA

Present the important environmental and geographic patterns discovered during analysis.

Explain how the findings influenced feature preparation, modelling, or interpretation.

## 8.7 Preprocessing

Document:

- Spatial alignment.
- Feature preparation.
- Feature engineering.
- Handling of invalid or missing values where applicable.
- Train/test separation.
- Leakage prevention.

The report must agree with the actual implementation.

## 8.8 Modelling

Describe the baseline, Balanced Logistic Regression, and Balanced Random Forest.

Explain their roles, relevant parameters, and reasons for inclusion.

## 8.9 Results/Evaluation

Report the actual evaluation metrics.

Explain the model comparison, important trade-offs, final model selection, susceptibility classes, and geographic results.

Discuss the limitations of interpreting the model outputs.

## 8.10 Challenges/Lessons

Discuss the real challenges encountered, such as:

- Integration of multiple geospatial datasets.
- Differences in native spatial resolutions.
- Class imbalance.
- Spatial differences between training and testing areas.
- Limitations of historical flood-extent information.
- Model interpretation and evaluation.

Only include challenges that accurately reflect the work performed.

## 8.11 Conclusion

Summarize:

- The problem addressed.
- The data and methods used.
- The final model.
- The principal findings.
- The contribution of the susceptibility map.
- Important limitations.
- What the project demonstrates.

Do not introduce new results in the conclusion.

## 8.12 References

Include sources actually used, including relevant data providers, technical documentation, research references, and other resources required by the course.

References must be verifiable and must not be fabricated.

## 8.13 Report and Notebook Consistency

Before submission, verify consistency between the report and notebook for:

- Dataset sources.
- Study area.
- Predictor variables.
- Target construction.
- Cleaning decisions.
- Preprocessing.
- Model configurations.
- Evaluation metrics.
- Final model.
- Susceptibility classes.
- Findings.
- Limitations.

## 8.14 Stage 8 Completion Criteria

The report is complete when it follows the required structure, documents the actual project, contains appropriate references, and agrees with the final notebook.

---

# STAGE 9 — OPTIONAL WEB DEMONSTRATION

## 9.1 Objective

Develop a lightweight web demonstration only if sufficient time remains after completing the required notebook, analysis, evaluation, and report.

The web interface is optional and must not replace the core capstone deliverables.

## 9.2 Required Work Comes First

Before starting the demo, ensure that the following are complete:

- Dataset preparation.
- Data cleaning.
- EDA.
- Preprocessing.
- Modelling.
- Evaluation.
- Model comparison.
- Final model selection.
- Final predictions.
- Geospatial visualization.
- Notebook documentation.
- Final report.

## 9.3 Possible Interface Features

If implemented, the demo may:

- Display the final flood-susceptibility map.
- Show selected environmental variables.
- Display existing model results.
- Demonstrate supported predictions.
- Display an FSI score or susceptibility class.
- Provide a simple explanation of the result.

The interface should remain simple and presentation-focused.

## 9.4 Model Integration

The web demo must use the final evaluated model and the correct preprocessing workflow.

It must not:

- Substitute a different model without documentation.
- Use different feature definitions.
- Bypass preprocessing.
- Fabricate predictions.
- Claim real-time forecasting capability without a validated forecasting system.

Any probability or score displayed must be described accurately.

## 9.5 Time Constraint

If developing the web interface threatens the completion or quality of the core deliverables, stop web development.

The notebook, preserved outputs, and report take priority.

## 9.6 Stage 9 Completion Criteria

If implemented, the interface must use the correct model and preprocessing, display results accurately, and avoid unsupported claims.

If time is insufficient, no web interface is required.

---

# FINAL PROJECT EXECUTION STRUCTURE

The complete project workflow is organized as follows:

**STAGE 1 — PROJECT DEFINITION**  
Define the problem, study area, goal, objectives, scope, inputs, outputs, and boundaries.

↓

**STAGE 2 — DATASET SELECTION & VALIDATION**  
Identify data sources, validate geographic coverage, inspect variables, and establish the flood-related target.

↓

**STAGE 3 — COLAB NOTEBOOK & DATA PIPELINE**  
Acquire, clean, explore, preprocess, align, and prepare the environmental and geospatial data.

↓

**STAGE 4 — MODELLING**  
Train the baseline, Balanced Logistic Regression, and Balanced Random Forest.

↓

**STAGE 5 — EVALUATION & MODEL COMPARISON**  
Evaluate model performance, examine trade-offs, select the final model, and document limitations.

↓

**STAGE 6 — FINAL RESULTS & PRESENTATION**  
Organize the final notebook, evaluation tables, predictions, maps, findings, and limitations.

↓

**STAGE 7 — DECISION DOCUMENTATION**  
Maintain an evidence-based record of important decisions throughout the project.

↓

**STAGE 8 — FINAL REPORT**  
Prepare the report using the required course structure.

↓

**STAGE 9 — OPTIONAL WEB DEMO**  
Build a simple demonstration interface only if sufficient time remains.

---

# CORE PROJECT DELIVERABLES

The principal project deliverables are:

1. Reproducible Colab/Jupyter notebook.
2. Documented data sources and processing workflow.
3. Model training and evaluation results.
4. Model comparison and final model selection.
5. Flood-susceptibility prediction outputs.
6. Geographic visualizations.
7. Findings and limitations.
8. Final project report.
9. References and supporting documentation.

A web interface is an optional extension.

---

# PROJECT QUALITY PRINCIPLES

The project should demonstrate:

- Appropriate use of real-world data.
- Clear understanding of the machine-learning workflow.
- Reproducible processing.
- Justified modelling decisions.
- Appropriate evaluation.
- Responsible interpretation of predictions.
- Transparent documentation.
- Recognition of dataset and model limitations.
- Consistency between code, results, maps, and report.

The project is evaluated on the quality and understanding of the complete workflow, not simply on obtaining the highest possible accuracy.

---

# IMPORTANT SCIENTIFIC LIMITATIONS

The final documentation must acknowledge the following:

1. The analysis is based on a specific historical flood reference event.
2. The reference flood-extent data does not represent every flood event.
3. CHIRPS rainfall data has a much coarser native resolution than the 90 m analysis grid.
4. Resampling coarse data does not create additional independent observations.
5. The permeability index is a land-cover-derived proxy, not direct soil-permeability measurement.
6. The spatial test region differs from the training regions.
7. The Flood Susceptibility Index is a model-derived score, not automatically a calibrated probability of future flooding.
8. The model does not directly predict flood onset, depth, duration, or timing.
9. The outputs should not be presented as a validated real-time flood-warning service.
10. Generalization beyond the study area and reference-data conditions requires further validation.

These limitations are necessary for responsible scientific communication.

---

# REPRODUCIBILITY AND VERSION CONTROL

The GitHub repository should preserve the actual project files and supporting documentation.

Recommended documentation includes:

- `README.md` — Project overview, setup, results, and repository navigation.
- `PROJECT_EXECUTION_PLAN.md` — The project workflow and execution requirements.
- `FINAL_EXPERIMENT_MANIFEST.md` — Technical experiment configuration and reported results.
- `SCIENTIFIC_CORRECTION_LOG.md` — Corrections, verified discrepancies, and scientific clarifications.
- `requirements.txt` — Dependencies, using versions verified against the actual environment.
- `decision_log.csv` — Important project decisions, where maintained.
- `limitations.csv` — Documented limitations, where maintained.

Recommended project outputs include the final notebook, relevant source code, and the map files actually generated by the workflow.

Do not include placeholder files as if they contain completed outputs. Document unavailable deliverables honestly.

---

# FINAL DEFINITION OF DONE

The project is ready for portfolio presentation when:

- The repository has a clear project description.
- The study area and data sources are documented.
- The notebook is available and understandable.
- The main processing workflow is documented.
- Model configurations are recorded.
- Evaluation metrics match the authoritative results.
- The final model selection is justified.
- Susceptibility classes and maps are explained.
- Important scientific limitations are visible.
- The report agrees with the notebook.
- References are included.
- The repository distinguishes group work from individual contributions.
- No results, sources, or capabilities have been fabricated.

**Final principle:** The purpose of this project is to demonstrate how AI/ML and GIS can be combined to analyse flood susceptibility responsibly, reproducibly, and transparently.
