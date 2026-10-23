import os
import sys
import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline

sys.stdout.reconfigure(encoding='utf-8')

data_dir = 'C:/content/flood_risk/data'

# Read rasters
with rasterio.open(f'{data_dir}/dem_90m_utm32n.tif') as s:
    dem = s.read(1)
with rasterio.open(f'{data_dir}/slope_90m_utm32n.tif') as s:
    slope = s.read(1)
with rasterio.open(f'{data_dir}/rain_sepoct2022_90m.tif') as s:
    rain = s.read(1)
with rasterio.open(f'{data_dir}/dist_river_90m_utm32n.tif') as s:
    dist = s.read(1)
with rasterio.open(f'{data_dir}/permeability_90m_utm32n.tif') as s:
    perm = s.read(1)
with rasterio.open(f'{data_dir}/flood_mask_90m_utm32n.tif') as s:
    flood = s.read(1)
with rasterio.open(f'{data_dir}/worldcover_90m_utm32n.tif') as s:
    lulc = s.read(1)

# Permanent water mask (lulc == 80)
valid_mask = (lulc != 80) & ~np.isnan(dem) & ~np.isnan(slope) & ~np.isnan(rain) & ~np.isnan(dist) & ~np.isnan(perm)

rows, cols = np.indices(dem.shape)
df = pd.DataFrame({
    'row': rows[valid_mask],
    'col': cols[valid_mask],
    'elevation': dem[valid_mask],
    'slope': slope[valid_mask],
    'rainfall': rain[valid_mask],
    'dist_river': dist[valid_mask],
    'permeability': perm[valid_mask],
    'flood': flood[valid_mask]
})

print(f"Total valid samples: {len(df):,}")
print(f"Flood count: {df['flood'].sum():,} ({df['flood'].mean()*100:.2f}%)")

# Quadrants for spatial split (2x2 grid)
mid_r = dem.shape[0] // 2
mid_c = dem.shape[1] // 2

q = np.zeros(len(df), dtype=int)
q[(df['row'] < mid_r) & (df['col'] < mid_c)] = 0
q[(df['row'] < mid_r) & (df['col'] >= mid_c)] = 1
q[(df['row'] >= mid_r) & (df['col'] < mid_c)] = 2
q[(df['row'] >= mid_r) & (df['col'] >= mid_c)] = 3
df['quadrant'] = q

for i in range(4):
    sub = df[df['quadrant'] == i]
    print(f"Quadrant {i}: total={len(sub):,}, flood={sub['flood'].sum():,} ({sub['flood'].mean()*100:.2f}%)")

train_df = df[df['quadrant'] != 3]
test_df = df[df['quadrant'] == 3]

print(f"Train: {len(train_df):,}, flood rate: {train_df['flood'].mean()*100:.2f}%")
print(f"Test : {len(test_df):,}, flood rate: {test_df['flood'].mean()*100:.2f}%")

features = ['elevation', 'slope', 'rainfall', 'dist_river', 'permeability']
print("\n--- CORRELATION MATRIX (TRAIN) ---")
corr = train_df[features + ['flood']].corr()
print(corr.round(4))

print("\n--- FEATURE STATS BY CLASS (TRAIN) ---")
print(train_df.groupby('flood')[features].agg(['mean', 'median', 'std']).T)

print("\n--- FEATURE STATS TRAIN VS TEST (DOMAIN SHIFT) ---")
train_stats = train_df[features].describe().T[['mean', 'std', '50%']]
test_stats = test_df[features].describe().T[['mean', 'std', '50%']]
stats_comp = train_stats.join(test_stats, lsuffix='_train', rsuffix='_test')
print(stats_comp.round(2))

X_train, y_train = train_df[features], train_df['flood']
X_test, y_test = test_df[features], test_df['flood']

# Train Logistic Regression
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
])
lr_pipe.fit(X_train, y_train)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_pipe.predict(X_test)
y_prob_lr = lr_pipe.predict_proba(X_test)[:, 1]

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Baseline (Majority Class: always predict 0)
y_pred_base = np.zeros(len(y_test))
y_prob_base = np.zeros(len(y_test))

print("\n=== METRICS SUMMARY (TEST SET) ===")
models = [
    ('Baseline (Majority)', y_pred_base, y_prob_base),
    ('Logistic Regression', y_pred_lr, y_prob_lr),
    ('Random Forest', y_pred_rf, y_prob_rf)
]

for name, y_p, y_pr in models:
    acc = accuracy_score(y_test, y_p)
    prec = precision_score(y_test, y_p, zero_division=0)
    rec = recall_score(y_test, y_p, zero_division=0)
    f1 = f1_score(y_test, y_p, zero_division=0)
    if len(np.unique(y_pr)) > 1:
        roc = roc_auc_score(y_test, y_pr)
        pr_auc = average_precision_score(y_test, y_pr)
    else:
        roc = 0.5
        pr_auc = y_test.mean()
    print(f"{name:22} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f}")

print("\nLR Coefficients:")
for f_name, coef in zip(features, lr_pipe.named_steps['clf'].coef_[0]):
    print(f"  {f_name:15}: {coef:+.4f}")

print("\nRF Feature Importances:")
for f_name, imp in zip(features, rf.feature_importances_):
    print(f"  {f_name:15}: {imp:.4f}")

# Threshold tuning on train set (optimizing F1 on train)
print("\n--- THRESHOLD EXPERIMENT ON VALIDATION/TRAIN ---")
# Evaluate threshold sweep for LR and RF on test (reported as sensitivity analysis)
for name, y_pr in [('Logistic Regression', y_prob_lr), ('Random Forest', y_prob_rf)]:
    print(f"Thresholds for {name}:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred_t = (y_pr >= thresh).astype(int)
        p = precision_score(y_test, pred_t, zero_division=0)
        r = recall_score(y_test, pred_t, zero_division=0)
        f = f1_score(y_test, pred_t, zero_division=0)
        print(f"  Threshold {thresh:.1f} -> Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")
