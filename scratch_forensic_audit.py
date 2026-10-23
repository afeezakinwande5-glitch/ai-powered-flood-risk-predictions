import os, sys, json, rasterio, geopandas as gpd, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss
)
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

sys.stdout.reconfigure(encoding='utf-8')

# Load all 90m rasters
data_dir = 'data'
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

mid_r = dem.shape[0] // 2
mid_c = dem.shape[1] // 2
q = np.zeros(len(df), dtype=int)
q[(df['row'] < mid_r) & (df['col'] < mid_c)] = 0
q[(df['row'] < mid_r) & (df['col'] >= mid_c)] = 1
q[(df['row'] >= mid_r) & (df['col'] < mid_c)] = 2
q[(df['row'] >= mid_r) & (df['col'] >= mid_c)] = 3
df['quadrant'] = q

train_df = df[df['quadrant'] != 3]
test_df = df[df['quadrant'] == 3]

features = ['elevation', 'slope', 'rainfall', 'dist_river', 'permeability']
X_train, y_train = train_df[features], train_df['flood']
X_test, y_test = test_df[features], test_df['flood']

print('=== 5. LOGISTIC REGRESSION: UNIVARIATE VS MULTIVARIABLE & VIF ===')
# Univariate Logistic Regressions
print('Univariate Logistic Regression (Standardized):')
for feat in features:
    scaler = StandardScaler()
    X_u = scaler.fit_transform(train_df[[feat]])
    lr_u = LogisticRegression(class_weight='balanced', random_state=42)
    lr_u.fit(X_u, y_train)
    coef = lr_u.coef_[0][0]
    corr = train_df[[feat, 'flood']].corr().iloc[0, 1]
    print(f'  {feat:15}: Bivariate r = {corr:+.4f} | Univariate LogReg Coef = {coef:+.4f}')

# Multivariable Logistic Regression
lr_full = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
lr_full.fit(X_train, y_train)
print('\nMultivariable Logistic Regression Coefs (Full Model):')
for feat, c in zip(features, lr_full.named_steps['lr'].coef_[0]):
    print(f'  {feat:15}: {c:+.4f}')

# Controlled removals
print('\nControlled Feature Removal Experiment (Multivariable Coefs):')
for omit_feat in ['none'] + features:
    sub_feats = [f for f in features if f != omit_feat]
    lr_sub = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    lr_sub.fit(train_df[sub_feats], y_train)
    coefs = dict(zip(sub_feats, lr_sub.named_steps['lr'].coef_[0]))
    coef_str = ', '.join([f'{f}: {coefs[f]:+.2f}' for f in sub_feats])
    print(f'  Omit {omit_feat:12} -> {coef_str}')

# VIF Calculation (on standardized train features)
scaler = StandardScaler()
X_tr_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
vifs = {}
for feat in features:
    # Regress feat against other features
    other_f = [f for f in features if f != feat]
    from sklearn.linear_model import LinearRegression
    lin = LinearRegression().fit(X_tr_sc[other_f], X_tr_sc[feat])
    r2 = lin.score(X_tr_sc[other_f], X_tr_sc[feat])
    vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    vifs[feat] = vif
print('\nVariance Inflation Factors (VIF):')
for f, v in vifs.items():
    print(f'  {f:15}: {v:.3f}')

print('\n=== 8. CALIBRATION CHECK ===')
y_prob_lr = lr_full.predict_proba(X_test)[:, 1]
brier_lr = brier_score_loss(y_test, y_prob_lr)
print(f'Logistic Regression Brier Score: {brier_lr:.4f}')
# Check calibration slope & intercept (Cox calibration)
from sklearn.linear_model import LogisticRegression as CalLogReg
cal_eval = CalLogReg().fit(y_prob_lr.reshape(-1, 1), y_test)
print(f'Calibration Logistic Fit: intercept = {cal_eval.intercept_[0]:.4f}, slope = {cal_eval.coef_[0][0]:.4f}')

print('\n=== 9 & 10. PR-AUC AND BASELINE PREVALENCE ===')
rf_model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=50, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

base_prev = y_test.mean()
pr_lr = average_precision_score(y_test, y_prob_lr)
pr_rf = average_precision_score(y_test, y_prob_rf)

print(f'Test Positive Prevalence: {base_prev:.4f} ({y_test.sum()}/{len(y_test)})')
print(f'Logistic Regression PR-AUC: {pr_lr:.4f}')
print(f'Random Forest PR-AUC      : {pr_rf:.4f}')
print(f'LR PR-AUC / Prevalence    : {pr_lr / base_prev:.2f}x')
print(f'RF PR-AUC / Prevalence    : {pr_rf / base_prev:.2f}x')

print('\n=== 13. SPATIAL HOLDOUT BOUNDARIES & AUTOCORRELATION ===')
print(f'Master Grid Shape: rows={dem.shape[0]}, cols={dem.shape[1]}')
print(f'Split Boundary   : row={mid_r} (490), col={mid_c} (493)')
print('Train rows/cols: NW (r<490, c<493), NE (r<490, c>=493), SW (r>=490, c<493)')
print('Test rows/cols : SE (r>=490, c>=493)')
print('Are there duplicate (row, col) coordinates across train and test?')
train_coords = set(zip(train_df['row'], train_df['col']))
test_coords = set(zip(test_df['row'], test_df['col']))
overlap = train_coords.intersection(test_coords)
print(f'Coordinate Overlap count: {len(overlap)}')

print('\n=== 14. DOMAIN SHIFT RECALCULATION ===')
for feat in features:
    tr_m, tr_s, tr_med = train_df[feat].mean(), train_df[feat].std(), train_df[feat].median()
    te_m, te_s, te_med = test_df[feat].mean(), test_df[feat].std(), test_df[feat].median()
    print(f'{feat:12} | Train: mean={tr_m:7.2f}, std={tr_s:6.2f}, med={tr_med:7.2f} | Test: mean={te_m:7.2f}, std={te_s:6.2f}, med={te_med:7.2f}')
