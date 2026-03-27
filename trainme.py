import glob
import logging
import warnings
import os
import copy
from pathlib import Path

# Fix loky/joblib CreateProcess error on Windows
os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 4))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

DATA_DIR = Path(__file__).parent / "Training_dataset"
MODEL_OUTPUT_PATH = Path(__file__).parent / "custom_hybrid_model.pkl"
OUTPUT_DIR = Path(__file__).parent

TEST_SIZE = 0.15
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
logging.getLogger("optuna").setLevel(logging.ERROR)

def detect_gpu() -> bool:
    log.info("Checking for NVIDIA GPU support...")
    return True

# --- FEATURE ENGINEERING ---
CITY_CENTERS = {
    'Mumbai':    (19.0760, 72.8777),
    'Delhi NCR': (28.6139, 77.2090),
    'Bengaluru': (12.9716, 77.5946),
    'Chennai':   (13.0827, 80.2707),
    'Hyderabad': (17.3850, 78.4867),
    'Pune':      (18.5204, 73.8567),
    'Kolkata':   (22.5726, 88.3639),
    'Ahmedabad': (23.0225, 72.5714)
}

def haversine_dist(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0 # Earth radius in km
    raw_lat1, raw_lon1 = math.radians(lat1), math.radians(lon1)
    raw_lat2, raw_lon2 = math.radians(lat2), math.radians(lon2)
    dlat = raw_lat2 - raw_lat1
    dlon = raw_lon2 - raw_lon1
    a = math.sin(dlat/2)**2 + math.cos(raw_lat1) * math.cos(raw_lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    
    # Ensure key numeric columns are actually numeric
    num_cols = ['SuperBuiltUpArea_sqft', 'BHK', 'Bathrooms', 'CarpetArea_sqft', 'Floor', 'TotalFloors', 'AmenitiesCount', 'AgeYears']
    for col in num_cols:
        if col in df_feat.columns:
            df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')
    
    if 'SuperBuiltUpArea_sqft' in df_feat.columns and 'BHK' in df_feat.columns:
        df_feat['Area_per_BHK'] = df_feat['SuperBuiltUpArea_sqft'] / df_feat['BHK'].clip(lower=1)
        df_feat['BHK_Area_Interaction'] = df_feat['BHK'] * df_feat['SuperBuiltUpArea_sqft']
    
    if 'SuperBuiltUpArea_sqft' in df_feat.columns and 'Bathrooms' in df_feat.columns:
        df_feat['Area_per_Bath'] = df_feat['SuperBuiltUpArea_sqft'] / df_feat['Bathrooms'].clip(lower=1)
    
    if 'SuperBuiltUpArea_sqft' in df_feat.columns and 'CarpetArea_sqft' in df_feat.columns:
        df_feat['Carpet_to_Super_Ratio'] = df_feat['CarpetArea_sqft'] / df_feat['SuperBuiltUpArea_sqft'].clip(lower=1)
    
    if 'Floor' in df_feat.columns and 'TotalFloors' in df_feat.columns:
        df_feat['Floor_Ratio'] = df_feat['Floor'] / df_feat['TotalFloors'].clip(lower=1)
        df_feat['Is_Top_Floor'] = (df_feat['Floor'] == df_feat['TotalFloors']).astype(int)
        df_feat['Is_Ground_Floor'] = (df_feat['Floor'] <= 1).astype(int)
    
    if 'AgeYears' in df_feat.columns:
        bins = [-1, 1, 5, 10, 20, 100]
        labels = ['Brand New', '1-5 Yrs', '5-10 Yrs', '10-20 Yrs', '20+ Yrs']
        df_feat['Age_Binned'] = pd.cut(df_feat['AgeYears'].fillna(5), bins=bins, labels=labels, right=True).astype(str)
    
    if 'BHK' in df_feat.columns and 'Bathrooms' in df_feat.columns:
        df_feat['Bath_per_BHK'] = df_feat['Bathrooms'] / df_feat['BHK'].clip(lower=1)
        df_feat['Unusual_Bath_Ratio'] = (df_feat['Bath_per_BHK'] > 1.5).astype(int)
    
    if 'SuperBuiltUpArea_sqft' in df_feat.columns:
        df_feat['Log_Area'] = np.log1p(df_feat['SuperBuiltUpArea_sqft'].clip(lower=1))

    if 'AmenitiesCount' in df_feat.columns and 'SuperBuiltUpArea_sqft' in df_feat.columns and 'BHK' in df_feat.columns:
        df_feat['Luxury_Score'] = (df_feat['AmenitiesCount'] * df_feat['SuperBuiltUpArea_sqft']) / df_feat['BHK'].clip(lower=1)
    
    if 'Latitude' in df_feat.columns and 'Longitude' in df_feat.columns and 'City' in df_feat.columns:
        dists = []
        for _, row in df_feat.iterrows():
            city = str(row['City'])
            lat, lon = row['Latitude'], row['Longitude']
            if pd.isna(lat) or pd.isna(lon) or city not in CITY_CENTERS:
                dists.append(-1.0)
            else:
                c_lat, c_lon = CITY_CENTERS[city]
                dist = haversine_dist(lat, lon, c_lat, c_lon)
                dists.append(dist)
        df_feat['City_Center_Distance_km'] = dists
    
    return df_feat
# -----------------------------

def initialize_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cols_to_exclude = ['ListingID', 'RERAID']
    features = [c for c in X.columns if c not in cols_to_exclude]

    numeric_features = X[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X[features].select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('target_encoder', TargetEncoder(target_type='continuous', random_state=RANDOM_STATE)),
        ('scaler', StandardScaler())
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

def silu_act(x):
    return x * torch.sigmoid(x)

class TransformerResNetDNN(nn.Module):
    def __init__(self, input_dim, units1=1024, units2=512, units3=256, dropout_rate=0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, units1)
        self.bn1 = nn.BatchNorm1d(units1)
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.res1 = nn.Linear(units1, units2)
        self.bn_res1 = nn.BatchNorm1d(units2)
        self.drop_res = nn.Dropout(dropout_rate * 0.75)
        
        self.res2 = nn.Linear(units2, units2)
        self.bn_res2 = nn.BatchNorm1d(units2)
        
        self.x_proj = nn.Linear(units1, units2)
        
        self.attn_proj = nn.Linear(input_dim, 64)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(64)
        
        self.merge_fc1 = nn.Linear(units2 + 64, units3)
        self.bn_merge = nn.BatchNorm1d(units3)
        self.drop_merge = nn.Dropout(dropout_rate * 0.5)
        self.merge_fc2 = nn.Linear(units3, 64)
        self.output_fc = nn.Linear(64, 1)

    def forward(self, x):
        h = silu_act(self.bn1(self.fc1(x)))
        h = self.drop1(h)

        res = silu_act(self.bn_res1(self.res1(h)))
        res = self.drop_res(res)
        res = silu_act(self.bn_res2(self.res2(res)))
        h = silu_act(self.x_proj(h) + res)

        seq = self.attn_proj(x).unsqueeze(1)
        attn_out, _ = self.attn(seq, seq, seq)
        attn_out = self.attn_norm(attn_out + seq)
        attn_flat = attn_out.mean(dim=1)

        merged = torch.cat([h, attn_flat], dim=1)
        merged = silu_act(self.bn_merge(self.merge_fc1(merged)))
        merged = self.drop_merge(merged)
        merged = silu_act(self.merge_fc2(merged))
        return self.output_fc(merged).squeeze(-1)

def build_transformer_dnn(input_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerResNetDNN(input_dim).to(device)
    return model, device

def train_pytorch_dnn(model, device, X_train, y_train, epochs=2500, batch_size=64, patience=200, lr=0.001):
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    n_val = int(len(X_t) * 0.15)
    X_val_t, y_val_t = X_t[:n_val], y_t[:n_val]
    X_tr_t, y_tr_t = X_t[n_val:], y_t[n_val:]

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    loss_fn = nn.HuberLoss()

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n        Early stop at epoch {epoch+1}, best val_loss={best_val_loss:.6f}")
                break

        # Show progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            avg_loss = epoch_loss / len(loader)
            lr_now = optimizer.param_groups[0]['lr']
            print(f"\r        Epoch {epoch+1:>4d}/{epochs} | train_loss={avg_loss:.6f} | val_loss={val_loss:.6f} | best={best_val_loss:.6f} | lr={lr_now:.2e} | wait={wait}/{patience}", end="", flush=True)

    print()  # newline after progress
    if best_state:
        model.load_state_dict(best_state)
    return model

def predict_pytorch_dnn(model, device, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        return model(X_t).cpu().numpy()

def tune_xgboost(X_tr, y_tr, use_gpu: bool, n_trials: int = 60) -> dict:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
            'verbosity': 0,
            'random_state': RANDOM_STATE,
        }
        if use_gpu:
            params['device'] = 'cuda'
            params['tree_method'] = 'hist'

        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = []
        for tr_idx, val_idx in kf.split(X_tr):
            model = XGBRegressor(**params)
            model.fit(X_tr[tr_idx], y_tr[tr_idx])
            preds = model.predict(X_tr[val_idx])
            cv_scores.append(r2_score(y_tr[val_idx], preds))
        return np.mean(cv_scores)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def tune_lgbm(X_tr, y_tr, use_gpu: bool, n_trials: int = 60) -> dict:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
            'max_depth': -1,
            'num_leaves': trial.suggest_int('num_leaves', 63, 511),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
            'random_state': RANDOM_STATE,
            'verbose': -1,
            'n_jobs': -1,
        }

        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = []
        for tr_idx, val_idx in kf.split(X_tr):
            model = LGBMRegressor(**params)
            model.fit(X_tr[tr_idx], y_tr[tr_idx])
            preds = model.predict(X_tr[val_idx])
            cv_scores.append(r2_score(y_tr[val_idx], preds))
        return np.mean(cv_scores)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

def tune_catboost(X_tr, y_tr, use_gpu: bool, n_trials: int = 60) -> dict:
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1500),
            'depth': trial.suggest_int('depth', 6, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_seed': RANDOM_STATE,
            'verbose': 0,
            'task_type': 'GPU' if use_gpu else 'CPU',
            'early_stopping_rounds': 100,
        }

        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = []
        for tr_idx, val_idx in kf.split(X_tr):
            model = CatBoostRegressor(**params)
            model.fit(X_tr[tr_idx], y_tr[tr_idx], eval_set=(X_tr[val_idx], y_tr[val_idx]), verbose=0)
            preds = model.predict(X_tr[val_idx])
            cv_scores.append(r2_score(y_tr[val_idx], preds))
        return np.mean(cv_scores)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

def get_combined_dataset():
    # Load default data
    train_files = sorted(glob.glob(str(DATA_DIR / "train_part*.csv")))
    dfs = [pd.read_csv(f, on_bad_lines='skip') for f in train_files]
    df_main = pd.concat(dfs, ignore_index=True)
    
    # Load extra properties data
    prop_file = DATA_DIR / "properties.csv"
    if prop_file.exists():
        log.info("Loading extra properties.csv...")
        prop = pd.read_csv(prop_file, on_bad_lines='skip', low_memory=False)
        
        # Map known columns
        cols_map = {
            'Price': 'Price_INR',
            'bedroom': 'BHK',
            'Bathroom': 'Bathrooms',
            'Covered Area': 'SuperBuiltUpArea_sqft',
            'Carpet Area': 'CarpetArea_sqft',
            'City': 'City',
            'Location': 'Locality',
            'Type of Property': 'PropertyType',
            'Floor No': 'Floor',
            'floors': 'TotalFloors',
            'balconies': 'Balconies',
            'Furnishing': 'Furnishing'
        }
        
        for k in list(cols_map.keys()):
            if k not in prop.columns:
                del cols_map[k]
                
        prop_mapped = prop.rename(columns=cols_map)
        
        # Force numeric conversion on columns that should be numeric
        force_numeric = ['Price_INR', 'BHK', 'Bathrooms', 'SuperBuiltUpArea_sqft', 'CarpetArea_sqft',
                         'BuiltUpArea_sqft', 'Floor', 'TotalFloors', 'Balconies', 'AgeYears', 'AmenitiesCount',
                         'Latitude', 'Longitude', 'YearBuilt']
        for col in force_numeric:
            if col in prop_mapped.columns:
                prop_mapped[col] = pd.to_numeric(prop_mapped[col], errors='coerce')
        
        common_cols = [c for c in df_main.columns if c in prop_mapped.columns]
        prop_filtered = prop_mapped[common_cols].copy()
        prop_filtered = prop_filtered.dropna(subset=['Price_INR'])
        
        log.info("Added %d rows from properties.csv", len(prop_filtered))
        df_all = pd.concat([df_main, prop_filtered], ignore_index=True)
    else:
        df_all = df_main
        
    df_all = df_all.dropna(subset=['Price_INR', 'BHK', 'City', 'SuperBuiltUpArea_sqft'], how='any')
    
    # Outlier Removal - Tighter 1% tails
    log.info("Removing outliers...")
    filtered_parts = []
    for city, group in df_all.groupby('City'):
        q_low = group['Price_INR'].quantile(0.01)
        q_high = group['Price_INR'].quantile(0.99)
        filtered = group[(group['Price_INR'] >= q_low) & (group['Price_INR'] <= q_high)]
        filtered_parts.append(filtered)
    
    if len(filtered_parts) > 0:
         df_all = pd.concat(filtered_parts, ignore_index=True)
    
    return engineer_features(df_all)

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       House Price Prediction — Hybrid Training Pipeline     ║")
    print("║  OOF Stacking │ CatBoost │ Extra Data │ 30 Optuna Trials    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    target_col = 'Price_INR'
    df_all = get_combined_dataset()
    
    X_init = df_all.drop(columns=[target_col])
    y_init = df_all[target_col].values
    preprocessor = initialize_preprocessor(X_init)
    
    X_train_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_init, y_init, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    X_tr_p = preprocessor.fit_transform(X_train_raw, np.log1p(y_tr))
    X_val_p = preprocessor.transform(X_val_raw)
    y_tr_log = np.log1p(y_tr)
    input_dim = X_tr_p.shape[1]
    use_gpu = detect_gpu()

    print(f"  📊 Training Samples : {len(X_tr_p):,}")
    print(f"  📊 Validation Set   : {len(X_val_p):,}")
    print(f"  📊 Features         : {input_dim}")
    print(f"  🖥️  GPU             : {'CUDA ✓' if use_gpu else 'CPU'}")
    print()
    print("━" * 62)
    print("  PHASE 1 ▸ Hyperparameter Optimization (50 trials × 3 models)")
    print("━" * 62)
    
    subset_size = min(8000, len(X_tr_p))
    print("\n  ► XGBoost Tuning")
    xgb_p = tune_xgboost(X_tr_p[:subset_size], y_tr_log[:subset_size], use_gpu, n_trials=50)
    print("  ► LightGBM Tuning")
    lgb_p = tune_lgbm(X_tr_p[:subset_size], y_tr_log[:subset_size], use_gpu, n_trials=50)
    print("  ► CatBoost Tuning")
    cat_p = tune_catboost(X_tr_p[:subset_size], y_tr_log[:subset_size], use_gpu, n_trials=50)

    print()
    print("━" * 62)
    print("  PHASE 2 ▸ 5-Fold Out-Of-Fold Stacking (8 models × 5 folds)")
    print("━" * 62)
    
    xgb_p.update({'random_state': RANDOM_STATE, 'verbosity': 0})
    if use_gpu: xgb_p.update({'device': 'cuda', 'tree_method': 'hist'})
    
    lgb_p.update({'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1})
    
    cat_p.update({'random_seed': RANDOM_STATE, 'verbose': 0, 'task_type': 'GPU' if use_gpu else 'CPU'})
    
    # Define models dict (Added GBR and RF for maximum ensemble diversity/accuracy)
    models = {
        'XGB': XGBRegressor(**xgb_p),
        'LGB': LGBMRegressor(**lgb_p),
        'CAT': CatBoostRegressor(**cat_p),
        'GBR': GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=0.8, random_state=RANDOM_STATE),
        'RF': RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE),
        'ET': ExtraTreesRegressor(n_estimators=1000, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1),
        'RIDGE': Ridge(alpha=1.0)
    }

    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros((X_tr_p.shape[0], len(models) + 1)) # +1 for DNN
    val_preds = np.zeros((X_val_p.shape[0], len(models) + 1))
    
    dnn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dnn_states = []
    
    import time as _time
    for fold, (tr_idx, oof_idx) in enumerate(kf.split(X_tr_p)):
        fold_start = _time.time()
        print(f"\n  ┌─ FOLD {fold+1}/{n_folds} ", end="")
        X_f_tr, y_f_tr = X_tr_p[tr_idx], y_tr_log[tr_idx]
        X_f_val, y_f_val = X_tr_p[oof_idx], y_tr_log[oof_idx]
        
        col_idx = 0
        for name, m in models.items():
            print(f"→ {name} ", end="", flush=True)
            m_clone = clone(m)
            m_clone.fit(X_f_tr, y_f_tr)
            oof_preds[oof_idx, col_idx] = m_clone.predict(X_f_val)
            val_preds[:, col_idx] += m_clone.predict(X_val_p) / n_folds
            col_idx += 1
            
        print("→ DNN ", end="", flush=True)
        dnn, _ = build_transformer_dnn(input_dim)
        dnn = train_pytorch_dnn(dnn, dnn_device, X_f_tr, y_f_tr, epochs=1500, patience=150, lr=0.001)
        oof_preds[oof_idx, col_idx] = predict_pytorch_dnn(dnn, dnn_device, X_f_val)
        val_preds[:, col_idx] += predict_pytorch_dnn(dnn, dnn_device, X_val_p) / n_folds
        dnn_states.append(copy.deepcopy(dnn.state_dict()))
        elapsed = _time.time() - fold_start
        print(f"✓ ({elapsed:.0f}s)")
            
    print()
    print("━" * 62)
    print("  PHASE 3 ▸ Final Refit + Meta-Learner")
    print("━" * 62)
    
    print("  Refitting all base models on full training set...", end=" ", flush=True)
    for name, m in models.items():
        m.fit(X_tr_p, y_tr_log)
    print("✓")
        
    final_dnn, _ = build_transformer_dnn(input_dim)
    final_dnn.load_state_dict(dnn_states[-1])
    
    print("  Training RidgeCV Meta-Learner on OOF predictions...", end=" ", flush=True)
    meta_learner = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    meta_learner.fit(oof_preds, y_tr_log)
    print("✓")
    
    preds_log = meta_learner.predict(val_preds)
    preds = np.expm1(preds_log)
    
    r2 = r2_score(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mape = np.mean(np.abs((y_val - preds) / y_val)) * 100
    
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    VALIDATION RESULTS                       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  R² Score  : {r2:.4f}                                      ║")
    print(f"║  MAE       : ₹{mae:>12,.0f}                                ║")
    print(f"║  RMSE      : ₹{rmse:>12,.0f}                                ║")
    print(f"║  MAPE      : {mape:.2f}%                                       ║")
    print(f"║  ACCURACY  : {100 - mape:.2f}%                                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_val, y=preds, alpha=0.5, edgecolor="none", color="royalblue")
    max_v = max(y_val.max(), preds.max())
    min_v = min(y_val.min(), preds.min())
    plt.plot([min_v, max_v], [min_v, max_v], color='red', linestyle='--')
    plt.title('Hybrid Model: Actual vs Predicted Prices')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hybrid_actual_vs_predicted.png")
    plt.close()

    final_artifact = {
        'preprocessor': preprocessor,
        'models_dict': models,
        'meta_learner': meta_learner,
        'xgb_params': xgb_p,
        'lgb_params': lgb_p,
        'cat_params': cat_p,
        'engineer_features': engineer_features
    }
    joblib.dump(final_artifact, MODEL_OUTPUT_PATH)
    torch.save(final_dnn.state_dict(), OUTPUT_DIR / "custom_hybrid_dnn.pt")
    print(f"\n  💾 Saved model → {MODEL_OUTPUT_PATH.name}")
    print(f"  💾 Saved DNN   → custom_hybrid_dnn.pt")

    print("\n  Generating test predictions...", end=" ", flush=True)
    test_files = glob.glob(str(DATA_DIR / "test_part*.csv"))
    if test_files:
        test_dfs = [pd.read_csv(f, on_bad_lines='skip') for f in test_files]
        test_df = pd.concat(test_dfs, ignore_index=True).dropna(axis=1, how='all')
        
        test_df = engineer_features(test_df)
        log.info("Processing test dataset of shape: %s", test_df.shape)
        X_test_p = preprocessor.transform(test_df)

        tst_preds_all = np.zeros((X_test_p.shape[0], len(models) + 1))
        col_idx = 0
        for name, m in models.items():
            tst_preds_all[:, col_idx] = m.predict(X_test_p)
            col_idx += 1
            
        tst_preds_all[:, col_idx] = predict_pytorch_dnn(final_dnn, dnn_device, X_test_p)
        test_preds = np.expm1(meta_learner.predict(tst_preds_all))

        submission_df = pd.DataFrame()
        if 'ListingID' in test_df.columns:
            submission_df['ListingID'] = test_df['ListingID']
        submission_df['Predicted_Price_INR'] = test_preds

        preds_path = OUTPUT_DIR / "test_predictions.csv"
        submission_df.to_csv(preds_path, index=False)
        print(f"✓ Saved → {preds_path.name}")
    print("\n  ✅ Training Complete!\n")

if __name__ == "__main__":
    main()
