import glob
import logging
import warnings
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     MultiHeadAttention, LayerNormalization,
                                     GlobalAveragePooling1D, Reshape)
from tensorflow.keras.models import Sequential

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

DATA_DIR = Path(__file__).parent / "Training_dataset"
MODEL_OUTPUT_PATH = Path(__file__).parent / "custom_hybrid_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def detect_gpu() -> bool:
    import subprocess
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            log.info("NVIDIA GPU detected - algorithms will use CUDA where possible.")
            return True
    except Exception:
        pass
    log.info("No NVIDIA GPU detected - running on CPU.")
    return False


def get_train_files() -> list:
    train_files = sorted(glob.glob(str(DATA_DIR / "train_part*.csv")))
    if not train_files:
        raise FileNotFoundError(f"No train_part*.csv files found in {DATA_DIR}")
    return train_files


def initialize_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cols_to_exclude = ['ListingID', 'RERAID']
    features = [c for c in X.columns if c not in cols_to_exclude]

    numeric_features = X[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X[features].select_dtypes(include=['object', 'category']).columns.tolist()

    log.info("Numeric Features (%d)", len(numeric_features))
    log.info("Categorical Features (%d)", len(categorical_features))

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


def build_transformer_dnn(input_dim: int, units1: int = 512, units2: int = 256,
                           units3: int = 128, dropout_rate: float = 0.3,
                           lr: float = 0.001) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))

    x = Dense(units1, activation='swish')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    res = Dense(units2, activation='swish')(x)
    res = BatchNormalization()(res)
    res = Dropout(dropout_rate * 0.75)(res)
    res = Dense(units2, activation='swish')(res)
    res = BatchNormalization()(res)

    x_proj = Dense(units2)(x)
    x = tf.keras.layers.Add()([x_proj, res])
    x = tf.keras.layers.Activation('swish')(x)

    seq = Reshape((1, input_dim))(inputs)
    d_model = 64
    seq = Dense(d_model)(seq)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=16)(seq, seq)
    attn_out = LayerNormalization()(attn_out + seq)
    attn_flat = GlobalAveragePooling1D()(attn_out)

    merged = tf.keras.layers.Concatenate()([x, attn_flat])
    merged = Dense(units3, activation='swish')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(dropout_rate * 0.5)(merged)
    merged = Dense(64, activation='swish')(merged)

    output = Dense(1, activation='linear')(merged)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    return model


def tune_xgboost(X_tr, y_tr, use_gpu: bool, n_trials: int = 30) -> dict:
    log.info("Running Optuna search for XGBoost (%d trials)...", n_trials)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
            'max_depth': trial.suggest_int('max_depth', 10, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
            'verbosity': 0,
            'random_state': RANDOM_STATE,
        }
        if use_gpu:
            params['device'] = 'cuda'
            params['tree_method'] = 'hist'

        X_np = np.array(X_tr) if not isinstance(X_tr, np.ndarray) else X_tr
        y_np = np.array(y_tr) if not isinstance(y_tr, np.ndarray) else y_tr

        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = []
        for tr_idx, val_idx in kf.split(X_np):
            model = XGBRegressor(**params)
            model.fit(X_np[tr_idx], y_np[tr_idx])
            preds = model.predict(X_np[val_idx])
            r2 = r2_score(y_np[val_idx], np.expm1(preds))
            cv_scores.append(r2)
        return np.mean(cv_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("Best XGBoost R2 (CV): %.4f | Params: %s", study.best_value, study.best_params)
    return study.best_params


def tune_lgbm(X_tr, y_tr, use_gpu: bool, n_trials: int = 30) -> dict:
    log.info("Running Optuna search for LightGBM (%d trials)...", n_trials)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
            'max_depth': -1,
            'num_leaves': trial.suggest_int('num_leaves', 127, 1023),
            'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
            'random_state': RANDOM_STATE,
            'device_type': 'gpu' if use_gpu else 'cpu',
            'verbose': -1,
        }

        X_np = np.array(X_tr) if not isinstance(X_tr, np.ndarray) else X_tr
        y_np = np.array(y_tr) if not isinstance(y_tr, np.ndarray) else y_tr

        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = []
        for tr_idx, val_idx in kf.split(X_np):
            model = LGBMRegressor(**params)
            model.fit(X_np[tr_idx], y_np[tr_idx])
            preds = model.predict(X_np[val_idx])
            r2 = r2_score(y_np[val_idx], np.expm1(preds))
            cv_scores.append(r2)
        return np.mean(cv_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("Best LightGBM R2 (CV): %.4f | Params: %s", study.best_value, study.best_params)
    return study.best_params


def main() -> None:
    log.info("=" * 65)
    log.info("House Price Prediction - Ultra Industry-Level Hybrid Pipeline")
    log.info("=" * 65)
    log.info("Training Target: ~60 minutes of thorough deep learning")
    log.info("=" * 65)

    target_col = 'Price_INR'

    try:
        train_files = get_train_files()
    except Exception as e:
        log.error(e)
        return

    log.info("Files to train on sequentially: %s", [Path(f).name for f in train_files])

    first_df = pd.read_csv(train_files[0], on_bad_lines='skip').dropna(axis=1, how='all')
    first_df = first_df.dropna(subset=[target_col])
    X_init = first_df.drop(columns=[target_col])

    log.info("Initializing preprocessor schema...")
    preprocessor = initialize_preprocessor(X_init)

    X_train_chunk, X_val, y_train_chunk, y_val = train_test_split(
        X_init, first_df[target_col], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    log.info("Fitting specialized preprocessor on chunk 1...")
    preprocessor.fit(X_train_chunk, y_train_chunk)
    X_val_processed = preprocessor.transform(X_val)
    input_dim = X_val_processed.shape[1]
    log.info("Final feature dimension: %d", input_dim)

    use_gpu = detect_gpu()

    log.info("=" * 65)
    log.info("PHASE 1: Optuna Hyperparameter Tuning (XGBoost + LightGBM)")
    log.info("Each model runs 30 trials x 3-Fold CV = 180 model fits per algo")
    log.info("=" * 65)
    X_tr_log = preprocessor.transform(X_train_chunk)
    y_tr_log = np.log1p(y_train_chunk)

    best_xgb_params = tune_xgboost(X_tr_log, y_tr_log, use_gpu, n_trials=30)
    best_lgb_params = tune_lgbm(X_tr_log, y_tr_log, use_gpu, n_trials=30)

    log.info("=" * 65)
    log.info("PHASE 2: Initializing Ultra-Heavy Base Learners")
    log.info("=" * 65)

    best_xgb_params['random_state'] = RANDOM_STATE
    best_xgb_params['verbosity'] = 0
    if use_gpu:
        best_xgb_params['device'] = 'cuda'
        best_xgb_params['tree_method'] = 'hist'

    best_lgb_params['random_state'] = RANDOM_STATE
    best_lgb_params['verbose'] = -1
    best_lgb_params['device_type'] = 'gpu' if use_gpu else 'cpu'

    model_xgb = XGBRegressor(**best_xgb_params)
    model_lgb = LGBMRegressor(**best_lgb_params)

    model_gbr = GradientBoostingRegressor(
        n_estimators=2000,
        learning_rate=0.005,
        max_depth=10,
        subsample=0.8,
        random_state=RANDOM_STATE,
        verbose=0
    )

    model_et = ExtraTreesRegressor(
        n_estimators=1000,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    log.info("Building Deep Transformer + ResNet DNN...")
    model_dnn = build_transformer_dnn(input_dim)
    model_dnn.summary()

    meta_xgb_kwargs = {
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.05,
        'random_state': RANDOM_STATE,
        'verbosity': 0
    }
    if use_gpu:
        meta_xgb_kwargs['device'] = 'cuda'
        meta_xgb_kwargs['tree_method'] = 'hist'
    meta_learner = XGBRegressor(**meta_xgb_kwargs)
    oof_meta_features = []
    global_y_train_log = []
    xgb_trained = None
    lgb_trained = None

    log.info("=" * 65)
    log.info("PHASE 3: Sequential Incremental Training (Dataset by Dataset)")
    log.info("=" * 65)

    for idx, f in enumerate(train_files):
        log.info("\n--- Processing Dataset %d/%d: %s ---", idx + 1, len(train_files), Path(f).name)

        if idx == 0:
            X_tr, y_tr = X_train_chunk, y_train_chunk
        else:
            df = pd.read_csv(f, on_bad_lines='skip').dropna(axis=1, how='all')
            df = df.dropna(subset=[target_col])
            y_tr = df[target_col]
            X_tr = df.drop(columns=[target_col])

        X_tr_p = preprocessor.transform(X_tr)
        y_tr_log = np.log1p(y_tr).values if hasattr(y_tr, 'values') else np.log1p(y_tr)

        log.info("[%d/%d] Training XGBoost (n_est=%d)...", idx+1, len(train_files),
                 best_xgb_params.get('n_estimators', 3000))
        if xgb_trained is None:
            model_xgb.fit(X_tr_p, y_tr_log)
        else:
            model_xgb.fit(X_tr_p, y_tr_log, xgb_model=model_xgb.get_booster())
        xgb_trained = True

        log.info("[%d/%d] Training LightGBM (n_est=%d)...", idx+1, len(train_files),
                 best_lgb_params.get('n_estimators', 3000))
        if lgb_trained is None:
            model_lgb.fit(X_tr_p, y_tr_log)
        else:
            model_lgb.fit(X_tr_p, y_tr_log, init_model=model_lgb.booster_)
        lgb_trained = True

        log.info("[%d/%d] Training GradientBoostingRegressor (1000 estimators)...", idx+1, len(train_files))
        model_gbr.fit(X_tr_p, y_tr_log)

        log.info("[%d/%d] Training ExtraTreesRegressor (500 estimators)...", idx+1, len(train_files))
        model_et.fit(X_tr_p, y_tr_log)

        log.info("[%d/%d] Training Deep Transformer ResNet (2000 Epochs w/ Early Stopping, batch_size=16)...", idx+1, len(train_files))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=30, verbose=1, min_lr=1e-7)
        ]
        model_dnn.fit(
            X_tr_p,
            y_tr_log,
            epochs=2000,
            batch_size=16,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1
        )

        xgb_p = model_xgb.predict(X_tr_p)
        lgb_p = model_lgb.predict(X_tr_p)
        gbr_p = model_gbr.predict(X_tr_p)
        et_p = model_et.predict(X_tr_p)
        dnn_p = model_dnn.predict(X_tr_p, verbose=0).flatten()

        chunk_meta = np.column_stack((xgb_p, lgb_p, gbr_p, et_p, dnn_p))
        oof_meta_features.append(chunk_meta)
        global_y_train_log.extend(y_tr_log)

    log.info("=" * 65)
    log.info("PHASE 4: Training XGBoost Meta-Learner on stacked predictions")
    log.info("=" * 65)
    X_meta = np.vstack(oof_meta_features)
    y_meta_log = np.array(global_y_train_log)
    meta_learner.fit(X_meta, y_meta_log)

    log.info("=" * 65)
    log.info("PHASE 5: Validation Metrics")
    log.info("=" * 65)

    val_xgb = model_xgb.predict(X_val_processed)
    val_lgb = model_lgb.predict(X_val_processed)
    val_gbr = model_gbr.predict(X_val_processed)
    val_et  = model_et.predict(X_val_processed)
    val_dnn = model_dnn.predict(X_val_processed, verbose=0).flatten()

    val_meta = np.column_stack((val_xgb, val_lgb, val_gbr, val_et, val_dnn))
    preds_log = meta_learner.predict(val_meta)
    preds = np.expm1(preds_log)

    r2   = r2_score(y_val, preds)
    mae  = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    log.info("--- Validation Metrics ---")
    log.info("R2 Score : %.4f", r2)
    log.info("MAE      : %.2f", mae)
    log.info("RMSE     : %.2f", rmse)
    log.info("--------------------------")

    log.info("Generating plots...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_val, y=preds, alpha=0.5)
    max_v = max(y_val.max(), preds.max())
    min_v = min(y_val.min(), preds.min())
    plt.plot([min_v, max_v], [min_v, max_v], color='red', linestyle='--')
    plt.title('Ultra Hybrid Model: Actual vs Predicted Prices')
    plt.xlabel('Actual Price (INR)')
    plt.ylabel('Predicted Price (INR)')
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "hybrid_actual_vs_predicted.png")
    plt.close()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    residuals = np.abs(np.array(y_val) - preds)
    ax.scatter(y_val, preds, residuals, c=residuals, cmap='plasma', alpha=0.6)
    ax.set_xlabel('Actual Price (INR)')
    ax.set_ylabel('Predicted Price (INR)')
    ax.set_zlabel('Absolute Error')
    ax.set_title('3D Plot: Predicted vs Actual vs Residual Error')
    plt.savefig(Path(__file__).parent / "hybrid_3d_residuals.png")
    plt.close()

    final_artifact = {
        'preprocessor': preprocessor,
        'model_xgb': model_xgb,
        'model_lgb': model_lgb,
        'model_gbr': model_gbr,
        'model_et': model_et,
        'meta_learner': meta_learner,
        'xgb_params': best_xgb_params,
        'lgb_params': best_lgb_params,
    }
    joblib.dump(final_artifact, MODEL_OUTPUT_PATH)
    model_dnn.save(Path(__file__).parent / "custom_hybrid_keras.keras")
    log.info("Saved Complete Hybrid Stack to %s", MODEL_OUTPUT_PATH)

    log.info("Loading test datasets (test_part*.csv) for final predictions...")
    test_files = glob.glob(str(DATA_DIR / "test_part*.csv"))
    if test_files:
        test_dfs = [pd.read_csv(f, on_bad_lines='skip') for f in test_files]
        test_df = pd.concat(test_dfs, ignore_index=True).dropna(axis=1, how='all')
        log.info("Processing test dataset of shape: %s", test_df.shape)
        X_test_p = preprocessor.transform(test_df)

        tst_xgb = model_xgb.predict(X_test_p)
        tst_lgb = model_lgb.predict(X_test_p)
        tst_gbr = model_gbr.predict(X_test_p)
        tst_et  = model_et.predict(X_test_p)
        tst_dnn = model_dnn.predict(X_test_p, verbose=0).flatten()

        tst_meta = np.column_stack((tst_xgb, tst_lgb, tst_gbr, tst_et, tst_dnn))
        test_preds = np.expm1(meta_learner.predict(tst_meta))

        submission_df = pd.DataFrame()
        if 'ListingID' in test_df.columns:
            submission_df['ListingID'] = test_df['ListingID']
        submission_df['Predicted_Price_INR'] = test_preds

        preds_path = Path(__file__).parent / "test_predictions.csv"
        submission_df.to_csv(preds_path, index=False)
        log.info("Saved predictions for all unseen test data to %s", preds_path)
    else:
        log.info("No test datasets found for prediction generation.")


if __name__ == "__main__":
    main()
