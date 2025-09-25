import json
import pathlib
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

from create_model import load_data


SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/kc_house_data.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode',
    'waterfront', 'view', 'grade', 'yr_built', 'yr_renovated', 'lat', 'long'
]
OUTPUT_DIR = "model"

# ==============================
# XGBoost training function
# ==============================
def train_xgboost(x: pd.DataFrame, y: pd.Series, n_trials: int):
    """
       Train an XGBoost regressor with Optuna hyperparameter tuning.
       Performs correlation-based feature elimination before training.
       """
     # Split data (same seed to keep consistency)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Drop highly correlated features (>0.90)
    corr_matrix = x_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.90)]
    x_train = x_train.drop(columns=to_drop)
    x_test = x_test.drop(columns=to_drop)
    features = x_train.columns


    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1
        }

        model = xgb.XGBRegressor(**params)

        # 5-fold cross-validation
        scores = cross_val_score(model, x_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        return scores.mean()

    # Optimize hyperparameters
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    print("Best parameters:", study.best_params)
    print("Best CV R²:", study.best_value)

    # Fit the final model
    best_model = xgb.XGBRegressor(**study.best_params)
    best_model.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=50, verbose=False)

    # Evaluate on test set
    y_test_pred = best_model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    print("\nTest set performance with Optuna-tuned XGBoost:")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  MAE:  {mae:,.2f}")
    print(f"  R²:   {r2:.3f}")

    return best_model, features


# ==============================
# LightGBM training function
# ==============================
def train_lgmb(x: pd.DataFrame, y: pd.Series, n_trials: int):
    """
    Train a LightGBM regressor with Optuna hyperparameter tuning.
    """

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # corr_matrix = X_train.corr().abs()
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # to_drop = [col for col in upper.columns if any(upper[col] > 0.90)]
    # X_train = X_train.drop(columns=to_drop)
    # X_test = X_test.drop(columns=to_drop)
    # features = X_train.columns

    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 1),
        }

        # 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            lgb_train = lgb.Dataset(X_tr, y_tr)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

            model = lgb.train(
                params, lgb_train, num_boost_round=5000, valid_sets=[lgb_val],
                callbacks=[early_stopping(stopping_rounds=100)]
            )

            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            rmse_scores.append(mean_squared_error(y_val, y_pred, squared=False))

        return np.mean(rmse_scores)

    # Hyperparams search
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best trial:")
    print(study.best_trial.params)

    # Train final model with best params
    best_params = study.best_trial.params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1
    })

    lgb_model = LGBMRegressor(**best_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=[early_stopping(stopping_rounds=100)]
    )

    # Eval on test set
    y_pred = lgb_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Test set performance with LightGBM:")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"R²:   {r2:.3f}")

    return lgb_model, X_train.columns


# ==============================
# Main function
# ==============================
def main(model_name='xgb', n_trials=100):
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    if model_name=='xgb':
        print('Training XGBoost model...')
        model, features = train_xgboost(x,y,n_trials)

    else:
        print('Training LGBM model...')
        model, features = train_lgmb(x,y,n_trials)
        model_name = 'lgb'

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Save model and features
    pickle.dump(model, open(output_dir / f"{model_name}_model.pkl", 'wb'))
    json.dump(list(features),
              open(output_dir / f"{model_name}_model_features.json", 'w'))


if __name__ == "__main__":
    main('lgb', 100)



