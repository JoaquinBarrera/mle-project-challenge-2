import joblib
import json
from app import config
from typing import Union
import pandas as pd
from app.schemas import BaseHouse, FullHouse, XGBHouse

# Load base model and features
base_model = joblib.load(config.MODEL_PATH)
with open(config.FEATURES_PATH) as f:
    base_features = json.load(f)

xgb_model = joblib.load(config.XGB_MODEL_PATH)
with open(config.XGB_FEATURES_PATH) as f:
    xgb_features = json.load(f)

# Load demographics data
demographics = pd.read_csv(config.DEMOGRAPHIC_PATH)

def predict_price(input_data: Union[BaseHouse, XGBHouse, FullHouse], model_name: str = 'base'):
    """
    Predict house price given a full set of features.
    """
    data = input_data.model_dump()

    # Filter demographics by zipcode
    zipcode = data["zipcode"]
    demo_row = demographics[demographics["zipcode"] == zipcode]

    if demo_row.empty:
        raise ValueError(f"Zipcode {zipcode} not found in demographics data")

    # Merge input with demographics info
    df = pd.DataFrame([data])
    df_merged = df.merge(demo_row, how="left", on="zipcode").drop(columns="zipcode")

    if model_name == 'xgb':
        model_local = xgb_model
        features_local = xgb_features
    else:
        model_local = base_model
        features_local = base_features

    # Align DataFrame with expected features
    df_merged = df_merged[features_local]

    prediction = model_local.predict(df_merged)[0]
    return round(float(prediction),2)
