import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model.pkl")
XGB_MODEL_PATH =  os.path.join(BASE_DIR, "..", "model", "xgb_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "model", "model_features.json")
XGB_FEATURES_PATH = os.path.join(BASE_DIR, "..", "model", "xgb_model_features.json")
DEMOGRAPHIC_PATH = os.path.join(BASE_DIR, "..", "data", "zipcode_demographics.csv")

ESSENTIAL_FEATURES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]

API_TITLE = "House Price Prediction API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Custom API for predicting housing prices in King County."
