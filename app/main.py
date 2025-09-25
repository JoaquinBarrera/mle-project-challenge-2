from fastapi import FastAPI
from app import config
from app.schemas import FullHouse, PredictionResponse
from app.model import predict_price

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="""
    API for predicting housing prices in King County
    """
)

@app.post("/predict",
          summary="Predict house price (baseline model)",
          description="Takes house features and returns the predicted price using the baseline model.",
          response_model=PredictionResponse)
def predict(house: FullHouse):
    prediction = predict_price(house)
    return {"prediction": prediction}


@app.post("/predict_xgb",
          summary="Predict house price (XGBoost model)",
          description="Uses an XGBoost model trained on King County dataset to estimate the price.",
          response_model=PredictionResponse)
def predict_xgb(house: FullHouse):
    prediction = predict_price(house, 'xgb')
    return {"prediction":  prediction,
            "model" : "XGBoost"
            }

@app.get("/health")
def health():
    return {"status_code": "200"}

