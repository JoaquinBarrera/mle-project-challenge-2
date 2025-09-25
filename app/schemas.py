from pydantic import BaseModel
from typing import Optional


class BaseHouse(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    sqft_above: float
    sqft_basement: float
    zipcode: int


class XGBHouse(BaseHouse):
    waterfront: Optional[int] = 0
    view: Optional[int] = 0
    grade: Optional[int] = None
    yr_built: Optional[int] = None
    yr_renovated: Optional[int] = 0
    lat: Optional[float] = None
    long: Optional[float] = None


class FullHouse(XGBHouse):
    condition: Optional[int] = None
    sqft_living15: Optional[int] = None
    sqft_lot15: Optional[int] = None


class PredictionResponse(BaseModel):
    prediction: float