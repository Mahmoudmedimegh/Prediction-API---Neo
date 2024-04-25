from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Define your FastAPI app
app = FastAPI()

# Pydantic model to validate incoming data
class ScoringItem(BaseModel):
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    ORGANIZATION_TYPE: str
    DAYS_BIRTH: int
    DAYS_ID_PUBLISH: int
    SK_ID_CURR: int
    REG_CITY_NOT_LIVE_CITY: int
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    YEARS_BEGINEXPLUATATION_MODE: float
    COMMONAREA_MODE: float
    FLOORSMAX_MODE: float
    LIVINGAPARTMENTS_MODE: float
    YEARS_BUILD_MEDI: float

# Load your model pipeline
model_pipeline = joblib.load('model_pipeline.pkl')
preprocessor = joblib.load('fitted_preprocessor_py3.pkl')
@app.post('/predict_proba', summary="Predict client score probability to loan eligibility")
async def scoring_endpoint(item: ScoringItem):
    # Convert Pydantic object to DataFrame
    df = pd.DataFrame([item.dict()], index=[0])

    try:
        # Preprocess the data
        preprocessed_data = preprocessor.transform(df)

        # Predict the probability of the positive class using the entire pipeline directly
        proba = model_pipeline.predict_proba(preprocessed_data)
        # Select the probability of the positive class (usually the second column in binary classification)
        positive_proba = float(proba[0][1])
    except Exception as e:
        return {"error": str(e)}

    return {"prediction": positive_proba}
