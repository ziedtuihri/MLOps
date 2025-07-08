from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Charger le modèle entraîné
model = joblib.load("model.pkl")

class Features(BaseModel):
    features: list

@app.post("/predict")
def predict(data: Features):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
