from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request, Form
import os

app = FastAPI()

# Charger le modèle entraîné
model = joblib.load("model.pkl")

# Set up templates directory
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

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

@app.get("/predict-form", response_class=HTMLResponse)
def get_predict_form(request: Request):
    return templates.TemplateResponse("predict_form.html", {"request": request, "prediction": None})

@app.post("/predict-form", response_class=HTMLResponse)
def post_predict_form(request: Request,
                     quarter: float = Form(...),
                     department: float = Form(...),
                     day: float = Form(...),
                     team: float = Form(...),
                     targeted_productivity: float = Form(...),
                     smv: float = Form(...),
                     wip: float = Form(...),
                     over_time: float = Form(...),
                     incentive: float = Form(...),
                     idle_time: float = Form(...),
                     idle_men: float = Form(...),
                     no_of_style_change: float = Form(...),
                     no_of_workers: float = Form(...)):
    features = [quarter, department, day, team, targeted_productivity, smv, wip, over_time, incentive, idle_time, idle_men, no_of_style_change, no_of_workers]
    try:
        features_np = np.array(features).reshape(1, -1)
        prediction = model.predict(features_np)[0]
    except Exception as e:
        prediction = f"Error: {str(e)}"
    return templates.TemplateResponse("predict_form.html", {"request": request, "prediction": prediction})
