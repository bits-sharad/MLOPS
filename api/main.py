from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("models/model.pkl")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(input: IrisInput):
    features = [[
        input.sepal_length, input.sepal_width,
        input.petal_length, input.petal_width
    ]]
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
