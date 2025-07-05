from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()
model = load_model("model.h5")

class DigitInput(BaseModel):
    pixels: list  # 28x28 flattened = 784 values

@app.get("/")
def welcome():
    return {"message": "Digit classifier is up!"}

@app.post("/predict")
def predict(data: DigitInput):
    x = np.array(data.pixels).reshape(1, 28, 28) / 255.0
    prediction = model.predict(x)
    return {"digit": int(np.argmax(prediction))}
