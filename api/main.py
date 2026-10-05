from fastapi import FastAPI
from pydantic import BaseModel
app=FastAPI()
import pandas as pd
import joblib



model=joblib.load("model/titanic_model.pkl")
@app.get("/home")
def home():
    return "this is the first ml model"

class Passenger(BaseModel):
    Pclass:int
    Sex:str
    Age:float
    Fare: float
    Embarked: str

@app.post("/predict")
def prediction(pas:Passenger):
    input_data=pd.DataFrame([
        {"Pclass":pas.Pclass,
         "Sex":pas.Sex,
         "Age":pas.Age,
         "Fare":pas.Fare,
         "Embarked":pas.Embarked
         }
    ]) 

    prediction=model.predict(input_data)

    return {
        "prediction":int(prediction[0])
    }