import joblib 
import pandas as pd

model=joblib.load("model/titanic_model.pkl")
print("model loaded successfully")


passenger = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "Fare": 7.25,
    "Embarked": "S"
}
input_data=pd.DataFrame([passenger])
print("model prediction....")

prediction=model.predict(input_data)
print("prediction",prediction[0])
