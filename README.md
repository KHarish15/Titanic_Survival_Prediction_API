# Titanic Survival Prediction API

## Project Overview

This project is a machine learning system that predicts whether a Titanic passenger survived based on passenger information.

The system includes:

- Data preprocessing
- Missing value handling
- Categorical feature encoding
- Numerical feature scaling
- Random Forest classification
- Hyperparameter tuning using GridSearchCV
- Model serialization using Joblib
- REST API using FastAPI
- Docker containerization

## Project Architecture

```text
Client
   |
   | HTTP POST /predict
   ↓
FastAPI
   |
   ↓
Pydantic Input Validation
   |
   ↓
Saved ML Pipeline
   |
   ├── Missing Value Handling
   ├── Numerical Scaling
   └── Categorical Encoding
   |
   ↓
Random Forest Model
   |
   ↓
Prediction
   |
   ↓
JSON Response
```

## Project Structure

```text
titanic_ml_project/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── src/
│   └── train.py
│
├── api/
│   ├── main.py
│   └── prediction.py
│
├── model/
│   └── titanic_model.pkl
│
├── requirements.txt
├── Dockerfile
└── README.md
```

## Dataset

The project uses the Titanic dataset.

The target variable is:

```text
Survived
```

where:

- `0` = Did not survive
- `1` = Survived

The model uses the following features:

| Feature | Description |
|---|---|
| Pclass | Passenger class |
| Sex | Passenger gender |
| Age | Passenger age |
| Fare | Ticket fare |
| Embarked | Port where the passenger boarded |

The `Cabin` feature was excluded because it contains a large number of missing values.

## Data Preprocessing

The preprocessing pipeline performs different operations for numerical and categorical features.

### Numerical Features

```text
Pclass
Age
Fare
```

Processing:

```text
Missing values → Median
        ↓
StandardScaler
```

### Categorical Features

```text
Sex
Embarked
```

Processing:

```text
Missing values → Most Frequent
        ↓
OneHotEncoder
```

The preprocessing and machine learning model are combined into a single Scikit-learn pipeline.

This allows raw input data to be passed directly to the trained model.

## Model

The project uses:

```text
RandomForestClassifier
```

Hyperparameters were optimized using:

```text
GridSearchCV
```

The best parameters found were:

```text
n_estimators = 200
max_depth = 10
min_samples_split = 5
min_samples_leaf = 2
```

Best cross-validation accuracy:

```text
82.59%
```

Final test-set accuracy:

```text
80.45%
```

### Test Results

```text
Accuracy  : 80.45%
Precision : 79.31%
Recall    : 66.67%
```

Confusion matrix:

```text
[[98 12]
 [23 46]]
```

## Model Serialization

After training, the complete trained pipeline is saved using Joblib:

```python
joblib.dump(best_model, "model/titanic_model.pkl")
```

The saved model contains both:

- Preprocessing
- Trained Random Forest model

Therefore, the API does not need to retrain the model.

The saved model is loaded using:

```python
model = joblib.load("model/titanic_model.pkl")
```

## Running the Project Locally

### 1. Create and activate the virtual environment

```powershell
python -m venv vnv
```

Windows PowerShell:

```powershell
.\vnv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the FastAPI application

```powershell
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## API Documentation

FastAPI automatically provides Swagger documentation.

Open:

```text
http://127.0.0.1:2000/docs
```

## API Endpoints

### GET `/`

Checks whether the API is running.

Example response:

```json
{
  "message": "Titanic Survival Prediction API"
}
```

### POST `/predict`

Predicts whether a passenger survived.

#### Request

```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 22,
  "Fare": 7.25,
  "Embarked": "S"
}
```

#### Response

```json
{
  "prediction": 1
}
```

Prediction values:

```text
0 → Did not survive
1 → Survived
```

## Docker

The application is containerized using Docker.

### Build the Docker image

From the project root:

```powershell
docker build -t titanic_ml_project .
```

### Run the container

```powershell
docker run -d -p 8000:8000 --name titanic_container titanic_ml_project
```

The port mapping is:

```text
Host Port 8000 → Container Port 8000
```

### Check the running container

```powershell
docker ps
```

### View container logs

```powershell
docker logs titanic_container
```

### Stop the container

```powershell
docker stop titanic_container
```

### Start the container again

```powershell
docker start titanic_container
```

## Testing the Dockerized API

After starting the container, open:

```text
http://localhost:2000/docs
```

Use the `/predict` endpoint with:

```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 22,
  "Fare": 7.25,
  "Embarked": "S"
}
```

Example response:

```json
{
  "prediction": 1
}
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Pydantic
- Uvicorn
- Docker

## Conclusion

This project demonstrates an end-to-end machine learning deployment workflow:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Train/Test Split
   ↓
Random Forest
   ↓
GridSearchCV
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
FastAPI REST API
   ↓
Docker Container
   ↓
Prediction
```

The resulting system provides a REST API that accepts passenger information and returns a Titanic survival prediction.