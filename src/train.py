import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import GridSearchCV
import joblib
# reading csv file

df=pd.read_csv("data/train.csv")
print(df)

# dataset shape

print("dataset shape:")
print(df.shape)

# Display column names
print("\nColumns:")
print(df.columns)

# Display missing values
print("\nMissing values:")
print(df.isnull().sum())

# -----------------------------
# 5. Select features and target
# -----------------------------

x = df[["Pclass", "Sex", "Age", "Fare", "Embarked"]]

y = df["Survived"]


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_features = ["Pclass", "Age", "Fare"]
categorical_features = ["Sex", "Embarked"]


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# random forest model

model_pipeline=Pipeline(
    [("preprocessor",preprocessor),
     ("model",RandomForestClassifier(random_state=42))  
    ]
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(x_train,y_train)



#best model
print("best model")
best_model=grid_search.best_estimator_
print("best_model",best_model)

print("Best parameters:")
print(grid_search.best_params_)

print("Best CV accuracy:")
print(grid_search.best_score_)

y_pred=best_model.predict(x_test)

# print("answer for testing",y_pred)

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
confusion_matrix=confusion_matrix(y_test,y_pred)
classification_report=classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("confusion_matirx",confusion_matrix)

print("classification_report",classification_report)



# serialization
joblib.dump(best_model,"model/titanic_model.pkl") 


