import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("titanic.csv")

print("Dataset loaded:", df.shape)

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

X = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "Embarked",
        "FamilySize",
        "IsAlone",
    ]
]

y = df["Fare"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

numeric_features = [
    "Age",
    "FamilySize",
    "IsAlone",
]

categorical_features = [
    "Pclass",
    "Sex",
    "Embarked",
]

numeric_pipeline = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

categorical_pipeline = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    [
        (
            "num",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)

model = Pipeline(
    [
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            LinearRegression(),
        ),
    ]
)

model.fit(X_train, y_train)

print("Titanic fare model trained successfully.")

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions,
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions,
    )
)

r2 = r2_score(
    y_test,
    predictions,
)

print("\nModel Evaluation")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 3))

joblib.dump(
    model,
    "titanic_fare_regression_model.joblib",
)

print("\nModel saved as:")
print("titanic_fare_regression_model.joblib")

test_passenger = pd.DataFrame(
    [
        {
            "Pclass": 1,
            "Sex": "female",
            "Age": 30,
            "Embarked": "S",
            "FamilySize": 1,
            "IsAlone": 1,
        }
    ]
)

predicted_fare = model.predict(
    test_passenger
)[0]

print(
    "\nTest predicted fare:",
    round(predicted_fare, 2),
)
