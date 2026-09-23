import joblib
import pandas as pd

from langchain_core.tools import tool


model = joblib.load(
    "titanic_fare_regression_model.joblib"
)


@tool
def titanic_fare_predictor(
    pclass: int,
    sex: str,
    age: float,
    embarked: str,
    family_size: int,
) -> str:
    """
    Predict Titanic passenger fare using a trained
    Linear Regression model.

    Inputs:
    pclass: Passenger class (1, 2, or 3)
    sex: male or female
    age: passenger age
    embarked: S, C, or Q
    family_size: total family size including passenger
    """

    sex = sex.lower().strip()
    embarked = embarked.upper().strip()

    if pclass not in [1, 2, 3]:
        return "Error: pclass must be 1, 2, or 3."

    if sex not in ["male", "female"]:
        return "Error: sex must be male or female."

    if embarked not in ["S", "C", "Q"]:
        return "Error: embarked must be S, C, or Q."

    if age < 0:
        return "Error: age cannot be negative."

    if family_size < 1:
        return "Error: family_size must be at least 1."

    is_alone = 1 if family_size == 1 else 0

    passenger = pd.DataFrame(
        [
            {
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "Embarked": embarked,
                "FamilySize": family_size,
                "IsAlone": is_alone,
            }
        ]
    )

    prediction = model.predict(
        passenger
    )[0]

    return (
        f"Predicted Titanic fare: "
        f"{prediction:.2f}"
    )


if __name__ == "__main__":

    result = titanic_fare_predictor.invoke(
        {
            "pclass": 1,
            "sex": "female",
            "age": 30,
            "embarked": "S",
            "family_size": 1,
        }
    )

    print(result)