import seaborn as sns

df = sns.load_dataset("titanic")

df = df[
    [
        "survived",
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
    ]
].copy()

df = df.rename(
    columns={
        "survived": "Survived",
        "pclass": "Pclass",
        "sex": "Sex",
        "age": "Age",
        "sibsp": "SibSp",
        "parch": "Parch",
        "fare": "Fare",
        "embarked": "Embarked",
    }
)

df.to_csv(
    "titanic.csv",
    index=False,
)

print("Titanic dataset downloaded.")
print("Shape:", df.shape)
print(df.head())