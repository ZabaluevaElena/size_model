import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

SEED: int = 42
MIN_WEIGHT: int = 36
MAX_WEIGHT: int = 100
MIN_AGE: int = 14
MAX_AGE: int = 80
FILE_NAME: str = "final_test.csv"
N_ESTIMATORS: int = 300
MAX_DEPTH: int = 12
SIZE_MAP: dict = {"XXS": 1, "S": 2, "M": 3, "L": 4, "XL": 5, "XXXL": 6}


def encode_target(y: str) -> int:
    return SIZE_MAP.get(y, -1)


def data_prepare(df: pd.DataFrame) -> tuple:
    df["age"] = df["age"].fillna(df["age"].median())
    df["height"] = df["height"].fillna(df["height"].median())
    df["weight"] = df["weight"].fillna(df["weight"].median())
    df = df.query("@MIN_WEIGHT <= weight <= @MAX_WEIGHT")
    df = df.query("@MIN_AGE <= age <= @MAX_AGE")
    df = df[df["size"] != "XXL"]
    df["size"] = df["size"].apply(encode_target)
    X = df.drop("size", axis=1)
    y = df["size"]
    return X, y


def main():
    df = pd.read_csv("./data/final_test.csv")
    X, y = data_prepare(df)
    model = RandomForestClassifier(
        random_state=SEED,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced",
        verbose=3,
    )
    model.fit(X, y)
    joblib.dump(model, "./model/model.joblib")


if __name__ == "__main__":
    main()
