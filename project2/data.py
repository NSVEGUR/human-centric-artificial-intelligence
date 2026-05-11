import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split


def load_and_preprocess():
    df = load_penguins()

    # drop missing rows - there are like 11 of them, checked manually
    df = df.dropna()

    # keeping track of the numerical ones separately
    # need this for MAD later (task 4 stuff)
    num_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

    # one hot encode island and sex
    # tried drop_first=True first but kept all categories in the end, easier to read
    df = pd.get_dummies(df, columns=["island", "sex"], drop_first=False)

    # year doesnt help predict species so drop it
    df = df.drop(columns=["year"])

    X = df.drop(columns=["species"])
    y = df["species"]

    # 80/20 split, stratify so all 3 species show up in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # MAD for each numerical feature
    # median absolute deviation - measures spread, used later for counterfactuals
    # if MAD comes out as 0 for some reason use 1.0 to avoid dividing by zero
    mad_vals = {}
    for col in num_cols:
        med = X_train[col].median()
        mad = (X_train[col] - med).abs().median()
        mad_vals[col] = mad if mad != 0 else 1.0

    return X_train, X_test, y_train, y_test, mad_vals, num_cols