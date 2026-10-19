import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = (
    r"C:\docx\Predictive_Maintenance\data"
    r"\ai4i+2020+predictive+maintenance+dataset\ai4i2020.csv"
)

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"

CATEGORICAL_FEATURES = ["Type"]

NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def prepare_data(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
