
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    X = df.drop("activity", axis=1)

    y = df["activity"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X, y_encoded, encoder
