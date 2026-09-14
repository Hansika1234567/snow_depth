"""
preprocessing.py
Functions for loading and preprocessing the snow depth dataset.
"""

import pandas as pd


def load_data(filepath):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df


def convert_date(df, date_column="date"):
    """Convert the date column to datetime format."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    return df


def extract_date_features(df, date_column="date"):
    """
    Extract year, month, day, and dayofyear from the date column.
    These temporal features are used as model inputs.
    """
    df = df.copy()
    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day
    df["dayofyear"] = df[date_column].dt.dayofyear
    return df


def drop_id_and_date(df, columns=["Station_ID", "date"]):
    """Drop Station_ID and date columns after feature extraction."""
    df = df.copy()
    df = df.drop(columns=[c for c in columns if c in df.columns])
    return df


def impute_missing_values(df):
    """
    Fill missing numeric values using median imputation.
    Each column is filled with its own median.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def preprocess(filepath):
    """
    Full preprocessing pipeline:
    load -> convert date -> extract date features -> drop columns -> impute missing values.
    """
    df = load_data(filepath)
    df = convert_date(df)
    df = extract_date_features(df)
    df = drop_id_and_date(df)
    df = impute_missing_values(df)
    return df
