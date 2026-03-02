import pandas as pd
import numpy as np
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.logger import logger


class DataPreprocessor:
    def __init__(self):
        self.pipeline = None

    def _separate_features(self, df: pd.DataFrame):
        logger.info("Separating features and target.")

        df = df.copy()

        # Drop ID column
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        # Convert TotalCharges to numeric (it may contain blanks)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        X = df.drop("Churn", axis=1)
        y = df["Churn"].map({"Yes": 1, "No": 0})

        return X, y

    def build_pipeline(self, X: pd.DataFrame):
        logger.info("Building preprocessing pipeline.")

        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features)
            ]
        )

        logger.info("Pipeline built successfully.")

    def fit_transform(self, df: pd.DataFrame):
        X, y = self._separate_features(df)
        self.build_pipeline(X)

        logger.info("Fitting preprocessing pipeline.")
        X_processed = self.pipeline.fit_transform(X)

        logger.info("Preprocessing completed.")
        return X_processed, y

    def transform(self, df: pd.DataFrame):
        X, y = self._separate_features(df)

        logger.info("Transforming new data using fitted pipeline.")
        X_processed = self.pipeline.transform(X)

        return X_processed, y

    def save_pipeline(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info(f"Preprocessing pipeline saved at {path}")

    def load_pipeline(self, path: str):
        self.pipeline = joblib.load(path)
        logger.info("Preprocessing pipeline loaded successfully.")
        