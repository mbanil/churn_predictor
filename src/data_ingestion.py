import os
import pandas as pd
from src.logger import logger


class DataIngestion:
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    def load_data(self) -> pd.DataFrame:
        try:
            logger.info("Loading raw dataset...")
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error("Error while loading dataset", exc_info=True)
            raise e

    def basic_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Running basic validation checks...")

            if df.empty:
                raise ValueError("Dataset is empty.")

            if "Churn" not in df.columns:
                raise ValueError("Target column 'Churn' not found.")

            logger.info("Validation passed successfully.")
            return df

        except Exception as e:
            logger.error("Validation failed", exc_info=True)
            raise e

    def save_processed_data(self, df: pd.DataFrame):
        try:
            os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
            df.to_csv(self.processed_data_path, index=False)
            logger.info(f"Processed data saved at {self.processed_data_path}")
        except Exception as e:
            logger.error("Failed to save processed data", exc_info=True)
            raise e

    def run(self):
        df = self.load_data()
        df = self.basic_validation(df)
        self.save_processed_data(df)
        logger.info("Data ingestion pipeline completed successfully.")


if __name__ == "__main__":
    ingestion = DataIngestion(
        raw_data_path="data/raw/Telco-Customer-Churn.csv",
        processed_data_path="data/processed/validated_data.csv"
    )
    ingestion.run()