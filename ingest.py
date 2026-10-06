import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# --- DATABASE CONFIGURATION ---
# Replace 'password' with your actual pgAdmin master password
DB_USER = "postgres"
DB_PASS = "postgres"  # <-- Change this to your PostgreSQL password
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "drift_sentinel"

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

SELECTED_COLS = [
    "tpep_pickup_datetime",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "payment_type",
    "PULocationID",
]


def fetch_and_clean_batch(year: int, month: int, sample_size: int = 50000):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    print(f"Downloading from {url} ...")

    # Read selected columns directly from the remote Parquet file
    df = pd.read_parquet(url, columns=SELECTED_COLS)

    # Standardize column names to lowercase to match PostgreSQL schema
    df.columns = [c.lower() for c in df.columns]

    # Sample rows for fast local execution
    sample_df = df.sample(
        n=min(sample_size, len(df)), random_state=42
    ).copy()
    return sample_df


def run_ingestion():
    # Additional chronological batches to build a rich multi-month timeline
    extra_batches = [
        (2024, 3, "batch_2024_03"),
        (2024, 9, "batch_2024_09"),
        (2024, 12, "batch_2024_12"),
    ]

    for year, month, batch_id in extra_batches:
        print(f"\n--- Ingesting Inference Stream {batch_id} ({year}-{month:02d}) ---")
        batch_df = fetch_and_clean_batch(year, month, sample_size=30000)
        batch_df["batch_id"] = batch_id
        batch_df.to_sql(
            "inference_stream",
            engine,
            if_exists="append",
            index=False,
            chunksize=5000,
        )
        print(f"Successfully loaded {len(batch_df)} rows as '{batch_id}'.")

    print("\nAdditional batches ingested successfully!")


if __name__ == "__main__":
    run_ingestion()
    print("\nAll data ingestion complete!")


if __name__ == "__main__":
    run_ingestion()