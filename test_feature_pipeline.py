"""Test the feature pipeline to debug the issue."""

import asyncio
import sys

import pandas as pd
import yfinance as yf

sys.path.append(".")

from robo_trader.config import Config
from robo_trader.features.feature_pipeline import FeaturePipeline


async def test():
    config = Config()
    pipeline = FeaturePipeline(config)

    # Get sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="1mo", interval="1d")
    df.columns = [col.lower() for col in df.columns]

    print(f"Data shape: {df.shape}")
    print(f"Data columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")

    # Calculate features
    features = await pipeline.calculate_features(symbol="AAPL", price_data=df, store_features=False)

    print(f"\nFeatures shape: {features.shape if features is not None else 'None'}")
    if features is not None and not features.empty:
        print(f"Features columns: {features.columns.tolist()}")
        print(f"First row:\n{features.iloc[0]}")
    else:
        print("No features returned!")

        # Check what the ML engine returns
        ml_features = pipeline.ml_engine.calculate_all_ml_features(
            symbol="AAPL", df=df, price_data=None, trades=None, quotes=None
        )
        print(f"\nML features: {ml_features}")


asyncio.run(test())
