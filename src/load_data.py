import pandas as pd
import os

def load_raw(filename="tiktok_data.csv"):
    path = os.path.join("data", "raw", filename)
    df = pd.read_csv(path)
    print(f"Loaded raw data {df.shape}")
    return df

def load_clean():
    path = os.path.join("data", "processed", "tiktok_clean.csv")
    df = pd.read_csv(path)
    df["human_time"] = pd.to_datetime(df["human_time"], errors="coerce")
    print(f"Loaded clean data: {df.shape}")
    return df

def load_sentiment():
    path = os.path.join("data", "processed", "tiktok_sentiment.csv")
    df = pd.read_csv(path)
    df["human_time"] = pd.to_datetime(df["human_time"], errors="coerce")
    print(f"Loaded sentiment data: {df.shape}")
    return df