import pandas as pd
import numpy as np

def fix_dtypes(df):
    df["views"]            = pd.to_numeric(df["views"],            errors="coerce")
    df["likes"]            = pd.to_numeric(df["likes"],            errors="coerce")
    df["author_followers"] = pd.to_numeric(df["author_followers"], errors="coerce")
    df["author_likes"]     = pd.to_numeric(df["author_likes"],     errors="coerce")
    df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")
    df["human_time"]       = pd.to_datetime(df["human_time"],      errors="coerce")
    return df

def drop_nulls(df):
    df = df.dropna(subset=["text_part"])
    df = df.dropna(subset=["views", "likes", "author_followers", "duration_seconds"])
    return df

def remove_duplicates(df):
    df = df.drop_duplicates(subset=["id_video"])
    return df

def add_engagement_rate(df):
    df["engagement_rate"] = df["likes"] / df["views"].replace(0, np.nan)
    return df

def add_date_parts(df):
    df["post_date"] = df["human_time"].dt.date
    df["post_hour"] = df["human_time"].dt.hour
    df["post_month"] = df["human_time"].dt.month
    return df

def add_creator_tier(df):
    median_followers = df["author_followers"].median()
    df["creator_tier"] = df["author_followers"].apply(
        lambda x: "high followers" if x >= median_followers else "low followers"
    )
    return df

def clean_text(df):
    df["text_part"] = df["text_part"].astype(str).str.strip()
    return df

def run_all(df):
    df = fix_dtypes(df)
    df = drop_nulls(df)
    df = remove_duplicates(df)
    df = add_engagement_rate(df)
    df = add_date_parts(df)
    df = add_creator_tier(df)
    df = clean_text(df)

    print(f"Cleaning complete: {df.shape}")
    return df