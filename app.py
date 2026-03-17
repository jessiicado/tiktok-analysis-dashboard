import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
import warnings
warnings.filterwarnings("ignore")

from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords")
nltk.download("punkt")

st.set_page_config(page_title="Tiktok Analytics", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/tiktok_sentiment.csv")
    df["human_time"] = pd.to_datetime(df["human_time"], errors="coerce")
    return df

df = load_data()

#sidebar
st.sidebar.title("Filters")

creator_tier = st.sidebar.selectbox(
    "Creator Tier",
    ["All", "Low Followers", "High Followers"]
)

sentiment_filter = st.sidebar.multiselect(
    "Sentiment",
    ["positive", "neutral", "negative"],
    default=["positive", "neutral", "negative"]
)

duration_range = st.sidebar.slider(
    "Video duration (seconds)",
    int(df["duration_seconds"].min()),
    int(df["duration_seconds"].max()),
    (int(df["duration_seconds"].min()), int(df["duration_seconds"].max()))      
)


