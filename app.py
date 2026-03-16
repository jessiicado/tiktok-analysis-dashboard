import streamlit as st
import pandas as pd
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon", quiet=True)

st.set_page_config(page_title="Tiktok Analytics", layout="wide")
st.title("TikTok Analytics Dashboard")
st.write("Setup successful. Ready to Build.")
