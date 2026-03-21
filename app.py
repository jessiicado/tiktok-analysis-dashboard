import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
import warnings
warnings.filterwarnings("ignore")

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src.load_data import load_sentiment
from src.sentiment import run_sentiment

df = load_sentiment()
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
    ["All", "low followers", "high followers"]
)

def safe_int(series):
    val = series.mean()
    return f"{int(val):,}" if pd.notna(val) else "N/A"

def safe_float(series, decimals=2):
    val = series.mean()
    return f"{val:.{decimals}f}" if pd.notna(val) else "N/A"
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

dff = df.copy()
if creator_tier != "All":
    dff = dff[dff["creator_tier"] == creator_tier]
dff = dff[dff["sentiment_label"].isin(sentiment_filter)]
dff = dff[
    (dff["duration_seconds"] >= duration_range[0]) &
    (dff["duration_seconds"] <= duration_range[1])
]

st.title("TikTok Analytics Dashboard")
st.caption(f"Showing {len(dff):,} videos after filters")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Videos", f"{len(dff):,}")
col2.metric("Avg Views", safe_int(dff['views']))
col3.metric("Avg likes", safe_int(dff['likes']))
col4.metric("Avg sentiment", safe_float(dff['sentiment']))

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment breakdown")
    counts = dff["sentiment_label"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.pie(counts, names="sentiment", values="count", color="sentiment", 
                 color_discrete_map={ "positive": "#1D9E75",
                     "neutral":  "#888780",
                     "negative": "#E24B4A"
                     })
    fig.update_layout(margin=dict(t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Avg views by sentiment")
    avg_views = dff.groupby("sentiment_label")["views"].mean().reset_index()
    fig = px.bar(avg_views, x="sentiment_label", y="views",
                 color = "sentiment_label",
                 color_discrete_map={ "positive": "#1D9E75",
                     "neutral":  "#888780",
                     "negative": "#E24B4A"
                     })
    fig.update_layout(showlegend=False, margin=dict(t=0))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Sentiment score vs. views")
fig = px.scatter(
    dff, x="sentiment", y="views",
    color="sentiment_label",
    color_discrete_map={ 
        "positive": "#1D9E75",
        "neutral":  "#888780",
         "negative": "#E24B4A"
    },
    opacity=0.4,
    trendline="ols",
    labels={"sentiment": "Sentiment Score", "views": "Views"}         
)

fig.update_layout(margin=dict(t=0))
st.plotly_chart(fig, use_container_width=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Duration vs. Views")
    fig = px.scatter(
        dff, x="duration_seconds", y="views", opacity=0.3, 
        labels={"duration_seconds": "Duration (seconds)", "views": "Views"}
    )
    fig.update_traces(marker_color="#7F77DD")
    fig.update_layout(margin=dict(t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Follower count vs. views")
    fig = px.scatter(
        dff, x="author_followers", y="views", 
        opacity=0.3, 
        labels={"author_followers": "Author_followers", "views": "Views"}
    )
    fig.update_traces(marker_color="#7F77DD")
    fig.update_layout(margin=dict(t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Views over time")
time_df = dff.dropna(subset=["human_time"]).copy()
time_df["date"] = time_df["human_time"].dt.date
daily = time_df.groupby("date")["views"].sum().reset_index()
fig = px.line(daily, x="date", y="views",
              labels={"date": "Date", "views": "Total views"})
fig.update_traces(line_color="#7F77DD")
fig.update_layout(margin=dict(t=0))
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Data explorer")
st.dataframe(
    dff[["text_part", "sentiment", "sentiment_label",
         "views", "likes", "engagement_rate",
         "duration_seconds", "author_followers"]].sort_values("views", ascending=False),
         use_container_width=True,
         height=300
)