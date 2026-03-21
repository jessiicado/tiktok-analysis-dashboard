import pandas as pd
import nltk

nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_sentiment_score(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(str(text))["compound"]

def get_sentiment_label(score):
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"
    
def run_sentiment(df, text_col="text_part"):
    sia = SentimentIntensityAnalyzer()
    df["sentiment"] = df[text_col].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    df["sentiment_label"] = df["sentiment"].apply(get_sentiment_label)
    print(f"Sentiment analysis complete")
    print(df["sentiment_label"].value_counts())
    return df