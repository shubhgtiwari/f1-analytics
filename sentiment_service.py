import feedparser
from textblob import TextBlob
import pandas as pd

def fetch_f1_news():
    """
    Fan Sentiment Analysis
    Fetches latest F1 news from Autosport RSS and calculates sentiment polarity.
    """
    # RSS Feed URL (Autosport F1)
    rss_url = "https://www.autosport.com/rss/feed/f1"
    
    feed = feedparser.parse(rss_url)
    
    news_data = []
    
    print(f"--- Fetching {len(feed.entries)} headlines... ---")
    
    for entry in feed.entries:
        title = entry.title
        summary = entry.description
        
        # AI Sentiment Analysis
        # Polarity: -1 (Negative) to +1 (Positive)
        blob = TextBlob(title + " " + summary)
        sentiment_score = blob.sentiment.polarity
        
        # Classify sentiment
        if sentiment_score > 0.1:
            mood = "Positive"
        elif sentiment_score < -0.1:
            mood = "Negative"
        else:
            mood = "Neutral"
            
        news_data.append({
            'title': title,
            'link': entry.link,
            'published': entry.published,
            'sentiment_score': sentiment_score,
            'mood': mood
        })
        
    return pd.DataFrame(news_data)

if __name__ == "__main__":
    df = fetch_f1_news()
    print(df[['title', 'mood', 'sentiment_score']].head())