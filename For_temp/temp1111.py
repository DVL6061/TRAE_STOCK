import os
from dotenv import load_dotenv
from GoogleNews import GoogleNews
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load environment variables from a .env file (if using other APIs)
load_dotenv()

# --- News Fetching Function ---
def get_stock_news(ticker):
    """
    Fetches the latest news headlines and associated details for a given stock ticker
    using the GoogleNews library, which scrapes news from Google News.
    """
    try:
        googlenews = GoogleNews(lang='en', region='IN')
        googlenews.search(f"{ticker} stock news")
        
        # Get the first page of results
        news_items = googlenews.result()
        
        if not news_items:
            return None, f"No recent news found for {ticker}."
        
        unique_news = []
        seen_titles = set()
        for item in news_items:
            title = item.get("title", "")
            if title and title not in seen_titles:
                unique_news.append({
                    "title": title,
                    "publisher": item.get("media", "N/A"),
                    "link": item.get("link", ""),
                    "published_date": item.get("date", "N/A")
                })
                seen_titles.add(title)

        if not unique_news:
            return None, f"No news to analyze for {ticker}."
            
        return unique_news, None
        
    except Exception as e:
        return None, f"Error fetching news from Google News: {e}"

# --- Sentiment Analysis with Hugging Face ---
def analyze_sentiment_huggingface(news_headlines):
    """
    Performs sentiment analysis on news headlines using a pre-trained
    Hugging Face model (FinBERT), returning sentiment labels and scores.
    """
    if not news_headlines:
        return []

    # Load the tokenizer and model from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    # Prepare the headlines for the model
    headlines = [item['title'] for item in news_headlines]
    
    # Tokenize and run the model
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the sentiment labels and scores
    predictions = torch.argmax(outputs.logits, dim=1)
    labels = [model.config.id2label[pred.item()] for pred in predictions]
    
    # Apply softmax to get probability scores
    scores = F.softmax(outputs.logits, dim=1)

    sentiment_results = []
    for i, headline in enumerate(headlines):
        sentiment_label = labels[i].capitalize()
        
        # Get the score for the predicted sentiment
        score = scores[i][predictions[i]].item()
        
        sentiment_results.append({
            "headline": headline,
            "sentiment": sentiment_label,
            "score": score
        })
    
    return sentiment_results

def combine_results(sentiment_results, initial_news):
    """
    Combines the sentiment analysis results with the initial news details
    (publisher, link, published date).
    """
    if not sentiment_results:
        return []

    combined_list = []
    
    # Create a mapping for easier combination
    sentiment_map = {item.get('headline', ''): {'sentiment': item.get('sentiment', 'Undetermined'), 'score': item.get('score', 0.0)} for item in sentiment_results}
    
    for news_item in initial_news:
        headline = news_item.get("title", "")
        sentiment_info = sentiment_map.get(headline, {'sentiment': "Undetermined", 'score': 0.0})
        
        combined_list.append({
            "headline": headline,
            "sentiment": sentiment_info['sentiment'],
            "score": f"{sentiment_info['score']:.4f}",  # Format score to 4 decimal places
            "publisher": news_item.get("publisher"),
            "published_date": news_item.get("published_date"),
            "link": news_item.get("link")
        })
        
    return combined_list

if __name__ == "__main__":
    stock_ticker = input("Enter a stock ticker (e.g., RELIANCE, TCS): ").upper()

    print(f"\nFetching recent news for {stock_ticker}...")
    
    news, error = get_stock_news(stock_ticker)
    
    if error:
        print(error)
    elif news:
        print(f"Analyzing sentiment for {len(news)} news items...")
        
        # Use the Hugging Face sentiment analysis function
        sentiment_analysis_results = analyze_sentiment_huggingface(news)
        
        if sentiment_analysis_results:
            final_news_list = combine_results(sentiment_analysis_results, news)
            
            print("\n--- Sentiment Analysis Results ---")
            for item in final_news_list:
                print(f"Headline: {item['headline']}")
                print(f"Sentiment: {item['sentiment']} (Score: {item['score']})")
                print(f"Source: {item['publisher']}")
                print(f"Published: {item['published_date']}")
                print(f"Link: {item['link']}")
                print("-" * 20)
        else:
            print("Failed to get sentiment analysis results.")
    else:
        print("No news to analyze.")




#RUN THAY CHE 
'''import os
import google.generativeai as genai
from dotenv import load_dotenv
from GoogleNews import GoogleNews
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load environment variables from a .env file (if using other APIs)
load_dotenv()

# --- News Fetching Function (Remains the same) ---
def get_stock_news(ticker):
    """
    Fetches the latest news headlines for a given stock ticker
    using the GoogleNews library.
    """
    try:
        googlenews = GoogleNews(lang='en', region='IN')
        googlenews.search(f"{ticker} stock news")
        
        # Get the first page of results
        news_items = googlenews.result()
        
        if not news_items:
            return None, f"No recent news found for {ticker}."
        
        unique_news = []
        seen_titles = set()
        for item in news_items:
            title = item.get("title", "")
            if title and title not in seen_titles:
                unique_news.append({
                    "title": title,
                    "publisher": item.get("media", "N/A"),
                    "link": item.get("link", "")
                })
                seen_titles.add(title)

        if not unique_news:
            return None, f"No news to analyze for {ticker}."
            
        return unique_news, None
        
    except Exception as e:
        return None, f"Error fetching news from Google News: {e}"

# --- Sentiment Analysis with Hugging Face ---
def analyze_sentiment_huggingface(news_headlines):
    """
    Performs sentiment analysis on news headlines using a pre-trained
    Hugging Face model (FinBERT) without requiring an API key.
    """
    if not news_headlines:
        return []

    # Load the tokenizer and model from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    # Prepare the headlines for the model
    headlines = [item['title'] for item in news_headlines]
    
    # Tokenize and run the model
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the sentiment labels
    predictions = torch.argmax(outputs.logits, dim=1)
    labels = [model.config.id2label[pred.item()] for pred in predictions]

    sentiment_results = []
    for i, headline in enumerate(headlines):
        # Map FinBERT's labels to our required format
        sentiment_label = labels[i].capitalize()
        sentiment_results.append({
            "headline": headline,
            "sentiment": sentiment_label
        })
    
    return sentiment_results

def classify_news(sentiment_results, initial_news):
    """
    Combines the sentiment analysis results with the initial news details.
    """
    if not sentiment_results:
        return []

    classified_news = []
    sentiment_map = {item.get('headline', ''): item.get('sentiment', 'Undetermined') for item in sentiment_results}
    
    for news_item in initial_news:
        headline = news_item.get("title", "")
        sentiment = sentiment_map.get(headline, "Undetermined")
        classified_news.append({
            "headline": headline,
            "sentiment": sentiment,
            "publisher": news_item.get("publisher"),
            "link": news_item.get("link")
        })
    return classified_news


if __name__ == "__main__":
    stock_ticker = input("Enter a stock ticker (e.g., RELIANCE, TCS): ").upper()

    print(f"\nFetching recent news for {stock_ticker}...")
    
    news, error = get_stock_news(stock_ticker)
    
    if error:
        print(error)
    elif news:
        print(f"Analyzing sentiment for {len(news)} news items...")
        
        # Use the Hugging Face sentiment analysis function
        sentiment_analysis_results = analyze_sentiment_huggingface(news)
        
        if sentiment_analysis_results:
            classified_news_list = classify_news(sentiment_analysis_results, news)
            
            print("\n--- Sentiment Analysis Results ---")
            for item in classified_news_list:
                print(f"Headline: {item['headline']}")
                print(f"Sentiment: {item['sentiment']}")
                print(f"Source: {item['publisher']}")
                print("-" * 20)
        else:
            print("Failed to get sentiment analysis results.")
    else:
        print("No news to analyze.")
'''