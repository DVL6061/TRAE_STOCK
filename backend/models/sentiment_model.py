import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import logging
from datetime import datetime, timedelta
import re
import joblib
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    BertTokenizer,
    BertForSequenceClassification
)
from torch.nn.functional import softmax
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class SentimentModel:
    """
    Sentiment analysis model for financial news using FinGPT and FinBERT.
    """
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the sentiment analysis model.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Sentiment labels mapping
        self.labels = ['negative', 'neutral', 'positive']
        self.label_mapping = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral', 
            'LABEL_2': 'positive',
            'negative': 'negative',
            'neutral': 'neutral',
            'positive': 'positive'
        }
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Load model
        self.load_model()
    
    def load_model(self) -> None:
        """
        Load the pre-trained FinBERT/FinGPT model from Hugging Face.
        """
        try:
            logger.info(f"Loading financial sentiment model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == 'cuda' else -1,
                return_all_scores=True
            )
            
            logger.info(f"Financial sentiment model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            logger.info("Falling back to rule-based sentiment analysis")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze the sentiment of a text using FinBERT/FinGPT.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment label and confidence score
        """
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        if self.pipeline is not None:
            try:
                # Use the transformer model for sentiment analysis
                results = self.pipeline(preprocessed_text)
                
                # Extract the best prediction
                best_result = max(results[0], key=lambda x: x['score'])
                
                # Map label to our format
                sentiment = self.label_mapping.get(best_result['label'], best_result['label'].lower())
                confidence = float(best_result['score'])
                
                return {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'sentiment_score': self._convert_to_score(sentiment, confidence),
                    'all_scores': {self.label_mapping.get(r['label'], r['label'].lower()): r['score'] for r in results[0]}
                }
                
            except Exception as e:
                logger.error(f"Error in model prediction: {str(e)}")
                # Fall back to rule-based approach
                return self._rule_based_sentiment(preprocessed_text)
        else:
            # Use rule-based approach as fallback
            return self._rule_based_sentiment(preprocessed_text)
    
    def _rule_based_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Rule-based sentiment analysis as fallback.
        
        Args:
            text: Preprocessed text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Define positive and negative keywords
        positive_keywords = ['up', 'rise', 'gain', 'profit', 'growth', 'positive', 'bullish', 'outperform', 
                            'beat', 'strong', 'success', 'improve', 'increase', 'higher', 'rally', 'surge',
                            'buy', 'upgrade', 'target', 'recommend', 'optimistic', 'boost', 'expand']
        negative_keywords = ['down', 'fall', 'loss', 'decline', 'negative', 'bearish', 'underperform', 
                            'miss', 'weak', 'fail', 'worsen', 'decrease', 'lower', 'drop', 'plunge',
                            'sell', 'downgrade', 'cut', 'warning', 'concern', 'risk', 'crash']
        
        # Count occurrences of positive and negative keywords
        positive_count = sum(1 for keyword in positive_keywords if keyword in text)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text)
        
        # Determine sentiment based on keyword counts
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = min(0.9, 0.5 + 0.1 * (positive_count - negative_count))
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = min(0.9, 0.5 + 0.1 * (negative_count - positive_count))
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'sentiment_score': self._convert_to_score(sentiment, confidence)
        }
    
    def _convert_to_score(self, sentiment: str, confidence: float) -> float:
        """
        Convert sentiment label and confidence to a numerical score.
        
        Args:
            sentiment: Sentiment label ('positive', 'neutral', 'negative')
            confidence: Confidence score
            
        Returns:
            Numerical sentiment score in the range [-1, 1]
        """
        if sentiment == 'positive':
            return confidence
        elif sentiment == 'negative':
            return -confidence
        else:  # neutral
            return 0.0
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Analyze sentiment for a batch of texts efficiently.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment analysis results
        """
        if not texts:
            return []
        
        if self.pipeline is not None:
            try:
                # Preprocess all texts
                preprocessed_texts = [self.preprocess_text(text) for text in texts]
                
                # Batch processing with the pipeline
                batch_results = self.pipeline(preprocessed_texts)
                
                results = []
                for i, text_results in enumerate(batch_results):
                    best_result = max(text_results, key=lambda x: x['score'])
                    sentiment = self.label_mapping.get(best_result['label'], best_result['label'].lower())
                    confidence = float(best_result['score'])
                    
                    results.append({
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'sentiment_score': self._convert_to_score(sentiment, confidence),
                        'all_scores': {self.label_mapping.get(r['label'], r['label'].lower()): r['score'] for r in text_results}
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"Error in batch prediction: {str(e)}")
                # Fall back to individual analysis
                return [self.analyze_sentiment(text) for text in texts]
        else:
            # Use individual analysis as fallback
            return [self.analyze_sentiment(text) for text in texts]
    
    async def async_analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Asynchronously analyze sentiment of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.analyze_sentiment, text)
    
    async def async_batch_analyze(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Asynchronously analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment analysis results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.batch_analyze, texts)
    
    def analyze_news_impact(self, news_data: pd.DataFrame, ticker: str = None) -> Dict[str, Union[float, int]]:
        """
        Analyze the overall sentiment impact of news articles.
        
        Args:
            news_data: DataFrame containing news articles with 'title', 'content', and optionally 'ticker' columns
            ticker: Optional ticker symbol to filter news by
            
        Returns:
            Dictionary with overall sentiment metrics
        """
        if news_data.empty:
            return {
                'overall_score': 0.0,
                'positive_count': 0,
                'neutral_count': 0,
                'negative_count': 0,
                'impact_score': 0.0
            }
        
        # Filter by ticker if provided
        if ticker is not None and 'ticker' in news_data.columns:
            filtered_news = news_data[news_data['ticker'] == ticker]
            if filtered_news.empty:
                filtered_news = news_data  # Use all news if no ticker-specific news found
        else:
            filtered_news = news_data
        
        # Analyze sentiment for each news article
        sentiments = []
        for _, row in filtered_news.iterrows():
            # Combine title and content for better context
            text = f"{row['title']} {row.get('content', '')}"
            sentiment_result = self.analyze_sentiment(text)
            sentiments.append(sentiment_result)
        
        # Calculate overall metrics
        sentiment_scores = [result['sentiment_score'] for result in sentiments]
        overall_score = np.mean(sentiment_scores) if sentiments else 0.0
        
        positive_count = sum(1 for result in sentiments if result['sentiment'] == 'positive')
        neutral_count = sum(1 for result in sentiments if result['sentiment'] == 'neutral')
        negative_count = sum(1 for result in sentiments if result['sentiment'] == 'negative')
        
        # Calculate impact score (weighted by recency and confidence)
        # In a real implementation, you would consider recency of news
        impact_score = overall_score * (1 + 0.1 * len(sentiments))  # Simple scaling by number of articles
        
        return {
            'overall_score': overall_score,
            'positive_count': positive_count,
            'neutral_count': neutral_count,
            'negative_count': negative_count,
            'impact_score': impact_score
        }
    
    def get_sentiment_trend(self, news_data: pd.DataFrame, ticker: str = None, days: int = 7) -> Dict[str, List]:
        """
        Get sentiment trend over time.
        
        Args:
            news_data: DataFrame containing news articles with 'title', 'content', 'date', and optionally 'ticker' columns
            ticker: Optional ticker symbol to filter news by
            days: Number of days to include in the trend
            
        Returns:
            Dictionary with dates and sentiment scores
        """
        if news_data.empty or 'date' not in news_data.columns:
            return {
                'dates': [],
                'sentiment_scores': [],
                'positive_counts': [],
                'neutral_counts': [],
                'negative_counts': []
            }
        
        # Filter by ticker if provided
        if ticker is not None and 'ticker' in news_data.columns:
            filtered_news = news_data[news_data['ticker'] == ticker]
            if filtered_news.empty:
                filtered_news = news_data  # Use all news if no ticker-specific news found
        else:
            filtered_news = news_data
        
        # Ensure date column is datetime
        filtered_news['date'] = pd.to_datetime(filtered_news['date'])
        
        # Get date range
        end_date = filtered_news['date'].max()
        start_date = end_date - timedelta(days=days-1)
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Initialize results
        dates = [date.strftime('%Y-%m-%d') for date in date_range]
        sentiment_scores = []
        positive_counts = []
        neutral_counts = []
        negative_counts = []
        
        # Calculate sentiment for each day
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            day_news = filtered_news[filtered_news['date'].dt.strftime('%Y-%m-%d') == date_str]
            
            if day_news.empty:
                sentiment_scores.append(0.0)
                positive_counts.append(0)
                neutral_counts.append(0)
                negative_counts.append(0)
                continue
            
            # Analyze sentiment for each news article
            day_sentiments = []
            for _, row in day_news.iterrows():
                text = f"{row['title']} {row.get('content', '')}"
                sentiment_result = self.analyze_sentiment(text)
                day_sentiments.append(sentiment_result)
            
            # Calculate metrics for the day
            day_scores = [result['sentiment_score'] for result in day_sentiments]
            day_score = np.mean(day_scores) if day_sentiments else 0.0
            
            day_positive = sum(1 for result in day_sentiments if result['sentiment'] == 'positive')
            day_neutral = sum(1 for result in day_sentiments if result['sentiment'] == 'neutral')
            day_negative = sum(1 for result in day_sentiments if result['sentiment'] == 'negative')
            
            sentiment_scores.append(day_score)
            positive_counts.append(day_positive)
            neutral_counts.append(day_neutral)
            negative_counts.append(day_negative)
        
        return {
            'dates': dates,
            'sentiment_scores': sentiment_scores,
            'positive_counts': positive_counts,
            'neutral_counts': neutral_counts,
            'negative_counts': negative_counts
        }
    
    def translate_sentiment_explanation(self, text: str, sentiment: str, language: str = 'english') -> str:
        """
        Generate an explanation of the sentiment analysis in the specified language.
        
        Args:
            text: The analyzed text
            sentiment: The sentiment label
            language: Target language ('english' or 'hindi')
            
        Returns:
            Explanation of the sentiment analysis
        """
        # In a real implementation, you would use a translation model or API
        # For this placeholder, we'll provide predefined explanations
        
        if language.lower() == 'english':
            if sentiment == 'positive':
                return f"The text contains positive financial indicators suggesting optimistic market outlook."
            elif sentiment == 'negative':
                return f"The text contains negative financial indicators suggesting pessimistic market outlook."
            else:  # neutral
                return f"The text contains balanced or neutral financial indicators."
        elif language.lower() == 'hindi':
            if sentiment == 'positive':
                return f"टेक्स्ट में सकारात्मक वित्तीय संकेतक हैं जो आशावादी बाजार दृष्टिकोण का सुझाव देते हैं।"
            elif sentiment == 'negative':
                return f"टेक्स्ट में नकारात्मक वित्तीय संकेतक हैं जो निराशावादी बाजार दृष्टिकोण का सुझाव देते हैं।"
            else:  # neutral
                return f"टेक्स्ट में संतुलित या तटस्थ वित्तीय संकेतक हैं।"
        else:
            return f"Sentiment analysis: {sentiment}"