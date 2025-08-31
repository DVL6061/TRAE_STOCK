import React, { useState, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { toast } from 'react-toastify';
import useWebSocket from '../hooks/useWebSocket';

// Sentiment color mapping
const getSentimentColor = (sentiment, score) => {
  if (sentiment === 'positive') {
    return score > 0.7 ? 'text-green-600 bg-green-100' : 'text-green-500 bg-green-50';
  } else if (sentiment === 'negative') {
    return score > 0.7 ? 'text-red-600 bg-red-100' : 'text-red-500 bg-red-50';
  } else {
    return 'text-gray-600 bg-gray-100';
  }
};

// Sentiment icon component
const SentimentIcon = ({ sentiment, score }) => {
  const getIcon = () => {
    if (sentiment === 'positive') {
      return score > 0.7 ? '😊' : '🙂';
    } else if (sentiment === 'negative') {
      return score > 0.7 ? '😞' : '😐';
    } else {
      return '😐';
    }
  };

  return (
    <span className="text-lg mr-2" title={`${sentiment} (${(score * 100).toFixed(1)}%)`}>
      {getIcon()}
    </span>
  );
};

// News item component
const NewsItem = ({ article, onSymbolClick }) => {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(false);

  const formatTimeAgo = (timestamp) => {
    const now = new Date();
    const articleTime = new Date(timestamp);
    const diffInMinutes = Math.floor((now - articleTime) / (1000 * 60));
    
    if (diffInMinutes < 1) return t('justNow');
    if (diffInMinutes < 60) return t('minutesAgo', { count: diffInMinutes });
    
    const diffInHours = Math.floor(diffInMinutes / 60);
    if (diffInHours < 24) return t('hoursAgo', { count: diffInHours });
    
    const diffInDays = Math.floor(diffInHours / 24);
    return t('daysAgo', { count: diffInDays });
  };

  const sentimentColorClass = getSentimentColor(article.sentiment, article.sentiment_score);

  return (
    <div className="bg-white dark:bg-gray-700 rounded-lg shadow-md p-4 mb-4 border-l-4 border-blue-500 hover:shadow-lg transition-shadow">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center">
          <SentimentIcon sentiment={article.sentiment} score={article.sentiment_score} />
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white text-sm md:text-base line-clamp-2">
              {article.title}
            </h3>
            <div className="flex items-center text-xs text-gray-500 dark:text-gray-400 mt-1">
              <span className="mr-3">{article.source}</span>
              <span>{formatTimeAgo(article.published_at)}</span>
            </div>
          </div>
        </div>
        
        {/* Sentiment Badge */}
        <div className={`px-2 py-1 rounded-full text-xs font-medium ${sentimentColorClass}`}>
          {t(article.sentiment)} ({(article.sentiment_score * 100).toFixed(0)}%)
        </div>
      </div>

      {/* Content */}
      <div className="mb-3">
        <p className={`text-gray-700 dark:text-gray-300 text-sm ${
          expanded ? '' : 'line-clamp-3'
        }`}>
          {article.content || article.description}
        </p>
        
        {(article.content || article.description) && (article.content || article.description).length > 200 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-blue-500 hover:text-blue-600 text-xs mt-1 font-medium"
          >
            {expanded ? t('showLess') : t('showMore')}
          </button>
        )}
      </div>

      {/* Related Symbols */}
      {article.related_symbols && article.related_symbols.length > 0 && (
        <div className="mb-3">
          <span className="text-xs text-gray-500 dark:text-gray-400 mr-2">
            {t('relatedStocks')}:
          </span>
          <div className="flex flex-wrap gap-1">
            {article.related_symbols.map((symbol, index) => (
              <button
                key={index}
                onClick={() => onSymbolClick(symbol)}
                className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded hover:bg-blue-200 transition-colors dark:bg-blue-900 dark:text-blue-200"
              >
                {symbol}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Keywords/Tags */}
      {article.keywords && article.keywords.length > 0 && (
        <div className="mb-3">
          <div className="flex flex-wrap gap-1">
            {article.keywords.slice(0, 5).map((keyword, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded dark:bg-gray-600 dark:text-gray-300"
              >
                #{keyword}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
        <div className="flex items-center space-x-4">
          {article.url && (
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 hover:text-blue-600 font-medium"
            >
              {t('readFull')}
            </a>
          )}
          
          {article.impact_score && (
            <span className="flex items-center">
              <span className="mr-1">📊</span>
              {t('impact')}: {(article.impact_score * 100).toFixed(0)}%
            </span>
          )}
        </div>
        
        {article.category && (
          <span className="px-2 py-1 bg-gray-200 text-gray-700 rounded dark:bg-gray-600 dark:text-gray-300">
            {t(article.category)}
          </span>
        )}
      </div>
    </div>
  );
};

// Sentiment summary component
const SentimentSummary = ({ articles }) => {
  const { t } = useTranslation();
  
  const sentimentStats = useMemo(() => {
    if (!articles.length) return { positive: 0, negative: 0, neutral: 0, avgScore: 0 };
    
    const stats = articles.reduce((acc, article) => {
      acc[article.sentiment] = (acc[article.sentiment] || 0) + 1;
      acc.totalScore += article.sentiment_score || 0;
      return acc;
    }, { positive: 0, negative: 0, neutral: 0, totalScore: 0 });
    
    stats.avgScore = stats.totalScore / articles.length;
    return stats;
  }, [articles]);

  const total = articles.length;
  if (total === 0) return null;

  return (
    <div className="bg-white dark:bg-gray-700 rounded-lg shadow-md p-4 mb-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        {t('sentimentOverview')}
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-green-600">
            {sentimentStats.positive}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {t('positive')} ({((sentimentStats.positive / total) * 100).toFixed(1)}%)
          </div>
        </div>
        
        <div className="text-center">
          <div className="text-2xl font-bold text-red-600">
            {sentimentStats.negative}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {t('negative')} ({((sentimentStats.negative / total) * 100).toFixed(1)}%)
          </div>
        </div>
        
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-600">
            {sentimentStats.neutral}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {t('neutral')} ({((sentimentStats.neutral / total) * 100).toFixed(1)}%)
          </div>
        </div>
        
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">
            {(sentimentStats.avgScore * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {t('avgConfidence')}
          </div>
        </div>
      </div>
      
      {/* Sentiment Bar */}
      <div className="mt-4">
        <div className="flex h-2 bg-gray-200 rounded-full overflow-hidden">
          <div 
            className="bg-green-500" 
            style={{ width: `${(sentimentStats.positive / total) * 100}%` }}
          ></div>
          <div 
            className="bg-red-500" 
            style={{ width: `${(sentimentStats.negative / total) * 100}%` }}
          ></div>
          <div 
            className="bg-gray-400" 
            style={{ width: `${(sentimentStats.neutral / total) * 100}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
};

// Main News Sentiment UI Component
const NewsSentimentUI = ({ selectedSymbol, onSymbolChange }) => {
  const { t } = useTranslation();
  const [articles, setArticles] = useState([]);
  const [filteredArticles, setFilteredArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    sentiment: 'all',
    category: 'all',
    timeRange: '24h',
    source: 'all'
  });
  const [searchQuery, setSearchQuery] = useState('');

  // WebSocket connection for real-time news
  const { lastMessage, connectionStatus } = useWebSocket(
    'ws://localhost:8000/ws/news',
    { maxReconnectAttempts: 5, reconnectInterval: 3000 }
  );

  // Process incoming WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      const { type, data } = lastMessage;
      
      switch (type) {
        case 'news_update':
          setArticles(prev => {
            const newArticles = Array.isArray(data) ? data : [data];
            const combined = [...newArticles, ...prev];
            // Remove duplicates and keep latest 100 articles
            const unique = combined.filter((article, index, self) => 
              index === self.findIndex(a => a.id === article.id)
            ).slice(0, 100);
            return unique;
          });
          setLoading(false);
          break;
          
        case 'sentiment_update':
          setArticles(prev => prev.map(article => 
            article.id === data.article_id 
              ? { ...article, sentiment: data.sentiment, sentiment_score: data.score }
              : article
          ));
          break;
          
        case 'error':
          toast.error(`News Error: ${data.message}`);
          break;
      }
    }
  }, [lastMessage]);

  // Filter articles based on current filters
  useEffect(() => {
    let filtered = articles;

    // Filter by sentiment
    if (filters.sentiment !== 'all') {
      filtered = filtered.filter(article => article.sentiment === filters.sentiment);
    }

    // Filter by category
    if (filters.category !== 'all') {
      filtered = filtered.filter(article => article.category === filters.category);
    }

    // Filter by time range
    if (filters.timeRange !== 'all') {
      const now = new Date();
      const timeRangeHours = {
        '1h': 1,
        '6h': 6,
        '24h': 24,
        '7d': 168
      }[filters.timeRange];
      
      if (timeRangeHours) {
        const cutoff = new Date(now.getTime() - timeRangeHours * 60 * 60 * 1000);
        filtered = filtered.filter(article => new Date(article.published_at) > cutoff);
      }
    }

    // Filter by source
    if (filters.source !== 'all') {
      filtered = filtered.filter(article => article.source === filters.source);
    }

    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(article => 
        article.title.toLowerCase().includes(query) ||
        (article.content && article.content.toLowerCase().includes(query)) ||
        (article.keywords && article.keywords.some(keyword => 
          keyword.toLowerCase().includes(query)
        ))
      );
    }

    // Filter by selected symbol
    if (selectedSymbol && selectedSymbol !== 'all') {
      filtered = filtered.filter(article => 
        article.related_symbols && article.related_symbols.includes(selectedSymbol)
      );
    }

    setFilteredArticles(filtered);
  }, [articles, filters, searchQuery, selectedSymbol]);

  // Get unique sources and categories for filter options
  const filterOptions = useMemo(() => {
    const sources = [...new Set(articles.map(a => a.source))].sort();
    const categories = [...new Set(articles.map(a => a.category))].filter(Boolean).sort();
    return { sources, categories };
  }, [articles]);

  const handleSymbolClick = (symbol) => {
    if (onSymbolChange) {
      onSymbolChange(symbol);
    }
  };

  return (
    <div className="bg-gray-50 dark:bg-gray-900 min-h-screen p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            {t('newsAndSentiment')}
          </h1>
          
          {/* Connection Status */}
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            connectionStatus === 'Connected' 
              ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
              : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
          }`}>
            {t('newsStream')}: {connectionStatus}
          </div>
        </div>

        {/* Search and Filters */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
          {/* Search Bar */}
          <div className="mb-4">
            <input
              type="text"
              placeholder={t('searchNews')}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            />
          </div>
          
          {/* Filter Controls */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <select
              value={filters.sentiment}
              onChange={(e) => setFilters(prev => ({ ...prev, sentiment: e.target.value }))}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              <option value="all">{t('allSentiments')}</option>
              <option value="positive">{t('positive')}</option>
              <option value="negative">{t('negative')}</option>
              <option value="neutral">{t('neutral')}</option>
            </select>
            
            <select
              value={filters.category}
              onChange={(e) => setFilters(prev => ({ ...prev, category: e.target.value }))}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              <option value="all">{t('allCategories')}</option>
              {filterOptions.categories.map(category => (
                <option key={category} value={category}>{t(category)}</option>
              ))}
            </select>
            
            <select
              value={filters.timeRange}
              onChange={(e) => setFilters(prev => ({ ...prev, timeRange: e.target.value }))}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              <option value="all">{t('allTime')}</option>
              <option value="1h">{t('lastHour')}</option>
              <option value="6h">{t('last6Hours')}</option>
              <option value="24h">{t('last24Hours')}</option>
              <option value="7d">{t('lastWeek')}</option>
            </select>
            
            <select
              value={filters.source}
              onChange={(e) => setFilters(prev => ({ ...prev, source: e.target.value }))}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              <option value="all">{t('allSources')}</option>
              {filterOptions.sources.map(source => (
                <option key={source} value={source}>{source}</option>
              ))}
            </select>
            
            <div className="text-sm text-gray-600 dark:text-gray-400 flex items-center">
              {filteredArticles.length} {t('articles')}
            </div>
          </div>
        </div>
      </div>

      {/* Sentiment Summary */}
      <SentimentSummary articles={filteredArticles} />

      {/* News Articles */}
      <div className="space-y-4">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          </div>
        ) : filteredArticles.length === 0 ? (
          <div className="text-center py-12 text-gray-500 dark:text-gray-400">
            <p className="text-lg">{t('noNewsFound')}</p>
            <p className="text-sm mt-2">{t('tryDifferentFilters')}</p>
          </div>
        ) : (
          filteredArticles.map((article, index) => (
            <NewsItem
              key={article.id || index}
              article={article}
              onSymbolClick={handleSymbolClick}
            />
          ))
        )}
      </div>
    </div>
  );
};

export default NewsSentimentUI;