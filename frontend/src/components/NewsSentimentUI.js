import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { toast } from 'react-toastify';
import { Line, Doughnut, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  BarElement,
} from 'chart.js';
import {
  FiTrendingUp,
  FiTrendingDown,
  FiActivity,
  FiBarChart3,
  FiPieChart,
  FiRefreshCw,
  FiFilter,
  FiSearch,
  FiExternalLink,
  FiClock,
  FiTag,
  FiAlertCircle,
  FiInfo,
  FiEye,
  FiEyeOff,
  FiAlertTriangle,
  FiX
} from 'react-icons/fi';
import useWebSocket from '../hooks/useWebSocket';
import axios from 'axios';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  BarElement
);

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
    <div className="bg-white dark:bg-gray-700 rounded-lg shadow-md p-3 sm:p-4 mb-3 sm:mb-4 border-l-4 border-blue-500 hover:shadow-lg transition-shadow">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between mb-2 sm:mb-3 space-y-2 sm:space-y-0">
        <div className="flex items-start space-x-2 sm:space-x-3 flex-1">
          <div className="flex-shrink-0 mt-1">
            <SentimentIcon sentiment={article.sentiment} score={article.sentiment_score} />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-gray-900 dark:text-white text-sm sm:text-base line-clamp-2 leading-tight">
              {article.title}
            </h3>
            <div className="flex flex-col sm:flex-row sm:items-center text-xs text-gray-500 dark:text-gray-400 mt-1 space-y-1 sm:space-y-0">
              <span className="sm:mr-3">{article.source}</span>
              <span className="sm:border-l sm:border-gray-300 sm:pl-3">{formatTimeAgo(article.published_at)}</span>
            </div>
          </div>
        </div>
        
        {/* Sentiment Badge */}
        <div className={`px-2 py-1 rounded-full text-xs font-medium self-start sm:self-auto flex-shrink-0 ${sentimentColorClass}`}>
          <span className="hidden sm:inline">{t(article.sentiment)} </span>({(article.sentiment_score * 100).toFixed(0)}%)
        </div>
      </div>

      {/* Content */}
      <div className="mb-2 sm:mb-3">
        <p className={`text-gray-700 dark:text-gray-300 text-xs sm:text-sm leading-relaxed ${
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
        <div className="mb-2 sm:mb-3">
          <span className="text-xs text-gray-500 dark:text-gray-400 block sm:inline sm:mr-2 mb-1 sm:mb-0">
            {t('relatedStocks')}:
          </span>
          <div className="flex flex-wrap gap-1 sm:gap-2">
            {article.related_symbols.map((symbol, index) => (
              <button
                key={index}
                onClick={() => onSymbolClick(symbol)}
                className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded hover:bg-blue-200 transition-colors dark:bg-blue-900 dark:text-blue-200 font-medium"
              >
                {symbol}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Keywords/Tags */}
      {article.keywords && article.keywords.length > 0 && (
        <div className="mb-2 sm:mb-3">
          <div className="flex flex-wrap gap-1 sm:gap-2">
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
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between text-xs text-gray-500 dark:text-gray-400 space-y-2 sm:space-y-0">
        <div className="flex flex-col sm:flex-row sm:items-center space-y-1 sm:space-y-0 sm:space-x-4">
          {article.url && (
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 hover:text-blue-600 font-medium inline-flex items-center"
            >
              <span className="mr-1">🔗</span>
              {t('readFull')}
            </a>
          )}
          
          {article.impact_score && (
            <span className="flex items-center">
              <span className="mr-1">📊</span>
              <span className="hidden sm:inline">{t('impact')}: </span>{(article.impact_score * 100).toFixed(0)}%
            </span>
          )}
        </div>
        
        {article.category && (
          <span className="px-2 py-1 bg-gray-200 text-gray-700 rounded dark:bg-gray-600 dark:text-gray-300 self-start sm:self-auto">
            {t(article.category)}
          </span>
        )}
      </div>
    </div>
  );
};

// Advanced Sentiment Trend Chart Component
const SentimentTrendChart = ({ articles, timeRange = '24h' }) => {
  const { t } = useTranslation();
  
  const chartData = useMemo(() => {
    if (!articles.length) return null;
    
    // Group articles by time intervals
    const now = new Date();
    const intervals = timeRange === '24h' ? 24 : timeRange === '7d' ? 7 : 30;
    const intervalMs = timeRange === '24h' ? 60 * 60 * 1000 : 24 * 60 * 60 * 1000;
    
    const timeSlots = Array.from({ length: intervals }, (_, i) => {
      const time = new Date(now.getTime() - (intervals - 1 - i) * intervalMs);
      return {
        time,
        label: timeRange === '24h' 
          ? time.getHours().toString().padStart(2, '0') + ':00'
          : time.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        positive: 0,
        negative: 0,
        neutral: 0,
        avgScore: 0,
        count: 0
      };
    });
    
    // Distribute articles into time slots
    articles.forEach(article => {
      const articleTime = new Date(article.published_at);
      const slotIndex = Math.floor((articleTime.getTime() - (now.getTime() - intervals * intervalMs)) / intervalMs);
      
      if (slotIndex >= 0 && slotIndex < intervals) {
        const slot = timeSlots[slotIndex];
        slot[article.sentiment]++;
        slot.avgScore += article.sentiment_score || 0;
        slot.count++;
      }
    });
    
    // Calculate average scores
    timeSlots.forEach(slot => {
      if (slot.count > 0) {
        slot.avgScore /= slot.count;
      }
    });
    
    return {
      labels: timeSlots.map(slot => slot.label),
      datasets: [
        {
          label: t('positive'),
          data: timeSlots.map(slot => slot.positive),
          borderColor: 'rgb(34, 197, 94)',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          tension: 0.4,
        },
        {
          label: t('negative'),
          data: timeSlots.map(slot => slot.negative),
          borderColor: 'rgb(239, 68, 68)',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          tension: 0.4,
        },
        {
          label: t('neutral'),
          data: timeSlots.map(slot => slot.neutral),
          borderColor: 'rgb(107, 114, 128)',
          backgroundColor: 'rgba(107, 114, 128, 0.1)',
          tension: 0.4,
        }
      ]
    };
  }, [articles, timeRange, t]);
  
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20,
        }
      },
      title: {
        display: true,
        text: t('sentimentTrends'),
        font: { size: 16, weight: 'bold' }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          afterBody: function(context) {
            const dataIndex = context[0].dataIndex;
            const total = context.reduce((sum, item) => sum + item.parsed.y, 0);
            return total > 0 ? [`${t('totalArticles')}: ${total}`] : [];
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: timeRange === '24h' ? t('hours') : t('days')
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: t('articleCount')
        },
        beginAtZero: true
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };
  
  if (!chartData) return null;
  
  return (
    <div className="bg-white dark:bg-gray-700 rounded-lg shadow-md p-4 mb-6">
      <div className="h-64 sm:h-80">
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
};

// Sentiment Distribution Pie Chart
const SentimentDistributionChart = ({ articles }) => {
  const { t } = useTranslation();
  
  const chartData = useMemo(() => {
    if (!articles.length) return null;
    
    const sentimentCounts = articles.reduce((acc, article) => {
      acc[article.sentiment] = (acc[article.sentiment] || 0) + 1;
      return acc;
    }, { positive: 0, negative: 0, neutral: 0 });
    
    return {
      labels: [t('positive'), t('negative'), t('neutral')],
      datasets: [{
        data: [sentimentCounts.positive, sentimentCounts.negative, sentimentCounts.neutral],
        backgroundColor: [
          'rgba(34, 197, 94, 0.8)',
          'rgba(239, 68, 68, 0.8)',
          'rgba(107, 114, 128, 0.8)'
        ],
        borderColor: [
          'rgb(34, 197, 94)',
          'rgb(239, 68, 68)',
          'rgb(107, 114, 128)'
        ],
        borderWidth: 2
      }]
    };
  }, [articles, t]);
  
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          usePointStyle: true,
          padding: 20,
        }
      },
      title: {
        display: true,
        text: t('sentimentDistribution'),
        font: { size: 16, weight: 'bold' }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const total = context.dataset.data.reduce((a, b) => a + b, 0);
            const percentage = ((context.parsed / total) * 100).toFixed(1);
            return `${context.label}: ${context.parsed} (${percentage}%)`;
          }
        }
      }
    }
  };
  
  if (!chartData) return null;
  
  return (
    <div className="bg-white dark:bg-gray-700 rounded-lg shadow-md p-4 mb-6">
      <div className="h-64">
        <Doughnut data={chartData} options={options} />
      </div>
    </div>
  );
};

// News Impact Analysis Component
const NewsImpactAnalysis = ({ articles, selectedSymbol }) => {
  const { t } = useTranslation();
  const [impactData, setImpactData] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const fetchImpactData = useCallback(async () => {
    if (!selectedSymbol || selectedSymbol === 'all') return;
    
    setLoading(true);
    try {
      const response = await axios.get(`/api/news/impact/${selectedSymbol}?days=7`);
      setImpactData(response.data);
    } catch (error) {
      console.error('Error fetching impact data:', error);
      toast.error(t('errorFetchingImpactData'));
    } finally {
      setLoading(false);
    }
  }, [selectedSymbol, t]);
  
  useEffect(() => {
    fetchImpactData();
  }, [fetchImpactData]);
  
  const impactChartData = useMemo(() => {
    if (!impactData?.daily_sentiment) return null;
    
    return {
      labels: impactData.daily_sentiment.map(day => 
        new Date(day.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
      ),
      datasets: [{
        label: t('sentimentScore'),
        data: impactData.daily_sentiment.map(day => day.avg_score),
        backgroundColor: impactData.daily_sentiment.map(day => 
          day.avg_score > 0 ? 'rgba(34, 197, 94, 0.6)' : 'rgba(239, 68, 68, 0.6)'
        ),
        borderColor: impactData.daily_sentiment.map(day => 
          day.avg_score > 0 ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)'
        ),
        borderWidth: 2
      }]
    };
  }, [impactData, t]);
  
  const impactOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: t('dailySentimentImpact'),
        font: { size: 16, weight: 'bold' }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const score = context.parsed.y;
            const impact = score > 0.5 ? t('highPositive') : 
                          score > 0 ? t('positive') : 
                          score > -0.5 ? t('negative') : t('highNegative');
            return `${t('sentimentScore')}: ${score.toFixed(2)} (${impact})`;
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: t('date')
        }
      },
      y: {
        title: {
          display: true,
          text: t('sentimentScore')
        },
        min: -1,
        max: 1,
        ticks: {
          callback: function(value) {
            return value.toFixed(1);
          }
        }
      }
    }
  };
  
  if (!selectedSymbol || selectedSymbol === 'all') {
    return (
      <div className="bg-white dark:bg-gray-700 rounded-lg shadow-md p-4 mb-6">
        <div className="text-center py-8">
          <FiInfo className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <p className="text-gray-600 dark:text-gray-400">{t('selectSymbolForImpact')}</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="bg-white dark:bg-gray-700 rounded-lg shadow-md p-4 mb-6">
      {loading ? (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      ) : impactChartData ? (
        <>
          <div className="h-64 mb-4">
            <Bar data={impactChartData} options={impactOptions} />
          </div>
          
          {/* Impact Summary */}
          {impactData && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-4 border-t border-gray-200 dark:border-gray-600">
              <div className="text-center">
                <div className="text-lg font-bold text-blue-600">
                  {impactData.news_count}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  {t('totalNews')}
                </div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-green-600">
                  {impactData.sentiment_distribution.positive}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  {t('positive')}
                </div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-red-600">
                  {impactData.sentiment_distribution.negative}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  {t('negative')}
                </div>
              </div>
              <div className="text-center">
                <div className={`text-lg font-bold ${
                  impactData.average_sentiment_score > 0 ? 'text-green-600' : 
                  impactData.average_sentiment_score < 0 ? 'text-red-600' : 'text-gray-600'
                }`}>
                  {impactData.average_sentiment_score.toFixed(2)}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  {t('avgSentiment')}
                </div>
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="text-center py-8">
          <FiAlertCircle className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <p className="text-gray-600 dark:text-gray-400">{t('noImpactDataAvailable')}</p>
        </div>
      )}
    </div>
  );
};

// Enhanced Sentiment summary component
const SentimentSummary = ({ articles, showCharts = true }) => {
  const { t } = useTranslation();
  const [activeView, setActiveView] = useState('overview'); // overview, trends, distribution, impact
  
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
    <div className="bg-white dark:bg-gray-700 rounded-lg shadow-md mb-4 sm:mb-6">
      {/* Header with View Toggle */}
      <div className="p-3 sm:p-4 border-b border-gray-200 dark:border-gray-600">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-2 sm:space-y-0">
          <h3 className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
            {t('sentimentAnalysis')}
          </h3>
          
          {showCharts && (
            <div className="flex space-x-1 bg-gray-100 dark:bg-gray-600 rounded-lg p-1">
              <button
                onClick={() => setActiveView('overview')}
                className={`px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-colors ${
                  activeView === 'overview'
                    ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm'
                    : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                <FiBarChart3 className="inline mr-1" />
                {t('overview')}
              </button>
              <button
                onClick={() => setActiveView('trends')}
                className={`px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-colors ${
                  activeView === 'trends'
                    ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm'
                    : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                <FiActivity className="inline mr-1" />
                {t('trends')}
              </button>
              <button
                onClick={() => setActiveView('distribution')}
                className={`px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-colors ${
                  activeView === 'distribution'
                    ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm'
                    : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                <FiPieChart className="inline mr-1" />
                {t('distribution')}
              </button>
            </div>
          )}
        </div>
      </div>
      
      {/* Content */}
      <div className="p-3 sm:p-4">
        {activeView === 'overview' && (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 mb-4">
              <div className="text-center">
                <div className="text-xl sm:text-2xl font-bold text-green-600">
                  {sentimentStats.positive}
                </div>
                <div className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 leading-tight">
                  <div className="sm:hidden">{t('positive')}</div>
                  <div className="hidden sm:block">{t('positive')} ({((sentimentStats.positive / total) * 100).toFixed(1)}%)</div>
                  <div className="sm:hidden text-xs">({((sentimentStats.positive / total) * 100).toFixed(1)}%)</div>
                </div>
              </div>
              
              <div className="text-center">
                <div className="text-xl sm:text-2xl font-bold text-red-600">
                  {sentimentStats.negative}
                </div>
                <div className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 leading-tight">
                  <div className="sm:hidden">{t('negative')}</div>
                  <div className="hidden sm:block">{t('negative')} ({((sentimentStats.negative / total) * 100).toFixed(1)}%)</div>
                  <div className="sm:hidden text-xs">({((sentimentStats.negative / total) * 100).toFixed(1)}%)</div>
                </div>
              </div>
              
              <div className="text-center">
                <div className="text-xl sm:text-2xl font-bold text-gray-600">
                  {sentimentStats.neutral}
                </div>
                <div className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 leading-tight">
                  <div className="sm:hidden">{t('neutral')}</div>
                  <div className="hidden sm:block">{t('neutral')} ({((sentimentStats.neutral / total) * 100).toFixed(1)}%)</div>
                  <div className="sm:hidden text-xs">({((sentimentStats.neutral / total) * 100).toFixed(1)}%)</div>
                </div>
              </div>
              
              <div className="text-center">
                <div className="text-xl sm:text-2xl font-bold text-blue-600">
                  {(sentimentStats.avgScore * 100).toFixed(1)}%
                </div>
                <div className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 leading-tight">
                  <div className="sm:hidden">{t('avgConfidence')}</div>
                  <div className="hidden sm:block">{t('avgConfidence')}</div>
                </div>
              </div>
            </div>
            
            {/* Sentiment Bar */}
            <div className="mt-3 sm:mt-4">
              <div className="flex h-2 sm:h-3 bg-gray-200 rounded-full overflow-hidden">
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
          </>
        )}
        
        {activeView === 'trends' && (
          <SentimentTrendChart articles={articles} />
        )}
        
        {activeView === 'distribution' && (
          <SentimentDistributionChart articles={articles} />
        )}
      </div>
    </div>
  );
};

// Real-time Sentiment Alerts Component
const SentimentAlerts = ({ articles, threshold = 0.7 }) => {
  const { t } = useTranslation();
  const [alerts, setAlerts] = useState([]);
  const [showAlerts, setShowAlerts] = useState(true);
  
  useEffect(() => {
    const highImpactArticles = articles.filter(article => 
      Math.abs(article.sentiment_score || 0) >= threshold && 
      article.impact && article.impact !== 'low'
    );
    
    const newAlerts = highImpactArticles.slice(0, 5).map(article => ({
      id: article.id,
      title: article.title,
      sentiment: article.sentiment,
      score: article.sentiment_score,
      impact: article.impact,
      symbol: article.related_symbols?.[0] || 'Market',
      timestamp: new Date(article.published_at)
    }));
    
    setAlerts(newAlerts);
  }, [articles, threshold]);
  
  if (!alerts.length || !showAlerts) return null;
  
  return (
    <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border border-yellow-200 dark:border-yellow-700 rounded-lg p-4 mb-6">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center">
          <FiAlertTriangle className="h-5 w-5 text-yellow-600 dark:text-yellow-400 mr-2" />
          <h4 className="text-sm font-semibold text-yellow-800 dark:text-yellow-200">
            {t('highImpactSentimentAlerts')}
          </h4>
        </div>
        <button
          onClick={() => setShowAlerts(false)}
          className="text-yellow-600 dark:text-yellow-400 hover:text-yellow-800 dark:hover:text-yellow-200"
        >
          <FiX className="h-4 w-4" />
        </button>
      </div>
      
      <div className="space-y-2">
        {alerts.map(alert => (
          <div key={alert.id} className="flex items-center justify-between bg-white dark:bg-gray-800 rounded-md p-2 text-sm">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                alert.sentiment === 'positive' ? 'bg-green-500' :
                alert.sentiment === 'negative' ? 'bg-red-500' : 'bg-gray-500'
              }`}></div>
              <span className="font-medium text-gray-900 dark:text-white">{alert.symbol}</span>
              <span className="text-gray-600 dark:text-gray-400 truncate max-w-xs">
                {alert.title}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <span className={`text-xs font-medium px-2 py-1 rounded ${
                alert.impact === 'high' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300' :
                alert.impact === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300' :
                'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'
              }`}>
                {t(alert.impact)}
              </span>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {formatTimeAgo(alert.timestamp)}
              </span>
            </div>
          </div>
        ))}
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
    <div className="bg-gray-50 dark:bg-gray-900 min-h-screen p-3 sm:p-4 lg:p-6">
      {/* Header */}
      <div className="mb-4 sm:mb-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-3 sm:mb-4 space-y-2 sm:space-y-0">
          <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold text-gray-900 dark:text-white">
            {t('newsAndSentiment')}
          </h1>
          
          {/* Connection Status */}
          <div className={`px-2 sm:px-3 py-1 rounded-full text-xs sm:text-sm font-medium self-start sm:self-auto ${
            connectionStatus === 'Connected' 
              ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
              : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
          }`}>
            <span className="hidden sm:inline">{t('newsStream')}: </span>{connectionStatus}
          </div>
        </div>

        {/* Search and Filters */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-3 sm:p-4">
          {/* Search Bar */}
          <div className="mb-3 sm:mb-4">
            <input
              type="text"
              placeholder={t('searchNews')}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-3 sm:px-4 py-2 text-sm sm:text-base border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            />
          </div>
          
          {/* Filter Controls */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2 sm:gap-3 lg:gap-4">
            <select
              value={filters.sentiment}
              onChange={(e) => setFilters(prev => ({ ...prev, sentiment: e.target.value }))}
              className="w-full px-2 sm:px-3 py-2 text-sm sm:text-base border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              <option value="all">{t('allSentiments')}</option>
              <option value="positive">{t('positive')}</option>
              <option value="negative">{t('negative')}</option>
              <option value="neutral">{t('neutral')}</option>
            </select>
            
            <select
              value={filters.category}
              onChange={(e) => setFilters(prev => ({ ...prev, category: e.target.value }))}
              className="w-full px-2 sm:px-3 py-2 text-sm sm:text-base border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              <option value="all">{t('allCategories')}</option>
              {filterOptions.categories.map(category => (
                <option key={category} value={category}>{t(category)}</option>
              ))}
            </select>
            
            <select
              value={filters.timeRange}
              onChange={(e) => setFilters(prev => ({ ...prev, timeRange: e.target.value }))}
              className="w-full px-2 sm:px-3 py-2 text-sm sm:text-base border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
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
              className="w-full px-2 sm:px-3 py-2 text-sm sm:text-base border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              <option value="all">{t('allSources')}</option>
              {filterOptions.sources.map(source => (
                <option key={source} value={source}>{source}</option>
              ))}
            </select>
            
            <div className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 flex items-center justify-center sm:justify-start col-span-1 sm:col-span-2 lg:col-span-1">
              {filteredArticles.length} {t('articles')}
            </div>
          </div>
        </div>
      </div>

      {/* High Impact Sentiment Alerts */}
      <SentimentAlerts articles={filteredArticles} />

      {/* Enhanced Sentiment Summary with Charts */}
      <SentimentSummary articles={filteredArticles} showCharts={true} />
      
      {/* Sentiment Trend Chart */}
      <SentimentTrendChart articles={filteredArticles} timeRange={filters.timeRange} />
      
      {/* Sentiment Distribution Chart */}
      <SentimentDistributionChart articles={filteredArticles} />
      
      {/* News Impact Analysis for Selected Symbol */}
      <NewsImpactAnalysis articles={filteredArticles} selectedSymbol={selectedSymbol} />

      {/* News Articles */}
      <div className="space-y-3 sm:space-y-4">
        {loading ? (
          <div className="flex items-center justify-center py-8 sm:py-12">
            <div className="animate-spin rounded-full h-8 w-8 sm:h-12 sm:w-12 border-b-2 border-blue-500"></div>
          </div>
        ) : filteredArticles.length === 0 ? (
          <div className="text-center py-8 sm:py-12 text-gray-500 dark:text-gray-400 px-4">
            <p className="text-base sm:text-lg">{t('noNewsFound')}</p>
            <p className="text-xs sm:text-sm mt-2">{t('tryDifferentFilters')}</p>
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