import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import {
  FiSearch,
  FiFilter,
  FiFileText,
  FiExternalLink,
  FiTrendingUp,
  FiTrendingDown,
  FiRefreshCw,
  FiCalendar,
  FiInfo,
  FiBarChart2,
  FiPieChart,
} from 'react-icons/fi';

// Mock news data
const mockNewsData = [
  {
    id: 1,
    title: 'Tata Motors reports strong Q4 results, profit jumps 46%',
    source: 'Economic Times',
    date: '2023-05-12T10:30:00Z',
    sentiment: 'positive',
    sentimentScore: 0.85,
    url: '#',
    impact: 'high',
    summary: 'Tata Motors reported a 46% increase in quarterly profit, driven by strong sales in the domestic market and improved performance of JLR.',
    relatedStocks: [
      { symbol: 'TATAMOTORS.NS', name: 'Tata Motors Ltd' },
      { symbol: 'TATASTEEL.NS', name: 'Tata Steel Ltd' },
    ],
    category: 'earnings',
  },
  {
    id: 2,
    title: 'Tata Motors launches new electric vehicle model with 500km range',
    source: 'Mint',
    date: '2023-05-08T14:15:00Z',
    sentiment: 'positive',
    sentimentScore: 0.78,
    url: '#',
    impact: 'medium',
    summary: 'Tata Motors has launched a new electric vehicle model with an impressive 500km range on a single charge, positioning itself as a leader in the EV segment.',
    relatedStocks: [
      { symbol: 'TATAMOTORS.NS', name: 'Tata Motors Ltd' },
    ],
    category: 'product_launch',
  },
  {
    id: 3,
    title: 'Global chip shortage continues to impact auto production',
    source: 'Business Standard',
    date: '2023-05-05T09:45:00Z',
    sentiment: 'negative',
    sentimentScore: -0.62,
    url: '#',
    impact: 'medium',
    summary: 'The ongoing global semiconductor shortage continues to affect production schedules of major automakers including Tata Motors.',
    relatedStocks: [
      { symbol: 'TATAMOTORS.NS', name: 'Tata Motors Ltd' },
      { symbol: 'M&M.NS', name: 'Mahindra & Mahindra Ltd' },
      { symbol: 'MARUTI.NS', name: 'Maruti Suzuki India Ltd' },
    ],
    category: 'supply_chain',
  },
  {
    id: 4,
    title: 'Tata Motors increases market share in commercial vehicle segment',
    source: 'CNBC',
    date: '2023-05-02T16:20:00Z',
    sentiment: 'positive',
    sentimentScore: 0.72,
    url: '#',
    impact: 'medium',
    summary: 'Tata Motors has increased its market share in the commercial vehicle segment to 45%, strengthening its leadership position.',
    relatedStocks: [
      { symbol: 'TATAMOTORS.NS', name: 'Tata Motors Ltd' },
      { symbol: 'ASHOKLEY.NS', name: 'Ashok Leyland Ltd' },
    ],
    category: 'market_share',
  },
  {
    id: 5,
    title: 'RBI keeps repo rate unchanged at 6.5%, maintains stance',
    source: 'Moneycontrol',
    date: '2023-05-10T11:00:00Z',
    sentiment: 'neutral',
    sentimentScore: 0.05,
    url: '#',
    impact: 'medium',
    summary: 'The Reserve Bank of India (RBI) has kept the repo rate unchanged at 6.5% in its latest monetary policy meeting, maintaining its stance on withdrawal of accommodation.',
    relatedStocks: [
      { symbol: 'NIFTY.NS', name: 'Nifty 50' },
      { symbol: 'SENSEX.NS', name: 'BSE Sensex' },
      { symbol: 'HDFCBANK.NS', name: 'HDFC Bank Ltd' },
      { symbol: 'ICICIBANK.NS', name: 'ICICI Bank Ltd' },
    ],
    category: 'economic_policy',
  },
  {
    id: 6,
    title: 'Infosys wins $1.5 billion deal from global telecom giant',
    source: 'Economic Times',
    date: '2023-05-11T08:30:00Z',
    sentiment: 'positive',
    sentimentScore: 0.91,
    url: '#',
    impact: 'high',
    summary: 'Infosys has secured a $1.5 billion deal from a global telecom company for digital transformation services, marking one of its largest deals in recent years.',
    relatedStocks: [
      { symbol: 'INFY.NS', name: 'Infosys Ltd' },
      { symbol: 'TCS.NS', name: 'Tata Consultancy Services Ltd' },
      { symbol: 'WIPRO.NS', name: 'Wipro Ltd' },
    ],
    category: 'deals',
  },
  {
    id: 7,
    title: 'Reliance Industries plans to expand renewable energy portfolio',
    source: 'Mint',
    date: '2023-05-09T13:45:00Z',
    sentiment: 'positive',
    sentimentScore: 0.82,
    url: '#',
    impact: 'medium',
    summary: 'Reliance Industries has announced plans to significantly expand its renewable energy portfolio with an investment of $10 billion over the next three years.',
    relatedStocks: [
      { symbol: 'RELIANCE.NS', name: 'Reliance Industries Ltd' },
      { symbol: 'ADANIGREEN.NS', name: 'Adani Green Energy Ltd' },
    ],
    category: 'investment',
  },
  {
    id: 8,
    title: 'HDFC Bank completes merger with HDFC Ltd',
    source: 'Business Standard',
    date: '2023-05-07T10:15:00Z',
    sentiment: 'positive',
    sentimentScore: 0.75,
    url: '#',
    impact: 'high',
    summary: 'HDFC Bank has completed its merger with HDFC Ltd, creating one of the largest banks in the world by market capitalization.',
    relatedStocks: [
      { symbol: 'HDFCBANK.NS', name: 'HDFC Bank Ltd' },
      { symbol: 'HDFC.NS', name: 'Housing Development Finance Corporation Ltd' },
    ],
    category: 'merger_acquisition',
  },
];

// Mock sentiment trends
const mockSentimentTrends = {
  daily: [
    { date: '2023-05-01', positive: 45, neutral: 35, negative: 20 },
    { date: '2023-05-02', positive: 50, neutral: 30, negative: 20 },
    { date: '2023-05-03', positive: 55, neutral: 25, negative: 20 },
    { date: '2023-05-04', positive: 40, neutral: 35, negative: 25 },
    { date: '2023-05-05', positive: 35, neutral: 40, negative: 25 },
    { date: '2023-05-06', positive: 45, neutral: 35, negative: 20 },
    { date: '2023-05-07', positive: 60, neutral: 25, negative: 15 },
    { date: '2023-05-08', positive: 65, neutral: 20, negative: 15 },
    { date: '2023-05-09', positive: 55, neutral: 30, negative: 15 },
    { date: '2023-05-10', positive: 50, neutral: 35, negative: 15 },
    { date: '2023-05-11', positive: 60, neutral: 25, negative: 15 },
    { date: '2023-05-12', positive: 65, neutral: 20, negative: 15 },
  ],
  sectors: [
    { name: 'Technology', positive: 65, neutral: 25, negative: 10 },
    { name: 'Banking', positive: 55, neutral: 30, negative: 15 },
    { name: 'Automotive', positive: 50, neutral: 30, negative: 20 },
    { name: 'Pharma', positive: 60, neutral: 25, negative: 15 },
    { name: 'Energy', positive: 45, neutral: 35, negative: 20 },
    { name: 'FMCG', positive: 50, neutral: 40, negative: 10 },
  ],
  topStocks: [
    { symbol: 'INFY.NS', name: 'Infosys Ltd', positive: 75, neutral: 15, negative: 10 },
    { symbol: 'RELIANCE.NS', name: 'Reliance Industries Ltd', positive: 70, neutral: 20, negative: 10 },
    { symbol: 'HDFCBANK.NS', name: 'HDFC Bank Ltd', positive: 65, neutral: 25, negative: 10 },
    { symbol: 'TATAMOTORS.NS', name: 'Tata Motors Ltd', positive: 60, neutral: 25, negative: 15 },
    { symbol: 'TCS.NS', name: 'Tata Consultancy Services Ltd', positive: 60, neutral: 30, negative: 10 },
  ],
};

const NewsPage = () => {
  const { t } = useTranslation();
  const [isLoading, setIsLoading] = useState(true);
  const [newsData, setNewsData] = useState([]);
  const [sentimentTrends, setSentimentTrends] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSentiment, setSelectedSentiment] = useState('all');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedImpact, setSelectedImpact] = useState('all');
  const [selectedStock, setSelectedStock] = useState('all');
  const [sortBy, setSortBy] = useState('latest');
  const [showFilters, setShowFilters] = useState(false);
  
  // Simulate data loading
  useEffect(() => {
    setIsLoading(true);
    
    // Simulate API call delay
    const timer = setTimeout(() => {
      setNewsData(mockNewsData);
      setSentimentTrends(mockSentimentTrends);
      setIsLoading(false);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Filter and sort news
  const filteredNews = newsData.filter(news => {
    // Search query filter
    if (searchQuery && !news.title.toLowerCase().includes(searchQuery.toLowerCase()) && 
        !news.summary.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false;
    }
    
    // Sentiment filter
    if (selectedSentiment !== 'all' && news.sentiment !== selectedSentiment) {
      return false;
    }
    
    // Category filter
    if (selectedCategory !== 'all' && news.category !== selectedCategory) {
      return false;
    }
    
    // Impact filter
    if (selectedImpact !== 'all' && news.impact !== selectedImpact) {
      return false;
    }
    
    // Stock filter
    if (selectedStock !== 'all' && !news.relatedStocks.some(stock => stock.symbol === selectedStock)) {
      return false;
    }
    
    return true;
  }).sort((a, b) => {
    if (sortBy === 'latest') {
      return new Date(b.date) - new Date(a.date);
    } else if (sortBy === 'oldest') {
      return new Date(a.date) - new Date(b.date);
    } else if (sortBy === 'sentiment_positive') {
      return b.sentimentScore - a.sentimentScore;
    } else if (sortBy === 'sentiment_negative') {
      return a.sentimentScore - b.sentimentScore;
    } else if (sortBy === 'impact') {
      const impactOrder = { high: 3, medium: 2, low: 1 };
      return impactOrder[b.impact] - impactOrder[a.impact];
    }
    return 0;
  });
  
  // Get unique categories
  const categories = [...new Set(newsData.map(news => news.category))];
  
  // Get unique stocks
  const stocks = [...new Set(newsData.flatMap(news => news.relatedStocks.map(stock => JSON.stringify(stock))))];
  const uniqueStocks = stocks.map(stock => JSON.parse(stock));
  
  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
  };
  
  // Loading skeleton
  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="animate-pulse">
          <div className="h-8 bg-neutral-200 dark:bg-neutral-700 rounded w-1/3 mb-6"></div>
          <div className="h-12 bg-neutral-200 dark:bg-neutral-700 rounded mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="h-64 bg-neutral-200 dark:bg-neutral-700 rounded"></div>
            <div className="h-64 bg-neutral-200 dark:bg-neutral-700 rounded md:col-span-2"></div>
          </div>
          <div className="h-8 bg-neutral-200 dark:bg-neutral-700 rounded w-1/4 mb-4"></div>
          <div className="space-y-4">
            {[...Array(5)].map((_, index) => (
              <div key={index} className="h-32 bg-neutral-200 dark:bg-neutral-700 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="container mx-auto px-3 sm:px-4 py-4 sm:py-8">
      {/* Header */}
      <div className="mb-4 sm:mb-8">
        <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold text-neutral-900 dark:text-neutral-100 mb-2">
          {t('financial_news_and_sentiment')}
        </h1>
        <p className="text-sm sm:text-base text-neutral-500 dark:text-neutral-400">
          {t('news_page_description')}
        </p>
      </div>
      
      {/* Search and filters */}
      <div className="mb-4 sm:mb-8">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 sm:gap-4 mb-3 sm:mb-4">
          <div className="relative lg:w-1/3">
            <input
              type="text"
              className="input w-full pl-8 sm:pl-10 text-sm sm:text-base"
              placeholder={t('search_news')}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <div className="absolute inset-y-0 left-0 flex items-center pl-2 sm:pl-3 pointer-events-none">
              <FiSearch className="text-neutral-500 h-4 w-4 sm:h-5 sm:w-5" />
            </div>
          </div>
          
          <div className="flex flex-wrap gap-2 sm:gap-3">
            <select
              className="input py-2 pl-2 sm:pl-3 pr-8 sm:pr-10 text-sm sm:text-base min-w-0 flex-1 sm:flex-none"
              value={selectedSentiment}
              onChange={(e) => setSelectedSentiment(e.target.value)}
            >
              <option value="all">{t('all_sentiment')}</option>
              <option value="positive">{t('positive')}</option>
              <option value="neutral">{t('neutral')}</option>
              <option value="negative">{t('negative')}</option>
            </select>
            
            <select
              className="input py-2 pl-2 sm:pl-3 pr-8 sm:pr-10 text-sm sm:text-base min-w-0 flex-1 sm:flex-none"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="latest">{t('latest')}</option>
              <option value="oldest">{t('oldest')}</option>
              <option value="sentiment_positive">{t('most_positive')}</option>
              <option value="sentiment_negative">{t('most_negative')}</option>
              <option value="impact">{t('highest_impact')}</option>
            </select>
            
            <button
              className="btn btn-outline flex items-center text-sm sm:text-base px-2 sm:px-4 py-2"
              onClick={() => setShowFilters(!showFilters)}
            >
              <FiFilter className="mr-1 sm:mr-2 h-4 w-4" />
              <span className="hidden sm:inline">{t('filters')}</span>
            </button>
            
            <button className="btn btn-outline flex items-center text-sm sm:text-base px-2 sm:px-4 py-2">
              <FiRefreshCw className="mr-1 sm:mr-2 h-4 w-4" />
              <span className="hidden sm:inline">{t('refresh')}</span>
            </button>
          </div>
        </div>
        
        {showFilters && (
          <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-3 sm:p-4 border border-neutral-200 dark:border-neutral-700 mb-3 sm:mb-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
              <div>
                <label className="block text-xs sm:text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                  {t('category')}
                </label>
                <select
                  className="input w-full text-sm sm:text-base"
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                >
                  <option value="all">{t('all_categories')}</option>
                  {categories.map(category => (
                    <option key={category} value={category}>
                      {t(category)}
                    </option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-xs sm:text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                  {t('impact')}
                </label>
                <select
                  className="input w-full text-sm sm:text-base"
                  value={selectedImpact}
                  onChange={(e) => setSelectedImpact(e.target.value)}
                >
                  <option value="all">{t('all_impact')}</option>
                  <option value="high">{t('high_impact')}</option>
                  <option value="medium">{t('medium_impact')}</option>
                  <option value="low">{t('low_impact')}</option>
                </select>
              </div>
              
              <div className="sm:col-span-2 lg:col-span-1">
                <label className="block text-xs sm:text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                  {t('related_stock')}
                </label>
                <select
                  className="input w-full text-sm sm:text-base"
                  value={selectedStock}
                  onChange={(e) => setSelectedStock(e.target.value)}
                >
                  <option value="all">{t('all_stocks')}</option>
                  {uniqueStocks.map(stock => (
                    <option key={stock.symbol} value={stock.symbol}>
                      {stock.name} ({stock.symbol})
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}
        
        <div className="text-sm text-neutral-500 dark:text-neutral-400">
          {t('showing_results', { count: filteredNews.length, total: newsData.length })}
        </div>
      </div>
      
      {/* Sentiment overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6 mb-6 md:mb-8">
        <div className="card p-4 md:p-6">
          <h3 className="text-base md:text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-3 md:mb-4 flex items-center">
            <FiPieChart className="mr-2 h-4 w-4 md:h-5 md:w-5 text-primary-500" />
            {t('sentiment_distribution')}
          </h3>
          
          <div className="space-y-3 md:space-y-4">
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs md:text-sm text-success-600 dark:text-success-400 font-medium">{t('positive')}</span>
                <span className="text-xs md:text-sm text-neutral-500 dark:text-neutral-400">
                  {sentimentTrends.daily[sentimentTrends.daily.length - 1].positive}%
                </span>
              </div>
              <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-1.5 md:h-2">
                <div
                  className="bg-success-500 h-1.5 md:h-2 rounded-full"
                  style={{ width: `${sentimentTrends.daily[sentimentTrends.daily.length - 1].positive}%` }}
                ></div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs md:text-sm text-neutral-600 dark:text-neutral-400 font-medium">{t('neutral')}</span>
                <span className="text-xs md:text-sm text-neutral-500 dark:text-neutral-400">
                  {sentimentTrends.daily[sentimentTrends.daily.length - 1].neutral}%
                </span>
              </div>
              <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-1.5 md:h-2">
                <div
                  className="bg-neutral-500 h-1.5 md:h-2 rounded-full"
                  style={{ width: `${sentimentTrends.daily[sentimentTrends.daily.length - 1].neutral}%` }}
                ></div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs md:text-sm text-danger-600 dark:text-danger-400 font-medium">{t('negative')}</span>
                <span className="text-xs md:text-sm text-neutral-500 dark:text-neutral-400">
                  {sentimentTrends.daily[sentimentTrends.daily.length - 1].negative}%
                </span>
              </div>
              <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-1.5 md:h-2">
                <div
                  className="bg-danger-500 h-1.5 md:h-2 rounded-full"
                  style={{ width: `${sentimentTrends.daily[sentimentTrends.daily.length - 1].negative}%` }}
                ></div>
              </div>
            </div>
          </div>
          
          <div className="mt-4 md:mt-6">
            <h4 className="text-sm md:text-base font-medium text-neutral-900 dark:text-neutral-100 mb-2 md:mb-3">
              {t('top_positive_stocks')}
            </h4>
            
            <div className="space-y-2 md:space-y-3">
              {sentimentTrends.topStocks.slice(0, 3).map((stock, index) => (
                <div key={index} className="flex items-center">
                  <Link
                    to={`/stock/${stock.symbol}`}
                    className="text-xs md:text-sm font-medium text-neutral-900 dark:text-neutral-100 hover:text-primary-600 dark:hover:text-primary-400 truncate flex-1 mr-2"
                  >
                    {stock.name}
                  </Link>
                  <div className="flex items-center flex-shrink-0">
                    <span className="text-xs text-success-600 dark:text-success-400 mr-1">
                      {stock.positive}%
                    </span>
                    <FiTrendingUp className="text-success-500 h-3 w-3 md:h-4 md:w-4" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        
        <div className="card p-4 md:p-6 md:col-span-2">
          <h3 className="text-base md:text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-3 md:mb-4 flex items-center">
            <FiBarChart2 className="mr-2 h-4 w-4 md:h-5 md:w-5 text-primary-500" />
            {t('sentiment_trends')}
          </h3>
          
          <div className="h-48 md:h-64 flex items-center justify-center">
            <div className="text-neutral-500 dark:text-neutral-400 text-center">
              <FiBarChart2 className="h-8 w-8 md:h-12 md:w-12 mx-auto mb-2 opacity-50" />
              <p className="text-sm md:text-base">{t('sentiment_chart_placeholder')}</p>
            </div>
          </div>
          
          <div className="mt-3 md:mt-4 grid grid-cols-2 md:grid-cols-3 gap-2 md:gap-4">
            {sentimentTrends.sectors.slice(0, 6).map((sector, index) => (
              <div key={index} className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-2 md:p-3">
                <div className="text-xs md:text-sm font-medium text-neutral-900 dark:text-neutral-100 mb-1 md:mb-2 truncate">
                  {sector.name}
                </div>
                <div className="flex items-center">
                  <div
                    className={`w-1.5 h-1.5 md:w-2 md:h-2 rounded-full mr-1 ${sector.positive > 50 ? 'bg-success-500' : sector.negative > 30 ? 'bg-danger-500' : 'bg-neutral-500'}`}
                  ></div>
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">
                    {sector.positive}% {t('positive')}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* News list */}
      <div className="mb-6 md:mb-8">
        <h2 className="text-lg md:text-xl font-bold text-neutral-900 dark:text-neutral-100 mb-4 md:mb-6">
          {t('latest_news')}
        </h2>
        
        <div className="space-y-4 md:space-y-6">
          {filteredNews.length > 0 ? (
            filteredNews.map((news) => (
              <div
                key={news.id}
                className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-3 md:p-4 hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors"
              >
                <div className="flex items-start">
                  <div
                    className={`mt-1 flex-shrink-0 w-2.5 h-2.5 md:w-3 md:h-3 rounded-full mr-2 md:mr-3 ${news.sentiment === 'positive' ? 'bg-success-500' : news.sentiment === 'negative' ? 'bg-danger-500' : 'bg-neutral-500'}`}
                  ></div>
                  <div className="flex-1 min-w-0">
                    <a
                      href={news.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-base md:text-lg font-medium text-neutral-900 dark:text-neutral-100 hover:text-primary-600 dark:hover:text-primary-400 flex items-start"
                    >
                      <span className="flex-1">{news.title}</span>
                      <FiExternalLink className="ml-2 h-3 w-3 md:h-4 md:w-4 flex-shrink-0 mt-1" />
                    </a>
                    
                    <div className="flex flex-wrap items-center mt-1 text-xs md:text-sm text-neutral-500 dark:text-neutral-400 gap-1">
                      <span className="font-medium">{news.source}</span>
                      <span className="hidden sm:inline">•</span>
                      <span>{formatDate(news.date)}</span>
                      <span className="hidden sm:inline">•</span>
                      <span
                        className={`${news.sentiment === 'positive' ? 'text-success-600 dark:text-success-400' : news.sentiment === 'negative' ? 'text-danger-600 dark:text-danger-400' : 'text-neutral-600 dark:text-neutral-400'}`}
                      >
                        {t(news.sentiment)} ({Math.abs(news.sentimentScore * 100).toFixed(0)}%)
                      </span>
                      <span
                        className={`px-1.5 md:px-2 py-0.5 rounded-full text-xs ${news.impact === 'high' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : news.impact === 'medium' ? 'bg-warning-100 dark:bg-warning-900 text-warning-800 dark:text-warning-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
                      >
                        {t(`${news.impact}_impact`)}
                      </span>
                      <span
                        className="px-1.5 md:px-2 py-0.5 rounded-full text-xs bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200"
                      >
                        {t(news.category)}
                      </span>
                    </div>
                    
                    <p className="mt-2 text-sm md:text-base text-neutral-600 dark:text-neutral-400 line-clamp-3">
                      {news.summary}
                    </p>
                    
                    <div className="mt-3 space-y-2">
                      <div>
                        <span className="text-xs text-neutral-500 dark:text-neutral-400 mr-2">
                          {t('related_stocks')}:
                        </span>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {news.relatedStocks.map((stock, index) => (
                            <Link
                              key={index}
                              to={`/stock/${stock.symbol}`}
                              className="inline-block px-1.5 md:px-2 py-0.5 md:py-1 bg-neutral-100 dark:bg-neutral-800 rounded text-xs text-neutral-700 dark:text-neutral-300 hover:bg-neutral-200 dark:hover:bg-neutral-700 truncate max-w-32 md:max-w-none"
                            >
                              {stock.name}
                            </Link>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex justify-end">
                        <Link
                          to={`/predictions?news=${news.id}`}
                          className="text-xs md:text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 flex items-center"
                        >
                          {t('see_impact_on_predictions')}
                          <FiTrendingUp className="ml-1 h-3 w-3 md:h-4 md:w-4" />
                        </Link>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-8 md:py-12 border border-neutral-200 dark:border-neutral-700 rounded-lg">
              <FiFileText className="h-8 w-8 md:h-12 md:w-12 mx-auto text-neutral-400 dark:text-neutral-600 mb-3 md:mb-4" />
              <h3 className="text-base md:text-lg font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                {t('no_news_found')}
              </h3>
              <p className="text-sm md:text-base text-neutral-500 dark:text-neutral-400 max-w-md mx-auto px-4">
                {t('no_news_found_description')}
              </p>
              <button
                className="btn btn-primary mt-4 text-sm md:text-base"
                onClick={() => {
                  setSearchQuery('');
                  setSelectedSentiment('all');
                  setSelectedCategory('all');
                  setSelectedImpact('all');
                  setSelectedStock('all');
                }}
              >
                {t('clear_filters')}
              </button>
            </div>
          )}
        </div>
      </div>
      
      {/* Load more button */}
      {filteredNews.length > 0 && (
        <div className="text-center mb-6 md:mb-8">
          <button className="btn btn-outline text-sm md:text-base">
            {t('load_more_news')}
          </button>
        </div>
      )}
      
      {/* Disclaimer */}
      <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-3 md:p-4 border border-neutral-200 dark:border-neutral-700">
        <h4 className="text-sm md:text-base font-medium text-neutral-900 dark:text-neutral-100 mb-2 flex items-center">
          <FiInfo className="mr-2 h-3 w-3 md:h-4 md:w-4 text-warning-500" />
          {t('disclaimer')}
        </h4>
        <p className="text-xs md:text-sm text-neutral-600 dark:text-neutral-400">
          {t('news_disclaimer')}
        </p>
      </div>
    </div>
  );
};

export default NewsPage;