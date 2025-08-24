import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  TimeScale
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';

// Icons
import {
  FiTrendingUp,
  FiTrendingDown,
  FiBarChart2,
  FiFileText,
  FiStar,
  FiInfo,
  FiClock,
  FiCalendar,
  FiDollarSign,
  FiActivity,
  FiPieChart,
  FiRefreshCw,
  FiAlertCircle,
  FiChevronDown,
  FiChevronUp,
  FiPlus,
  FiMinus
} from 'react-icons/fi';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  TimeScale
);

// Mock data for Tata Motors
const mockStockData = {
  symbol: 'TATAMOTORS.NS',
  name: 'Tata Motors Limited',
  price: 900.50,
  change: 15.75,
  changePercent: 1.78,
  open: 885.25,
  high: 905.80,
  low: 882.10,
  close: 900.50,
  volume: 5432100,
  marketCap: 302500000000,
  peRatio: 28.5,
  dividendYield: 0.8,
  eps: 31.6,
  high52w: 950.25,
  low52w: 720.30,
  avgVolume: 4875000,
  isInWatchlist: true
};

// Mock historical data
const generateHistoricalData = (days) => {
  const data = [];
  const today = new Date();
  let price = 900.50;
  
  for (let i = days; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    
    // Skip weekends
    if (date.getDay() === 0 || date.getDay() === 6) {
      continue;
    }
    
    // Random price change between -2% and 2%
    const change = price * (Math.random() * 0.04 - 0.02);
    price += change;
    
    // Generate OHLC data
    const open = price;
    const high = price * (1 + Math.random() * 0.01);
    const low = price * (1 - Math.random() * 0.01);
    const close = price + (Math.random() * 0.02 - 0.01) * price;
    
    // Random volume between 3M and 6M
    const volume = Math.floor(Math.random() * 3000000) + 3000000;
    
    data.push({
      date: date.toISOString().split('T')[0],
      open,
      high,
      low,
      close,
      volume
    });
    
    price = close;
  }
  
  return data;
};

const mockHistoricalData = generateHistoricalData(30);

// Mock technical indicators
const mockTechnicalIndicators = {
  rsi: {
    value: 62.5,
    signal: 'neutral',
    description: 'RSI is in neutral territory, indicating balanced buying and selling pressure.'
  },
  macd: {
    value: 3.2,
    signal: 'bullish',
    description: 'MACD is above the signal line, suggesting bullish momentum.'
  },
  ema20: {
    value: 885.30,
    signal: 'bullish',
    description: 'Price is above the 20-day EMA, indicating a short-term uptrend.'
  },
  sma50: {
    value: 860.75,
    signal: 'bullish',
    description: 'Price is above the 50-day SMA, suggesting a medium-term uptrend.'
  },
  sma200: {
    value: 820.40,
    signal: 'bullish',
    description: 'Price is above the 200-day SMA, indicating a long-term uptrend.'
  },
  bollingerBands: {
    upper: 920.30,
    middle: 890.50,
    lower: 860.70,
    signal: 'neutral',
    description: 'Price is near the middle band, suggesting a neutral trend.'
  },
  adx: {
    value: 28.5,
    signal: 'strong_trend',
    description: 'ADX above 25 indicates a strong trend is present.'
  },
  stochastic: {
    k: 75.3,
    d: 68.7,
    signal: 'neutral',
    description: 'Stochastic oscillator is in neutral territory, not yet overbought.'
  }
};

// Mock news data
const mockNewsData = [
  {
    id: 1,
    title: 'Tata Motors reports strong Q4 results, profit jumps 46%',
    source: 'Economic Times',
    date: '2023-05-12',
    sentiment: 'positive',
    url: '#',
    impact: 'high',
    summary: 'Tata Motors reported a 46% increase in quarterly profit, driven by strong sales in the domestic market and improved performance of JLR.'
  },
  {
    id: 2,
    title: 'Tata Motors launches new electric vehicle model with 500km range',
    source: 'Mint',
    date: '2023-05-08',
    sentiment: 'positive',
    url: '#',
    impact: 'medium',
    summary: 'Tata Motors has launched a new electric vehicle model with an impressive 500km range on a single charge, positioning itself as a leader in the EV segment.'
  },
  {
    id: 3,
    title: 'Global chip shortage continues to impact auto production',
    source: 'Business Standard',
    date: '2023-05-05',
    sentiment: 'negative',
    url: '#',
    impact: 'medium',
    summary: 'The ongoing global semiconductor shortage continues to affect production schedules of major automakers including Tata Motors.'
  },
  {
    id: 4,
    title: 'Tata Motors increases market share in commercial vehicle segment',
    source: 'CNBC',
    date: '2023-05-02',
    sentiment: 'positive',
    url: '#',
    impact: 'medium',
    summary: 'Tata Motors has increased its market share in the commercial vehicle segment to 45%, strengthening its leadership position.'
  }
];

// Mock predictions
const mockPredictions = {
  shortTerm: {
    prediction: 'buy',
    confidence: 87,
    predictedPrice: 950.25,
    priceRange: { low: 930.50, high: 970.00 },
    timeframe: '1-7 days',
    factors: [
      { name: 'Technical Indicators', impact: 'positive', weight: 35 },
      { name: 'Recent News Sentiment', impact: 'positive', weight: 25 },
      { name: 'Volume Analysis', impact: 'positive', weight: 20 },
      { name: 'Market Trend', impact: 'neutral', weight: 15 },
      { name: 'Sector Performance', impact: 'neutral', weight: 5 }
    ]
  },
  mediumTerm: {
    prediction: 'hold',
    confidence: 65,
    predictedPrice: 920.75,
    priceRange: { low: 880.00, high: 960.00 },
    timeframe: '15-30 days',
    factors: [
      { name: 'Technical Indicators', impact: 'neutral', weight: 30 },
      { name: 'Fundamental Analysis', impact: 'positive', weight: 25 },
      { name: 'Sector Outlook', impact: 'neutral', weight: 20 },
      { name: 'Market Sentiment', impact: 'negative', weight: 15 },
      { name: 'Economic Indicators', impact: 'neutral', weight: 10 }
    ]
  },
  longTerm: {
    prediction: 'strong_buy',
    confidence: 78,
    predictedPrice: 1050.30,
    priceRange: { low: 950.00, high: 1150.00 },
    timeframe: '3-6 months',
    factors: [
      { name: 'Fundamental Analysis', impact: 'positive', weight: 40 },
      { name: 'Industry Growth', impact: 'positive', weight: 25 },
      { name: 'Company Roadmap', impact: 'positive', weight: 15 },
      { name: 'Economic Outlook', impact: 'neutral', weight: 10 },
      { name: 'Competitive Position', impact: 'positive', weight: 10 }
    ]
  }
};

const StockDetail = () => {
  const { ticker } = useParams();
  const { t, i18n } = useTranslation();
  const [isLoading, setIsLoading] = useState(true);
  const [stockData, setStockData] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [technicalIndicators, setTechnicalIndicators] = useState(null);
  const [newsData, setNewsData] = useState([]);
  const [predictions, setPredictions] = useState(null);
  const [timeframe, setTimeframe] = useState('1M'); // 1D, 1W, 1M, 3M, 6M, 1Y, 5Y
  const [chartType, setChartType] = useState('line'); // line, candle
  const [activeTab, setActiveTab] = useState('overview'); // overview, technical, fundamental, news, predictions
  const [isInWatchlist, setIsInWatchlist] = useState(false);
  const [expandedIndicators, setExpandedIndicators] = useState(false);
  const [activePredictionTab, setActivePredictionTab] = useState('short_term'); // short_term, medium_term, long_term

  // Simulate data loading
  useEffect(() => {
    setIsLoading(true);
    
    // Simulate API call delay
    const timer = setTimeout(() => {
      setStockData(mockStockData);
      setHistoricalData(mockHistoricalData);
      setTechnicalIndicators(mockTechnicalIndicators);
      setNewsData(mockNewsData);
      setPredictions(mockPredictions);
      setIsInWatchlist(mockStockData.isInWatchlist);
      setIsLoading(false);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, [ticker]);

  // Handle timeframe change
  const handleTimeframeChange = (newTimeframe) => {
    setTimeframe(newTimeframe);
    // In a real app, this would fetch new historical data based on the timeframe
  };

  // Handle chart type change
  const handleChartTypeChange = (newType) => {
    setChartType(newType);
  };

  // Toggle watchlist status
  const toggleWatchlist = () => {
    setIsInWatchlist(!isInWatchlist);
    // In a real app, this would call an API to update the watchlist
  };

  // Toggle expanded indicators
  const toggleExpandedIndicators = () => {
    setExpandedIndicators(!expandedIndicators);
  };

  // Prepare chart data
  const chartData = {
    labels: historicalData.map(item => item.date),
    datasets: [
      {
        label: stockData?.name || '',
        data: historicalData.map(item => item.close),
        borderColor: 'rgba(59, 130, 246, 1)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 5
      }
    ]
  };

  // Volume chart data
  const volumeChartData = {
    labels: historicalData.map(item => item.date),
    datasets: [
      {
        label: t('volume'),
        data: historicalData.map(item => item.volume),
        backgroundColor: 'rgba(99, 102, 241, 0.5)',
        borderColor: 'rgba(99, 102, 241, 1)',
        borderWidth: 1
      }
    ]
  };

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: function(context) {
            if (context.dataset.label === t('volume')) {
              return `${context.dataset.label}: ${context.raw.toLocaleString()}`;
            }
            return `${context.dataset.label}: ₹${context.raw.toFixed(2)}`;
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          display: false
        }
      },
      y: {
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        },
        ticks: {
          callback: (value) => `₹${value.toLocaleString()}`
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  // Volume chart options
  const volumeChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        mode: 'index',
        intersect: false
      }
    },
    scales: {
      x: {
        display: false
      },
      y: {
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        },
        ticks: {
          callback: (value) => `${(value / 1000000).toFixed(1)}M`
        }
      }
    }
  };

  // Render loading skeleton
  if (isLoading) {
    return (
      <div className="animate-pulse">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
          <div className="h-8 bg-neutral-200 dark:bg-neutral-700 rounded w-1/3 mb-4 md:mb-0"></div>
          <div className="h-10 bg-neutral-200 dark:bg-neutral-700 rounded w-1/4"></div>
        </div>
        
        <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm p-4 mb-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
            <div className="h-6 bg-neutral-200 dark:bg-neutral-700 rounded w-1/4 mb-2 md:mb-0"></div>
            <div className="h-6 bg-neutral-200 dark:bg-neutral-700 rounded w-1/5"></div>
          </div>
          <div className="h-64 bg-neutral-200 dark:bg-neutral-700 rounded mb-4"></div>
          <div className="h-24 bg-neutral-200 dark:bg-neutral-700 rounded"></div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm p-4">
              <div className="h-6 bg-neutral-200 dark:bg-neutral-700 rounded w-1/3 mb-4"></div>
              <div className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-full mb-2"></div>
              <div className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-2/3"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div>
      {/* Page header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center">
            {stockData.name}
            <span className="ml-2 text-sm font-normal text-neutral-500 dark:text-neutral-400">
              {stockData.symbol}
            </span>
          </h1>
          <div className="flex items-center mt-1">
            <span className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
              ₹{stockData.price.toLocaleString()}
            </span>
            <span
              className={`ml-2 text-sm font-medium ${stockData.change > 0 ? 'text-success-600 dark:text-success-400' : 'text-danger-600 dark:text-danger-400'}`}
            >
              {stockData.change > 0 ? '+' : ''}{stockData.change.toFixed(2)} ({stockData.change > 0 ? '+' : ''}{stockData.changePercent.toFixed(2)}%)
            </span>
          </div>
        </div>
        
        <div className="flex items-center space-x-2 mt-4 md:mt-0">
          <button
            onClick={toggleWatchlist}
            className={`btn ${isInWatchlist ? 'btn-warning' : 'btn-outline'} flex items-center`}
          >
            {isInWatchlist ? (
              <>
                <FiMinus className="mr-1 h-4 w-4" />
                {t('remove_from_watchlist')}
              </>
            ) : (
              <>
                <FiPlus className="mr-1 h-4 w-4" />
                {t('add_to_watchlist')}
              </>
            )}
          </button>
          
          <Link
            to={`/predictions?ticker=${stockData.symbol}`}
            className="btn btn-primary flex items-center"
          >
            <FiBarChart2 className="mr-1 h-4 w-4" />
            {t('predict')}
          </Link>
        </div>
      </div>

      {/* Stock chart */}
      <div className="card mb-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
          <div className="flex items-center space-x-2 mb-2 md:mb-0">
            <button
              onClick={() => handleChartTypeChange('line')}
              className={`px-2 py-1 text-xs font-medium rounded ${chartType === 'line' ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-700'}`}
            >
              {t('line')}
            </button>
            <button
              onClick={() => handleChartTypeChange('candle')}
              className={`px-2 py-1 text-xs font-medium rounded ${chartType === 'candle' ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-700'}`}
            >
              {t('candlestick')}
            </button>
          </div>
          
          <div className="flex space-x-2">
            <button
              onClick={() => handleTimeframeChange('1D')}
              className={`px-2 py-1 text-xs font-medium rounded ${timeframe === '1D' ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-700'}`}
            >
              1D
            </button>
            <button
              onClick={() => handleTimeframeChange('1W')}
              className={`px-2 py-1 text-xs font-medium rounded ${timeframe === '1W' ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-700'}`}
            >
              1W
            </button>
            <button
              onClick={() => handleTimeframeChange('1M')}
              className={`px-2 py-1 text-xs font-medium rounded ${timeframe === '1M' ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-700'}`}
            >
              1M
            </button>
            <button
              onClick={() => handleTimeframeChange('3M')}
              className={`px-2 py-1 text-xs font-medium rounded ${timeframe === '3M' ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-700'}`}
            >
              3M
            </button>
            <button
              onClick={() => handleTimeframeChange('6M')}
              className={`px-2 py-1 text-xs font-medium rounded ${timeframe === '6M' ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-700'}`}
            >
              6M
            </button>
            <button
              onClick={() => handleTimeframeChange('1Y')}
              className={`px-2 py-1 text-xs font-medium rounded ${timeframe === '1Y' ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-700'}`}
            >
              1Y
            </button>
          </div>
        </div>
        
        <div className="h-64 mb-4">
          <Line data={chartData} options={chartOptions} />
        </div>
        
        <div className="h-24">
          <Bar data={volumeChartData} options={volumeChartOptions} />
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-neutral-200 dark:border-neutral-700 mb-6">
        <button
          onClick={() => setActiveTab('overview')}
          className={`px-4 py-2 text-sm font-medium ${activeTab === 'overview' ? 'border-b-2 border-primary-500 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'}`}
        >
          {t('overview')}
        </button>
        <button
          onClick={() => setActiveTab('technical')}
          className={`px-4 py-2 text-sm font-medium ${activeTab === 'technical' ? 'border-b-2 border-primary-500 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'}`}
        >
          {t('technical_indicators')}
        </button>
        <button
          onClick={() => setActiveTab('fundamental')}
          className={`px-4 py-2 text-sm font-medium ${activeTab === 'fundamental' ? 'border-b-2 border-primary-500 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'}`}
        >
          {t('fundamentals')}
        </button>
        <button
          onClick={() => setActiveTab('news')}
          className={`px-4 py-2 text-sm font-medium ${activeTab === 'news' ? 'border-b-2 border-primary-500 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'}`}
        >
          {t('news')}
        </button>
        <button
          onClick={() => setActiveTab('predictions')}
          className={`px-4 py-2 text-sm font-medium ${activeTab === 'predictions' ? 'border-b-2 border-primary-500 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'}`}
        >
          {t('predictions')}
        </button>
      </div>

      {/* Tab content */}
      {activeTab === 'overview' && (
        <div>
          {/* Stock info cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            {/* Price info */}
            <div className="card">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
                <FiDollarSign className="mr-2 h-5 w-5 text-primary-500" />
                {t('price_information')}
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('open')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">₹{stockData.open.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('high')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">₹{stockData.high.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('low')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">₹{stockData.low.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('close')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">₹{stockData.close.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('volume')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">{stockData.volume.toLocaleString()}</span>
                </div>
              </div>
            </div>

            {/* Key stats */}
            <div className="card">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
                <FiActivity className="mr-2 h-5 w-5 text-primary-500" />
                {t('key_statistics')}
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('market_cap')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">₹{(stockData.marketCap / 10000000).toFixed(2)} Cr</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('pe_ratio')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">{stockData.peRatio.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('eps')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">₹{stockData.eps.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('dividend_yield')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">{stockData.dividendYield.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('avg_volume')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">{stockData.avgVolume.toLocaleString()}</span>
                </div>
              </div>
            </div>

            {/* 52 week range */}
            <div className="card">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
                <FiCalendar className="mr-2 h-5 w-5 text-primary-500" />
                {t('52_week_range')}
              </h3>
              <div className="mb-4">
                <div className="flex justify-between mb-2">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('52w_low')}</span>
                  <span className="text-neutral-600 dark:text-neutral-400">{t('52w_high')}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">₹{stockData.low52w.toLocaleString()}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">₹{stockData.high52w.toLocaleString()}</span>
                </div>
              </div>
              
              {/* Progress bar showing current price in 52w range */}
              <div className="mt-4">
                <div className="relative h-2 bg-neutral-200 dark:bg-neutral-700 rounded-full">
                  <div
                    className="absolute h-2 bg-primary-500 rounded-full"
                    style={{
                      width: `${((stockData.price - stockData.low52w) / (stockData.high52w - stockData.low52w) * 100).toFixed(2)}%`
                    }}
                  ></div>
                  <div
                    className="absolute w-2 h-4 bg-primary-600 rounded-full -mt-1"
                    style={{
                      left: `calc(${((stockData.price - stockData.low52w) / (stockData.high52w - stockData.low52w) * 100).toFixed(2)}% - 4px)`
                    }}
                  ></div>
                </div>
                <div className="flex justify-between mt-2 text-xs text-neutral-500 dark:text-neutral-400">
                  <span>{t('52w_low')}</span>
                  <span>{t('current')}: ₹{stockData.price.toLocaleString()}</span>
                  <span>{t('52w_high')}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Technical indicators summary */}
          <div className="card mb-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 flex items-center">
                <FiBarChart2 className="mr-2 h-5 w-5 text-primary-500" />
                {t('technical_indicators_summary')}
              </h3>
              <button
                onClick={() => setActiveTab('technical')}
                className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300"
              >
                {t('view_all')}
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {Object.entries(technicalIndicators).slice(0, 4).map(([key, indicator]) => (
                <div key={key} className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">
                      {t(key.toLowerCase())}
                    </span>
                    <span
                      className={`text-xs font-medium px-2 py-0.5 rounded-full ${indicator.signal === 'bullish' ? 'bg-success-100 dark:bg-success-900 text-success-800 dark:text-success-200' : indicator.signal === 'bearish' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
                    >
                      {t(indicator.signal)}
                    </span>
                  </div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    {key === 'bollingerBands' ? (
                      <span>U: {indicator.upper.toFixed(2)} M: {indicator.middle.toFixed(2)} L: {indicator.lower.toFixed(2)}</span>
                    ) : key === 'stochastic' ? (
                      <span>K: {indicator.k.toFixed(2)} D: {indicator.d.toFixed(2)}</span>
                    ) : (
                      <span>{indicator.value.toFixed(2)}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Latest news */}
          <div className="card mb-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 flex items-center">
                <FiFileText className="mr-2 h-5 w-5 text-primary-500" />
                {t('latest_news')}
              </h3>
              <button
                onClick={() => setActiveTab('news')}
                className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300"
              >
                {t('view_all')}
              </button>
            </div>
            
            <div className="space-y-4">
              {newsData.slice(0, 2).map((news) => (
                <div
                  key={news.id}
                  className="border-b border-neutral-200 dark:border-neutral-700 last:border-0 pb-3 last:pb-0"
                >
                  <div className="flex items-start mb-2">
                    <div
                      className={`mt-1 flex-shrink-0 w-2 h-2 rounded-full mr-2 ${news.sentiment === 'positive' ? 'bg-success-500' : news.sentiment === 'negative' ? 'bg-danger-500' : 'bg-neutral-500'}`}
                    ></div>
                    <div>
                      <a
                        href={news.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-medium text-neutral-900 dark:text-neutral-100 hover:text-primary-600 dark:hover:text-primary-400"
                      >
                        {news.title}
                      </a>
                      <div className="flex items-center mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                        <span>{news.source}</span>
                        <span className="mx-1">•</span>
                        <span>{new Date(news.date).toLocaleDateString()}</span>
                        <span className="mx-1">•</span>
                        <span
                          className={`${news.sentiment === 'positive' ? 'text-success-600 dark:text-success-400' : news.sentiment === 'negative' ? 'text-danger-600 dark:text-danger-400' : 'text-neutral-600 dark:text-neutral-400'}`}
                        >
                          {t(news.sentiment)}
                        </span>
                      </div>
                    </div>
                  </div>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400 ml-4">
                    {news.summary}
                  </p>
                  <div className="flex justify-between items-center mt-2 ml-4">
                    <div
                      className={`text-xs px-2 py-0.5 rounded-full ${news.impact === 'high' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : news.impact === 'medium' ? 'bg-warning-100 dark:bg-warning-900 text-warning-800 dark:text-warning-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
                    >
                      {t(`${news.impact}_impact`)}
                    </div>
                    <a
                      href={news.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300"
                    >
                      {t('read_more')} →
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Prediction summary */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 flex items-center">
                <FiPieChart className="mr-2 h-5 w-5 text-primary-500" />
                {t('prediction_summary')}
              </h3>
              <button
                onClick={() => setActiveTab('predictions')}
                className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300"
              >
                {t('view_details')}
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Short-term prediction */}
              <div className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-neutral-600 dark:text-neutral-400">
                    {t('short_term')} ({predictions.shortTerm.timeframe})
                  </span>
                  <div
                    className={`px-2 py-1 rounded-full text-xs font-medium ${predictions.shortTerm.prediction === 'buy' || predictions.shortTerm.prediction === 'strong_buy' ? 'bg-success-100 dark:bg-success-900 text-success-800 dark:text-success-200' : predictions.shortTerm.prediction === 'sell' || predictions.shortTerm.prediction === 'strong_sell' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
                  >
                    {t(predictions.shortTerm.prediction)}
                  </div>
                </div>
                <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                  ₹{predictions.shortTerm.predictedPrice.toLocaleString()}
                </div>
                <div className="text-xs text-neutral-500 dark:text-neutral-400">
                  {t('range')}: ₹{predictions.shortTerm.priceRange.low.toLocaleString()} - ₹{predictions.shortTerm.priceRange.high.toLocaleString()}
                </div>
                <div className="flex items-center mt-2">
                  <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-1.5 mr-2">
                    <div
                      className={`h-1.5 rounded-full ${predictions.shortTerm.confidence >= 80 ? 'bg-success-500' : predictions.shortTerm.confidence >= 60 ? 'bg-warning-500' : 'bg-danger-500'}`}
                      style={{ width: `${predictions.shortTerm.confidence}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-neutral-500 dark:text-neutral-400">
                    {predictions.shortTerm.confidence}%
                  </span>
                </div>
              </div>

              {/* Medium-term prediction */}
              <div className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-neutral-600 dark:text-neutral-400">
                    {t('medium_term')} ({predictions.mediumTerm.timeframe})
                  </span>
                  <div
                    className={`px-2 py-1 rounded-full text-xs font-medium ${predictions.mediumTerm.prediction === 'buy' || predictions.mediumTerm.prediction === 'strong_buy' ? 'bg-success-100 dark:bg-success-900 text-success-800 dark:text-success-200' : predictions.mediumTerm.prediction === 'sell' || predictions.mediumTerm.prediction === 'strong_sell' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
                  >
                    {t(predictions.mediumTerm.prediction)}
                  </div>
                </div>
                <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                  ₹{predictions.mediumTerm.predictedPrice.toLocaleString()}
                </div>
                <div className="text-xs text-neutral-500 dark:text-neutral-400">
                  {t('range')}: ₹{predictions.mediumTerm.priceRange.low.toLocaleString()} - ₹{predictions.mediumTerm.priceRange.high.toLocaleString()}
                </div>
                <div className="flex items-center mt-2">
                  <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-1.5 mr-2">
                    <div
                      className={`h-1.5 rounded-full ${predictions.mediumTerm.confidence >= 80 ? 'bg-success-500' : predictions.mediumTerm.confidence >= 60 ? 'bg-warning-500' : 'bg-danger-500'}`}
                      style={{ width: `${predictions.mediumTerm.confidence}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-neutral-500 dark:text-neutral-400">
                    {predictions.mediumTerm.confidence}%
                  </span>
                </div>
              </div>

              {/* Long-term prediction */}
              <div className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-neutral-600 dark:text-neutral-400">
                    {t('long_term')} ({predictions.longTerm.timeframe})
                  </span>
                  <div
                    className={`px-2 py-1 rounded-full text-xs font-medium ${predictions.longTerm.prediction === 'buy' || predictions.longTerm.prediction === 'strong_buy' ? 'bg-success-100 dark:bg-success-900 text-success-800 dark:text-success-200' : predictions.longTerm.prediction === 'sell' || predictions.longTerm.prediction === 'strong_sell' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
                  >
                    {t(predictions.longTerm.prediction)}
                  </div>
                </div>
                <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                  ₹{predictions.longTerm.predictedPrice.toLocaleString()}
                </div>
                <div className="text-xs text-neutral-500 dark:text-neutral-400">
                  {t('range')}: ₹{predictions.longTerm.priceRange.low.toLocaleString()} - ₹{predictions.longTerm.priceRange.high.toLocaleString()}
                </div>
                <div className="flex items-center mt-2">
                  <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-1.5 mr-2">
                    <div
                      className={`h-1.5 rounded-full ${predictions.longTerm.confidence >= 80 ? 'bg-success-500' : predictions.longTerm.confidence >= 60 ? 'bg-warning-500' : 'bg-danger-500'}`}
                      style={{ width: `${predictions.longTerm.confidence}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-neutral-500 dark:text-neutral-400">
                    {predictions.longTerm.confidence}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'technical' && (
        <div className="card">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 flex items-center">
              <FiBarChart2 className="mr-2 h-5 w-5 text-primary-500" />
              {t('technical_indicators')}
            </h3>
            <button
              onClick={toggleExpandedIndicators}
              className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 flex items-center"
            >
              {expandedIndicators ? t('show_less') : t('show_more')}
              {expandedIndicators ? <FiChevronUp className="ml-1" /> : <FiChevronDown className="ml-1" />}
            </button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(technicalIndicators).map(([key, indicator]) => (
              <div key={key} className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {t(key.toLowerCase())}
                  </span>
                  <span
                    className={`text-xs font-medium px-2 py-0.5 rounded-full ${indicator.signal === 'bullish' ? 'bg-success-100 dark:bg-success-900 text-success-800 dark:text-success-200' : indicator.signal === 'bearish' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
                  >
                    {t(indicator.signal)}
                  </span>
                </div>
                
                <div className="text-neutral-900 dark:text-neutral-100 font-bold mb-2">
                  {key === 'bollingerBands' ? (
                    <div className="flex flex-col">
                      <span>Upper: ₹{indicator.upper.toFixed(2)}</span>
                      <span>Middle: ₹{indicator.middle.toFixed(2)}</span>
                      <span>Lower: ₹{indicator.lower.toFixed(2)}</span>
                    </div>
                  ) : key === 'stochastic' ? (
                    <div className="flex flex-col">
                      <span>%K: {indicator.k.toFixed(2)}</span>
                      <span>%D: {indicator.d.toFixed(2)}</span>
                    </div>
                  ) : (
                    <span>{indicator.value.toFixed(2)}</span>
                  )}
                </div>
                
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  {indicator.description}
                </p>
                
                {expandedIndicators && (
                  <div className="mt-4 pt-4 border-t border-neutral-200 dark:border-neutral-700">
                    <h4 className="text-sm font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                      {t('interpretation')}
                    </h4>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      {key === 'rsi' && (
                        <>RSI values above 70 indicate overbought conditions, while values below 30 indicate oversold conditions. Current value of {indicator.value.toFixed(2)} suggests {indicator.value > 70 ? 'potential reversal or correction' : indicator.value < 30 ? 'potential buying opportunity' : 'balanced market conditions'}.</>
                      )}
                      {key === 'macd' && (
                        <>MACD crossing above the signal line is bullish, while crossing below is bearish. Current value of {indicator.value.toFixed(2)} indicates {indicator.value > 0 ? 'positive momentum' : 'negative momentum'}.</>
                      )}
                      {key === 'ema20' && (
                        <>Price above 20-day EMA suggests bullish short-term trend, while price below suggests bearish short-term trend. Current price is {stockData.price > indicator.value ? 'above' : 'below'} the EMA20 value of ₹{indicator.value.toFixed(2)}.</>
                      )}
                      {key === 'sma50' && (
                        <>Price above 50-day SMA suggests bullish medium-term trend, while price below suggests bearish medium-term trend. Current price is {stockData.price > indicator.value ? 'above' : 'below'} the SMA50 value of ₹{indicator.value.toFixed(2)}.</>
                      )}
                      {key === 'sma200' && (
                        <>Price above 200-day SMA suggests bullish long-term trend, while price below suggests bearish long-term trend. Current price is {stockData.price > indicator.value ? 'above' : 'below'} the SMA200 value of ₹{indicator.value.toFixed(2)}.</>
                      )}
                      {key === 'bollingerBands' && (
                        <>Price near upper band suggests overbought conditions, while price near lower band suggests oversold conditions. Current price is {stockData.price > indicator.upper ? 'above the upper band' : stockData.price < indicator.lower ? 'below the lower band' : 'between the bands'}, indicating {stockData.price > indicator.upper ? 'potential resistance or continuation of strong uptrend' : stockData.price < indicator.lower ? 'potential support or continuation of strong downtrend' : 'neutral conditions'}.</>
                      )}
                      {key === 'adx' && (
                        <>ADX values above 25 indicate a strong trend, while values below 20 indicate a weak trend. Current value of {indicator.value.toFixed(2)} suggests {indicator.value > 25 ? 'a strong trend is present' : indicator.value < 20 ? 'a weak or absent trend' : 'a developing trend'}.</>
                      )}
                      {key === 'stochastic' && (
                        <>Stochastic values above 80 indicate overbought conditions, while values below 20 indicate oversold conditions. Current %K of {indicator.k.toFixed(2)} and %D of {indicator.d.toFixed(2)} suggest {indicator.k > 80 ? 'potential reversal or correction' : indicator.k < 20 ? 'potential buying opportunity' : 'neutral market conditions'}.</>
                      )}
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'fundamental' && (
        <div className="card">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-6 flex items-center">
            <FiInfo className="mr-2 h-5 w-5 text-primary-500" />
            {t('fundamental_analysis')}
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Financial ratios */}
            <div className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-4">
              <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-4">
                {t('financial_ratios')}
              </h4>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('pe_ratio')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">{stockData.peRatio.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('eps')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">₹{stockData.eps.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('dividend_yield')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">{stockData.dividendYield.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('price_to_book')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">3.2</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('debt_to_equity')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">0.85</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('roe')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">18.5%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">{t('profit_margin')}</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">12.3%</span>
                </div>
              </div>
            </div>

            {/* Quarterly results */}
            <div className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-4">
              <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-4">
                {t('quarterly_results')}
              </h4>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-neutral-200 dark:divide-neutral-700">
                  <thead>
                    <tr>
                      <th className="px-2 py-2 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">{t('quarter')}</th>
                      <th className="px-2 py-2 text-right text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">{t('revenue')}</th>
                      <th className="px-2 py-2 text-right text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">{t('profit')}</th>
                      <th className="px-2 py-2 text-right text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">{t('growth')}</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-200 dark:divide-neutral-700">
                    <tr>
                      <td className="px-2 py-2 text-sm text-neutral-600 dark:text-neutral-400">Q1 2023</td>
                      <td className="px-2 py-2 text-sm text-right text-neutral-900 dark:text-neutral-100">₹15,200 Cr</td>
                      <td className="px-2 py-2 text-sm text-right text-neutral-900 dark:text-neutral-100">₹2,100 Cr</td>
                      <td className="px-2 py-2 text-sm text-right text-success-600 dark:text-success-400">+12.5%</td>
                    </tr>
                    <tr>
                      <td className="px-2 py-2 text-sm text-neutral-600 dark:text-neutral-400">Q4 2022</td>
                      <td className="px-2 py-2 text-sm text-right text-neutral-900 dark:text-neutral-100">₹14,800 Cr</td>
                      <td className="px-2 py-2 text-sm text-right text-neutral-900 dark:text-neutral-100">₹1,950 Cr</td>
                      <td className="px-2 py-2 text-sm text-right text-success-600 dark:text-success-400">+8.2%</td>
                    </tr>
                    <tr>
                      <td className="px-2 py-2 text-sm text-neutral-600 dark:text-neutral-400">Q3 2022</td>
                      <td className="px-2 py-2 text-sm text-right text-neutral-900 dark:text-neutral-100">₹14,200 Cr</td>
                      <td className="px-2 py-2 text-sm text-right text-neutral-900 dark:text-neutral-100">₹1,850 Cr</td>
                      <td className="px-2 py-2 text-sm text-right text-success-600 dark:text-success-400">+5.7%</td>
                    </tr>
                    <tr>
                      <td className="px-2 py-2 text-sm text-neutral-600 dark:text-neutral-400">Q2 2022</td>
                      <td className="px-2 py-2 text-sm text-right text-neutral-900 dark:text-neutral-100">₹13,500 Cr</td>
                      <td className="px-2 py-2 text-sm text-right text-neutral-900 dark:text-neutral-100">₹1,720 Cr</td>
                      <td className="px-2 py-2 text-sm text-right text-danger-600 dark:text-danger-400">-2.3%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Company information */}
          <div className="mt-6 border border-neutral-200 dark:border-neutral-700 rounded-lg p-4">
            <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-4">
              {t('company_information')}
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h5 className="text-sm font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                  {t('about')}
                </h5>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  Tata Motors Limited is an Indian multinational automotive manufacturing company headquartered in Mumbai, India. It is a part of Tata Group, an Indian conglomerate. Its products include passenger cars, trucks, vans, coaches, buses, luxury cars, sports cars, and construction equipment.
                </p>
              </div>
              <div>
                <h5 className="text-sm font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                  {t('key_information')}
                </h5>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-neutral-600 dark:text-neutral-400">{t('sector')}</span>
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">Automotive</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-neutral-600 dark:text-neutral-400">{t('industry')}</span>
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">Auto Manufacturers</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-neutral-600 dark:text-neutral-400">{t('employees')}</span>
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">75,000+</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-neutral-600 dark:text-neutral-400">{t('founded')}</span>
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">1945</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-neutral-600 dark:text-neutral-400">{t('headquarters')}</span>
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">Mumbai, India</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'news' && (
        <div className="card">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-6 flex items-center">
            <FiFileText className="mr-2 h-5 w-5 text-primary-500" />
            {t('news_and_events')}
          </h3>
          
          {/* Search and filter */}
          <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
            <div className="relative mb-4 md:mb-0 md:w-1/3">
              <input
                type="text"
                className="input w-full pl-10"
                placeholder={t('search_news')}
              />
              <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                <svg className="w-5 h-5 text-neutral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                </svg>
              </div>
            </div>
            
            <div className="flex space-x-2">
              <select className="input py-2 pl-3 pr-10">
                <option value="all">{t('all_sentiment')}</option>
                <option value="positive">{t('positive')}</option>
                <option value="neutral">{t('neutral')}</option>
                <option value="negative">{t('negative')}</option>
              </select>
              
              <select className="input py-2 pl-3 pr-10">
                <option value="latest">{t('latest')}</option>
                <option value="oldest">{t('oldest')}</option>
                <option value="impact">{t('impact')}</option>
              </select>
            </div>
          </div>
          
          {/* News list */}
          <div className="space-y-6">
            {newsData.map((news) => (
              <div
                key={news.id}
                className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-4 hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors"
              >
                <div className="flex items-start">
                  <div
                    className={`mt-1 flex-shrink-0 w-3 h-3 rounded-full mr-3 ${news.sentiment === 'positive' ? 'bg-success-500' : news.sentiment === 'negative' ? 'bg-danger-500' : 'bg-neutral-500'}`}
                  ></div>
                  <div className="flex-1">
                    <a
                      href={news.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-lg font-medium text-neutral-900 dark:text-neutral-100 hover:text-primary-600 dark:hover:text-primary-400"
                    >
                      {news.title}
                    </a>
                    <div className="flex items-center mt-1 text-sm text-neutral-500 dark:text-neutral-400">
                      <span className="font-medium">{news.source}</span>
                      <span className="mx-1">•</span>
                      <span>{new Date(news.date).toLocaleDateString()}</span>
                      <span className="mx-1">•</span>
                      <span
                        className={`${news.sentiment === 'positive' ? 'text-success-600 dark:text-success-400' : news.sentiment === 'negative' ? 'text-danger-600 dark:text-danger-400' : 'text-neutral-600 dark:text-neutral-400'}`}
                      >
                        {t(news.sentiment)}
                      </span>
                      <span className="mx-1">•</span>
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs ${news.impact === 'high' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : news.impact === 'medium' ? 'bg-warning-100 dark:bg-warning-900 text-warning-800 dark:text-warning-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
                      >
                        {t(`${news.impact}_impact`)}
                      </span>
                    </div>
                    <p className="mt-2 text-neutral-600 dark:text-neutral-400">
                      {news.summary}
                    </p>
                    <div className="mt-3 flex justify-end">
                      <a
                        href={news.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 flex items-center"
                      >
                        {t('read_full_article')}
                        <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path>
                        </svg>
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {/* Load more button */}
          <div className="mt-6 text-center">
            <button className="btn btn-outline">
              {t('load_more_news')}
            </button>
          </div>
        </div>
      )}

      {activeTab === 'predictions' && (
        <div className="card">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-6 flex items-center">
            <FiPieChart className="mr-2 h-5 w-5 text-primary-500" />
            {t('ai_predictions')}
          </h3>
          
          {/* Prediction tabs */}
          <div className="flex border-b border-neutral-200 dark:border-neutral-700 mb-6">
            <button
              onClick={() => setActivePredictionTab('short_term')}
              className={`px-4 py-2 text-sm font-medium ${activePredictionTab === 'short_term' ? 'border-b-2 border-primary-500 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'}`}
            >
              {t('short_term')} (1-7 {t('days')})
            </button>
            <button
              onClick={() => setActivePredictionTab('medium_term')}
              className={`px-4 py-2 text-sm font-medium ${activePredictionTab === 'medium_term' ? 'border-b-2 border-primary-500 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'}`}
            >
              {t('medium_term')} (15-30 {t('days')})
            </button>
            <button
              onClick={() => setActivePredictionTab('long_term')}
              className={`px-4 py-2 text-sm font-medium ${activePredictionTab === 'long_term' ? 'border-b-2 border-primary-500 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'}`}
            >
              {t('long_term')} (3-6 {t('months')})
            </button>
          </div>
          
          {/* Prediction content */}
          {activePredictionTab === 'short_term' && (
            <div>
              <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
                <div>
                  <div className="flex items-center">
                    <div
                      className={`px-3 py-1 rounded-full text-sm font-medium ${predictions.shortTerm.prediction === 'buy' || predictions.shortTerm.prediction === 'strong_buy' ? 'bg-success-100 dark:bg-success-900 text-success-800 dark:text-success-200' : predictions.shortTerm.prediction === 'sell' || predictions.shortTerm.prediction === 'strong_sell' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
                    >
                      {t(predictions.shortTerm.prediction)}
                    </div>
                    <div className="ml-3 text-sm text-neutral-500 dark:text-neutral-400">
                      {t('confidence')}: {predictions.shortTerm.confidence}%
                    </div>
                  </div>
                  <div className="mt-2">
                    <span className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                      ₹{predictions.shortTerm.predictedPrice.toLocaleString()}
                    </span>
                    <span className="ml-2 text-sm text-neutral-500 dark:text-neutral-400">
                      {t('range')}: ₹{predictions.shortTerm.priceRange.low.toLocaleString()} - ₹{predictions.shortTerm.priceRange.high.toLocaleString()}
                    </span>
                  </div>
                </div>
                
                <Link
                  to={`/predictions?ticker=${stockData.symbol}&timeframe=short_term`}
                  className="btn btn-primary mt-4 md:mt-0"
                >
                  {t('detailed_analysis')}
                </Link>
              </div>
              
              <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-4">
                {t('prediction_factors')}
              </h4>
              
              <div className="space-y-4 mb-6">
                {predictions.shortTerm.factors.map((factor, index) => (
                  <div key={index} className="flex items-center">
                    <div className="w-1/3 text-sm text-neutral-600 dark:text-neutral-400">
                      {t(factor.name.toLowerCase().replace(/ /g, '_'))}
                    </div>
                    <div className="w-2/3">
                      <div className="flex items-center">
                        <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2 mr-2">
                          <div
                            className={`h-2 rounded-full ${factor.impact === 'positive' ? 'bg-success-500' : factor.impact === 'negative' ? 'bg-danger-500' : 'bg-neutral-500'}`}
                            style={{ width: `${factor.weight}%` }}
                          ></div>
                        </div>
                        <span className="text-xs text-neutral-500 dark:text-neutral-400 w-10 text-right">
                          {factor.weight}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4 border border-neutral-200 dark:border-neutral-700">
                <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2 flex items-center">
                  <FiInfo className="mr-2 h-4 w-4 text-primary-500" />
                  {t('prediction_explanation')}
                </h4>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  {t('short_term_prediction_explanation', { stock: stockData.name })}
                </p>
              </div>
            </div>
          )}
          
          {activePredictionTab === 'long_term' && (
            <div>
              <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
                <div>
                  <div className="flex items-center">
                    <div
                      className={`px-3 py-1 rounded-full text-sm font-medium ${predictions.longTerm.prediction === 'buy' || predictions.longTerm.prediction === 'strong_buy' ? 'bg-success-100 dark:bg-success-900 text-success-800 dark:text-success-200' : predictions.longTerm.prediction === 'sell' || predictions.longTerm.prediction === 'strong_sell' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
                    >
                      {t(predictions.longTerm.prediction)}
                    </div>
                    <div className="ml-3 text-sm text-neutral-500 dark:text-neutral-400">
                      {t('confidence')}: {predictions.longTerm.confidence}%
                    </div>
                  </div>
                  <div className="mt-2">
                    <span className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                      ₹{predictions.longTerm.predictedPrice.toLocaleString()}
                    </span>
                    <span className="ml-2 text-sm text-neutral-500 dark:text-neutral-400">
                      {t('range')}: ₹{predictions.longTerm.priceRange.low.toLocaleString()} - ₹{predictions.longTerm.priceRange.high.toLocaleString()}
                    </span>
                  </div>
                </div>
                
                <Link
                  to={`/predictions?ticker=${stockData.symbol}&timeframe=long_term`}
                  className="btn btn-primary mt-4 md:mt-0"
                >
                  {t('detailed_analysis')}
                </Link>
              </div>
              
              <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-4">
                {t('prediction_factors')}
              </h4>
              
              <div className="space-y-4 mb-6">
                {predictions.longTerm.factors.map((factor, index) => (
                  <div key={index} className="flex items-center">
                    <div className="w-1/3 text-sm text-neutral-600 dark:text-neutral-400">
                      {t(factor.name.toLowerCase().replace(/ /g, '_'))}
                    </div>
                    <div className="w-2/3">
                      <div className="flex items-center">
                        <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2 mr-2">
                          <div
                            className={`h-2 rounded-full ${factor.impact === 'positive' ? 'bg-success-500' : factor.impact === 'negative' ? 'bg-danger-500' : 'bg-neutral-500'}`}
                            style={{ width: `${factor.weight}%` }}
                          ></div>
                        </div>
                        <span className="text-xs text-neutral-500 dark:text-neutral-400 w-10 text-right">
                          {factor.weight}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4 border border-neutral-200 dark:border-neutral-700">
                <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2 flex items-center">
                  <FiInfo className="mr-2 h-4 w-4 text-primary-500" />
                  {t('prediction_explanation')}
                </h4>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  {t('long_term_prediction_explanation', { stock: stockData.name })}
                </p>
              </div>
            </div>
          )}
          
          {activePredictionTab === 'medium_term' && (
            <div>
              <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
                <div>
                  <div className="flex items-center">
                    <div
                      className={`px-3 py-1 rounded-full text-sm font-medium ${predictions.mediumTerm.prediction === 'buy' || predictions.mediumTerm.prediction === 'strong_buy' ? 'bg-success-100 dark:bg-success-900 text-success-800 dark:text-success-200' : predictions.mediumTerm.prediction === 'sell' || predictions.mediumTerm.prediction === 'strong_sell' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
                    >
                      {t(predictions.mediumTerm.prediction)}
                    </div>
                    <div className="ml-3 text-sm text-neutral-500 dark:text-neutral-400">
                      {t('confidence')}: {predictions.mediumTerm.confidence}%
                    </div>
                  </div>
                  <div className="mt-2">
                    <span className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                      ₹{predictions.mediumTerm.predictedPrice.toLocaleString()}
                    </span>
                    <span className="ml-2 text-sm text-neutral-500 dark:text-neutral-400">
                      {t('range')}: ₹{predictions.mediumTerm.priceRange.low.toLocaleString()} - ₹{predictions.mediumTerm.priceRange.high.toLocaleString()}
                    </span>
                  </div>
                </div>
                
                <Link
                  to={`/predictions?ticker=${stockData.symbol}&timeframe=medium_term`}
                  className="btn btn-primary mt-4 md:mt-0"
                >
                  {t('detailed_analysis')}
                </Link>
              </div>
              
              <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-4">
                {t('prediction_factors')}
              </h4>
              
              <div className="space-y-4 mb-6">
                {predictions.mediumTerm.factors.map((factor, index) => (
                  <div key={index} className="flex items-center">
                    <div className="w-1/3 text-sm text-neutral-600 dark:text-neutral-400">
                      {t(factor.name.toLowerCase().replace(/ /g, '_'))}
                    </div>
                    <div className="w-2/3">
                      <div className="flex items-center">
                        <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2 mr-2">
                          <div
                            className={`h-2 rounded-full ${factor.impact === 'positive' ? 'bg-success-500' : factor.impact === 'negative' ? 'bg-danger-500' : 'bg-neutral-500'}`}
                            style={{ width: `${factor.weight}%` }}
                          ></div>
                        </div>
                        <span className="text-xs text-neutral-500 dark:text-neutral-400 w-10 text-right">
                          {factor.weight}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4 border border-neutral-200 dark:border-neutral-700">
                <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2 flex items-center">
                  <FiInfo className="mr-2 h-4 w-4 text-primary-500" />
                  {t('prediction_explanation')}
                </h4>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  {t('medium_term_prediction_explanation', { stock: stockData.name })}
                </p>
              </div>
            </div>
          )}