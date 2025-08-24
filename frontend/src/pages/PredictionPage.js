import React, { useState, useEffect } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import {
  FiArrowLeft,
  FiInfo,
  FiTrendingUp,
  FiBarChart2,
  FiPieChart,
  FiAlertCircle,
  FiCheckCircle,
  FiFileText,
  FiDownload,
  FiRefreshCw,
  FiCalendar,
} from 'react-icons/fi';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Mock prediction data
const mockPredictions = {
  short_term: {
    ticker: 'TATAMOTORS.NS',
    name: 'Tata Motors Limited',
    prediction: 'buy',
    confidence: 87,
    predictedPrice: 950.25,
    priceRange: { low: 930.50, high: 970.00 },
    timeframe: '1-7 days',
    currentPrice: 900.50,
    potentialReturn: 5.52,
    riskLevel: 'medium',
    lastUpdated: '2023-05-15T10:30:00Z',
    factors: [
      { name: 'Technical Indicators', impact: 'positive', weight: 35, description: 'RSI, MACD, and EMA all show bullish signals' },
      { name: 'Recent News Sentiment', impact: 'positive', weight: 25, description: 'Recent earnings report exceeded expectations' },
      { name: 'Volume Analysis', impact: 'positive', weight: 20, description: 'Increasing volume on up days indicates strong buying pressure' },
      { name: 'Market Trend', impact: 'neutral', weight: 15, description: 'Overall market is showing mixed signals' },
      { name: 'Sector Performance', impact: 'neutral', weight: 5, description: 'Auto sector is performing in line with the broader market' }
    ],
    technicalSignals: {
      rsi: { value: 62.5, interpretation: 'neutral', description: 'RSI is in neutral territory' },
      macd: { value: 3.2, interpretation: 'bullish', description: 'MACD is above the signal line' },
      ema20: { value: 885.30, interpretation: 'bullish', description: 'Price is above the 20-day EMA' },
      sma50: { value: 860.75, interpretation: 'bullish', description: 'Price is above the 50-day SMA' },
      bollingerBands: { value: 'middle', interpretation: 'neutral', description: 'Price is near the middle band' },
    },
    newsImpact: [
      { title: 'Tata Motors reports strong Q4 results, profit jumps 46%', sentiment: 'positive', impact: 'high', source: 'Economic Times', date: '2023-05-12' },
      { title: 'Tata Motors launches new electric vehicle model with 500km range', sentiment: 'positive', impact: 'medium', source: 'Mint', date: '2023-05-08' },
    ],
    predictionHistory: [
      { date: '2023-05-15', prediction: 'buy', predictedPrice: 950.25, actualPrice: 900.50, confidence: 87 },
      { date: '2023-05-08', prediction: 'buy', predictedPrice: 920.75, actualPrice: 885.25, confidence: 82 },
      { date: '2023-05-01', prediction: 'hold', predictedPrice: 880.50, actualPrice: 875.30, confidence: 65 },
      { date: '2023-04-24', prediction: 'hold', predictedPrice: 870.25, actualPrice: 865.75, confidence: 60 },
    ],
    forecastData: [
      { date: '2023-05-15', actual: 900.50, predicted: null, lower: null, upper: null },
      { date: '2023-05-16', actual: null, predicted: 910.25, lower: 905.50, upper: 915.00 },
      { date: '2023-05-17', actual: null, predicted: 920.75, lower: 912.25, upper: 929.25 },
      { date: '2023-05-18', actual: null, predicted: 930.50, lower: 918.75, upper: 942.25 },
      { date: '2023-05-19', actual: null, predicted: 940.25, lower: 925.50, upper: 955.00 },
      { date: '2023-05-22', actual: null, predicted: 950.25, lower: 930.50, upper: 970.00 },
    ],
  },
  medium_term: {
    ticker: 'TATAMOTORS.NS',
    name: 'Tata Motors Limited',
    prediction: 'hold',
    confidence: 65,
    predictedPrice: 920.75,
    priceRange: { low: 880.00, high: 960.00 },
    timeframe: '15-30 days',
    currentPrice: 900.50,
    potentialReturn: 2.25,
    riskLevel: 'medium',
    lastUpdated: '2023-05-15T10:30:00Z',
    factors: [
      { name: 'Technical Indicators', impact: 'neutral', weight: 30, description: 'Mixed signals from technical indicators' },
      { name: 'Fundamental Analysis', impact: 'positive', weight: 25, description: 'Strong fundamentals with good P/E ratio and growth prospects' },
      { name: 'Sector Outlook', impact: 'neutral', weight: 20, description: 'Auto sector facing challenges but also opportunities' },
      { name: 'Market Sentiment', impact: 'negative', weight: 15, description: 'Market volatility expected in the coming weeks' },
      { name: 'Economic Indicators', impact: 'neutral', weight: 10, description: 'Economic data shows mixed signals for consumer spending' }
    ],
    technicalSignals: {
      rsi: { value: 62.5, interpretation: 'neutral', description: 'RSI is in neutral territory' },
      macd: { value: 3.2, interpretation: 'bullish', description: 'MACD is above the signal line' },
      ema20: { value: 885.30, interpretation: 'bullish', description: 'Price is above the 20-day EMA' },
      sma50: { value: 860.75, interpretation: 'bullish', description: 'Price is above the 50-day SMA' },
      bollingerBands: { value: 'middle', interpretation: 'neutral', description: 'Price is near the middle band' },
    },
    newsImpact: [
      { title: 'Tata Motors reports strong Q4 results, profit jumps 46%', sentiment: 'positive', impact: 'high', source: 'Economic Times', date: '2023-05-12' },
      { title: 'Global chip shortage continues to impact auto production', sentiment: 'negative', impact: 'medium', source: 'Business Standard', date: '2023-05-05' },
    ],
    predictionHistory: [
      { date: '2023-05-15', prediction: 'hold', predictedPrice: 920.75, actualPrice: 900.50, confidence: 65 },
      { date: '2023-05-01', prediction: 'hold', predictedPrice: 890.50, actualPrice: 875.30, confidence: 60 },
      { date: '2023-04-15', prediction: 'buy', predictedPrice: 870.25, actualPrice: 865.75, confidence: 75 },
      { date: '2023-04-01', prediction: 'buy', predictedPrice: 850.50, actualPrice: 845.25, confidence: 70 },
    ],
    forecastData: [
      { date: '2023-05-15', actual: 900.50, predicted: null, lower: null, upper: null },
      { date: '2023-05-22', actual: null, predicted: 905.25, lower: 890.50, upper: 920.00 },
      { date: '2023-05-29', actual: null, predicted: 910.75, lower: 885.25, upper: 935.25 },
      { date: '2023-06-05', actual: null, predicted: 915.50, lower: 880.75, upper: 950.25 },
      { date: '2023-06-12', actual: null, predicted: 920.75, lower: 880.00, upper: 960.00 },
    ],
  },
  long_term: {
    ticker: 'TATAMOTORS.NS',
    name: 'Tata Motors Limited',
    prediction: 'strong_buy',
    confidence: 78,
    predictedPrice: 1050.30,
    priceRange: { low: 950.00, high: 1150.00 },
    timeframe: '3-6 months',
    currentPrice: 900.50,
    potentialReturn: 16.64,
    riskLevel: 'medium',
    lastUpdated: '2023-05-15T10:30:00Z',
    factors: [
      { name: 'Fundamental Analysis', impact: 'positive', weight: 40, description: 'Strong balance sheet and growth in EV segment' },
      { name: 'Industry Growth', impact: 'positive', weight: 25, description: 'Auto industry expected to grow with EV transition' },
      { name: 'Company Roadmap', impact: 'positive', weight: 15, description: 'Ambitious plans for new models and technologies' },
      { name: 'Economic Outlook', impact: 'neutral', weight: 10, description: 'Economic growth projections are moderate' },
      { name: 'Competitive Position', impact: 'positive', weight: 10, description: 'Strong market position in key segments' }
    ],
    technicalSignals: {
      rsi: { value: 62.5, interpretation: 'neutral', description: 'RSI is in neutral territory' },
      macd: { value: 3.2, interpretation: 'bullish', description: 'MACD is above the signal line' },
      sma50: { value: 860.75, interpretation: 'bullish', description: 'Price is above the 50-day SMA' },
      sma200: { value: 820.40, interpretation: 'bullish', description: 'Price is above the 200-day SMA' },
      adx: { value: 28.5, interpretation: 'strong_trend', description: 'ADX above 25 indicates a strong trend' },
    },
    newsImpact: [
      { title: 'Tata Motors reports strong Q4 results, profit jumps 46%', sentiment: 'positive', impact: 'high', source: 'Economic Times', date: '2023-05-12' },
      { title: 'Tata Motors launches new electric vehicle model with 500km range', sentiment: 'positive', impact: 'medium', source: 'Mint', date: '2023-05-08' },
      { title: 'Tata Motors increases market share in commercial vehicle segment', sentiment: 'positive', impact: 'medium', source: 'CNBC', date: '2023-05-02' },
    ],
    predictionHistory: [
      { date: '2023-05-15', prediction: 'strong_buy', predictedPrice: 1050.30, actualPrice: 900.50, confidence: 78 },
      { date: '2023-04-15', prediction: 'buy', predictedPrice: 1000.25, actualPrice: 865.75, confidence: 72 },
      { date: '2023-03-15', prediction: 'buy', predictedPrice: 950.50, actualPrice: 830.25, confidence: 68 },
      { date: '2023-02-15', prediction: 'hold', predictedPrice: 900.75, actualPrice: 810.50, confidence: 60 },
    ],
    forecastData: [
      { date: '2023-05-15', actual: 900.50, predicted: null, lower: null, upper: null },
      { date: '2023-06-15', actual: null, predicted: 950.25, lower: 920.50, upper: 980.00 },
      { date: '2023-07-15', actual: null, predicted: 980.75, lower: 935.25, upper: 1025.25 },
      { date: '2023-08-15', actual: null, predicted: 1010.50, lower: 945.75, upper: 1075.25 },
      { date: '2023-09-15', actual: null, predicted: 1030.25, lower: 950.00, upper: 1110.50 },
      { date: '2023-10-15', actual: null, predicted: 1050.30, lower: 950.00, upper: 1150.00 },
    ],
  }
};

const PredictionPage = () => {
  const { t } = useTranslation();
  const location = useLocation();
  const [isLoading, setIsLoading] = useState(true);
  const [predictionData, setPredictionData] = useState(null);
  const [timeframe, setTimeframe] = useState('short_term');
  
  // Parse query parameters
  useEffect(() => {
    const searchParams = new URLSearchParams(location.search);
    const ticker = searchParams.get('ticker') || 'TATAMOTORS.NS';
    const tf = searchParams.get('timeframe') || 'short_term';
    
    setTimeframe(tf);
    
    // Simulate API call
    setIsLoading(true);
    setTimeout(() => {
      setPredictionData(mockPredictions[tf]);
      setIsLoading(false);
    }, 1000);
  }, [location.search]);
  
  // Handle timeframe change
  const handleTimeframeChange = (newTimeframe) => {
    setTimeframe(newTimeframe);
    setPredictionData(mockPredictions[newTimeframe]);
  };
  
  // Prepare chart data for forecast
  const prepareForecastChartData = () => {
    if (!predictionData) return null;
    
    const labels = predictionData.forecastData.map(item => item.date);
    
    return {
      labels,
      datasets: [
        {
          label: t('actual_price'),
          data: predictionData.forecastData.map(item => item.actual),
          borderColor: '#6366F1',
          backgroundColor: '#6366F1',
          pointBackgroundColor: '#6366F1',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: '#6366F1',
          pointRadius: 5,
          pointHoverRadius: 7,
          fill: false,
        },
        {
          label: t('predicted_price'),
          data: predictionData.forecastData.map(item => item.predicted),
          borderColor: '#10B981',
          backgroundColor: '#10B981',
          pointBackgroundColor: '#10B981',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: '#10B981',
          pointRadius: 5,
          pointHoverRadius: 7,
          borderDash: [5, 5],
          fill: false,
        },
        {
          label: t('prediction_range'),
          data: predictionData.forecastData.map(item => item.upper),
          borderColor: 'rgba(16, 185, 129, 0.2)',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          pointRadius: 0,
          borderWidth: 1,
          fill: '+1',
        },
        {
          label: '',
          data: predictionData.forecastData.map(item => item.lower),
          borderColor: 'rgba(16, 185, 129, 0.2)',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          pointRadius: 0,
          borderWidth: 1,
          fill: false,
        },
      ],
    };
  };
  
  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          boxWidth: 6,
        },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
    scales: {
      x: {
        grid: {
          display: false,
        },
      },
      y: {
        grid: {
          borderDash: [2, 4],
          color: 'rgba(0, 0, 0, 0.06)',
        },
        ticks: {
          callback: (value) => `₹${value.toLocaleString()}`,
        },
      },
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false,
    },
  };
  
  // Loading skeleton
  if (isLoading || !predictionData) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="animate-pulse">
          <div className="h-8 bg-neutral-200 dark:bg-neutral-700 rounded w-1/3 mb-6"></div>
          <div className="h-64 bg-neutral-200 dark:bg-neutral-700 rounded mb-6"></div>
          <div className="h-8 bg-neutral-200 dark:bg-neutral-700 rounded w-1/4 mb-4"></div>
          <div className="h-32 bg-neutral-200 dark:bg-neutral-700 rounded mb-6"></div>
          <div className="h-8 bg-neutral-200 dark:bg-neutral-700 rounded w-1/4 mb-4"></div>
          <div className="h-48 bg-neutral-200 dark:bg-neutral-700 rounded"></div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <Link to={`/stock/${predictionData.ticker}`} className="text-primary-600 dark:text-primary-400 flex items-center mb-4">
          <FiArrowLeft className="mr-2" />
          {t('back_to_stock_details')}
        </Link>
        
        <div className="flex flex-col md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
              {t('ai_prediction_for')} {predictionData.name} ({predictionData.ticker})
            </h1>
            <p className="text-neutral-500 dark:text-neutral-400 mt-1">
              {t('last_updated')}: {new Date(predictionData.lastUpdated).toLocaleString()}
            </p>
          </div>
          
          <div className="mt-4 md:mt-0">
            <button className="btn btn-outline flex items-center">
              <FiDownload className="mr-2" />
              {t('export_report')}
            </button>
          </div>
        </div>
      </div>
      
      {/* Timeframe selector */}
      <div className="flex border-b border-neutral-200 dark:border-neutral-700 mb-8">
        <button
          onClick={() => handleTimeframeChange('short_term')}
          className={`px-4 py-2 text-sm font-medium ${timeframe === 'short_term' ? 'border-b-2 border-primary-500 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'}`}
        >
          {t('short_term')} (1-7 {t('days')})
        </button>
        <button
          onClick={() => handleTimeframeChange('medium_term')}
          className={`px-4 py-2 text-sm font-medium ${timeframe === 'medium_term' ? 'border-b-2 border-primary-500 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'}`}
        >
          {t('medium_term')} (15-30 {t('days')})
        </button>
        <button
          onClick={() => handleTimeframeChange('long_term')}
          className={`px-4 py-2 text-sm font-medium ${timeframe === 'long_term' ? 'border-b-2 border-primary-500 text-primary-600 dark:text-primary-400' : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'}`}
        >
          {t('long_term')} (3-6 {t('months')})
        </button>
      </div>
      
      {/* Prediction summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
            <FiPieChart className="mr-2 h-5 w-5 text-primary-500" />
            {t('prediction_summary')}
          </h3>
          
          <div className="mb-4">
            <div
              className={`inline-block px-3 py-1 rounded-full text-sm font-medium mb-2 ${predictionData.prediction === 'buy' || predictionData.prediction === 'strong_buy' ? 'bg-success-100 dark:bg-success-900 text-success-800 dark:text-success-200' : predictionData.prediction === 'sell' || predictionData.prediction === 'strong_sell' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
            >
              {t(predictionData.prediction)}
            </div>
            <div className="text-sm text-neutral-500 dark:text-neutral-400">
              {t('confidence')}: {predictionData.confidence}%
            </div>
          </div>
          
          <div className="space-y-3">
            <div>
              <div className="text-sm text-neutral-500 dark:text-neutral-400">{t('current_price')}</div>
              <div className="text-xl font-bold text-neutral-900 dark:text-neutral-100">₹{predictionData.currentPrice.toLocaleString()}</div>
            </div>
            
            <div>
              <div className="text-sm text-neutral-500 dark:text-neutral-400">{t('predicted_price')}</div>
              <div className="text-xl font-bold text-neutral-900 dark:text-neutral-100">₹{predictionData.predictedPrice.toLocaleString()}</div>
            </div>
            
            <div>
              <div className="text-sm text-neutral-500 dark:text-neutral-400">{t('price_range')}</div>
              <div className="text-neutral-900 dark:text-neutral-100">₹{predictionData.priceRange.low.toLocaleString()} - ₹{predictionData.priceRange.high.toLocaleString()}</div>
            </div>
            
            <div>
              <div className="text-sm text-neutral-500 dark:text-neutral-400">{t('potential_return')}</div>
              <div className="text-success-600 dark:text-success-400 font-medium">+{predictionData.potentialReturn}%</div>
            </div>
            
            <div>
              <div className="text-sm text-neutral-500 dark:text-neutral-400">{t('risk_level')}</div>
              <div className="text-warning-600 dark:text-warning-400 font-medium">{t(predictionData.riskLevel)}</div>
            </div>
          </div>
        </div>
        
        <div className="card p-6 md:col-span-2">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
            <FiTrendingUp className="mr-2 h-5 w-5 text-primary-500" />
            {t('price_forecast')}
          </h3>
          
          <div className="h-64">
            <Line data={prepareForecastChartData()} options={chartOptions} />
          </div>
          
          <div className="mt-4 text-sm text-neutral-500 dark:text-neutral-400 flex items-center">
            <FiInfo className="mr-2" />
            {t('forecast_explanation')}
          </div>
        </div>
      </div>
      
      {/* Prediction factors */}
      <div className="card p-6 mb-8">
        <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-6 flex items-center">
          <FiBarChart2 className="mr-2 h-5 w-5 text-primary-500" />
          {t('prediction_factors')}
        </h3>
        
        <div className="space-y-6">
          {predictionData.factors.map((factor, index) => (
            <div key={index}>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  <div
                    className={`w-3 h-3 rounded-full mr-2 ${factor.impact === 'positive' ? 'bg-success-500' : factor.impact === 'negative' ? 'bg-danger-500' : 'bg-neutral-500'}`}
                  ></div>
                  <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                    {t(factor.name.toLowerCase().replace(/ /g, '_'))} ({factor.weight}%)
                  </h4>
                </div>
                <div
                  className={`text-sm ${factor.impact === 'positive' ? 'text-success-600 dark:text-success-400' : factor.impact === 'negative' ? 'text-danger-600 dark:text-danger-400' : 'text-neutral-600 dark:text-neutral-400'}`}
                >
                  {t(factor.impact)}
                </div>
              </div>
              
              <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2 mb-2">
                <div
                  className={`h-2 rounded-full ${factor.impact === 'positive' ? 'bg-success-500' : factor.impact === 'negative' ? 'bg-danger-500' : 'bg-neutral-500'}`}
                  style={{ width: `${factor.weight}%` }}
                ></div>
              </div>
              
              <p className="text-sm text-neutral-600 dark:text-neutral-400">
                {factor.description}
              </p>
            </div>
          ))}
        </div>
      </div>
      
      {/* Technical signals and news impact */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
            <FiActivity className="mr-2 h-5 w-5 text-primary-500" />
            {t('technical_signals')}
          </h3>
          
          <div className="space-y-4">
            {Object.entries(predictionData.technicalSignals).map(([key, signal]) => (
              <div key={key} className="flex items-start">
                <div
                  className={`mt-1 flex-shrink-0 w-3 h-3 rounded-full mr-3 ${signal.interpretation === 'bullish' ? 'bg-success-500' : signal.interpretation === 'bearish' ? 'bg-danger-500' : 'bg-neutral-500'}`}
                ></div>
                <div>
                  <div className="flex items-center">
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">
                      {t(key.toUpperCase())}
                    </span>
                    <span className="ml-2 text-sm text-neutral-500 dark:text-neutral-400">
                      {typeof signal.value === 'number' ? signal.value.toFixed(2) : signal.value}
                    </span>
                  </div>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
                    {signal.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
            <FiFileText className="mr-2 h-5 w-5 text-primary-500" />
            {t('news_impact')}
          </h3>
          
          <div className="space-y-4">
            {predictionData.newsImpact.map((news, index) => (
              <div key={index} className="flex items-start">
                <div
                  className={`mt-1 flex-shrink-0 w-3 h-3 rounded-full mr-3 ${news.sentiment === 'positive' ? 'bg-success-500' : news.sentiment === 'negative' ? 'bg-danger-500' : 'bg-neutral-500'}`}
                ></div>
                <div>
                  <div className="font-medium text-neutral-900 dark:text-neutral-100">
                    {news.title}
                  </div>
                  <div className="flex items-center mt-1 text-sm text-neutral-500 dark:text-neutral-400">
                    <span>{news.source}</span>
                    <span className="mx-1">•</span>
                    <span>{news.date}</span>
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
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* Prediction history */}
      <div className="card p-6 mb-8">
        <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center">
          <FiCalendar className="mr-2 h-5 w-5 text-primary-500" />
          {t('prediction_history')}
        </h3>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-neutral-200 dark:border-neutral-700">
                <th className="py-3 px-4 text-left text-sm font-medium text-neutral-500 dark:text-neutral-400">{t('date')}</th>
                <th className="py-3 px-4 text-left text-sm font-medium text-neutral-500 dark:text-neutral-400">{t('prediction')}</th>
                <th className="py-3 px-4 text-left text-sm font-medium text-neutral-500 dark:text-neutral-400">{t('predicted_price')}</th>
                <th className="py-3 px-4 text-left text-sm font-medium text-neutral-500 dark:text-neutral-400">{t('actual_price')}</th>
                <th className="py-3 px-4 text-left text-sm font-medium text-neutral-500 dark:text-neutral-400">{t('confidence')}</th>
                <th className="py-3 px-4 text-left text-sm font-medium text-neutral-500 dark:text-neutral-400">{t('accuracy')}</th>
              </tr>
            </thead>
            <tbody>
              {predictionData.predictionHistory.map((history, index) => {
                const accuracy = history.actualPrice ? (100 - Math.abs((history.predictedPrice - history.actualPrice) / history.actualPrice * 100)).toFixed(2) : '-';
                
                return (
                  <tr key={index} className="border-b border-neutral-200 dark:border-neutral-700">
                    <td className="py-3 px-4 text-sm text-neutral-900 dark:text-neutral-100">{history.date}</td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-2 py-1 rounded-full text-xs font-medium ${history.prediction === 'buy' || history.prediction === 'strong_buy' ? 'bg-success-100 dark:bg-success-900 text-success-800 dark:text-success-200' : history.prediction === 'sell' || history.prediction === 'strong_sell' ? 'bg-danger-100 dark:bg-danger-900 text-danger-800 dark:text-danger-200' : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-800 dark:text-neutral-200'}`}
                      >
                        {t(history.prediction)}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm text-neutral-900 dark:text-neutral-100">₹{history.predictedPrice.toLocaleString()}</td>
                    <td className="py-3 px-4 text-sm text-neutral-900 dark:text-neutral-100">₹{history.actualPrice.toLocaleString()}</td>
                    <td className="py-3 px-4 text-sm text-neutral-900 dark:text-neutral-100">{history.confidence}%</td>
                    <td className="py-3 px-4">
                      <span
                        className={`text-sm ${accuracy > 95 ? 'text-success-600 dark:text-success-400' : accuracy > 85 ? 'text-warning-600 dark:text-warning-400' : 'text-danger-600 dark:text-danger-400'}`}
                      >
                        {accuracy}%
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Disclaimer */}
      <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4 border border-neutral-200 dark:border-neutral-700">
        <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2 flex items-center">
          <FiAlertCircle className="mr-2 h-4 w-4 text-warning-500" />
          {t('disclaimer')}
        </h4>
        <p className="text-sm text-neutral-600 dark:text-neutral-400">
          {t('prediction_disclaimer')}
        </p>
      </div>
    </div>
  );
};

export default PredictionPage;