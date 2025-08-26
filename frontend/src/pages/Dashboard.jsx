import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import useWebSocket from 'react-use-websocket';
import {
  ChartBarIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  CurrencyRupeeIcon,
  EyeIcon,
  BoltIcon,
  CpuChipIcon,
  GlobeAltIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ArrowUpIcon,
  ArrowDownIcon
} from '@heroicons/react/24/outline';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar
} from 'recharts';

const Dashboard = () => {
  const { t, i18n } = useTranslation();
  const [marketData, setMarketData] = useState(null);
  const [portfolioData, setPortfolioData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D');

  // WebSocket connection
  const { lastMessage, readyState } = useWebSocket('ws://localhost:8000/ws/market', {
    onOpen: () => console.log('WebSocket Connected'),
    onClose: () => console.log('WebSocket Disconnected'),
    onError: (error) => console.error('WebSocket Error:', error),
    shouldReconnect: (closeEvent) => true,
    reconnectInterval: 3000
  });

  // Handle real-time updates
  useEffect(() => {
    if (lastMessage !== null) {
      const data = JSON.parse(lastMessage.data);
      
      if (data.type === 'market_update') {
        setMarketData(data.payload);
      } else if (data.type === 'prediction_update') {
        setPredictions(data.payload);
      }
    }
  }, [lastMessage]);

  // Initial data fetch
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        
        // Fetch initial market data
        const marketResponse = await fetch('http://localhost:8000/api/market/current');
        const marketData = await marketResponse.json();
        setMarketData(marketData);

        // Fetch portfolio data
        const portfolioResponse = await fetch('http://localhost:8000/api/portfolio/summary');
        const portfolioData = await portfolioResponse.json();
        setPortfolioData(portfolioData);

        // Fetch AI predictions
        const predictionsResponse = await fetch('http://localhost:8000/api/predictions/latest');
        const predictionsData = await predictionsResponse.json();
        setPredictions(predictionsData);

      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchDashboardData();
  }, []);

  // Mock chart data
  const chartData = [
    { time: '09:15', nifty: 19720, volume: 1200000 },
    { time: '10:00', nifty: 19750, volume: 1500000 },
    { time: '11:00', nifty: 19780, volume: 1800000 },
    { time: '12:00', nifty: 19820, volume: 2100000 },
    { time: '13:00', nifty: 19845, volume: 2300000 },
    { time: '14:00', nifty: 19860, volume: 2500000 },
    { time: '15:30', nifty: 19845, volume: 2800000 }
  ];

  const sectorData = [
    { name: t('sectors.banking'), value: 35, color: '#3B82F6' },
    { name: t('sectors.it'), value: 25, color: '#10B981' },
    { name: t('sectors.pharma'), value: 15, color: '#F59E0B' },
    { name: t('sectors.auto'), value: 12, color: '#EF4444' },
    { name: t('sectors.fmcg'), value: 8, color: '#8B5CF6' },
    { name: t('sectors.metals'), value: 5, color: '#06B6D4' }
  ];

  const formatCurrency = (value) => {
    return new Intl.NumberFormat(i18n.language === 'hi' ? 'hi-IN' : 'en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatNumber = (value) => {
    return new Intl.NumberFormat(i18n.language === 'hi' ? 'hi-IN' : 'en-IN').format(value);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-lg text-gray-600 dark:text-gray-400">
          {t('common.loading')}
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            {t('dashboard.title')}
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            {t('dashboard.subtitle')}
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            <span>{t('dashboard.liveData')}</span>
          </div>
          
          <select 
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value)}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="1D">{t('timeframes.1d')}</option>
            <option value="1W">{t('timeframes.1w')}</option>
            <option value="1M">{t('timeframes.1m')}</option>
            <option value="3M">{t('timeframes.3m')}</option>
          </select>
        </div>
      </div>

      {/* Market Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {marketData && Object.entries(marketData).map(([key, data]) => (
          <div key={key} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white uppercase">
                {key === 'nifty' ? 'NIFTY 50' : key === 'sensex' ? 'SENSEX' : 'BANK NIFTY'}
              </h3>
              <div className={`p-2 rounded-lg ${
                data.change >= 0 ? 'bg-green-100 dark:bg-green-900/20' : 'bg-red-100 dark:bg-red-900/20'
              }`}>
                {data.change >= 0 ? (
                  <TrendingUpIcon className="h-5 w-5 text-green-600 dark:text-green-400" />
                ) : (
                  <TrendingDownIcon className="h-5 w-5 text-red-600 dark:text-red-400" />
                )}
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {formatNumber(data.value)}
              </div>
              
              <div className="flex items-center gap-2">
                <span className={`text-sm font-medium ${
                  data.change >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                }`}>
                  {data.change >= 0 ? '+' : ''}{formatNumber(data.change)}
                </span>
                <span className={`text-sm ${
                  data.change >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                }`}>
                  ({data.changePercent >= 0 ? '+' : ''}{data.changePercent}%)
                </span>
              </div>
              
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {t('dashboard.volume')}: {data.volume}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart Section */}
        <div className="lg:col-span-2 space-y-6">
          {/* Price Chart */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                {t('dashboard.priceChart')}
              </h3>
              <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                <ClockIcon className="h-4 w-4" />
                <span>{t('dashboard.realTime')}</span>
              </div>
            </div>
            
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis 
                    dataKey="time" 
                    className="text-xs"
                    tick={{ fill: 'currentColor' }}
                  />
                  <YAxis 
                    className="text-xs"
                    tick={{ fill: 'currentColor' }}
                    domain={['dataMin - 50', 'dataMax + 50']}
                  />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: 'var(--tooltip-bg)',
                      border: '1px solid var(--tooltip-border)',
                      borderRadius: '8px'
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="nifty" 
                    stroke="#3B82F6" 
                    fill="#3B82F6" 
                    fillOpacity={0.1}
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* AI Predictions */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-6">
              <CpuChipIcon className="h-6 w-6 text-purple-600" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                {t('dashboard.aiPredictions')}
              </h3>
            </div>
            
            {predictions && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* XGBoost */}
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <ChartBarIcon className="h-4 w-4 text-blue-600" />
                    <span className="text-sm font-medium text-blue-900 dark:text-blue-300">
                      XGBoost
                    </span>
                  </div>
                  <div className={`text-lg font-bold ${
                    predictions.xgboost.signal === 'BUY' ? 'text-green-600' : 
                    predictions.xgboost.signal === 'SELL' ? 'text-red-600' : 'text-yellow-600'
                  }`}>
                    {t(`signals.${predictions.xgboost.signal.toLowerCase()}`)}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    {t('dashboard.confidence')}: {(predictions.xgboost.confidence * 100).toFixed(0)}%
                  </div>
                </div>

                {/* Informer */}
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <BoltIcon className="h-4 w-4 text-green-600" />
                    <span className="text-sm font-medium text-green-900 dark:text-green-300">
                      Informer
                    </span>
                  </div>
                  <div className={`text-lg font-bold ${
                    predictions.informer.signal === 'BUY' ? 'text-green-600' : 
                    predictions.informer.signal === 'SELL' ? 'text-red-600' : 'text-yellow-600'
                  }`}>
                    {t(`signals.${predictions.informer.signal.toLowerCase()}`)}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    {t('dashboard.confidence')}: {(predictions.informer.confidence * 100).toFixed(0)}%
                  </div>
                </div>

                {/* DQN */}
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <CpuChipIcon className="h-4 w-4 text-purple-600" />
                    <span className="text-sm font-medium text-purple-900 dark:text-purple-300">
                      DQN
                    </span>
                  </div>
                  <div className={`text-lg font-bold ${
                    predictions.dqn.signal === 'BUY' ? 'text-green-600' : 
                    predictions.dqn.signal === 'SELL' ? 'text-red-600' : 'text-yellow-600'
                  }`}>
                    {t(`signals.${predictions.dqn.signal.toLowerCase()}`)}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    {t('dashboard.confidence')}: {(predictions.dqn.confidence * 100).toFixed(0)}%
                  </div>
                </div>

                {/* Sentiment */}
                <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <GlobeAltIcon className="h-4 w-4 text-orange-600" />
                    <span className="text-sm font-medium text-orange-900 dark:text-orange-300">
                      {t('dashboard.sentiment')}
                    </span>
                  </div>
                  <div className={`text-lg font-bold ${
                    predictions.sentiment.label === 'POSITIVE' ? 'text-green-600' : 
                    predictions.sentiment.label === 'NEGATIVE' ? 'text-red-600' : 'text-yellow-600'
                  }`}>
                    {t(`sentiment.${predictions.sentiment.label.toLowerCase()}`)}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    {predictions.sentiment.newsCount} {t('dashboard.newsAnalyzed')}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Portfolio Summary */}
          {portfolioData && (
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                {t('dashboard.portfolio')}
              </h3>
              
              <div className="space-y-4">
                <div>
                  <div className="text-2xl font-bold text-gray-900 dark:text-white">
                    {formatCurrency(portfolioData.totalValue)}
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <span className={`text-sm font-medium ${
                      portfolioData.dayChange >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {portfolioData.dayChange >= 0 ? '+' : ''}{formatCurrency(portfolioData.dayChange)}
                    </span>
                    <span className={`text-sm ${
                      portfolioData.dayChange >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      ({portfolioData.dayChangePercent >= 0 ? '+' : ''}{portfolioData.dayChangePercent}%)
                    </span>
                  </div>
                </div>
                
                <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                    {t('dashboard.totalReturn')}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`font-medium ${
                      portfolioData.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {portfolioData.totalReturn >= 0 ? '+' : ''}{formatCurrency(portfolioData.totalReturn)}
                    </span>
                    <span className={`text-sm ${
                      portfolioData.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      ({portfolioData.totalReturnPercent >= 0 ? '+' : ''}{portfolioData.totalReturnPercent}%)
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Sector Allocation */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              {t('dashboard.sectorAllocation')}
            </h3>
            
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={sectorData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {sectorData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            
            <div className="space-y-2 mt-4">
              {sectorData.map((sector) => (
                <div key={sector.name} className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: sector.color }}
                    />
                    <span className="text-gray-700 dark:text-gray-300">{sector.name}</span>
                  </div>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {sector.value}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              {t('dashboard.quickActions')}
            </h3>
            
            <div className="space-y-3">
              <button className="w-full flex items-center gap-3 p-3 text-left bg-blue-50 dark:bg-blue-900/20 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors">
                <TrendingUpIcon className="h-5 w-5 text-blue-600" />
                <span className="text-blue-900 dark:text-blue-300 font-medium">
                  {t('dashboard.viewPredictions')}
                </span>
              </button>
              
              <button className="w-full flex items-center gap-3 p-3 text-left bg-green-50 dark:bg-green-900/20 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors">
                <EyeIcon className="h-5 w-5 text-green-600" />
                <span className="text-green-900 dark:text-green-300 font-medium">
                  {t('dashboard.manageWatchlist')}
                </span>
              </button>
              
              <button className="w-full flex items-center gap-3 p-3 text-left bg-purple-50 dark:bg-purple-900/20 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors">
                <ChartBarIcon className="h-5 w-5 text-purple-600" />
                <span className="text-purple-900 dark:text-purple-300 font-medium">
                  {t('dashboard.technicalAnalysis')}
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;