import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js';

// Custom metrics
const errorRate = new Rate('error_rate');
const responseTime = new Trend('response_time');
const requestCount = new Counter('request_count');
const activeUsers = new Gauge('active_users');
const throughput = new Rate('throughput');

// Stress test configuration - gradually increase load to find breaking point
export const options = {
  stages: [
    { duration: '1m', target: 5 },   // Warm up
    { duration: '2m', target: 10 },  // Normal load
    { duration: '2m', target: 20 },  // Increased load
    { duration: '2m', target: 50 },  // High load
    { duration: '3m', target: 100 }, // Stress load
    { duration: '3m', target: 150 }, // Heavy stress
    { duration: '2m', target: 200 }, // Breaking point test
    { duration: '2m', target: 100 }, // Recovery test
    { duration: '2m', target: 0 },   // Cool down
  ],
  thresholds: {
    http_req_duration: {
      'p(50)<1000': true,  // 50% of requests should be below 1s
      'p(90)<3000': true,  // 90% of requests should be below 3s
      'p(95)<5000': true,  // 95% of requests should be below 5s
      'p(99)<10000': true, // 99% of requests should be below 10s
    },
    http_req_failed: {
      'rate<0.1': true,    // Error rate should be less than 10% under stress
    },
    error_rate: {
      'rate<0.15': true,   // Custom error rate threshold
    },
    response_time: {
      'p(95)<8000': true,  // 95% response time under stress
    },
    throughput: {
      'rate>10': true,     // Minimum throughput requirement
    },
  },
};

// Base URL from environment
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test data pools
const stockSymbols = [
  'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX',
  'ORCL', 'CRM', 'ADBE', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO',
  'IBM', 'HPQ', 'DELL', 'VMW', 'SNOW', 'PLTR', 'COIN', 'SQ'
];

const timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'];
const indicators = ['RSI', 'MACD', 'EMA', 'SMA', 'BB', 'STOCH', 'ADX', 'CCI'];
const predictionTypes = ['price', 'trend', 'volatility', 'support_resistance'];
const modelTypes = ['xgboost', 'lstm', 'transformer', 'ensemble'];

// User session data
let userSession = {
  id: Math.random().toString(36).substring(7),
  startTime: Date.now(),
  requestCount: 0,
  errors: 0
};

export function setup() {
  console.log('Starting stress test setup...');
  
  // Verify system is ready
  const healthCheck = http.get(`${BASE_URL}/health`);
  if (healthCheck.status !== 200) {
    throw new Error(`System not ready. Health check failed with status: ${healthCheck.status}`);
  }
  
  console.log('System health check passed. Starting stress test...');
  return { baseUrl: BASE_URL, startTime: Date.now() };
}

export default function(data) {
  activeUsers.add(1);
  
  // Simulate different user behavior patterns under stress
  const userType = getUserType();
  
  switch (userType) {
    case 'heavy_trader':
      heavyTraderScenario(data.baseUrl);
      break;
    case 'data_analyst':
      dataAnalystScenario(data.baseUrl);
      break;
    case 'casual_user':
      casualUserScenario(data.baseUrl);
      break;
    case 'api_consumer':
      apiConsumerScenario(data.baseUrl);
      break;
    case 'bot_trader':
      botTraderScenario(data.baseUrl);
      break;
    default:
      mixedScenario(data.baseUrl);
  }
  
  // Variable sleep based on current load
  const currentVUs = __ENV.K6_VUS || 1;
  const sleepTime = Math.max(0.1, 2 - (currentVUs / 100)); // Reduce sleep as load increases
  sleep(sleepTime);
}

// User type distribution
function getUserType() {
  const rand = Math.random();
  if (rand < 0.3) return 'heavy_trader';
  if (rand < 0.5) return 'data_analyst';
  if (rand < 0.7) return 'casual_user';
  if (rand < 0.85) return 'api_consumer';
  if (rand < 0.95) return 'bot_trader';
  return 'mixed';
}

// Heavy trader scenario - frequent predictions and real-time data
function heavyTraderScenario(baseUrl) {
  const actions = [
    () => getRealTimeData(baseUrl),
    () => generatePrediction(baseUrl),
    () => getTechnicalIndicators(baseUrl),
    () => getNewsSentiment(baseUrl),
    () => getRealTimeData(baseUrl), // Double weight for real-time data
  ];
  
  // Execute 3-5 actions rapidly
  const actionCount = Math.floor(Math.random() * 3) + 3;
  for (let i = 0; i < actionCount; i++) {
    const action = actions[Math.floor(Math.random() * actions.length)];
    action();
    sleep(0.1); // Very short sleep between actions
  }
}

// Data analyst scenario - complex queries and batch operations
function dataAnalystScenario(baseUrl) {
  // Batch request multiple stocks
  const batchSize = Math.floor(Math.random() * 5) + 3;
  const symbols = getRandomSymbols(batchSize);
  
  symbols.forEach(symbol => {
    getHistoricalData(baseUrl, symbol);
    getTechnicalIndicators(baseUrl, symbol);
  });
  
  // Generate complex prediction
  generateComplexPrediction(baseUrl);
}

// Casual user scenario - browsing and simple queries
function casualUserScenario(baseUrl) {
  getStockList(baseUrl);
  sleep(0.5);
  
  const symbol = getRandomSymbol();
  getBasicStockData(baseUrl, symbol);
  sleep(0.3);
  
  if (Math.random() < 0.6) {
    getNewsSentiment(baseUrl, symbol);
  }
}

// API consumer scenario - systematic API calls
function apiConsumerScenario(baseUrl) {
  // Systematic data collection
  const symbols = getRandomSymbols(3);
  
  symbols.forEach(symbol => {
    makeApiCall(() => http.get(`${baseUrl}/api/v1/stocks/${symbol}/data`), 'stock_data');
    makeApiCall(() => http.get(`${baseUrl}/api/v1/indicators/${symbol}/RSI`), 'indicators');
  });
}

// Bot trader scenario - high frequency, predictable patterns
function botTraderScenario(baseUrl) {
  const symbol = getRandomSymbol();
  
  // Rapid fire requests
  for (let i = 0; i < 5; i++) {
    getRealTimeData(baseUrl, symbol);
    if (i % 2 === 0) {
      generatePrediction(baseUrl, symbol);
    }
  }
}

// Mixed scenario - random combination
function mixedScenario(baseUrl) {
  const actions = [
    () => getStockList(baseUrl),
    () => getRealTimeData(baseUrl),
    () => generatePrediction(baseUrl),
    () => getTechnicalIndicators(baseUrl),
    () => getNewsSentiment(baseUrl),
  ];
  
  const action = actions[Math.floor(Math.random() * actions.length)];
  action();
}

// Individual test functions
function getStockList(baseUrl) {
  makeApiCall(() => http.get(`${baseUrl}/api/v1/stocks/list`), 'stock_list');
}

function getRealTimeData(baseUrl, symbol = null) {
  const stock = symbol || getRandomSymbol();
  const timeframe = timeframes[Math.floor(Math.random() * 3)]; // Prefer shorter timeframes
  
  makeApiCall(
    () => http.get(`${baseUrl}/api/v1/stocks/${stock}/realtime?timeframe=${timeframe}`),
    'realtime_data'
  );
}

function getBasicStockData(baseUrl, symbol = null) {
  const stock = symbol || getRandomSymbol();
  const timeframe = getRandomTimeframe();
  
  makeApiCall(
    () => http.get(`${baseUrl}/api/v1/stocks/${stock}/data?timeframe=${timeframe}`),
    'stock_data'
  );
}

function getHistoricalData(baseUrl, symbol = null) {
  const stock = symbol || getRandomSymbol();
  const timeframe = timeframes[Math.floor(Math.random() * timeframes.length)];
  
  makeApiCall(
    () => http.get(`${baseUrl}/api/v1/stocks/${stock}/historical?timeframe=${timeframe}&limit=1000`),
    'historical_data'
  );
}

function generatePrediction(baseUrl, symbol = null) {
  const stock = symbol || getRandomSymbol();
  const predictionType = predictionTypes[Math.floor(Math.random() * predictionTypes.length)];
  const modelType = modelTypes[Math.floor(Math.random() * modelTypes.length)];
  
  const payload = {
    symbol: stock,
    prediction_type: predictionType,
    model_type: modelType,
    timeframe: getRandomTimeframe(),
    horizon: Math.floor(Math.random() * 30) + 1
  };
  
  makeApiCall(
    () => http.post(`${baseUrl}/api/v1/predictions/generate`, JSON.stringify(payload), {
      headers: { 'Content-Type': 'application/json' }
    }),
    'prediction'
  );
}

function generateComplexPrediction(baseUrl) {
  const symbols = getRandomSymbols(3);
  const payload = {
    symbols: symbols,
    prediction_type: 'portfolio',
    model_type: 'ensemble',
    timeframe: '1d',
    horizon: 7,
    include_confidence: true,
    include_shap: true
  };
  
  makeApiCall(
    () => http.post(`${baseUrl}/api/v1/predictions/batch`, JSON.stringify(payload), {
      headers: { 'Content-Type': 'application/json' }
    }),
    'complex_prediction'
  );
}

function getTechnicalIndicators(baseUrl, symbol = null) {
  const stock = symbol || getRandomSymbol();
  const indicator = indicators[Math.floor(Math.random() * indicators.length)];
  
  makeApiCall(
    () => http.get(`${baseUrl}/api/v1/indicators/${stock}/${indicator}`),
    'technical_indicators'
  );
}

function getNewsSentiment(baseUrl, symbol = null) {
  const stock = symbol || getRandomSymbol();
  
  makeApiCall(
    () => http.get(`${baseUrl}/api/v1/news/${stock}/sentiment?limit=20`),
    'news_sentiment'
  );
}

// Helper functions
function makeApiCall(requestFunc, scenario) {
  const startTime = Date.now();
  userSession.requestCount++;
  
  try {
    const response = requestFunc();
    const duration = Date.now() - startTime;
    
    const success = check(response, {
      [`${scenario} status is success`]: (r) => r.status >= 200 && r.status < 400,
      [`${scenario} response time acceptable`]: () => duration < 10000, // 10s max under stress
    });
    
    if (!success) {
      userSession.errors++;
    }
    
    // Record metrics
    requestCount.add(1, { scenario, user_type: getUserTypeFromSession() });
    responseTime.add(duration, { scenario });
    errorRate.add(!success, { scenario });
    throughput.add(1);
    
    return response;
  } catch (error) {
    userSession.errors++;
    errorRate.add(1, { scenario });
    console.error(`Request failed for ${scenario}:`, error.message);
    return null;
  }
}

function getRandomSymbol() {
  return stockSymbols[Math.floor(Math.random() * stockSymbols.length)];
}

function getRandomSymbols(count) {
  const shuffled = [...stockSymbols].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, Math.min(count, stockSymbols.length));
}

function getRandomTimeframe() {
  return timeframes[Math.floor(Math.random() * timeframes.length)];
}

function getUserTypeFromSession() {
  // Determine user type based on session behavior
  const requestRate = userSession.requestCount / ((Date.now() - userSession.startTime) / 1000);
  const errorRate = userSession.errors / userSession.requestCount;
  
  if (requestRate > 5) return 'bot_trader';
  if (requestRate > 2) return 'heavy_trader';
  if (errorRate > 0.1) return 'problematic_user';
  return 'normal_user';
}

export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  
  console.log('Stress test completed');
  console.log(`Duration: ${duration}s`);
  console.log(`Base URL: ${data.baseUrl}`);
  
  // Generate stress test report
  const report = {
    test_type: 'stress_test',
    base_url: data.baseUrl,
    duration_seconds: duration,
    timestamp: new Date().toISOString(),
    test_stages: [
      { stage: 'warmup', target_vus: 5, duration: '1m' },
      { stage: 'normal_load', target_vus: 10, duration: '2m' },
      { stage: 'increased_load', target_vus: 20, duration: '2m' },
      { stage: 'high_load', target_vus: 50, duration: '2m' },
      { stage: 'stress_load', target_vus: 100, duration: '3m' },
      { stage: 'heavy_stress', target_vus: 150, duration: '3m' },
      { stage: 'breaking_point', target_vus: 200, duration: '2m' },
      { stage: 'recovery', target_vus: 100, duration: '2m' },
      { stage: 'cooldown', target_vus: 0, duration: '2m' }
    ],
    user_scenarios: [
      'heavy_trader',
      'data_analyst', 
      'casual_user',
      'api_consumer',
      'bot_trader',
      'mixed'
    ]
  };
  
  console.log('Stress Test Report:', JSON.stringify(report, null, 2));
}

// Generate HTML report
export function handleSummary(data) {
  return {
    'stress-test-report.html': htmlReport(data),
    'stress-test-summary.json': JSON.stringify(data, null, 2),
  };
}