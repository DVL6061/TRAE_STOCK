import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('error_rate');
const responseTime = new Trend('response_time');
const requestCount = new Counter('request_count');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 }, // Ramp up to 10 users
    { duration: '5m', target: 10 }, // Stay at 10 users
    { duration: '2m', target: 20 }, // Ramp up to 20 users
    { duration: '5m', target: 20 }, // Stay at 20 users
    { duration: '2m', target: 0 },  // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests should be below 2s
    http_req_failed: ['rate<0.05'],    // Error rate should be less than 5%
    error_rate: ['rate<0.05'],
    response_time: ['p(95)<2000'],
  },
};

// Base URL from environment or default
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test data
const testStocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX'];
const testTimeframes = ['1d', '1w', '1m', '3m', '6m', '1y'];
const testIndicators = ['RSI', 'MACD', 'EMA', 'SMA', 'BB'];

// Authentication token (mock for testing)
let authToken = null;

// Setup function - runs once per VU
export function setup() {
  console.log('Starting load test setup...');
  
  // Test basic connectivity
  const healthResponse = http.get(`${BASE_URL}/health`);
  check(healthResponse, {
    'Health check status is 200': (r) => r.status === 200,
  });
  
  return { baseUrl: BASE_URL };
}

// Main test function
export default function(data) {
  const baseUrl = data.baseUrl;
  
  // Test scenarios with different weights
  const scenarios = [
    { name: 'health_check', weight: 10, func: testHealthCheck },
    { name: 'stock_list', weight: 15, func: testStockList },
    { name: 'stock_data', weight: 20, func: testStockData },
    { name: 'predictions', weight: 25, func: testPredictions },
    { name: 'technical_indicators', weight: 15, func: testTechnicalIndicators },
    { name: 'news_sentiment', weight: 10, func: testNewsSentiment },
    { name: 'websocket_simulation', weight: 5, func: testWebSocketSimulation },
  ];
  
  // Select scenario based on weight
  const totalWeight = scenarios.reduce((sum, s) => sum + s.weight, 0);
  const random = Math.random() * totalWeight;
  let currentWeight = 0;
  
  for (const scenario of scenarios) {
    currentWeight += scenario.weight;
    if (random <= currentWeight) {
      scenario.func(baseUrl);
      break;
    }
  }
  
  // Random sleep between 1-3 seconds
  sleep(Math.random() * 2 + 1);
}

// Test functions
function testHealthCheck(baseUrl) {
  const startTime = Date.now();
  const response = http.get(`${baseUrl}/health`);
  const duration = Date.now() - startTime;
  
  const success = check(response, {
    'Health check status is 200': (r) => r.status === 200,
    'Health check response time < 500ms': () => duration < 500,
    'Health check has correct content-type': (r) => r.headers['Content-Type'].includes('application/json'),
  });
  
  recordMetrics(success, duration, 'health_check');
}

function testStockList(baseUrl) {
  const startTime = Date.now();
  const response = http.get(`${baseUrl}/api/v1/stocks/list`, {
    headers: getHeaders(),
  });
  const duration = Date.now() - startTime;
  
  const success = check(response, {
    'Stock list status is 200': (r) => r.status === 200,
    'Stock list response time < 1000ms': () => duration < 1000,
    'Stock list returns array': (r) => {
      try {
        const data = JSON.parse(r.body);
        return Array.isArray(data.stocks);
      } catch (e) {
        return false;
      }
    },
  });
  
  recordMetrics(success, duration, 'stock_list');
}

function testStockData(baseUrl) {
  const stock = testStocks[Math.floor(Math.random() * testStocks.length)];
  const timeframe = testTimeframes[Math.floor(Math.random() * testTimeframes.length)];
  
  const startTime = Date.now();
  const response = http.get(`${baseUrl}/api/v1/stocks/${stock}/data?timeframe=${timeframe}`, {
    headers: getHeaders(),
  });
  const duration = Date.now() - startTime;
  
  const success = check(response, {
    'Stock data status is 200 or 404': (r) => r.status === 200 || r.status === 404,
    'Stock data response time < 2000ms': () => duration < 2000,
    'Stock data has valid structure': (r) => {
      if (r.status !== 200) return true;
      try {
        const data = JSON.parse(r.body);
        return data.hasOwnProperty('ohlcv') || data.hasOwnProperty('error');
      } catch (e) {
        return false;
      }
    },
  });
  
  recordMetrics(success, duration, 'stock_data');
}

function testPredictions(baseUrl) {
  const stock = testStocks[Math.floor(Math.random() * testStocks.length)];
  const timeframe = testTimeframes[Math.floor(Math.random() * testTimeframes.length)];
  
  const payload = {
    symbol: stock,
    timeframe: timeframe,
    prediction_type: 'price',
    model_type: 'xgboost'
  };
  
  const startTime = Date.now();
  const response = http.post(`${baseUrl}/api/v1/predictions/generate`, JSON.stringify(payload), {
    headers: {
      ...getHeaders(),
      'Content-Type': 'application/json',
    },
  });
  const duration = Date.now() - startTime;
  
  const success = check(response, {
    'Prediction status is 200 or 202': (r) => r.status === 200 || r.status === 202,
    'Prediction response time < 5000ms': () => duration < 5000,
    'Prediction has valid response': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.hasOwnProperty('prediction_id') || data.hasOwnProperty('prediction');
      } catch (e) {
        return false;
      }
    },
  });
  
  recordMetrics(success, duration, 'predictions');
}

function testTechnicalIndicators(baseUrl) {
  const stock = testStocks[Math.floor(Math.random() * testStocks.length)];
  const indicator = testIndicators[Math.floor(Math.random() * testIndicators.length)];
  
  const startTime = Date.now();
  const response = http.get(`${baseUrl}/api/v1/indicators/${stock}/${indicator}`, {
    headers: getHeaders(),
  });
  const duration = Date.now() - startTime;
  
  const success = check(response, {
    'Indicator status is 200 or 404': (r) => r.status === 200 || r.status === 404,
    'Indicator response time < 1500ms': () => duration < 1500,
    'Indicator has valid data': (r) => {
      if (r.status !== 200) return true;
      try {
        const data = JSON.parse(r.body);
        return data.hasOwnProperty('values') || data.hasOwnProperty('error');
      } catch (e) {
        return false;
      }
    },
  });
  
  recordMetrics(success, duration, 'technical_indicators');
}

function testNewsSentiment(baseUrl) {
  const stock = testStocks[Math.floor(Math.random() * testStocks.length)];
  
  const startTime = Date.now();
  const response = http.get(`${baseUrl}/api/v1/news/${stock}/sentiment`, {
    headers: getHeaders(),
  });
  const duration = Date.now() - startTime;
  
  const success = check(response, {
    'News sentiment status is 200 or 404': (r) => r.status === 200 || r.status === 404,
    'News sentiment response time < 3000ms': () => duration < 3000,
    'News sentiment has valid structure': (r) => {
      if (r.status !== 200) return true;
      try {
        const data = JSON.parse(r.body);
        return data.hasOwnProperty('sentiment') || data.hasOwnProperty('news');
      } catch (e) {
        return false;
      }
    },
  });
  
  recordMetrics(success, duration, 'news_sentiment');
}

function testWebSocketSimulation(baseUrl) {
  // Simulate WebSocket connection by testing the WebSocket endpoint
  const startTime = Date.now();
  const response = http.get(`${baseUrl}/ws/info`, {
    headers: getHeaders(),
  });
  const duration = Date.now() - startTime;
  
  const success = check(response, {
    'WebSocket info status is 200': (r) => r.status === 200,
    'WebSocket info response time < 500ms': () => duration < 500,
  });
  
  recordMetrics(success, duration, 'websocket_simulation');
}

// Helper functions
function getHeaders() {
  const headers = {
    'User-Agent': 'k6-load-test/1.0',
    'Accept': 'application/json',
  };
  
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }
  
  return headers;
}

function recordMetrics(success, duration, scenario) {
  requestCount.add(1, { scenario });
  responseTime.add(duration, { scenario });
  errorRate.add(!success, { scenario });
}

// Teardown function - runs once after all VUs finish
export function teardown(data) {
  console.log('Load test completed');
  console.log(`Base URL: ${data.baseUrl}`);
  
  // Generate summary report
  const summary = {
    test_type: 'load_test',
    base_url: data.baseUrl,
    timestamp: new Date().toISOString(),
    scenarios: [
      'health_check',
      'stock_list', 
      'stock_data',
      'predictions',
      'technical_indicators',
      'news_sentiment',
      'websocket_simulation'
    ]
  };
  
  console.log('Test Summary:', JSON.stringify(summary, null, 2));
}