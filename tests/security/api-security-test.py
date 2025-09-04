#!/usr/bin/env python3
"""
API Security Testing Script for Stock Prediction System
Focuses on REST API endpoints, WebSocket security, and business logic vulnerabilities
"""

import requests
import websocket
import json
import jwt
import time
import threading
import ssl
from datetime import datetime, timedelta
import logging
import argparse
import hashlib
import hmac
import base64
from urllib.parse import urlencode
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APISecurityTester:
    def __init__(self, base_url="http://localhost:8000", ws_url="ws://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.ws_url = ws_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'APISecurityTester/1.0',
            'Content-Type': 'application/json'
        })
        self.test_results = []
        self.vulnerabilities = []
        self.auth_token = None
    
    def log_test_result(self, test_name, status, severity='info', details=None, vulnerability=False):
        """Log test result"""
        result = {
            'test': test_name,
            'status': status,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.test_results.append(result)
        
        if vulnerability:
            self.vulnerabilities.append(result)
        
        logger.info(f"{test_name}: {status} ({severity})")
    
    def test_api_authentication(self):
        """Test API authentication mechanisms"""
        logger.info("Testing API authentication...")
        
        # Test 1: Access protected endpoints without authentication
        protected_endpoints = [
            '/api/v1/user/profile',
            '/api/v1/predictions/create',
            '/api/v1/admin/users',
            '/api/v1/user/settings'
        ]
        
        auth_bypass_found = False
        for endpoint in protected_endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    auth_bypass_found = True
                    self.log_test_result(
                        f"Authentication Bypass - {endpoint}",
                        'FAIL',
                        'critical',
                        {'endpoint': endpoint, 'status_code': response.status_code},
                        vulnerability=True
                    )
            except Exception as e:
                logger.warning(f"Error testing {endpoint}: {e}")
        
        if not auth_bypass_found:
            self.log_test_result('Authentication Protection', 'PASS', 'info')
        
        # Test 2: JWT token manipulation
        self.test_jwt_vulnerabilities()
        
        # Test 3: Session management
        self.test_session_management()
    
    def test_jwt_vulnerabilities(self):
        """Test JWT token vulnerabilities"""
        logger.info("Testing JWT vulnerabilities...")
        
        # Try to get a valid token first
        try:
            login_response = self.session.post(
                f"{self.base_url}/api/v1/auth/login",
                json={'username': 'test@example.com', 'password': 'testpassword'}
            )
            
            if login_response.status_code == 200:
                token_data = login_response.json()
                self.auth_token = token_data.get('access_token')
        except Exception as e:
            logger.warning(f"Could not obtain valid token: {e}")
        
        jwt_vulnerabilities = []
        
        # Test 1: None algorithm attack
        if self.auth_token:
            try:
                # Decode the token to get payload
                decoded = jwt.decode(self.auth_token, options={"verify_signature": False})
                
                # Create token with 'none' algorithm
                none_token = jwt.encode(decoded, '', algorithm='none')
                
                # Test with none algorithm token
                headers = {'Authorization': f'Bearer {none_token}'}
                response = self.session.get(
                    f"{self.base_url}/api/v1/user/profile",
                    headers=headers
                )
                
                if response.status_code == 200:
                    jwt_vulnerabilities.append({
                        'type': 'None Algorithm Attack',
                        'description': 'Server accepts JWT tokens with none algorithm'
                    })
            
            except Exception as e:
                logger.warning(f"Error testing none algorithm: {e}")
        
        # Test 2: Weak secret brute force
        weak_secrets = ['secret', '123456', 'password', 'jwt_secret', 'key']
        if self.auth_token:
            for secret in weak_secrets:
                try:
                    decoded = jwt.decode(self.auth_token, secret, algorithms=['HS256'])
                    jwt_vulnerabilities.append({
                        'type': 'Weak JWT Secret',
                        'secret': secret,
                        'description': f'JWT can be decoded with weak secret: {secret}'
                    })
                    break
                except jwt.InvalidSignatureError:
                    continue
                except Exception:
                    continue
        
        # Test 3: Token expiration
        if self.auth_token:
            try:
                decoded = jwt.decode(self.auth_token, options={"verify_signature": False})
                exp = decoded.get('exp')
                if not exp:
                    jwt_vulnerabilities.append({
                        'type': 'No Token Expiration',
                        'description': 'JWT token does not have expiration time'
                    })
                else:
                    exp_time = datetime.fromtimestamp(exp)
                    if exp_time > datetime.now() + timedelta(days=1):
                        jwt_vulnerabilities.append({
                            'type': 'Long Token Expiration',
                            'expiration': exp_time.isoformat(),
                            'description': 'JWT token has very long expiration time'
                        })
            except Exception as e:
                logger.warning(f"Error checking token expiration: {e}")
        
        if jwt_vulnerabilities:
            self.log_test_result(
                'JWT Security Test',
                'FAIL',
                'high',
                {'vulnerabilities': jwt_vulnerabilities},
                vulnerability=True
            )
        else:
            self.log_test_result('JWT Security Test', 'PASS', 'info')
    
    def test_session_management(self):
        """Test session management security"""
        logger.info("Testing session management...")
        
        session_issues = []
        
        # Test 1: Session fixation
        try:
            # Get initial session
            response1 = self.session.get(f"{self.base_url}/api/v1/health")
            initial_cookies = self.session.cookies.copy()
            
            # Login
            login_response = self.session.post(
                f"{self.base_url}/api/v1/auth/login",
                json={'username': 'test@example.com', 'password': 'testpassword'}
            )
            
            # Check if session ID changed after login
            if login_response.status_code == 200:
                post_login_cookies = self.session.cookies.copy()
                
                # Compare session cookies
                session_changed = False
                for cookie in initial_cookies:
                    if cookie.name.lower() in ['sessionid', 'jsessionid', 'phpsessid']:
                        if cookie.value == post_login_cookies.get(cookie.name):
                            session_issues.append({
                                'type': 'Session Fixation',
                                'description': 'Session ID does not change after login'
                            })
                            break
        
        except Exception as e:
            logger.warning(f"Error testing session fixation: {e}")
        
        # Test 2: Session timeout
        try:
            # Check if there's a session timeout mechanism
            if self.auth_token:
                headers = {'Authorization': f'Bearer {self.auth_token}'}
                
                # Make request after some time
                time.sleep(2)
                response = self.session.get(
                    f"{self.base_url}/api/v1/user/profile",
                    headers=headers
                )
                
                # This is a basic test - in real scenario, you'd wait longer
                # or manipulate token timestamps
        
        except Exception as e:
            logger.warning(f"Error testing session timeout: {e}")
        
        if session_issues:
            self.log_test_result(
                'Session Management Test',
                'FAIL',
                'medium',
                {'issues': session_issues},
                vulnerability=True
            )
        else:
            self.log_test_result('Session Management Test', 'PASS', 'info')
    
    def test_api_rate_limiting(self):
        """Test API rate limiting"""
        logger.info("Testing API rate limiting...")
        
        endpoints_to_test = [
            '/api/v1/stocks/AAPL',
            '/api/v1/predictions',
            '/api/v1/news',
            '/api/v1/auth/login'
        ]
        
        rate_limit_issues = []
        
        for endpoint in endpoints_to_test:
            try:
                # Send rapid requests
                responses = []
                start_time = time.time()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                    futures = []
                    for i in range(50):
                        if endpoint == '/api/v1/auth/login':
                            future = executor.submit(
                                self.session.post,
                                f"{self.base_url}{endpoint}",
                                json={'username': f'user{i}@test.com', 'password': 'password'}
                            )
                        else:
                            future = executor.submit(
                                self.session.get,
                                f"{self.base_url}{endpoint}"
                            )
                        futures.append(future)
                    
                    for future in concurrent.futures.as_completed(futures, timeout=30):
                        try:
                            response = future.result()
                            responses.append(response.status_code)
                        except Exception:
                            responses.append(0)
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Analyze responses
                rate_limited_count = responses.count(429)  # Too Many Requests
                success_count = responses.count(200)
                
                if rate_limited_count == 0 and success_count > 40:
                    rate_limit_issues.append({
                        'endpoint': endpoint,
                        'requests_sent': len(responses),
                        'success_responses': success_count,
                        'rate_limited_responses': rate_limited_count,
                        'duration': duration,
                        'requests_per_second': len(responses) / duration
                    })
            
            except Exception as e:
                logger.warning(f"Error testing rate limiting on {endpoint}: {e}")
        
        if rate_limit_issues:
            self.log_test_result(
                'API Rate Limiting Test',
                'FAIL',
                'medium',
                {'issues': rate_limit_issues},
                vulnerability=True
            )
        else:
            self.log_test_result('API Rate Limiting Test', 'PASS', 'info')
    
    def test_business_logic_vulnerabilities(self):
        """Test business logic vulnerabilities specific to stock prediction system"""
        logger.info("Testing business logic vulnerabilities...")
        
        business_logic_issues = []
        
        # Test 1: Price manipulation through prediction requests
        try:
            # Test if we can manipulate predictions by sending crafted requests
            malicious_predictions = [
                {'symbol': 'AAPL', 'predicted_price': -1000},  # Negative price
                {'symbol': 'GOOGL', 'predicted_price': 999999999},  # Unrealistic price
                {'symbol': '../../../etc/passwd', 'predicted_price': 100},  # Path traversal
            ]
            
            for prediction in malicious_predictions:
                response = self.session.post(
                    f"{self.base_url}/api/v1/predictions",
                    json=prediction
                )
                
                if response.status_code == 200:
                    business_logic_issues.append({
                        'type': 'Invalid Prediction Accepted',
                        'payload': prediction,
                        'response_code': response.status_code
                    })
        
        except Exception as e:
            logger.warning(f"Error testing prediction manipulation: {e}")
        
        # Test 2: Historical data manipulation
        try:
            # Test if we can inject false historical data
            false_data = {
                'symbol': 'TEST',
                'date': '2030-01-01',  # Future date
                'price': 'invalid_price',  # Invalid price format
                'volume': -1000  # Negative volume
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/stocks/historical",
                json=false_data
            )
            
            if response.status_code == 200:
                business_logic_issues.append({
                    'type': 'Invalid Historical Data Accepted',
                    'payload': false_data,
                    'response_code': response.status_code
                })
        
        except Exception as e:
            logger.warning(f"Error testing historical data manipulation: {e}")
        
        # Test 3: News sentiment manipulation
        try:
            # Test if we can inject false news with manipulated sentiment
            false_news = {
                'title': '<script>alert("XSS")</script>',
                'content': 'Malicious content with SQL injection \'; DROP TABLE news; --',
                'sentiment': 999,  # Invalid sentiment score
                'source': 'fake_source'
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/news",
                json=false_news
            )
            
            if response.status_code == 200:
                business_logic_issues.append({
                    'type': 'Malicious News Content Accepted',
                    'payload': false_news,
                    'response_code': response.status_code
                })
        
        except Exception as e:
            logger.warning(f"Error testing news manipulation: {e}")
        
        if business_logic_issues:
            self.log_test_result(
                'Business Logic Vulnerabilities Test',
                'FAIL',
                'high',
                {'issues': business_logic_issues},
                vulnerability=True
            )
        else:
            self.log_test_result('Business Logic Vulnerabilities Test', 'PASS', 'info')
    
    def test_websocket_security(self):
        """Test WebSocket security"""
        logger.info("Testing WebSocket security...")
        
        ws_issues = []
        
        # Test 1: WebSocket authentication
        try:
            def on_message(ws, message):
                logger.info(f"Received WebSocket message: {message}")
            
            def on_error(ws, error):
                logger.warning(f"WebSocket error: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                logger.info("WebSocket connection closed")
            
            def on_open(ws):
                logger.info("WebSocket connection opened")
                # Try to send malicious messages
                malicious_messages = [
                    '{"type": "subscribe", "symbol": "../../../etc/passwd"}',
                    '{"type": "admin_command", "command": "shutdown"}',
                    '{"type": "inject", "payload": "<script>alert(1)</script>"}'
                ]
                
                for msg in malicious_messages:
                    ws.send(msg)
                    time.sleep(0.1)
                
                ws.close()
            
            # Test WebSocket without authentication
            ws_url = self.ws_url.replace('http://', 'ws://').replace('https://', 'wss://')
            ws = websocket.WebSocketApp(
                f"{ws_url}/ws/stocks",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Run WebSocket in a separate thread with timeout
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            ws_thread.join(timeout=5)
            
            # If connection was successful without auth, it's a vulnerability
            if ws_thread.is_alive():
                ws_issues.append({
                    'type': 'Unauthenticated WebSocket Access',
                    'description': 'WebSocket accepts connections without authentication'
                })
        
        except Exception as e:
            logger.warning(f"Error testing WebSocket security: {e}")
        
        # Test 2: WebSocket message injection
        # This would be more complex and require actual WebSocket connection
        
        if ws_issues:
            self.log_test_result(
                'WebSocket Security Test',
                'FAIL',
                'medium',
                {'issues': ws_issues},
                vulnerability=True
            )
        else:
            self.log_test_result('WebSocket Security Test', 'PASS', 'info')
    
    def test_api_versioning_security(self):
        """Test API versioning security issues"""
        logger.info("Testing API versioning security...")
        
        versioning_issues = []
        
        # Test different API versions
        api_versions = ['v1', 'v2', 'v0', 'beta', 'alpha', 'dev', 'test']
        
        for version in api_versions:
            try:
                response = self.session.get(f"{self.base_url}/api/{version}/stocks")
                
                if response.status_code == 200 and version in ['v0', 'dev', 'test', 'alpha', 'beta']:
                    versioning_issues.append({
                        'version': version,
                        'status_code': response.status_code,
                        'description': f'Potentially insecure API version {version} is accessible'
                    })
            
            except Exception as e:
                logger.warning(f"Error testing API version {version}: {e}")
        
        if versioning_issues:
            self.log_test_result(
                'API Versioning Security Test',
                'FAIL',
                'low',
                {'issues': versioning_issues},
                vulnerability=True
            )
        else:
            self.log_test_result('API Versioning Security Test', 'PASS', 'info')
    
    def test_data_exposure(self):
        """Test for sensitive data exposure"""
        logger.info("Testing data exposure...")
        
        exposure_issues = []
        
        # Test 1: Check if API responses contain sensitive information
        endpoints_to_check = [
            '/api/v1/stocks/AAPL',
            '/api/v1/predictions',
            '/api/v1/news',
            '/api/v1/user/profile'
        ]
        
        sensitive_patterns = [
            r'password',
            r'secret',
            r'key',
            r'token',
            r'api_key',
            r'private',
            r'confidential',
            r'internal',
            r'debug',
            r'stack.*trace',
            r'exception'
        ]
        
        for endpoint in endpoints_to_check:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                
                if response.status_code == 200:
                    response_text = response.text.lower()
                    
                    for pattern in sensitive_patterns:
                        import re
                        if re.search(pattern, response_text):
                            exposure_issues.append({
                                'endpoint': endpoint,
                                'pattern': pattern,
                                'description': f'Potentially sensitive information found: {pattern}'
                            })
            
            except Exception as e:
                logger.warning(f"Error checking data exposure on {endpoint}: {e}")
        
        if exposure_issues:
            self.log_test_result(
                'Data Exposure Test',
                'FAIL',
                'medium',
                {'issues': exposure_issues},
                vulnerability=True
            )
        else:
            self.log_test_result('Data Exposure Test', 'PASS', 'info')
    
    def run_all_tests(self):
        """Run all API security tests"""
        logger.info("Starting comprehensive API security testing...")
        
        test_methods = [
            self.test_api_authentication,
            self.test_api_rate_limiting,
            self.test_business_logic_vulnerabilities,
            self.test_websocket_security,
            self.test_api_versioning_security,
            self.test_data_exposure
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                logger.error(f"Error running {test_method.__name__}: {e}")
                self.log_test_result(
                    test_method.__name__.replace('test_', '').replace('_', ' ').title(),
                    'ERROR',
                    'info',
                    {'error': str(e)}
                )
        
        logger.info("API security testing completed")
    
    def generate_report(self, output_file='api_security_report.json'):
        """Generate API security test report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'target': self.base_url,
            'test_results': self.test_results,
            'vulnerabilities': self.vulnerabilities,
            'summary': {
                'total_tests': len(self.test_results),
                'vulnerabilities_found': len(self.vulnerabilities),
                'critical': len([v for v in self.vulnerabilities if v['severity'] == 'critical']),
                'high': len([v for v in self.vulnerabilities if v['severity'] == 'high']),
                'medium': len([v for v in self.vulnerabilities if v['severity'] == 'medium']),
                'low': len([v for v in self.vulnerabilities if v['severity'] == 'low'])
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"API security report generated: {output_file}")
        
        # Print summary
        summary = report['summary']
        print("\n" + "="*50)
        print("API SECURITY TESTING SUMMARY")
        print("="*50)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Vulnerabilities Found: {summary['vulnerabilities_found']}")
        print(f"\nSeverity Breakdown:")
        print(f"Critical: {summary['critical']}")
        print(f"High: {summary['high']}")
        print(f"Medium: {summary['medium']}")
        print(f"Low: {summary['low']}")
        print("="*50)
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Stock Prediction System API Security Testing')
    parser.add_argument('--url', default='http://localhost:8000', help='Target API URL')
    parser.add_argument('--ws-url', default='ws://localhost:8000', help='WebSocket URL')
    parser.add_argument('--output', default='api_security_report.json', help='Output file')
    
    args = parser.parse_args()
    
    tester = APISecurityTester(args.url, args.ws_url)
    tester.run_all_tests()
    tester.generate_report(args.output)

if __name__ == '__main__':
    main()