#!/usr/bin/env python3
"""
Security Configuration Scanner for Stock Prediction System
Scans deployment configurations, Docker files, and environment settings for security issues
"""

import os
import json
import yaml
import re
import logging
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityConfigScanner:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.findings = []
        self.config_files = []
        self.scan_results = {
            'docker': [],
            'environment': [],
            'nginx': [],
            'database': [],
            'api': [],
            'secrets': [],
            'ssl_tls': [],
            'monitoring': []
        }
    
    def log_finding(self, category: str, severity: str, title: str, description: str, 
                   file_path: str = None, line_number: int = None, recommendation: str = None):
        """Log a security finding"""
        finding = {
            'category': category,
            'severity': severity,
            'title': title,
            'description': description,
            'file_path': str(file_path) if file_path else None,
            'line_number': line_number,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
        
        self.findings.append(finding)
        self.scan_results[category].append(finding)
        
        logger.info(f"{severity.upper()}: {title} in {file_path or 'configuration'}")
    
    def find_config_files(self):
        """Find all configuration files in the project"""
        config_patterns = [
            '**/*.yml',
            '**/*.yaml',
            '**/*.json',
            '**/Dockerfile*',
            '**/.env*',
            '**/nginx.conf',
            '**/docker-compose*.yml',
            '**/requirements.txt',
            '**/package.json',
            '**/terraform/*.tf',
            '**/k8s/*.yaml'
        ]
        
        for pattern in config_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file() and not any(exclude in str(file_path) for exclude in 
                    ['.git', 'node_modules', '__pycache__', '.pytest_cache', 'venv']):
                    self.config_files.append(file_path)
        
        logger.info(f"Found {len(self.config_files)} configuration files")
    
    def scan_docker_configurations(self):
        """Scan Docker configurations for security issues"""
        logger.info("Scanning Docker configurations...")
        
        docker_files = [f for f in self.config_files if 'Dockerfile' in f.name or 'docker-compose' in f.name]
        
        for docker_file in docker_files:
            try:
                content = docker_file.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    line_lower = line.lower().strip()
                    
                    # Check for running as root
                    if line_lower.startswith('user root') or 'user 0' in line_lower:
                        self.log_finding(
                            'docker', 'high', 'Running as Root User',
                            'Container is configured to run as root user',
                            docker_file, i,
                            'Use a non-root user: USER 1000:1000 or create a specific user'
                        )
                    
                    # Check for privileged mode
                    if 'privileged: true' in line_lower or '--privileged' in line_lower:
                        self.log_finding(
                            'docker', 'critical', 'Privileged Container',
                            'Container is running in privileged mode',
                            docker_file, i,
                            'Remove privileged mode unless absolutely necessary'
                        )
                    
                    # Check for exposed sensitive ports
                    sensitive_ports = ['22', '3389', '5432', '3306', '6379', '27017']
                    for port in sensitive_ports:
                        if f':{port}' in line and ('ports:' in line_lower or 'expose' in line_lower):
                            self.log_finding(
                                'docker', 'medium', 'Sensitive Port Exposed',
                                f'Sensitive port {port} is exposed',
                                docker_file, i,
                                'Avoid exposing database and SSH ports directly'
                            )
                    
                    # Check for latest tag usage
                    if ':latest' in line and 'from' in line_lower:
                        self.log_finding(
                            'docker', 'low', 'Using Latest Tag',
                            'Using :latest tag for base image',
                            docker_file, i,
                            'Use specific version tags for reproducible builds'
                        )
                    
                    # Check for ADD instead of COPY
                    if line_lower.startswith('add ') and not line_lower.startswith('add --'):
                        self.log_finding(
                            'docker', 'low', 'Using ADD Instead of COPY',
                            'ADD command has additional features that may be unnecessary',
                            docker_file, i,
                            'Use COPY instead of ADD unless you need URL fetching or tar extraction'
                        )
                    
                    # Check for secrets in environment variables
                    if ('password' in line_lower or 'secret' in line_lower or 'key' in line_lower) and '=' in line:
                        if not ('${' in line or '$(' in line):  # Not using variable substitution
                            self.log_finding(
                                'docker', 'high', 'Hardcoded Secrets',
                                'Potential hardcoded secrets in Docker configuration',
                                docker_file, i,
                                'Use environment variables or Docker secrets'
                            )
            
            except Exception as e:
                logger.warning(f"Error scanning {docker_file}: {e}")
    
    def scan_environment_files(self):
        """Scan environment files for security issues"""
        logger.info("Scanning environment files...")
        
        env_files = [f for f in self.config_files if '.env' in f.name]
        
        for env_file in env_files:
            try:
                content = env_file.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Check for weak passwords
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key_lower = key.lower()
                        
                        if 'password' in key_lower or 'secret' in key_lower or 'key' in key_lower:
                            # Check for weak passwords
                            if len(value) < 8:
                                self.log_finding(
                                    'environment', 'medium', 'Weak Password',
                                    f'Short password/secret in {key}',
                                    env_file, i,
                                    'Use passwords with at least 12 characters'
                                )
                            
                            # Check for common weak passwords
                            weak_passwords = ['password', '123456', 'admin', 'root', 'test']
                            if value.lower() in weak_passwords:
                                self.log_finding(
                                    'environment', 'high', 'Common Weak Password',
                                    f'Common weak password used in {key}',
                                    env_file, i,
                                    'Use strong, unique passwords'
                                )
                        
                        # Check for debug mode in production
                        if key_lower in ['debug', 'development'] and value.lower() in ['true', '1', 'yes']:
                            self.log_finding(
                                'environment', 'medium', 'Debug Mode Enabled',
                                'Debug mode may be enabled in production',
                                env_file, i,
                                'Disable debug mode in production environments'
                            )
                        
                        # Check for insecure database connections
                        if 'database_url' in key_lower or 'db_url' in key_lower:
                            if 'sslmode=disable' in value or 'ssl=false' in value:
                                self.log_finding(
                                    'environment', 'medium', 'Insecure Database Connection',
                                    'Database connection without SSL',
                                    env_file, i,
                                    'Enable SSL for database connections'
                                )
            
            except Exception as e:
                logger.warning(f"Error scanning {env_file}: {e}")
    
    def scan_nginx_configurations(self):
        """Scan Nginx configurations for security issues"""
        logger.info("Scanning Nginx configurations...")
        
        nginx_files = [f for f in self.config_files if 'nginx' in str(f).lower() and f.suffix in ['.conf', '.cfg']]
        
        for nginx_file in nginx_files:
            try:
                content = nginx_file.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                security_headers = {
                    'x-frame-options': False,
                    'x-content-type-options': False,
                    'x-xss-protection': False,
                    'strict-transport-security': False,
                    'content-security-policy': False
                }
                
                ssl_configured = False
                server_tokens_off = False
                
                for i, line in enumerate(lines, 1):
                    line_lower = line.lower().strip()
                    
                    # Check for security headers
                    for header in security_headers:
                        if header in line_lower:
                            security_headers[header] = True
                    
                    # Check for SSL configuration
                    if 'ssl_certificate' in line_lower or 'listen 443' in line_lower:
                        ssl_configured = True
                    
                    # Check for server tokens
                    if 'server_tokens off' in line_lower:
                        server_tokens_off = True
                    
                    # Check for weak SSL protocols
                    if 'ssl_protocols' in line_lower:
                        if 'sslv2' in line_lower or 'sslv3' in line_lower or 'tlsv1' in line_lower:
                            self.log_finding(
                                'nginx', 'high', 'Weak SSL Protocols',
                                'Weak SSL/TLS protocols enabled',
                                nginx_file, i,
                                'Use only TLSv1.2 and TLSv1.3'
                            )
                    
                    # Check for weak ciphers
                    if 'ssl_ciphers' in line_lower and ('rc4' in line_lower or 'des' in line_lower):
                        self.log_finding(
                            'nginx', 'high', 'Weak SSL Ciphers',
                            'Weak SSL ciphers configured',
                            nginx_file, i,
                            'Use strong cipher suites only'
                        )
                
                # Check for missing security headers
                for header, present in security_headers.items():
                    if not present:
                        self.log_finding(
                            'nginx', 'medium', f'Missing Security Header',
                            f'Missing {header} security header',
                            nginx_file, None,
                            f'Add {header} header for better security'
                        )
                
                # Check for server tokens
                if not server_tokens_off:
                    self.log_finding(
                        'nginx', 'low', 'Server Tokens Enabled',
                        'Nginx version information is exposed',
                        nginx_file, None,
                        'Add "server_tokens off;" to hide version information'
                    )
            
            except Exception as e:
                logger.warning(f"Error scanning {nginx_file}: {e}")
    
    def scan_database_configurations(self):
        """Scan database configurations for security issues"""
        logger.info("Scanning database configurations...")
        
        # Check docker-compose files for database configurations
        compose_files = [f for f in self.config_files if 'docker-compose' in f.name]
        
        for compose_file in compose_files:
            try:
                with open(compose_file, 'r') as f:
                    compose_data = yaml.safe_load(f)
                
                if 'services' in compose_data:
                    for service_name, service_config in compose_data['services'].items():
                        if any(db in service_name.lower() for db in ['postgres', 'mysql', 'redis', 'mongo']):
                            # Check for default passwords
                            env_vars = service_config.get('environment', {})
                            if isinstance(env_vars, list):
                                env_dict = {}
                                for env in env_vars:
                                    if '=' in env:
                                        key, value = env.split('=', 1)
                                        env_dict[key] = value
                                env_vars = env_dict
                            
                            for key, value in env_vars.items():
                                if 'password' in key.lower():
                                    weak_passwords = ['password', 'admin', 'root', '123456']
                                    if value in weak_passwords:
                                        self.log_finding(
                                            'database', 'high', 'Default Database Password',
                                            f'Default password used for {service_name}',
                                            compose_file, None,
                                            'Use strong, unique passwords for database services'
                                        )
                            
                            # Check for exposed database ports
                            ports = service_config.get('ports', [])
                            db_ports = {'5432': 'PostgreSQL', '3306': 'MySQL', '6379': 'Redis', '27017': 'MongoDB'}
                            
                            for port_mapping in ports:
                                if isinstance(port_mapping, str) and ':' in port_mapping:
                                    external_port = port_mapping.split(':')[0]
                                    if external_port in db_ports:
                                        self.log_finding(
                                            'database', 'medium', 'Database Port Exposed',
                                            f'{db_ports[external_port]} port {external_port} is exposed',
                                            compose_file, None,
                                            'Avoid exposing database ports directly to the host'
                                        )
            
            except Exception as e:
                logger.warning(f"Error scanning {compose_file}: {e}")
    
    def scan_api_configurations(self):
        """Scan API configurations for security issues"""
        logger.info("Scanning API configurations...")
        
        # Check FastAPI configuration files
        python_files = [f for f in self.config_files if f.suffix == '.py' and 'config' in f.name.lower()]
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    line_lower = line.lower().strip()
                    
                    # Check for CORS configuration
                    if 'allow_origins' in line_lower and '*' in line:
                        self.log_finding(
                            'api', 'medium', 'Permissive CORS Configuration',
                            'CORS allows all origins (*)',
                            py_file, i,
                            'Specify exact origins instead of using wildcard'
                        )
                    
                    # Check for debug mode
                    if 'debug=true' in line_lower or 'debug = true' in line_lower:
                        self.log_finding(
                            'api', 'medium', 'Debug Mode Enabled',
                            'API debug mode is enabled',
                            py_file, i,
                            'Disable debug mode in production'
                        )
                    
                    # Check for hardcoded secrets
                    if any(secret in line_lower for secret in ['secret_key', 'api_key', 'password']) and '=' in line:
                        if not ('os.environ' in line or 'getenv' in line or 'config' in line_lower):
                            self.log_finding(
                                'api', 'high', 'Hardcoded Secret',
                                'Potential hardcoded secret in API configuration',
                                py_file, i,
                                'Use environment variables for secrets'
                            )
            
            except Exception as e:
                logger.warning(f"Error scanning {py_file}: {e}")
    
    def scan_secrets_exposure(self):
        """Scan for exposed secrets and sensitive information"""
        logger.info("Scanning for exposed secrets...")
        
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\'\n\r]{1,}["\']', 'Password'),
            (r'api[_-]?key\s*=\s*["\'][^"\'\n\r]{10,}["\']', 'API Key'),
            (r'secret[_-]?key\s*=\s*["\'][^"\'\n\r]{10,}["\']', 'Secret Key'),
            (r'access[_-]?token\s*=\s*["\'][^"\'\n\r]{10,}["\']', 'Access Token'),
            (r'private[_-]?key\s*=\s*["\'][^"\'\n\r]{10,}["\']', 'Private Key'),
            (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----', 'Private Key'),
            (r'sk_live_[a-zA-Z0-9]{24,}', 'Stripe Secret Key'),
            (r'pk_live_[a-zA-Z0-9]{24,}', 'Stripe Public Key'),
            (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
            (r'[0-9a-f]{32}', 'MD5 Hash (potential secret)')
        ]
        for config_file in self.config_files:
            if config_file.suffix in ['.py', '.js', '.json', '.yml', '.yaml', '.env', '.conf']:
                try:
                    content = config_file.read_text(encoding='utf-8')
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines, 1):
                        for pattern, secret_type in secret_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                # Skip if it's a comment or example
                                if line.strip().startswith('#') or 'example' in line.lower():
                                    continue
                                
                                self.log_finding(
                                    'secrets', 'high', f'Potential {secret_type} Exposure',
                                    f'Potential {secret_type.lower()} found in configuration',
                                    config_file, i,
                                    'Move secrets to environment variables or secure secret management'
                                )
                
                except Exception as e:
                    logger.warning(f"Error scanning {config_file} for secrets: {e}")
    
    def scan_ssl_tls_configurations(self):
        """Scan SSL/TLS configurations"""
        logger.info("Scanning SSL/TLS configurations...")
        
        # Check for SSL certificate configurations
        for config_file in self.config_files:
            try:
                content = config_file.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    line_lower = line.lower().strip()
                    
                    # Check for self-signed certificates
                    if 'self-signed' in line_lower or 'selfsigned' in line_lower:
                        self.log_finding(
                            'ssl_tls', 'medium', 'Self-Signed Certificate',
                            'Self-signed SSL certificate detected',
                            config_file, i,
                            'Use certificates from trusted CA for production'
                        )
                    
                    # Check for weak SSL/TLS versions
                    if 'ssl_protocols' in line_lower or 'tls_protocols' in line_lower:
                        weak_protocols = ['sslv2', 'sslv3', 'tlsv1.0', 'tlsv1.1']
                        for protocol in weak_protocols:
                            if protocol in line_lower:
                                self.log_finding(
                                    'ssl_tls', 'high', 'Weak SSL/TLS Protocol',
                                    f'Weak protocol {protocol} enabled',
                                    config_file, i,
                                    'Use only TLSv1.2 and TLSv1.3'
                                )
                    
                    # Check for insecure SSL settings
                    if 'ssl_verify' in line_lower and ('false' in line_lower or 'off' in line_lower):
                        self.log_finding(
                            'ssl_tls', 'high', 'SSL Verification Disabled',
                            'SSL certificate verification is disabled',
                            config_file, i,
                            'Enable SSL certificate verification'
                        )
            
            except Exception as e:
                logger.warning(f"Error scanning {config_file} for SSL/TLS: {e}")
    
    def scan_monitoring_configurations(self):
        """Scan monitoring and logging configurations"""
        logger.info("Scanning monitoring configurations...")
        
        # Check Prometheus and Grafana configurations
        monitoring_files = [f for f in self.config_files if any(tool in str(f).lower() 
                           for tool in ['prometheus', 'grafana', 'alertmanager'])]
        
        for monitoring_file in monitoring_files:
            try:
                if monitoring_file.suffix in ['.yml', '.yaml']:
                    with open(monitoring_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                    
                    # Check for authentication in monitoring tools
                    if 'grafana' in str(monitoring_file).lower():
                        if isinstance(config_data, dict):
                            # Check for default admin credentials
                            if 'GF_SECURITY_ADMIN_PASSWORD' in str(config_data):
                                admin_pass = str(config_data).lower()
                                if 'admin' in admin_pass or 'password' in admin_pass:
                                    self.log_finding(
                                        'monitoring', 'medium', 'Default Grafana Credentials',
                                        'Default admin credentials may be in use',
                                        monitoring_file, None,
                                        'Change default admin credentials'
                                    )
                    
                    # Check for exposed metrics endpoints
                    if 'prometheus' in str(monitoring_file).lower():
                        if isinstance(config_data, dict) and 'scrape_configs' in config_data:
                            for scrape_config in config_data['scrape_configs']:
                                if 'basic_auth' not in scrape_config and 'bearer_token' not in scrape_config:
                                    self.log_finding(
                                        'monitoring', 'low', 'Unprotected Metrics Endpoint',
                                        'Metrics endpoint without authentication',
                                        monitoring_file, None,
                                        'Add authentication to metrics endpoints'
                                    )
            
            except Exception as e:
                logger.warning(f"Error scanning {monitoring_file}: {e}")
    
    def run_all_scans(self):
        """Run all security configuration scans"""
        logger.info("Starting security configuration scanning...")
        
        self.find_config_files()
        
        scan_methods = [
            self.scan_docker_configurations,
            self.scan_environment_files,
            self.scan_nginx_configurations,
            self.scan_database_configurations,
            self.scan_api_configurations,
            self.scan_secrets_exposure,
            self.scan_ssl_tls_configurations,
            self.scan_monitoring_configurations
        ]
        
        for scan_method in scan_methods:
            try:
                scan_method()
            except Exception as e:
                logger.error(f"Error running {scan_method.__name__}: {e}")
        
        logger.info("Security configuration scanning completed")
    
    def generate_report(self, output_file='security_config_report.json'):
        """Generate security configuration report"""
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        category_counts = {category: len(findings) for category, findings in self.scan_results.items()}
        
        for finding in self.findings:
            severity_counts[finding['severity']] += 1
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'total_files_scanned': len(self.config_files),
            'total_findings': len(self.findings),
            'severity_breakdown': severity_counts,
            'category_breakdown': category_counts,
            'findings': self.findings,
            'scan_results': self.scan_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Security configuration report generated: {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("SECURITY CONFIGURATION SCANNING SUMMARY")
        print("="*60)
        print(f"Files Scanned: {len(self.config_files)}")
        print(f"Total Findings: {len(self.findings)}")
        print(f"\nSeverity Breakdown:")
        for severity, count in severity_counts.items():
            print(f"  {severity.capitalize()}: {count}")
        print(f"\nCategory Breakdown:")
        for category, count in category_counts.items():
            if count > 0:
                print(f"  {category.capitalize()}: {count}")
        print("="*60)
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Security Configuration Scanner')
    parser.add_argument('--project-root', default='.', help='Project root directory')
    parser.add_argument('--output', default='security_config_report.json', help='Output file')
    
    args = parser.parse_args()
    
    scanner = SecurityConfigScanner(args.project_root)
    scanner.run_all_scans()
    scanner.generate_report(args.output)

if __name__ == '__main__':
    main()