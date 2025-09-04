import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

#from backend.utils.config import YAHOO_FINANCE_CONFIG

logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    """
    Comprehensive fundamental analysis for Indian stocks with financial ratios,
    valuation metrics, and financial health indicators.
    """
    
    def __init__(self):
        """Initialize the fundamental analyzer."""
        self.cache = {}
        self.cache_duration = timedelta(hours=6)  # Cache for 6 hours
        
    def get_stock_info(self, ticker: str) -> Dict:
        """
        Get comprehensive stock information including fundamental metrics.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'RELIANCE.NS')
            
        Returns:
            Dictionary containing fundamental analysis data
        """
        try:
            # Check cache first
            cache_key = f"info_{ticker}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']
            
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get financial statements
            financials = self._get_financial_statements(stock)
            
            # Calculate fundamental metrics
            fundamental_data = self._calculate_fundamental_metrics(info, financials)
            
            # Cache the result
            self.cache[cache_key] = {
                'data': fundamental_data,
                'timestamp': datetime.now()
            }
            
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {ticker}: {e}")
            return {}
    
    def _get_financial_statements(self, stock) -> Dict:
        """
        Get financial statements from Yahoo Finance.
        
        Args:
            stock: yfinance Ticker object
            
        Returns:
            Dictionary containing financial statements
        """
        try:
            financials = {
                'income_statement': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cashflow,
                'quarterly_financials': stock.quarterly_financials,
                'quarterly_balance_sheet': stock.quarterly_balance_sheet,
                'quarterly_cashflow': stock.quarterly_cashflow
            }
            return financials
        except Exception as e:
            logger.warning(f"Error fetching financial statements: {e}")
            return {}
    
    def _calculate_fundamental_metrics(self, info: Dict, financials: Dict) -> Dict:
        """
        Calculate comprehensive fundamental analysis metrics.
        
        Args:
            info: Stock info from Yahoo Finance
            financials: Financial statements
            
        Returns:
            Dictionary containing calculated metrics
        """
        metrics = {
            'basic_info': self._extract_basic_info(info),
            'valuation_ratios': self._calculate_valuation_ratios(info),
            'profitability_ratios': self._calculate_profitability_ratios(info, financials),
            'liquidity_ratios': self._calculate_liquidity_ratios(info, financials),
            'leverage_ratios': self._calculate_leverage_ratios(info, financials),
            'efficiency_ratios': self._calculate_efficiency_ratios(info, financials),
            'growth_metrics': self._calculate_growth_metrics(info, financials),
            'dividend_metrics': self._calculate_dividend_metrics(info),
            'financial_health': self._assess_financial_health(info, financials)
        }
        
        return metrics
    
    def _extract_basic_info(self, info: Dict) -> Dict:
        """
        Extract basic company information.
        
        Args:
            info: Stock info dictionary
            
        Returns:
            Dictionary with basic company info
        """
        return {
            'company_name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'enterprise_value': info.get('enterpriseValue', 0),
            'employees': info.get('fullTimeEmployees', 0),
            'country': info.get('country', 'N/A'),
            'currency': info.get('currency', 'INR'),
            'exchange': info.get('exchange', 'NSI'),
            'website': info.get('website', 'N/A')
        }
    
    def _calculate_valuation_ratios(self, info: Dict) -> Dict:
        """
        Calculate valuation ratios.
        
        Args:
            info: Stock info dictionary
            
        Returns:
            Dictionary with valuation ratios
        """
        return {
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'price_to_book': info.get('priceToBook', 0),
            'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
            'enterprise_to_revenue': info.get('enterpriseToRevenue', 0),
            'enterprise_to_ebitda': info.get('enterpriseToEbitda', 0),
            'price_to_cash_flow': self._safe_divide(info.get('marketCap', 0), 
                                                   info.get('operatingCashflow', 1)),
            'ev_to_sales': self._safe_divide(info.get('enterpriseValue', 0), 
                                           info.get('totalRevenue', 1))
        }
    
    def _calculate_profitability_ratios(self, info: Dict, financials: Dict) -> Dict:
        """
        Calculate profitability ratios.
        
        Args:
            info: Stock info dictionary
            financials: Financial statements
            
        Returns:
            Dictionary with profitability ratios
        """
        return {
            'gross_margin': info.get('grossMargins', 0),
            'operating_margin': info.get('operatingMargins', 0),
            'profit_margin': info.get('profitMargins', 0),
            'ebitda_margin': info.get('ebitdaMargins', 0),
            'roe': info.get('returnOnEquity', 0),
            'roa': info.get('returnOnAssets', 0),
            'roic': self._calculate_roic(info),
            'net_income_margin': self._safe_divide(info.get('netIncomeToCommon', 0), 
                                                 info.get('totalRevenue', 1)),
            'ebit_margin': self._safe_divide(info.get('ebit', 0), 
                                           info.get('totalRevenue', 1))
        }
    
    def _calculate_liquidity_ratios(self, info: Dict, financials: Dict) -> Dict:
        """
        Calculate liquidity ratios.
        
        Args:
            info: Stock info dictionary
            financials: Financial statements
            
        Returns:
            Dictionary with liquidity ratios
        """
        current_ratio = self._safe_divide(info.get('totalCurrentAssets', 0), 
                                        info.get('totalCurrentLiabilities', 1))
        
        quick_ratio = self._safe_divide(
            info.get('totalCurrentAssets', 0) - info.get('inventory', 0),
            info.get('totalCurrentLiabilities', 1)
        )
        
        return {
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'cash_ratio': self._safe_divide(info.get('totalCash', 0), 
                                          info.get('totalCurrentLiabilities', 1)),
            'operating_cash_flow_ratio': self._safe_divide(info.get('operatingCashflow', 0), 
                                                         info.get('totalCurrentLiabilities', 1)),
            'working_capital': info.get('totalCurrentAssets', 0) - info.get('totalCurrentLiabilities', 0)
        }
    
    def _calculate_leverage_ratios(self, info: Dict, financials: Dict) -> Dict:
        """
        Calculate leverage/debt ratios.
        
        Args:
            info: Stock info dictionary
            financials: Financial statements
            
        Returns:
            Dictionary with leverage ratios
        """
        total_debt = info.get('totalDebt', 0)
        total_equity = info.get('totalStockholderEquity', 0)
        total_assets = info.get('totalAssets', 0)
        
        return {
            'debt_to_equity': self._safe_divide(total_debt, total_equity),
            'debt_to_assets': self._safe_divide(total_debt, total_assets),
            'equity_ratio': self._safe_divide(total_equity, total_assets),
            'debt_to_capital': self._safe_divide(total_debt, total_debt + total_equity),
            'interest_coverage': self._safe_divide(info.get('ebit', 0), 
                                                 info.get('interestExpense', 1)),
            'debt_service_coverage': self._safe_divide(info.get('operatingCashflow', 0), 
                                                     total_debt),
            'financial_leverage': self._safe_divide(total_assets, total_equity)
        }
    
    def _calculate_efficiency_ratios(self, info: Dict, financials: Dict) -> Dict:
        """
        Calculate efficiency/activity ratios.
        
        Args:
            info: Stock info dictionary
            financials: Financial statements
            
        Returns:
            Dictionary with efficiency ratios
        """
        return {
            'asset_turnover': self._safe_divide(info.get('totalRevenue', 0), 
                                              info.get('totalAssets', 1)),
            'inventory_turnover': self._safe_divide(info.get('costOfRevenue', 0), 
                                                  info.get('inventory', 1)),
            'receivables_turnover': self._safe_divide(info.get('totalRevenue', 0), 
                                                    info.get('netReceivables', 1)),
            'payables_turnover': self._safe_divide(info.get('costOfRevenue', 0), 
                                                 info.get('accountsPayable', 1)),
            'equity_turnover': self._safe_divide(info.get('totalRevenue', 0), 
                                               info.get('totalStockholderEquity', 1)),
            'working_capital_turnover': self._safe_divide(
                info.get('totalRevenue', 0),
                info.get('totalCurrentAssets', 0) - info.get('totalCurrentLiabilities', 0)
            )
        }
    
    def _calculate_growth_metrics(self, info: Dict, financials: Dict) -> Dict:
        """
        Calculate growth metrics.
        
        Args:
            info: Stock info dictionary
            financials: Financial statements
            
        Returns:
            Dictionary with growth metrics
        """
        return {
            'revenue_growth': info.get('revenueGrowth', 0),
            'earnings_growth': info.get('earningsGrowth', 0),
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', 0),
            'revenue_quarterly_growth': info.get('revenueQuarterlyGrowth', 0),
            'book_value_growth': self._calculate_book_value_growth(financials),
            'dividend_growth': self._calculate_dividend_growth(info),
            'eps_growth': self._calculate_eps_growth(financials)
        }
    
    def _calculate_dividend_metrics(self, info: Dict) -> Dict:
        """
        Calculate dividend-related metrics.
        
        Args:
            info: Stock info dictionary
            
        Returns:
            Dictionary with dividend metrics
        """
        return {
            'dividend_yield': info.get('dividendYield', 0),
            'dividend_rate': info.get('dividendRate', 0),
            'payout_ratio': info.get('payoutRatio', 0),
            'ex_dividend_date': info.get('exDividendDate', None),
            'dividend_date': info.get('dividendDate', None),
            'five_year_avg_dividend_yield': info.get('fiveYearAvgDividendYield', 0)
        }
    
    def _assess_financial_health(self, info: Dict, financials: Dict) -> Dict:
        """
        Assess overall financial health with scoring.
        
        Args:
            info: Stock info dictionary
            financials: Financial statements
            
        Returns:
            Dictionary with financial health assessment
        """
        # Calculate individual scores (0-100)
        profitability_score = self._score_profitability(info)
        liquidity_score = self._score_liquidity(info)
        leverage_score = self._score_leverage(info)
        efficiency_score = self._score_efficiency(info)
        growth_score = self._score_growth(info)
        
        # Overall financial health score (weighted average)
        overall_score = (
            profitability_score * 0.25 +
            liquidity_score * 0.20 +
            leverage_score * 0.20 +
            efficiency_score * 0.15 +
            growth_score * 0.20
        )
        
        # Determine rating
        if overall_score >= 80:
            rating = 'Excellent'
        elif overall_score >= 65:
            rating = 'Good'
        elif overall_score >= 50:
            rating = 'Average'
        elif overall_score >= 35:
            rating = 'Below Average'
        else:
            rating = 'Poor'
        
        return {
            'overall_score': round(overall_score, 2),
            'rating': rating,
            'profitability_score': round(profitability_score, 2),
            'liquidity_score': round(liquidity_score, 2),
            'leverage_score': round(leverage_score, 2),
            'efficiency_score': round(efficiency_score, 2),
            'growth_score': round(growth_score, 2),
            'strengths': self._identify_strengths(info),
            'weaknesses': self._identify_weaknesses(info),
            'recommendations': self._generate_recommendations(info)
        }
    
    def _calculate_roic(self, info: Dict) -> float:
        """
        Calculate Return on Invested Capital.
        
        Args:
            info: Stock info dictionary
            
        Returns:
            ROIC value
        """
        nopat = info.get('ebit', 0) * (1 - 0.25)  # Assuming 25% tax rate
        invested_capital = info.get('totalStockholderEquity', 0) + info.get('totalDebt', 0)
        return self._safe_divide(nopat, invested_capital)
    
    def _calculate_book_value_growth(self, financials: Dict) -> float:
        """
        Calculate book value growth rate.
        
        Args:
            financials: Financial statements
            
        Returns:
            Book value growth rate
        """
        try:
            balance_sheet = financials.get('balance_sheet', pd.DataFrame())
            if balance_sheet.empty or len(balance_sheet.columns) < 2:
                return 0.0
            
            # Get book value for last two years
            equity_values = balance_sheet.loc['Total Stockholder Equity'] if 'Total Stockholder Equity' in balance_sheet.index else None
            if equity_values is not None and len(equity_values) >= 2:
                current_bv = equity_values.iloc[0]
                previous_bv = equity_values.iloc[1]
                return self._safe_divide(current_bv - previous_bv, previous_bv)
        except Exception:
            pass
        return 0.0
    
    def _calculate_dividend_growth(self, info: Dict) -> float:
        """
        Calculate dividend growth rate.
        
        Args:
            info: Stock info dictionary
            
        Returns:
            Dividend growth rate
        """
        # This would require historical dividend data
        # For now, return 0 as placeholder
        return 0.0
    
    def _calculate_eps_growth(self, financials: Dict) -> float:
        """
        Calculate EPS growth rate.
        
        Args:
            financials: Financial statements
            
        Returns:
            EPS growth rate
        """
        try:
            income_statement = financials.get('income_statement', pd.DataFrame())
            if income_statement.empty or len(income_statement.columns) < 2:
                return 0.0
            
            # Get net income for last two years
            net_income_values = income_statement.loc['Net Income'] if 'Net Income' in income_statement.index else None
            if net_income_values is not None and len(net_income_values) >= 2:
                current_ni = net_income_values.iloc[0]
                previous_ni = net_income_values.iloc[1]
                return self._safe_divide(current_ni - previous_ni, abs(previous_ni))
        except Exception:
            pass
        return 0.0
    
    def _score_profitability(self, info: Dict) -> float:
        """
        Score profitability metrics (0-100).
        
        Args:
            info: Stock info dictionary
            
        Returns:
            Profitability score
        """
        roe = info.get('returnOnEquity', 0) or 0
        roa = info.get('returnOnAssets', 0) or 0
        profit_margin = info.get('profitMargins', 0) or 0
        
        # Convert to percentages and score
        roe_score = min(roe * 100 / 20, 100) if roe > 0 else 0  # 20% ROE = 100 points
        roa_score = min(roa * 100 / 10, 100) if roa > 0 else 0   # 10% ROA = 100 points
        margin_score = min(profit_margin * 100 / 15, 100) if profit_margin > 0 else 0  # 15% margin = 100 points
        
        return (roe_score + roa_score + margin_score) / 3
    
    def _score_liquidity(self, info: Dict) -> float:
        """
        Score liquidity metrics (0-100).
        
        Args:
            info: Stock info dictionary
            
        Returns:
            Liquidity score
        """
        current_ratio = self._safe_divide(info.get('totalCurrentAssets', 0), 
                                        info.get('totalCurrentLiabilities', 1))
        
        # Score current ratio (2.0 = 100 points, 1.0 = 50 points)
        if current_ratio >= 2.0:
            score = 100
        elif current_ratio >= 1.0:
            score = 50 + (current_ratio - 1.0) * 50
        else:
            score = current_ratio * 50
        
        return min(score, 100)
    
    def _score_leverage(self, info: Dict) -> float:
        """
        Score leverage metrics (0-100). Lower debt = higher score.
        
        Args:
            info: Stock info dictionary
            
        Returns:
            Leverage score
        """
        debt_to_equity = self._safe_divide(info.get('totalDebt', 0), 
                                         info.get('totalStockholderEquity', 1))
        
        # Score debt-to-equity (0 = 100 points, 1.0 = 50 points, 2.0+ = 0 points)
        if debt_to_equity <= 0.5:
            score = 100 - debt_to_equity * 50
        elif debt_to_equity <= 2.0:
            score = 75 - (debt_to_equity - 0.5) * 50
        else:
            score = 0
        
        return max(score, 0)
    
    def _score_efficiency(self, info: Dict) -> float:
        """
        Score efficiency metrics (0-100).
        
        Args:
            info: Stock info dictionary
            
        Returns:
            Efficiency score
        """
        asset_turnover = self._safe_divide(info.get('totalRevenue', 0), 
                                         info.get('totalAssets', 1))
        
        # Score asset turnover (1.0 = 100 points)
        score = min(asset_turnover * 100, 100)
        return score
    
    def _score_growth(self, info: Dict) -> float:
        """
        Score growth metrics (0-100).
        
        Args:
            info: Stock info dictionary
            
        Returns:
            Growth score
        """
        revenue_growth = info.get('revenueGrowth', 0) or 0
        earnings_growth = info.get('earningsGrowth', 0) or 0
        
        # Convert to percentages and score (20% growth = 100 points)
        rev_score = min(max(revenue_growth * 100 / 0.2, 0), 100) if revenue_growth else 0
        earn_score = min(max(earnings_growth * 100 / 0.2, 0), 100) if earnings_growth else 0
        
        return (rev_score + earn_score) / 2
    
    def _identify_strengths(self, info: Dict) -> List[str]:
        """
        Identify company strengths based on metrics.
        
        Args:
            info: Stock info dictionary
            
        Returns:
            List of identified strengths
        """
        strengths = []
        
        if (info.get('returnOnEquity', 0) or 0) > 0.15:
            strengths.append('High Return on Equity')
        if (info.get('profitMargins', 0) or 0) > 0.10:
            strengths.append('Strong Profit Margins')
        if (info.get('revenueGrowth', 0) or 0) > 0.10:
            strengths.append('Strong Revenue Growth')
        if self._safe_divide(info.get('totalCurrentAssets', 0), info.get('totalCurrentLiabilities', 1)) > 2.0:
            strengths.append('Strong Liquidity Position')
        if self._safe_divide(info.get('totalDebt', 0), info.get('totalStockholderEquity', 1)) < 0.5:
            strengths.append('Low Debt Levels')
        
        return strengths
    
    def _identify_weaknesses(self, info: Dict) -> List[str]:
        """
        Identify company weaknesses based on metrics.
        
        Args:
            info: Stock info dictionary
            
        Returns:
            List of identified weaknesses
        """
        weaknesses = []
        
        if (info.get('returnOnEquity', 0) or 0) < 0.05:
            weaknesses.append('Low Return on Equity')
        if (info.get('profitMargins', 0) or 0) < 0.02:
            weaknesses.append('Weak Profit Margins')
        if (info.get('revenueGrowth', 0) or 0) < 0:
            weaknesses.append('Declining Revenue')
        if self._safe_divide(info.get('totalCurrentAssets', 0), info.get('totalCurrentLiabilities', 1)) < 1.0:
            weaknesses.append('Poor Liquidity Position')
        if self._safe_divide(info.get('totalDebt', 0), info.get('totalStockholderEquity', 1)) > 2.0:
            weaknesses.append('High Debt Levels')
        
        return weaknesses
    
    def _generate_recommendations(self, info: Dict) -> List[str]:
        """
        Generate investment recommendations based on analysis.
        
        Args:
            info: Stock info dictionary
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        pe_ratio = info.get('trailingPE', 0) or 0
        if pe_ratio > 30:
            recommendations.append('Stock appears overvalued based on P/E ratio')
        elif pe_ratio < 10 and pe_ratio > 0:
            recommendations.append('Stock appears undervalued based on P/E ratio')
        
        debt_to_equity = self._safe_divide(info.get('totalDebt', 0), info.get('totalStockholderEquity', 1))
        if debt_to_equity > 1.5:
            recommendations.append('Monitor debt levels closely')
        
        roe = info.get('returnOnEquity', 0) or 0
        if roe > 0.15:
            recommendations.append('Strong profitability metrics support investment')
        
        return recommendations
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """
        Safely divide two numbers, returning 0 if denominator is 0.
        
        Args:
            numerator: Numerator value
            denominator: Denominator value
            
        Returns:
            Division result or 0 if denominator is 0
        """
        try:
            if denominator == 0 or denominator is None:
                return 0.0
            return float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if cache is valid, False otherwise
        """
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return datetime.now() - cache_time < self.cache_duration
    
    def get_fundamental_features(self, ticker: str) -> Dict[str, float]:
        """
        Get fundamental features for ML model training.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary of fundamental features for ML models
        """
        try:
            fundamental_data = self.get_stock_info(ticker)
            
            if not fundamental_data:
                return {}
            
            # Extract key features for ML model
            features = {}
            
            # Valuation ratios
            valuation = fundamental_data.get('valuation_ratios', {})
            features.update({
                'pe_ratio': valuation.get('pe_ratio', 0),
                'price_to_book': valuation.get('price_to_book', 0),
                'price_to_sales': valuation.get('price_to_sales', 0),
                'enterprise_to_ebitda': valuation.get('enterprise_to_ebitda', 0)
            })
            
            # Profitability ratios
            profitability = fundamental_data.get('profitability_ratios', {})
            features.update({
                'roe': profitability.get('roe', 0),
                'roa': profitability.get('roa', 0),
                'profit_margin': profitability.get('profit_margin', 0),
                'operating_margin': profitability.get('operating_margin', 0)
            })
            
            # Liquidity ratios
            liquidity = fundamental_data.get('liquidity_ratios', {})
            features.update({
                'current_ratio': liquidity.get('current_ratio', 0),
                'quick_ratio': liquidity.get('quick_ratio', 0)
            })
            
            # Leverage ratios
            leverage = fundamental_data.get('leverage_ratios', {})
            features.update({
                'debt_to_equity': leverage.get('debt_to_equity', 0),
                'debt_to_assets': leverage.get('debt_to_assets', 0)
            })
            
            # Growth metrics
            growth = fundamental_data.get('growth_metrics', {})
            features.update({
                'revenue_growth': growth.get('revenue_growth', 0),
                'earnings_growth': growth.get('earnings_growth', 0)
            })
            
            # Financial health scores
            health = fundamental_data.get('financial_health', {})
            features.update({
                'financial_health_score': health.get('overall_score', 0),
                'profitability_score': health.get('profitability_score', 0),
                'liquidity_score': health.get('liquidity_score', 0)
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting fundamental features for {ticker}: {e}")
            return {}
