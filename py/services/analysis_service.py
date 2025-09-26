"""투자 분석 서비스"""
import logging

import numpy as np


class InvestmentAnalyzer:
    """투자 의사결정 분석 클래스"""
    
    def __init__(self, ticker, current_price, price_predictions, stock_data):
        self.ticker = ticker
        self.current_price = current_price
        self.price_predictions = price_predictions
        self.stock_data = stock_data
    
    def calculate_metrics(self):
        """주요 지표 계산"""
        pred_avg = np.mean(self.price_predictions)
        pred_max = max(self.price_predictions)
        pred_min = min(self.price_predictions)
        
        total_return = (self.price_predictions[-1] - self.current_price) / self.current_price * 100
        avg_daily_return = np.mean([(pred - self.current_price) / self.current_price * 100 
                                   for pred in self.price_predictions])
        volatility = np.std(self.price_predictions)
        upside_prob = sum(1 for pred in self.price_predictions if pred > self.current_price) / len(self.price_predictions) * 100
        
        return {
            'current_price': float(self.current_price),
            'predicted_avg_price': float(pred_avg),
            'predicted_max_price': float(pred_max),
            'predicted_min_price': float(pred_min),
            'expected_total_return': float(total_return),
            'expected_avg_daily_return': float(avg_daily_return),
            'predicted_volatility': float(volatility),
            'upside_probability': float(upside_prob)
        }
    
    def risk_analysis(self):
        """리스크 분석"""
        returns = self.stock_data['close'].pct_change().dropna()
        
        # 예측 수익률 계산
        pred_returns = []
        current_price = self.current_price
        for pred_price in self.price_predictions:
            ret = (pred_price - current_price) / current_price
            pred_returns.append(ret)
            current_price = pred_price

        print('Sharpe Ratio : ', pred_returns, np.mean(pred_returns), np.std(pred_returns))
        sharpe_ratio = (np.mean(pred_returns) / np.std(pred_returns)) if np.std(pred_returns) > 0 else 0
        
        risk_metrics = {
            'historical_volatility_annualized': float(returns.std() * np.sqrt(252) * 100),
            'predicted_volatility': float(np.std(pred_returns) * 100),
            'var_95': float(np.percentile(pred_returns, 5) * 100),
            'max_expected_loss': float(min(pred_returns) * 100),
            'max_expected_gain': float(max(pred_returns) * 100),
            'estimated_sharpe_ratio': float(sharpe_ratio)
        }
        
        return risk_metrics
    
    def generate_recommendation(self):
        """투자 추천 생성"""
        metrics = self.calculate_metrics()
        risk_metrics = self.risk_analysis()
        
        score = 0
        signals = []
        
        # 수익률 평가
        expected_return = metrics['expected_total_return']
        if expected_return > 5:
            signals.append("높은 수익률 기대 (+5% 이상)")
            score += 2
        elif expected_return > 2:
            signals.append("보통 수익률 기대 (2-5%)")
            score += 1
        elif expected_return > -2:
            signals.append("낮은 수익률 기대 (-2% ~ 2%)")
            score -= 1
        else:
            signals.append("매우 낮은 수익률 기대 (-2% 미만)")
            score -= 2
        
        # 상승 확률 평가
        upside_prob = metrics['upside_probability']
        if upside_prob > 70:
            signals.append("높은 상승 확률 (70% 이상)")
            score += 2
        elif upside_prob > 50:
            signals.append("보통 상승 확률 (50-70%)")
            score += 1
        elif upside_prob > 30:
            signals.append("낮은 상승 확률 (30-50%)")
            score -= 1
        else:
            signals.append("매우 낮은 상승 확률 (30% 미만)")
            score -= 2
        
        # VaR 평가
        var_95 = risk_metrics['var_95']
        if var_95 > -3:
            signals.append("낮은 리스크 (VaR -3% 이상)")
            score += 1
        elif var_95 > -5:
            signals.append("보통 리스크 (VaR -3% ~ -5%)")
        else:
            signals.append("높은 리스크 (VaR -5% 미만)")
            score -= 1
        
        # 최종 추천
        if score >= 4:
            recommendation = "강력 매수 추천"
            action = "BUY"
            confidence = "HIGH"
        elif score >= 2:
            recommendation = "매수 추천"
            action = "BUY"
            confidence = "MEDIUM"
        elif score >= 0:
            recommendation = "관망 추천"
            action = "HOLD"
            confidence = "LOW"
        else:
            recommendation = "매도 고려"
            action = "SELL"
            confidence = "MEDIUM"
        
        return {
            'recommendation': recommendation,
            'action': action,
            'confidence': confidence,
            'score': score,
            'max_score': 5,
            'min_score': -5,
            'signals': signals,
            'metrics': metrics,
            'risk_metrics': risk_metrics
        }
