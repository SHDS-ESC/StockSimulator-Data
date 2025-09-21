"""주식 예측 메인 서비스"""

import datetime as dt
from datetime import timedelta
import pandas as pd

from .database_service import DatabaseService
from .prediction_service import StockPredictor
from .analysis_service import InvestmentAnalyzer
from .chart_service import ChartService


class StockService:
    """주식 예측 메인 서비스 클래스"""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.chart_service = ChartService()
    
    def predict_stock(self, ticker, train_days=500, predict_steps=5, today=None, save_image=True):
        """주식 예측 실행 함수"""
        
        if today is None:
            today = dt.date.today()
        
        # 1. 데이터 로드 (이미 report_date가 인덱스로 설정됨)
        stock_data = self.db_service.get_stock_data(ticker)
        
        # 2. 모델 초기화
        predictor = StockPredictor()
        
        # 3. 피처 준비
        prepared_data = predictor.prepare_features(stock_data)
        
        # 4. 날짜 필터링
        today_ts = pd.to_datetime(today)
        today_idx = (abs(stock_data.index - today_ts)).argmin()
        target_idx = today_idx + predict_steps
        if target_idx < len(stock_data.index):
            end_date = stock_data.index[target_idx]
        else:
            missing_days = int(target_idx - len(stock_data.index) + 1)
            print(f"Missing days: {missing_days}")
            end_date = stock_data.index[-1] + timedelta(days=missing_days)
        start_date = today - timedelta(days=train_days)
        
        date_mask = (prepared_data.index >= pd.to_datetime(start_date)) & \
                    (prepared_data.index <= pd.to_datetime(end_date))
        filtered_data = prepared_data[date_mask]

        # 5. 훈련 데이터 선택
        split_idx = int(len(filtered_data) * 1)
        train_data = filtered_data.iloc[:split_idx-predict_steps]
        
        # 6. 피처와 타겟 분리
        feature_cols = predictor.get_feature_columns(train_data)
        X_train = train_data[feature_cols].copy()
        y_train = train_data['target'].copy()
        
        # 7. 모델 학습
        predictor.train(X_train, y_train)
        
        # 8. 미래 예측
        last_features = X_train.iloc[[-1]]
        return_predictions = predictor.predict_next_returns(last_features, predict_steps)
        
        # 9. 수익률을 가격으로 변환
        last_price = train_data['close'].iloc[-1]
        price_predictions = predictor.convert_returns_to_prices(last_price, return_predictions)
        
        # 10. 예측 날짜 계산 (filtered_data의 close index 패턴 추종)
        pred_dates = self._calculate_prediction_dates(today, predict_steps, filtered_data)
        
        # 11. 투자 분석 실행
        analyzer = InvestmentAnalyzer(
            ticker=ticker,
            current_price=last_price,
            price_predictions=price_predictions,
            stock_data=stock_data
        )
        investment_analysis = analyzer.generate_recommendation()
        
        # 12. 차트 생성
        chart_data = None
        if save_image:
            chart_data = self.chart_service.create_prediction_charts(
                ticker, today, price_predictions, filtered_data['close'], pred_dates
            )
        
        result = {
            'ticker': ticker,
            'base_date': today,
            'last_price': float(last_price),
            'return_predictions': return_predictions,
            'price_predictions': price_predictions,
            'prediction_dates': [date.date() if hasattr(date, 'date') else date for date in pred_dates],  # date 객체로 변환
            'train_data_count': len(X_train),
            'feature_count': len(feature_cols),
            'investment_analysis': investment_analysis
        }
        
        # 차트 데이터 추가 (있는 경우)
        if chart_data:
            result.update(chart_data)
        
        return result
    
    def _calculate_prediction_dates(self, today, predict_steps, filtered_data):
        """예측 날짜 계산"""
        pred_dates = []
        today_ts = pd.Timestamp(today)
        close_data = filtered_data['close']
        
        # close_data에서 today 이후의 날짜들을 찾기
        future_dates_in_data = close_data.index[close_data.index > today_ts]
        
        if len(future_dates_in_data) >= predict_steps:
            # 충분한 미래 날짜가 데이터에 있는 경우 (과거 예측)
            pred_dates = future_dates_in_data[:predict_steps].tolist()
            print(f"🔍 과거 예측 모드: close_data의 실제 날짜 사용")
        else:
            # 미래 예측의 경우: 기존 패턴 추종 후 단순 증가
            if len(future_dates_in_data) > 0:
                # 일부는 실제 날짜, 나머지는 추정
                pred_dates.extend(future_dates_in_data.tolist())
                last_date = future_dates_in_data[-1]
                remaining_days = predict_steps - len(future_dates_in_data)
            else:
                # 완전 미래 예측: today부터 시작
                last_date = today_ts
                remaining_days = predict_steps
            
            # 나머지 날짜들을 영업일 기준으로 생성
            current_date = last_date
            for i in range(remaining_days):
                current_date += timedelta(days=1)
                # 주말 건너뛰기 (간단한 영업일 계산)
                while current_date.weekday() >= 5:  # 5=토요일, 6=일요일
                    current_date += timedelta(days=1)
                pred_dates.append(current_date)
            
            print(f"🔍 미래 예측 모드: 실제 {len(future_dates_in_data)}개 + 추정 {remaining_days}개")
        
        return pred_dates
