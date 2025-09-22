"""주식 예측 메인 서비스"""

import datetime as dt
from datetime import timedelta

import lightgbm
import pandas as pd
import numpy as np
from websockets.legacy.framing import prepare_data

from .database_service import DatabaseService
from .prediction_service import StockPredictor
from .analysis_service import InvestmentAnalyzer
from .chart_service import ChartService


class StockService:
    """주식 예측 메인 서비스 클래스"""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.chart_service = ChartService()
    
    def predict_stock(self, ticker, train_days=200, predict_steps=5, today=None, save_image=True,
                     window_size=25, step_size=3, max_training_days=500):
        """주식 예측 실행 함수 - 슬라이딩 윈도우 방식"""
        
        if today is None:
            today = dt.date.today()
        
        # 파라미터 검증
        # if train_days > max_training_days:
        #     raise ValueError(f"train_days({train_days})는 max_training_days({max_training_days})보다 작아야 합니다")
        
        # 학습 횟수 계산
        training_count = (train_days - window_size) // step_size + 1
        # if training_count > 100:
        #     raise ValueError(f"학습 횟수({training_count})가 너무 많습니다. step_size를 늘리거나 train_days를 줄이세요")
        
        print(f"슬라이딩 윈도우 설정: window_size={window_size}, step_size={step_size}, train_days={train_days}")
        print(f"예상 학습 횟수: {training_count}")
        
        # 1. 데이터 로드
        stock_data = self.db_service.get_stock_data(ticker)
        
        # 2. 모델 초기화
        lgbm = lightgbm.LGBMRegressor(n_estimators=500,
                                      max_depth=-1,
                                      learning_rate=0.01,
                                      n_jobs=-1,
                                      verbose=-1,
                                      random_state=42)

        predictor = StockPredictor(model=lgbm)

        # # 3. 사용 데이터 범위 산정 (메모리/시각화 최적화)
        today_dt = pd.to_datetime(today)
        today_idx = (abs(stock_data.index - today_dt)).argmin()
        
        start_idx = max(0, today_idx - train_days)
        end_idx = today_idx + predict_steps
        
        # 인덱스의 날짜 문자열 보관
        idx_max = len(stock_data.index) - 1
        today_str = stock_data.index[today_idx].strftime('%Y-%m-%d')
        start_str = stock_data.index[start_idx].strftime('%Y-%m-%d')
        end_str = stock_data.index[min(end_idx, idx_max)].strftime('%Y-%m-%d')
        
        print(f"start_idx={start_idx}({start_str}), end_idx={end_idx}({end_str}), today_idx={today_idx}({today_str})")
        
        # 4. 피처 준비: (start-50) ~ end 구간만 사용해 특성 생성 후
        # 생성된 데이터에서 start_str ~ end_str 범위만 재슬라이싱하여 사용
        buffer_start_idx = max(0, start_idx - 50)
        buffer_slice = stock_data.iloc[buffer_start_idx:end_idx+1]
        print(f"buffer slice: {buffer_slice.index[0]} ~ {buffer_slice.index[-1]} (len={len(buffer_slice)})")
        
        prepared_data = predictor.prepare_features(buffer_slice)
        stock_data = prepared_data.loc[start_str:end_str]
        
        print(f"prepared sliced len: {len(stock_data)}")
        print(f"stock_data.head(): {stock_data.head()}")
        print(f"stock_data.tail(): {stock_data.tail()}")

        # today_dt = pd.to_datetime(today)
        # 입력된 날짜보다 작거나 같은 날짜 중 마지막 날짜
        # today_idx = stock_data.index[stock_data.index <= today_dt].max()

        # 오늘까지의 데이터 -> 학습용
        train_data = stock_data.iloc[ : today_idx]
        print(f"훈련 데이터: {len(train_data)}일 ({train_data.index[0]} ~ {train_data.index[-1]})")

        # 내일부터 ~ 예측일까지의 데이터 -> 검증용 (미래 시제는 검증 못함)
        valid_data = stock_data.iloc[today_idx : ]
        # print(f"검증 데이터: {predict_steps}일 ({valid_data.index[0]} ~ {valid_data.index[-1]})")
        print(f"valid_data: {valid_data}")

        # 5. 슬라이딩 윈도우 예측 실행
        price_predictions, metrics_summary = self._sliding_window_predict(
            train_data, predictor, predict_steps, 
            window_size, step_size
        )
        print(f"price_predictions: {price_predictions}")
        
        # 10. 예측 날짜 계산 (가용 구간 기준)
        pred_dates = self._calculate_prediction_dates(today, predict_steps, train_data)
        # pred_dates = valid_data.index
        print(f"pred_dates={pred_dates}")
        
        # 11. 투자 분석 실행
        # 현재 가격은 train_data의 마지막 close 가격 사용
        current_price = train_data['close'].iloc[-1]
        
        analyzer = InvestmentAnalyzer(
            ticker=ticker,
            current_price=current_price,
            price_predictions=price_predictions,
            stock_data=stock_data
        )
        investment_analysis = analyzer.generate_recommendation()
        
        # 12. 차트 생성
        chart_data = None
        if save_image:
            model_name = type(predictor.model).__name__ if hasattr(predictor, 'model') else 'Model'
            dataset_label = f"{ticker}-{stock_data.index[0].strftime('%Y-%m-%d')}~{stock_data.index[-1].strftime('%Y-%m-%d')}"
            created_at = dt.date.today().strftime('%Y-%m-%d')
            metadata = {
                'model_name': model_name,
                'metrics': metrics_summary or {},
                'dataset': dataset_label,
                'created_at': created_at,
                'params': {
                    'window_size': window_size,
                    'step_size': step_size,
                    'train_days': train_days,
                    'predict_steps': predict_steps,
                    'feature_count': len(predictor.get_feature_columns(train_data))
                }
            }
            chart_data = self.chart_service.create_prediction_charts(
                ticker, today, price_predictions, pd.concat([train_data, valid_data])['close'], pred_dates,
                metadata=metadata
            )
        
        result = {
            'ticker': ticker,
            'base_date': today,
            'last_price': float(current_price),
            'return_predictions': [],  # 슬라이딩 윈도우에서는 수익률 예측을 별도로 계산하지 않음
            'price_predictions': price_predictions,
            'prediction_dates': [date.date() if hasattr(date, 'date') else date for date in pred_dates],  # date 객체로 변환
            'train_data_count': len(train_data),  # 사용 데이터 수
            'feature_count': len(predictor.get_feature_columns(train_data)),  # 피처 수
            'investment_analysis': investment_analysis
        }
        
        # 차트 데이터 추가 (있는 경우)
        if chart_data:
            result.update(chart_data)
        
        return result
    
    def _sliding_window_predict(self, train_data, predictor, predict_steps, 
                               window_size, step_size):
        """성능 검증이 포함된 슬라이딩 윈도우 예측 실행 (사전 산정된 가용 구간 사용)

        Returns:
            tuple[list[float], dict]: (final_predictions, metrics_summary)
        """
        
        # 슬라이딩 윈도우로 예측 수행 (성능 검증 포함)
        all_predictions = []
        validation_scores = []
        performance_metrics = []

        for i in range(0, len(train_data) - window_size + 1, step_size):
            # 현재 윈도우 데이터 선택
            window_data = train_data.iloc[i:i + window_size]
            if len(window_data) < window_size:
                break

            # 학습/검증 분할: window_size=25라면 24일 학습, 1일 검증
            train_df = window_data.iloc[:-1]
            val_df = window_data.iloc[-1:]

            # 피처와 타겟 분리 (학습)
            feature_cols = predictor.get_feature_columns(train_df)
            X_train = train_df[feature_cols].copy()
            y_train = train_df['target'].copy()

            # 모델 학습
            predictor.train(X_train, y_train)

            # 1-step 검증: 마지막 학습 피처로 1일 예측
            last_features = X_train.iloc[[-1]]
            val_pred_1 = predictor.predict_next_returns(last_features, steps=1)
            y_true_1 = val_df['target'].values  # 길이 1

            # 검증 성능 (스칼라 비교)
            mae_score = self._calculate_mae(y_true_1, val_pred_1)
            rmse_score = self._calculate_rmse(y_true_1, val_pred_1)
            direction_acc = 1.0 if np.sign(val_pred_1[0]) == np.sign(y_true_1[0]) else 0.0

            validation_scores.append(mae_score)
            performance_metrics.append({
                'mae': mae_score,
                'rmse': rmse_score,
                'direction_accuracy': direction_acc
            })

            # 실제 예측: recursive로 predict_steps일 예측
            return_predictions = predictor.predict_next_returns(last_features, steps=predict_steps)

            # 수익률을 가격으로 변환
            last_price = train_df['close'].iloc[-1]
            price_predictions = predictor.convert_returns_to_prices(last_price, return_predictions)

            all_predictions.append(price_predictions)

            print(f"{train_df.index[0]} - {train_df.index[-1]}", sep=' / ')
            print(f"윈도우 {i//step_size + 1}: MAE={mae_score:.4f}, RMSE={rmse_score:.4f}, 방향정확도={direction_acc:.3f}")
            print(f"val_pred_1: {val_pred_1}, return_predictions: {return_predictions}, y_true_1: {y_true_1}")
        
        # 예측 결과 검증
        if not all_predictions:
            raise ValueError("예측할 수 있는 충분한 데이터가 없습니다")
        
        # 성능 기반 가중치 계산
        weights = self._calculate_performance_weights(validation_scores)
        
        # 가중 앙상블로 최종 예측
        final_predictions = self._weighted_ensemble(all_predictions, weights)
        
        # 성능 요약 출력
        avg_mae = np.mean([m['mae'] for m in performance_metrics])
        avg_rmse = np.mean([m['rmse'] for m in performance_metrics])
        avg_direction_acc = np.mean([m['direction_accuracy'] for m in performance_metrics])
        
        print(f"총 {len(all_predictions)}개 윈도우로 예측 완료")
        print(f"평균 성능: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, 방향정확도={avg_direction_acc:.3f}")

        metrics_summary = {
            'mae': float(avg_mae) if not np.isnan(avg_mae) else None,
            'rmse': float(avg_rmse) if not np.isnan(avg_rmse) else None,
            'direction_accuracy': float(avg_direction_acc) if not np.isnan(avg_direction_acc) else None,
            'num_windows': len(performance_metrics)
        }
        
        return final_predictions, metrics_summary
    
    def _calculate_mae(self, y_true, y_pred):
        """Mean Absolute Error 계산"""
        return np.mean(np.abs(y_true - y_pred))
    
    def _calculate_rmse(self, y_true, y_pred):
        """Root Mean Square Error 계산"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def _calculate_direction_accuracy(self, y_true, y_pred):
        """방향 정확도 계산 (상승/하락 예측 정확도)"""
        if len(y_true) < 2 or len(y_pred) < 2:
            return 0.0
        
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        return np.mean(true_direction == pred_direction)
    
    def _calculate_performance_weights(self, validation_scores):
        """검증 성능 기반 가중치 계산 (성능이 좋을수록 높은 가중치)"""
        if not validation_scores:
            return []
        
        # MAE가 낮을수록 좋으므로 역수 사용
        weights = [1 / (score + 0.001) for score in validation_scores]
        
        # 정규화
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        return weights
    
    def _weighted_ensemble(self, predictions, weights):
        """가중 앙상블로 최종 예측 계산"""
        if not predictions or not weights:
            return predictions[0] if predictions else []
        
        if len(predictions) != len(weights):
            # 가중치가 없으면 단순 평균
            return [np.mean([pred[i] for pred in predictions]) for i in range(len(predictions[0]))]
        
        final_predictions = []
        for day in range(len(predictions[0])):
            weighted_sum = sum(pred[day] * weight for pred, weight in zip(predictions, weights))
            final_predictions.append(weighted_sum)
        
        return final_predictions
    
    def _calculate_prediction_dates(self, today, predict_steps, stock_data):
        """예측 날짜 계산"""
        pred_dates = []
        today_ts = pd.Timestamp(today)
        close_data = stock_data['close']
        
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
