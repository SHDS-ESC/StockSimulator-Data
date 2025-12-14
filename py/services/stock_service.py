"""주식 예측 메인 서비스"""

import datetime as dt
from datetime import timedelta
import warnings
import logging

import lightgbm
import pandas as pd
import numpy as np
from websockets.legacy.framing import prepare_data

# LightGBM 경고 억제
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('lightgbm').setLevel(logging.ERROR)

from .database_service import DatabaseService
from .prediction_service import StockPredictor
from .analysis_service import InvestmentAnalyzer
from .chart_service import ChartService

logger = logging.getLogger(__name__)


class StockService:
    """주식 예측 메인 서비스 클래스"""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.chart_service = ChartService()
    
    def get_available_tickers(self):
        """사용 가능한 티커 목록 조회"""
        try:
            # stock 테이블에서 티커 목록 조회
            stock_data = self.db_service.load_data_from_db('stock')
            tickers = stock_data['ticker'].tolist()

            return {
                'tickers': sorted(tickers),
                'count': len(tickers)
            }
        except Exception as e:
            raise ValueError(f"티커 목록 조회 실패: {str(e)}")
    
    def _find_nearest_date(self, target_date, date_index, allow_fallback=True, fallback_to_first=False):
        """인덱스에서 가장 가까운 날짜를 찾는 헬퍼 함수
        
        Args:
            target_date: 찾고자 하는 날짜 (date 객체 또는 datetime)
            date_index: pandas DatetimeIndex 또는 날짜 인덱스
            allow_fallback: 정확한 날짜가 없을 때 가장 가까운 날짜 사용 여부
            fallback_to_first: 정확한 날짜가 없고 allow_fallback=True일 때, 
                              가장 가까운 날짜도 없으면 첫 번째 날짜 사용 여부
        
        Returns:
            tuple: (찾은 날짜, 날짜가 변경되었는지 여부)
        
        Raises:
            ValueError: 날짜를 찾을 수 없고 fallback도 불가능한 경우
        """
        target_date = pd.to_datetime(target_date).date() if not isinstance(target_date, dt.date) else target_date
        
        # 정확한 날짜가 있는지 확인
        if target_date in date_index:
            return target_date, False
        
        # 정확한 날짜가 없으면 가장 가까운 이전 날짜 찾기
        available_dates = date_index[date_index <= target_date]
        
        if len(available_dates) > 0:
            nearest_date = available_dates.max()
            if allow_fallback:
                return nearest_date, True
            else:
                raise ValueError(
                    f"정확한 날짜 ({target_date})를 찾을 수 없습니다. "
                    f"가장 가까운 날짜: {nearest_date}. "
                    f"데이터 범위: {date_index.min()} ~ {date_index.max()}"
                )
        
        # 가장 가까운 날짜도 없으면
        if fallback_to_first and len(date_index) > 0:
            return date_index[0], True
        
        # 모든 fallback이 실패하면 에러
        raise ValueError(
            f"날짜 ({target_date})를 찾을 수 없습니다. "
            f"데이터 범위: {date_index.min()} ~ {date_index.max()}"
        )
    
    def _ensure_date_in_index(self, target_date, date_index, context_name="날짜", fallback_to_boundary=False):
        """인덱스에 날짜가 존재하는지 확인하고, 없으면 가장 가까운 날짜로 대체
        
        Args:
            target_date: 확인할 날짜
            date_index: pandas DatetimeIndex 또는 날짜 인덱스
            context_name: 에러 메시지에 사용할 컨텍스트 이름 (예: "today_str", "start_str")
            fallback_to_boundary: 가장 가까운 날짜도 없을 때 첫/마지막 날짜 사용 여부
        
        Returns:
            tuple: (유효한 날짜, 원래 날짜와 다른지 여부)
        """
        target_date = pd.to_datetime(target_date).date() if not isinstance(target_date, dt.date) else target_date
        
        # 날짜가 인덱스에 있는지 확인
        if target_date in date_index:
            return target_date, False
        
        # 없으면 가장 가까운 이전 날짜 찾기
        available_dates = date_index[date_index <= target_date]
        
        if len(available_dates) > 0:
            nearest_date = available_dates.max()
            print(f"⚠️ {context_name} ({target_date})이(가) dropna()로 제거되어 가장 가까운 날짜 사용: {nearest_date}")
            return nearest_date, True
        
        # 가장 가까운 날짜도 없으면
        if fallback_to_boundary:
            if target_date < date_index.min():
                fallback_date = date_index[0]
                print(f"⚠️ {context_name} ({target_date})이(가) dropna()로 제거되어 첫 번째 날짜 사용: {fallback_date}")
                return fallback_date, True
            else:
                fallback_date = date_index[-1]
                print(f"⚠️ {context_name} ({target_date})이(가) dropna()로 제거되어 마지막 날짜 사용: {fallback_date}")
                return fallback_date, True
        
        # 모든 fallback이 실패하면 에러
        raise ValueError(
            f"{context_name} ({target_date}) 이후의 데이터가 dropna()로 모두 제거되었습니다. "
            f"데이터 범위: {date_index.min()} ~ {date_index.max()}"
        )

    def predict_stock(self, request):
        """주식 예측 실행 함수 - 슬라이딩 윈도우 방식"""
        
        # request에서 파라미터 추출
        ticker = request.ticker
        train_days = request.train_days
        predict_steps = request.predict_steps
        today = request.today
        save_image = request.save_image
        batch_size = request.batch_size
        step_size = request.step_size
        model_params_request = request.model_params
        
        # 기본값 설정: batch_size, step_size가 설정되지 않은 경우 train_days와 같은 값으로 설정
        if batch_size is None:
            batch_size = train_days
        if step_size is None:
            step_size = train_days
        if today is None:
            today = dt.date.today()

        # 학습 횟수 계산
        training_count = (train_days - batch_size) // step_size + 1
        print(f"슬라이딩 윈도우 설정: batch_size={batch_size}, step_size={step_size}, train_days={train_days}")
        print(f"예상 학습 횟수: {training_count}")
        
        # 파라미터 로깅 (차트 생성 시 참조용)
        logger.info(f"예측 파라미터: ticker={ticker}, train_days={train_days}, predict_steps={predict_steps}, batch_size={batch_size}, step_size={step_size}, today={today}")
        if model_params_request:
            logger.info(f"요청된 모델 파라미터: {model_params_request}")

        # 2. 모델 초기화 - 파라미터 변수화
        # 모델 파라미터 설정 (변수화) - 요청 파라미터와 기본값 병합
        default_model_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42
        }
        
        # 요청된 모델 파라미터가 있으면 기본값과 병합
        model_params = default_model_params.copy()
        if model_params_request:
            model_params.update(model_params_request)
        
        lgbm = lightgbm.LGBMRegressor(
            n_estimators=model_params['n_estimators'],
            learning_rate=model_params['learning_rate'],
            max_depth=model_params['max_depth'],
            n_jobs=-1,
            verbose=-1,
            # warm_start=True,
            # num_leaves=63,
            min_child_samples=model_params['min_child_samples'],
            # min_split_gain=0.0,
            reg_alpha=model_params['reg_alpha'],
            reg_lambda=model_params['reg_lambda'],
            random_state=model_params['random_state']
        )
        logger.info(f"모델 파라미터: {model_params}")

        predictor = StockPredictor(model=lgbm)

        # 1. 데이터 로드
        stock_data = self.db_service.get_stock_data(ticker)
        feature_cols = predictor.get_feature_columns(stock_data)

        # 3. 사용 데이터 범위 산정 (메모리/시각화 최적화)
        today_dt = pd.to_datetime(today).date() # 입력된, 전달받은 today (메서드 호출)
        
        # today_dt 이하의 날짜 중 가장 가까운 날짜 찾기 (Boolean Indexing 사용 -> today가 휴일일 수 있으므로)
        today_str, date_changed = self._find_nearest_date(today_dt, stock_data.index, allow_fallback=True)
        
        if date_changed:
            print(f"⚠️ 요청한 날짜 ({today_dt})에 데이터가 없어 가장 가까운 이전 날짜 사용: {today_str}")
        
        today_idx = stock_data.index.get_loc(today_str)
        start_idx = max(0, today_idx - train_days) # train_days => 영업일 기준임. today_idx를 빼고 n일 전부터 하루 전까지 학습 데이터로 사용
        end_idx = min(today_idx + predict_steps, len(stock_data) - 1) # predict_steps => 오늘로부터 예측일. 존재한다면 영업일 기준으로 설정

        # 인덱스의 날짜 문자열 보관
        start_str = stock_data.index[start_idx]
        end_str = stock_data.index[end_idx]

        print(f"start_idx={start_idx}({start_str}) - today_idx={today_idx}({today_str}) - end_idx={end_idx}({end_str})")
        
        # 4. 학습 피처 & 예측 피처(일반 수익률, 로그 수익률) 준비: (start-50) ~ end 구간만 사용해 특성 생성 후 => 이평선 구하는 로직에서 NAN이 발생하는데, dropna에서 학습 데이터 빠지는 것을 방지하려고
        # 생성된 데이터에서 start_str ~ end_str 범위만 재슬라이싱하여 사용
        buffer_start_idx = max(0, start_idx - 50)
        buffer_slice = stock_data.iloc[buffer_start_idx:end_idx+2]
        print(f"buffer slice: {buffer_slice.index[0]} ~ {buffer_slice.index[-1]} (len={len(buffer_slice)}) # end_idx+2한 이유는 dropna 때문..")

        # 피처 만들기. loc으로 범위 재지정
        prepared_data = predictor.prepare_returns(buffer_slice)
        stock_data = prepared_data.loc[start_str:end_str]

        # index 갱신
        # dropna()로 인해 날짜들이 제거될 수 있으므로, 존재 여부 확인 후 가장 가까운 날짜 사용
        today_str, _ = self._ensure_date_in_index(today_str, stock_data.index, context_name="today_str")
        today_idx = stock_data.index.get_loc(today_str)
        
        start_str, _ = self._ensure_date_in_index(start_str, stock_data.index, context_name="start_str", fallback_to_boundary=True)
        end_str, _ = self._ensure_date_in_index(end_str, stock_data.index, context_name="end_str", fallback_to_boundary=True)
        
        start_idx = stock_data.index.get_loc(start_str)
        end_idx = stock_data.index.get_loc(end_str)

        print(f"prepared stock_data len: {len(stock_data)}, start_idx={start_idx}({start_str}), end_idx={end_idx}({end_str})")
        print(f"stock_data.head(): \n{stock_data[feature_cols].head()}")
        print(f"stock_data.tail(): \n{stock_data[feature_cols].tail()}")

        # 어제까지의 데이터 -> 학습용
        train_data = stock_data.iloc[ : today_idx]
        print(f"훈련 데이터: {len(train_data)}일 ({train_data.index[0]} ~ {train_data.index[-1]})")

        # 오늘부터 ~ 예측일까지의 데이터 -> 검증용 (미래 시제는 검증 못함)
        valid_data = stock_data.iloc[today_idx : ]
        print(f"검증 데이터: {len(valid_data)}일 ({valid_data.index[0]} ~ {valid_data.index[-1]})")
        print(f"valid_data.head(): \n{valid_data[feature_cols].head()}")
        print(f"valid_data.tail(): \n{valid_data[feature_cols].tail()}")

        # 5. 슬라이딩 윈도우 예측 실행
        price_predictions, metrics_summary = self._sliding_window_predict(
            train_data, predictor, predict_steps, 
            batch_size, step_size
        )
        print(f"price_predictions: {price_predictions}")
        
        # 10. 예측 날짜 계산 (가용 구간 기준)
        pred_dates = self._calculate_prediction_dates(today, predict_steps, stock_data)
        # pred_dates = valid_data.index
        print(f"pred_dates={pred_dates}")
        
        # 11. 투자 분석 실행
        # 현재 가격
        current_price = stock_data.loc[today_str]['close']
        
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
            dataset_label = f"{ticker}-{stock_data.index[0]}~{stock_data.index[-1]}"
            created_at = dt.date.today().strftime('%Y-%m-%d')
            metadata = {
                'model_name': model_name,
                'metrics': metrics_summary or {},
                'dataset': dataset_label,
                'created_at': created_at,
                'params': {
                    'batch_size': batch_size,
                    'step_size': step_size,
                    'train_days': train_days,
                    'predict_steps': predict_steps,
                    'feature_count': len(predictor.get_feature_columns(train_data))
                },
                'model_params': model_params
            }
            chart_data = self.chart_service.create_prediction_charts(
                ticker, today, price_predictions, stock_data.loc[start_str:end_str]['close'], pred_dates,
                metadata=metadata
            )
        
        # Feature importance 추출
        feature_importance_data = self._get_feature_importance(predictor, train_data)
        feature_importance = None
        if "error" not in feature_importance_data:
            from ..models.dto import FeatureImportance
            feature_importance = FeatureImportance(**feature_importance_data)
        
        result = {
            'ticker': ticker,
            'base_date': today,
            'last_price': float(current_price),
            'return_predictions': [],  # 슬라이딩 윈도우에서는 수익률 예측을 별도로 계산하지 않음
            'price_predictions': price_predictions,
            'prediction_dates': [date.date() if hasattr(date, 'date') else date for date in pred_dates],  # date 객체로 변환
            'train_data_count': len(train_data),  # 사용 데이터 수
            'feature_count': len(predictor.get_feature_columns(train_data)),  # 피처 수
            'feature_importance': feature_importance,  # 피처 중요도 추가
            'investment_analysis': investment_analysis
        }
        
        # 차트 데이터 추가 (있는 경우)
        if chart_data:
            result.update(chart_data)
        
        return result
    
    def _get_feature_importance(self, predictor, train_data):
        """Feature importance 추출"""
        try:
            if not hasattr(predictor.model, 'feature_importances_'):
                return {"error": "모델이 feature importance를 지원하지 않습니다."}
            
            # 피처 컬럼 가져오기
            feature_cols = predictor.get_feature_columns(train_data)
            
            if not feature_cols:
                return {"error": "피처 컬럼을 찾을 수 없습니다."}
            
            # Feature importance 추출
            importances = predictor.model.feature_importances_
            
            # 피처명과 중요도를 매핑
            feature_importance_dict = dict(zip(feature_cols, importances))
            
            # 중요도 순으로 정렬 (상위 20개)
            sorted_features = sorted(feature_importance_dict.items(), 
                                   key=lambda x: x[1], reverse=True)[:20]
            
            return {
                "top_features": sorted_features,
                "total_features": len(feature_cols),
                "importance_sum": float(importances.sum()),
                "max_importance": float(importances.max()),
                "min_importance": float(importances.min())
            }
            
        except Exception as e:
            return {"error": f"Feature importance 추출 실패: {str(e)}"}
    
    def _sliding_window_predict(self, train_data, predictor, predict_steps, 
                               batch_size, step_size):
        """성능 검증이 포함된 슬라이딩 윈도우 예측 실행 (사전 산정된 가용 구간 사용)

        Returns:
            tuple[list[float], dict]: (final_predictions, metrics_summary)
        """
        
        # 슬라이딩 윈도우로 예측 수행 (성능 검증 포함)
        all_predictions = []
        validation_scores = []
        performance_metrics = []

        print(f"학습 시작.. len(train_data): {len(train_data)}, train_index range: {train_data.index[0]} ~ {train_data.index[-1]}")
        for i in range(0, len(train_data) - batch_size + 1, step_size):
            # 현재 윈도우 데이터 선택
            window_data = train_data.iloc[i:i + batch_size]
            if len(window_data) < batch_size:
                break

            # 학습/검증 분할: 예측 기간에 비례한 검증 데이터 크기
            val_ratio = predict_steps / batch_size
            val_size = max(1, int(len(window_data) * val_ratio))
            val_size = min(val_size, len(window_data) - 1)  # 최소 1일은 학습용으로 남김
            
            train_df = window_data.iloc[:-val_size]
            val_df = window_data.iloc[-val_size:]

            print(f"train_df range: {train_df.index[0]} ~ {train_df.index[-1]}, val_df range: {val_df.index[0]} ~ {val_df.index[-1]}")

            # 피처와 타겟 분리 (학습)
            feature_cols = predictor.get_feature_columns(train_df)
            pred_col = predictor.get_predict_column()

            print(f"feature_cols: {feature_cols}, pred_col: {pred_col}")

            # Target 변수와 Feature-Target 상관관계 분석
            self._analyze_target_features(train_df, feature_cols, pred_col)

            X_train = train_df[feature_cols].copy()
            # y는 1D Series로 사용 (스칼라 메트릭 비교 용이)
            y_train = train_df[pred_col].copy()
            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.squeeze()

            # 모델 학습 (일반 학습)
            predictor.train(X_train, y_train)
            
            # 모델 학습 (warm_start): 윈도우마다 트리를 Δ개 추가하며 이어학습
            # trees_per_window = 200  # 윈도우당 추가 트리 수 (정책)
            # predictor.train(
            #     X_train, y_train,
            #     X_val=val_df[feature_cols].copy(), y_val=val_df[pred_col].copy(),
            #     warm_start=True, add_estimators=trees_per_window,
            #     early_stopping_rounds=50, eval_metric="l2"
            # )

            # # 학습 상태 로깅: 트리 추가 여부/규모 점검
            # try:
            #     model_obj = predictor.model
            #     n_estimators_attr = getattr(model_obj, 'n_estimators_', None)
            #     best_iter = getattr(model_obj, 'best_iteration_', None)
            #     booster = getattr(model_obj, 'booster_', None)
            #     num_trees = booster.num_trees() if booster is not None else None
            #     print(f"[train status] trees_per_window={trees_per_window} n_estimators_={n_estimators_attr} best_iteration_={best_iter} num_trees={num_trees}")
            # except Exception as e:
            #     print(f"[train status] logging error: {e}")

            # 실제 예측: recursive로 predict_steps일 예측
            X_val = val_df[feature_cols].iloc[[-1]].copy()
            y_val = float(val_df[pred_col].iloc[-1]) # 실제 값(사용 시 주의: 단일 값)
            if isinstance(y_val, pd.DataFrame):
                y_val = y_val.squeeze()
            # 히스토리 버퍼(window_data)를 넘겨 지표를 스텝마다 재계산하며 예측
            return_predictions = predictor.predict_next_returns(
                X_val, steps=predict_steps, history_df=window_data.copy()
            )

            print(f"X_val: {X_val}, y_val: {y_val}")

            # 1-step 검증을 별도 호출 없이 처리: 실제 recursive 예측의 첫 스텝을 사용
            # - 시점 정합성: X_val(t-1) → r_{t-1→t}
            # - 운영 일관성: 생산 예측과 동일 경로에서 평가
            if len(return_predictions) < 1:
                raise ValueError("predict_next_returns가 빈 결과를 반환했습니다")
            pred1_scalar = float(np.asarray(return_predictions[0]).ravel()[0])
            y_val_scalar = float(np.asarray(y_val).ravel()[0])

            mae_score_1 = self._calculate_mae(np.array([y_val_scalar]), np.array([pred1_scalar]))
            rmse_score_1 = self._calculate_rmse(np.array([y_val_scalar]), np.array([pred1_scalar]))
            direction_acc_1 = 1.0 if np.sign(pred1_scalar) == np.sign(y_val_scalar) else 0.0

            # 멀티스텝 검증: 예측한 스텝 수 만큼 실제 값과 비교
            k_steps = min(predict_steps, len(val_df))
            if k_steps > 0:
                y_true_seq = val_df[pred_col].iloc[:k_steps].values.astype(float)
                y_pred_seq = np.asarray(return_predictions[:k_steps], dtype=float)
                mae_score_ms = self._calculate_mae(y_true_seq, y_pred_seq)
                rmse_score_ms = self._calculate_rmse(y_true_seq, y_pred_seq)
                direction_acc_ms = float(np.mean(np.sign(y_pred_seq) == np.sign(y_true_seq)))
            else:
                mae_score_ms = np.nan
                rmse_score_ms = np.nan
                direction_acc_ms = np.nan

            validation_scores.append(mae_score_1)
            performance_metrics.append({
                'mae_1': mae_score_1,
                'rmse_1': rmse_score_1,
                'direction_accuracy_1': direction_acc_1,
                'mae_ms': mae_score_ms,
                'rmse_ms': rmse_score_ms,
                'direction_accuracy_ms': direction_acc_ms
            })

            # 수익률을 가격으로 변환
            last_price = window_data['close'].iloc[-1]
            price_predictions = predictor.convert_returns_to_prices(last_price, return_predictions)

            all_predictions.append(price_predictions)

            window_no = i // step_size + 1
            start_dt, end_dt = train_df.index[0], train_df.index[-1]
            # 1-step 결과 요약 (recursive 첫 스텝 기반)
            print(f"[Window {window_no}] {start_dt} ~ {end_dt}")
            print(f"  1-step: pred={pred1_scalar:.6f}, true={y_val_scalar:.6f}, MAE={mae_score_1:.6f}, RMSE={rmse_score_1:.6f}, DirAcc={direction_acc_1:.3f}")

            # 멀티스텝 결과 요약
            steps_len = len(return_predictions) if hasattr(return_predictions, '__len__') else predict_steps
            ret_first = return_predictions[0] if steps_len > 0 else None
            ret_last = return_predictions[-1] if steps_len > 0 else None
            # 가격 기준 RMSE(멀티스텝) 계산 및 로그
            if k_steps > 0:
                actual_prices_seq = val_df['close'].iloc[:k_steps].values.astype(float)
                pred_prices_seq = np.asarray(price_predictions[:k_steps], dtype=float)
                price_rmse_ms = self._calculate_rmse(actual_prices_seq, pred_prices_seq)
                performance_metrics[-1]['price_rmse_ms'] = price_rmse_ms
                print(f"  multi-step: steps={steps_len}, rtn_first={ret_first:.6f} rtn_last={ret_last:.6f}, price_RMSE={price_rmse_ms:.6f}")
            else:
                print(f"  multi-step: steps={steps_len}, rtn_first={ret_first:.6f} rtn_last={ret_last:.6f}")
            print(f"  base_price={last_price:.4f}, history=on")
        
        # 예측 결과 검증
        if not all_predictions:
            raise ValueError("예측할 수 있는 충분한 데이터가 없습니다")

        # 생산 예측 정책:
        # - 서로 다른 기준시점(윈도우)의 예측을 혼합하지 않는다.
        # - 최종 예측은 "가장 최신 윈도우"(마지막 윈도우)의 멀티스텝 결과를 사용한다.
        # - 윈도우별 성능(MAE 등)은 리포팅/모니터링/가중치 산정용 참고 값으로만 사용.
        final_predictions = all_predictions[-1]

        # 성능 요약 출력 (1-step 및 멀티스텝 각각)
        def _safe_mean(values):
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                return np.nan
            return float(np.nanmean(arr))

        avg_mae_1 = _safe_mean([m.get('mae_1') for m in performance_metrics])
        avg_rmse_1 = _safe_mean([m.get('rmse_1') for m in performance_metrics])
        avg_direction_acc_1 = _safe_mean([m.get('direction_accuracy_1') for m in performance_metrics])

        avg_mae_ms = _safe_mean([m.get('mae_ms') for m in performance_metrics])
        avg_rmse_ms = _safe_mean([m.get('rmse_ms') for m in performance_metrics])
        avg_direction_acc_ms = _safe_mean([m.get('direction_accuracy_ms') for m in performance_metrics])
        avg_price_rmse_ms = _safe_mean([m.get('price_rmse_ms') for m in performance_metrics])

        print(f"총 {len(all_predictions)}개 윈도우로 예측 완료")
        print(f"평균 성능(1-step): MAE={avg_mae_1:.4f}, RMSE={avg_rmse_1:.4f}, 방향정확도={avg_direction_acc_1:.3f}")
        print(f"평균 성능(multi-step): MAE={avg_mae_ms:.4f}, RMSE={avg_rmse_ms:.4f}, 방향정확도={avg_direction_acc_ms:.3f}, 가격RMSE={avg_price_rmse_ms:.4f}")

        metrics_summary = {
            'mae_1': avg_mae_1 if not np.isnan(avg_mae_1) else None,
            'rmse_1': avg_rmse_1 if not np.isnan(avg_rmse_1) else None,
            'direction_accuracy_1': avg_direction_acc_1 if not np.isnan(avg_direction_acc_1) else None,
            'mae_ms': avg_mae_ms if not np.isnan(avg_mae_ms) else None,
            'rmse_ms': avg_rmse_ms if not np.isnan(avg_rmse_ms) else None,
            'direction_accuracy_ms': avg_direction_acc_ms if not np.isnan(avg_direction_acc_ms) else None,
            'price_rmse_ms': avg_price_rmse_ms if not np.isnan(avg_price_rmse_ms) else None,
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
    
    def _analyze_target_features(self, train_df, feature_cols, pred_col):
        """Target 변수와 Feature-Target 상관관계 분석"""
        try:
            print(f"\n🔍 Target 변수 분석: {pred_col}")
            print("=" * 50)
            
            # Target 변수 통계
            target_stats = train_df[pred_col].describe()
            print(f"Target 변수 통계:")
            print(f"  개수: {target_stats['count']:.0f}")
            print(f"  평균: {target_stats['mean']:.6f}")
            print(f"  표준편차: {target_stats['std']:.6f}")
            print(f"  최솟값: {target_stats['min']:.6f}")
            print(f"  최댓값: {target_stats['max']:.6f}")
            print(f"  25%: {target_stats['25%']:.6f}")
            print(f"  50%: {target_stats['50%']:.6f}")
            print(f"  75%: {target_stats['75%']:.6f}")
            
            # Target 변수 샘플
            print(f"\nTarget 변수 샘플 (처음 10개):")
            sample_values = train_df[pred_col].head(10).values
            for i, val in enumerate(sample_values):
                print(f"  [{i}] {val:.6f}")
            
            # Target 변수 분포
            positive_count = (train_df[pred_col] > 0).sum()
            negative_count = (train_df[pred_col] < 0).sum()
            zero_count = (train_df[pred_col] == 0).sum()
            total_count = len(train_df[pred_col])
            
            print(f"\nTarget 변수 분포:")
            print(f"  양수: {positive_count}개 ({positive_count/total_count*100:.1f}%)")
            print(f"  음수: {negative_count}개 ({negative_count/total_count*100:.1f}%)")
            print(f"  영: {zero_count}개 ({zero_count/total_count*100:.1f}%)")
            
            # Feature-Target 상관관계
            print(f"\nFeature-Target 상관관계 (상위 10개):")
            correlations = []
            for col in feature_cols:
                try:
                    corr = train_df[col].corr(train_df[pred_col])
                    if not np.isnan(corr):
                        correlations.append((col, corr))
                except:
                    continue
            
            # 상관관계 절댓값 기준으로 정렬
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for i, (col, corr) in enumerate(correlations[:10]):
                direction = "📈" if corr > 0 else "📉"
                print(f"  {i+1:2d}. {col:20s}: {corr:8.4f} {direction}")
            
            # 가장 높은 상관관계 피처들의 실제 값 확인
            if correlations:
                top_feature = correlations[0][0]
                print(f"\n최고 상관관계 피처 '{top_feature}' 샘플:")
                sample_data = train_df[[top_feature, pred_col]].head(5)
                for idx, row in sample_data.iterrows():
                    print(f"  {idx}: {top_feature}={row[top_feature]:.6f}, {pred_col}={row[pred_col]:.6f}")
            
            # 피처들의 변화량과 Target 상관관계 분석
            print(f"\n피처 변화량 vs Target 상관관계 분석:")
            print("-" * 50)
            
            # 주요 피처들의 변화량 계산
            change_features = {}
            for col in ['rsi', 'macd', 'close', 'volume']:
                if col in train_df.columns:
                    change_col = f"{col}_change"
                    train_df[change_col] = train_df[col] - train_df[col].shift(1)
                    change_features[change_col] = train_df[change_col]
            
            # 가격 변화량 (수익률)
            if 'close' in train_df.columns:
                train_df['price_change'] = train_df['close'] / train_df['close'].shift(1) - 1
                change_features['price_change'] = train_df['price_change']
            
            # 변화량 피처들과 Target의 상관관계
            change_correlations = []
            for col, values in change_features.items():
                try:
                    corr = values.corr(train_df[pred_col])
                    if not np.isnan(corr):
                        change_correlations.append((col, corr))
                except:
                    continue
            
            change_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print("변화량 피처 상관관계 (상위 5개):")
            for i, (col, corr) in enumerate(change_correlations[:5]):
                direction = "📈" if corr > 0 else "📉"
                print(f"  {i+1}. {col:20s}: {corr:8.4f} {direction}")
            
            # 피처들의 실제 값과 Target의 관계 패턴 분석
            print(f"\n피처-타겟 관계 패턴 분석:")
            print("-" * 50)
            
            # RSI와 Target의 관계
            if 'rsi' in train_df.columns:
                print("RSI vs Target 관계:")
                rsi_ranges = [(0, 30), (30, 50), (50, 70), (70, 100)]
                for low, high in rsi_ranges:
                    mask = (train_df['rsi'] >= low) & (train_df['rsi'] < high)
                    if mask.sum() > 0:
                        avg_target = train_df[mask][pred_col].mean()
                        count = mask.sum()
                        print(f"  RSI {low}-{high}: 평균 Target={avg_target:.4f} (n={count})")
            
            # 가격 변화량과 Target의 관계
            if 'price_change' in train_df.columns:
                print("\n가격 변화량 vs Target 관계:")
                price_ranges = [(-0.1, -0.05), (-0.05, 0), (0, 0.05), (0.05, 0.1)]
                for low, high in price_ranges:
                    mask = (train_df['price_change'] >= low) & (train_df['price_change'] < high)
                    if mask.sum() > 0:
                        avg_target = train_df[mask][pred_col].mean()
                        count = mask.sum()
                        print(f"  가격변화 {low:.2f}-{high:.2f}: 평균 Target={avg_target:.4f} (n={count})")
            
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Target-Feature 분석 중 오류: {e}")
    
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
        close_data = stock_data['close']
        
        # close_data에서 today 이후의 날짜들을 찾기
        future_dates_in_data = close_data.index[close_data.index > today]
        
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
                last_date = today
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

    # ===== 포트폴리오 누적수익률 =====
    def portfolio_analysis(self, request):
        """여러 포트폴리오에 대한 일별 누적가치 계산

        Args:
            request (PortfolioCumulativeReturnsRequest): 기간과 포트폴리오 사양

        Returns:
            dict: { 'series': [ { 'id': str, 'series': [ { 'date': date, 'value': float } ] } ], 'metrics': dict }
        """
        # TODO: 실제 구현 - 데이터 로드, 수익률 계산, 가중합, 리밸런싱, 누적가치 변환
        try:
            if request.start_date > request.end_date:
                raise ValueError("start_date는 end_date보다 이전이어야 합니다")
            if not request.portfolios:
                raise ValueError("포트폴리오 목록이 비어 있습니다")

            series_list = []
            metrics_list = []
            for pf in request.portfolios:
                if len(pf.tickers) != len(pf.weights):
                    raise ValueError(f"포트폴리오 '{pf.id}'의 tickers와 weights 길이가 다릅니다")

                # 일별 수익률 Series
                portfolio_daily_returns = self.make_portfolio_cumulative(pf.tickers, pf.weights, request.start_date, request.end_date)
                
                # quantstats 지표 계산
                qs_metrics = self.portfolio_reports_metrics(portfolio_daily_returns)
                metrics_payload = self._convert_quantstats_to_dto(qs_metrics)
                # 변동성 폴백: 1년 미만 등으로 비어 있으면 일변동성의 연율화로 보충
                try:
                    if metrics_payload.get('volatility_annualized') is None:
                        if len(portfolio_daily_returns) >= 2:
                            daily_vol = float(pd.Series(portfolio_daily_returns).std(ddof=1))
                            ann_vol = float(daily_vol * np.sqrt(252))
                            metrics_payload['volatility_annualized'] = ann_vol
                except Exception:
                    pass
                
                # 누적수익률 Series
                portfolio_cum = (1 + portfolio_daily_returns).cumprod() - 1
                portfolio_cum = portfolio_cum.sort_index()
                items = []
                for idx, val in portfolio_cum.items():
                    # idx를 date로 변환 보장
                    if hasattr(idx, 'date'):
                        d = idx.date()
                    else:
                        try:
                            d = pd.to_datetime(idx).date()
                        except Exception:
                            d = idx
                    items.append({'date': d, 'value': float(val)})

                series_list.append({
                    'id': pf.id,
                    'series': items
                })
                metrics_list.append({
                    'id': pf.id,
                    'metrics': metrics_payload
                })

            return { 'series': series_list, 'metrics': metrics_list }
        except Exception:
            # 상위에서 로깅/에러 변환
            raise


    def make_portfolio_cumulative(self, ticker_list, weight_list, start_date, end_date):
        returns = (
            self.db_service
                .get_multi_stock_data(ticker_list, columns=['close'], wide=True)
                .loc[start_date:end_date]
                .pct_change()
                .dropna()
        )
        portfolio_returns = pd.Series(np.dot(weight_list, returns.T), index=returns.index)
        return portfolio_returns

    def portfolio_reports_metrics(self, portfolio_returns):
        import quantstats as qs
        # metrics 함수를 사용하여 주요 지표 값을 한 번에 얻기
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
        metrics_series = qs.reports.metrics(
            portfolio_returns,
            mode='full',
            display=False,
            annualize=True  # 연율화된 지표 포함
        )
        print(f'metrics_series: {metrics_series}')
        # basic 모드에 변동성이 포함되지 않을 수 있어, 직접 계산하여 테이블에 추가
        try:
            vol_ann = float(pd.Series(portfolio_returns).std(ddof=1) * np.sqrt(252))
            if isinstance(metrics_series, pd.Series):
                if 'Volatility (ann.)' not in metrics_series.index and 'Volatility' not in metrics_series.index:
                    metrics_series['Volatility (ann.)'] = vol_ann
            else:
                col = metrics_series.columns[0] if len(metrics_series.columns) > 0 else None
                if col is not None and 'Volatility (ann.)' not in metrics_series.index and 'Volatility' not in metrics_series.index:
                    metrics_series.loc['Volatility (ann.)', col] = vol_ann
        except Exception:
            pass
        
        return metrics_series

    def _convert_quantstats_to_dto(self, metrics_obj):
        """QuantStats metrics(DataFrame/Series)을 API DTO(dict)로 변환"""
        def _get_val(name):
            try:
                if isinstance(metrics_obj, pd.Series):
                    return metrics_obj.get(name)
                # DataFrame: 첫 컬럼 기준 값 추출
                col = metrics_obj.columns[0] if len(metrics_obj.columns) > 0 else None
                if col is None:
                    return None
                if name in metrics_obj.index:
                    return metrics_obj.loc[name, col]
                return None
            except Exception:
                return None

        def _to_float(v):
            try:
                if v is None:
                    return None
                s = str(v).strip()
                if s == '' or s.lower() in ('nan', 'none'):
                    return None
                is_percent = s.endswith('%')
                if is_percent:
                    s = s[:-1]
                s = s.replace(',', '')
                num = float(s)
                return num / 100.0 if is_percent else num
            except Exception:
                try:
                    return float(v)
                except Exception:
                    return None

        def _to_int(v):
            try:
                if v is None:
                    return None
                return int(float(v))
            except Exception:
                return None

        def _to_date(v):
            try:
                if v is None:
                    return None
                dtv = pd.to_datetime(v, errors='coerce')
                if pd.isna(dtv):
                    return None
                return dtv.date()
            except Exception:
                return None

        # 키 매핑 및 변환
        payload = {
            'start_period': _to_date(_get_val('Start Period')),
            'end_period': _to_date(_get_val('End Period')),
            'time_in_market': _to_float(_get_val('Time in Market')),
            'cumulative_return': _to_float(_get_val('Cumulative Return')),
            'cagr': _to_float(_get_val('CAGR﹪') or _get_val('CAGR%') or _get_val('CAGR')),
            'sharpe': _to_float(_get_val('Sharpe')),
            'prob_sharpe_ratio': _to_float(_get_val('Prob. Sharpe Ratio')),
            'sortino': _to_float(_get_val('Sortino')),
            'omega': _to_float(_get_val('Omega')),
            'max_drawdown': _to_float(_get_val('Max Drawdown')),
            'max_dd_date': _to_date(_get_val('Max DD Date')),
            'max_dd_period_start': _to_date(_get_val('Max DD Period Start')),
            'max_dd_period_end': _to_date(_get_val('Max DD Period End')),
            'longest_dd_days': _to_int(_get_val('Longest DD Days')),
            'gain_pain_ratio': _to_float(_get_val('Gain/Pain Ratio')),
            'payoff_ratio': _to_float(_get_val('Payoff Ratio')),
            'profit_factor': _to_float(_get_val('Profit Factor')),
            'cpc_index': _to_float(_get_val('CPC Index')),
            'tail_ratio': _to_float(_get_val('Tail Ratio')),
            'outlier_win_ratio': _to_float(_get_val('Outlier Win Ratio')),
            'outlier_loss_ratio': _to_float(_get_val('Outlier Loss Ratio')),
            'mtd': _to_float(_get_val('MTD')),
            'three_m': _to_float(_get_val('3M')),
            'six_m': _to_float(_get_val('6M')),
            'ytd': _to_float(_get_val('YTD')),
            'one_y': _to_float(_get_val('1Y')),
            'three_y_ann': _to_float(_get_val('3Y (ann.)')),
            'five_y_ann': _to_float(_get_val('5Y (ann.)')),
            'ten_y_ann': _to_float(_get_val('10Y (ann.)')),
            'all_time_ann': _to_float(_get_val('All-time (ann.)') or _get_val('All-Time (ann.)')),
            'avg_drawdown': _to_float(_get_val('Avg. Drawdown')),
            'avg_drawdown_days': _to_int(_get_val('Avg. Drawdown Days')),
            'recovery_factor': _to_float(_get_val('Recovery Factor')),
            'ulcer_index': _to_float(_get_val('Ulcer Index')),
            'serenity_index': _to_float(_get_val('Serenity Index')),
            'volatility_annualized': _to_float(_get_val('Volatility (ann.)') or _get_val('Volatility')),
        }

        return payload