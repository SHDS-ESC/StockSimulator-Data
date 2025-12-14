"""주식 예측 서비스"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 기술분석 라이브러리
import talib as ta
import lightgbm as lgb


class StockPredictor:
    """주식 가격 예측 모델"""
    
    def __init__(self, model=None, use_scaler=True, random_state=42):
        self.model = model if model else RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            max_features="sqrt",
            min_samples_split=2,
            min_samples_leaf=5,
            random_state=random_state
        )
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None
        self.feature_columns = None
        
    def calculate_technical_indicators(self, df):
        """기술 지표 계산"""
        df_temp = df.copy()
        
        # 이미 계산된 기술지표가 있는지 확인
        required_indicators = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'atr', 'stoch_k', 'stoch_d', 'obv']
        missing_indicators = [col for col in required_indicators if col not in df_temp.columns]
        
        if not missing_indicators:
            print("✅ 기술지표가 이미 계산되어 있음, 재계산 건너뛰기")
            return df_temp
        
        print(f"🔄 누락된 기술지표 계산 중: {missing_indicators}")
        
        # 누락된 지표만 계산
        if 'rsi' in missing_indicators:
            df_temp['rsi'] = ta.RSI(df_temp['close'].values, timeperiod=14)
        if any(x in missing_indicators for x in ['macd', 'macd_signal', 'macd_hist']):
            df_temp['macd'], df_temp['macd_signal'], df_temp['macd_hist'] = ta.MACD(
                df_temp['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        if 'atr' in missing_indicators:
            df_temp['atr'] = ta.ATR(df_temp['high'].values, df_temp['low'].values, 
                                   df_temp['close'].values, timeperiod=14)
        if any(x in missing_indicators for x in ['stoch_k', 'stoch_d']):
            df_temp['stoch_k'], df_temp['stoch_d'] = ta.STOCH(
                df_temp['high'].values, df_temp['low'].values, df_temp['close'].values,
                fastk_period=5, slowk_period=3, slowd_period=3)
        if 'obv' in missing_indicators:
            df_temp['obv'] = ta.OBV(df_temp['close'].values, df_temp['volume'].values)
        
        return df_temp
    
    def add_engineered_features(self, df):
        """추가 피처 생성"""
        df_temp = df.copy()
        
        # 이동평균 및 가격 비율
        for period in [5, 10, 20, 50]:
            df_temp[f'sma_{period}'] = df_temp['close'].rolling(window=period).mean()
            df_temp[f'price_to_sma_{period}'] = df_temp['close'] / df_temp[f'sma_{period}']
        
        # 볼린저 밴드
        bb_period = 20
        bb_std = 2
        bb_sma = df_temp['close'].rolling(window=bb_period).mean()
        bb_std_val = df_temp['close'].rolling(window=bb_period).std()
        df_temp['bb_upper'] = bb_sma + (bb_std_val * bb_std)
        df_temp['bb_lower'] = bb_sma - (bb_std_val * bb_std)
        df_temp['bb_position'] = (df_temp['close'] - df_temp['bb_lower']) / (df_temp['bb_upper'] - df_temp['bb_lower'])
        
        # 수익률 피처 (단위: 소수)
        for period in [1, 3, 5, 10]:
            df_temp[f'return_{period}d'] = (df_temp['close'] / df_temp['close'].shift(period) - 1)
        
        # 변동성
        for period in [10, 20]:
            df_temp[f'volatility_{period}d'] = df_temp[f'return_1d'].rolling(window=period).std()
        
        # 거래량 분석
        df_temp['volume_sma_10'] = df_temp['volume'].rolling(window=10).mean()
        df_temp['volume_ratio'] = df_temp['volume'] / df_temp['volume_sma_10']

        # 타겟 변수 생성 (다음날 수익률, 단위: 소수)
        # shift(-1)은 아래행을 위로 끌어 올리는 것. 즉, 내일의 종가 / 오늘의 종가 == 내일 수익률 을 y로 두고
        # x로는 오늘의 피처 변수들을 두는 것. 로그 수익률에 -1을 하지 않는건, np.log(내일 종가) - np.log(오늘 종가) == np.log(내일종가/오늘종가) 이므로...
        df_temp['simple_rtn'] = (df_temp['close'].shift(-1) / df_temp['close'] - 1)
        df_temp['log_rtn'] = np.log(df_temp['close'].shift(-1) / df_temp['close'])

        # Lagged Feature 생성 (t-1일 수익률, t-2일 수익률 ..)
        lags = [1, 2, 3, 5, 10]
        for l in lags:
            df_temp[f'rtn_lag_{l}'] = df_temp['simple_rtn'].shift(l)
        
        return df_temp
    
    def time_series_split_validate(self, X, y, n_splits=5):
        """시계열 교차검증"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # 임시 모델 생성
            temp_model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
            
            if self.use_scaler:
                temp_scaler = StandardScaler()
                X_train_scaled = temp_scaler.fit_transform(X_train_cv.values)
                X_val_scaled = temp_scaler.transform(X_val_cv.values)
            else:
                X_train_scaled = X_train_cv.values
                X_val_scaled = X_val_cv.values
            
            temp_model.fit(X_train_scaled, y_train_cv)
            y_pred = temp_model.predict(X_val_scaled)
            
            mse = mean_squared_error(y_val_cv, y_pred)
            scores.append(mse)
        
        return np.mean(scores), np.std(scores)

    # simple_rtn, log_rtn
    def prepare_returns(self, df):
        """피처 준비"""
        df_temp = df.copy()
        
        # 컬럼명을 소문자로 변환
        df_temp.columns = df_temp.columns.str.lower()
        
        df_temp = self.calculate_technical_indicators(df_temp)
        df_temp = self.add_engineered_features(df_temp)


        # NaN 제거
        df_temp = df_temp.dropna()
        
        return df_temp

    # simple_rtn, log_rtn
    def get_predict_column(self):
        # 'simple_rtn' 또는 'log_rtn' 중 선택 (단일 컬럼명 문자열 반환)
        return 'simple_rtn'


    def get_feature_columns(self, df):
        """피처 컬럼 선택"""
        # OHLC 및 타깃 컬럼 제외(모델 입력은 정규화 파생 위주)
        exclude_cols = [
            'report_id', 'report_date', 'stock_id',
            'open', 'high', 'low', 'close',
            'simple_rtn', 'log_rtn'
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              warm_start=False, add_estimators=0, early_stopping_rounds=None,
              eval_metric="l2"):
        """모델 학습
        warm_start가 True이면, n_estimators를 add_estimators만큼 증가시켜 이어학습을 시도합니다.
        early_stopping_rounds가 설정되면 eval_set을 사용해 조기 종료를 활성화합니다.
        """
        # 피처 컬럼 저장
        self.feature_columns = X_train.columns.tolist()
        
        # 교차검증(옵션): 필요 시 유지. 현재는 결과를 저장만 하고 사용하지 않음
        # cv_mean, cv_std = self.time_series_split_validate(X_train, y_train)
        
        # 스케일링
        if self.use_scaler:
            X_train_scaled = self.scaler.fit_transform(X_train.values)
        else:
            X_train_scaled = X_train.values
        
        eval_set = None
        if X_val is not None and y_val is not None:
            if isinstance(y_val, pd.DataFrame):
                y_val = y_val.squeeze()
            if self.use_scaler:
                X_val_scaled = self.scaler.transform(X_val.values)
            else:
                X_val_scaled = X_val.values
            eval_set = [(X_val_scaled, y_val)]
        
        # warm_start 이어학습 설정
        if warm_start:
            try:
                current_n = self.model.get_params().get('n_estimators', 0) or 0
                self.model.set_params(warm_start=True, n_estimators=current_n + int(add_estimators))
            except Exception:
                pass
        
        # 최종 모델 학습
        fit_kwargs = {}
        callbacks = []
        if eval_set is not None:
            fit_kwargs['eval_set'] = eval_set
            fit_kwargs['eval_metric'] = eval_metric
            if early_stopping_rounds is not None:
                callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))
        if callbacks:
            fit_kwargs['callbacks'] = callbacks
        
        self.model.fit(X_train_scaled, y_train, **fit_kwargs)
        
        return self
    
    def predict_next_returns(self, X_last, steps=5, history_df=None):
        """미래 수익률 예측"""
        if self.feature_columns is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        predictions = []
        current_features = X_last.copy()
        use_history = history_df is not None and isinstance(history_df, pd.DataFrame) and len(history_df) > 0
        if use_history:
            history = history_df.copy()
            # history는 이미 외부에서 피처가 계산된 상태(prepare_returns)라고 가정
        
        for step in range(steps):
            # 현재 피처로 수익률 예측
            if self.use_scaler:
                features_scaled = self.scaler.transform(current_features[self.feature_columns].values)
            else:
                features_scaled = current_features[self.feature_columns].values
            
            return_pred = self.model.predict(features_scaled)[0]
            predictions.append(return_pred)
            
            # 다음 스텝을 위한 피처 업데이트
            if use_history:
                # 1) 예측 close를 히스토리에 추가
                last_row = history.iloc[[-1]].copy()
                prev_close = float(last_row['close'].iloc[0]) if 'close' in last_row.columns else None
                next_close = prev_close * (1 + return_pred) if prev_close is not None else None
                new_row = last_row.copy()
                if next_close is not None:
                    new_row.loc[:, 'close'] = next_close
                # open/high/low/volume은 보수적으로 직전값 유지(없으면 생성하지 않음)
                history = pd.concat([history, new_row], axis=0)

                # 2) close 기반 지표 재계산(마지막 행만)
                # SMA 및 price_to_sma
                for period in [5, 10, 20, 50]:
                    sma_col = f'sma_{period}'
                    history.loc[:, sma_col] = history['close'].rolling(window=period, min_periods=period).mean()
                    pts_col = f'price_to_sma_{period}'
                    if sma_col in history.columns:
                        sma_vals = history[sma_col]
                        history.loc[:, pts_col] = history['close'] / sma_vals

                # 볼린저 밴드 및 포지션
                bb_period = 20
                bb_std = 2
                bb_sma = history['close'].rolling(window=bb_period, min_periods=bb_period).mean()
                bb_std_val = history['close'].rolling(window=bb_period, min_periods=bb_period).std()
                history.loc[:, 'bb_upper'] = bb_sma + (bb_std_val * bb_std)
                history.loc[:, 'bb_lower'] = bb_sma - (bb_std_val * bb_std)
                denom = (history['bb_upper'] - history['bb_lower'])
                history.loc[:, 'bb_position'] = (history['close'] - history['bb_lower']) / denom.replace(0, np.nan)

                # 수익률 및 변동성(단위: 소수)
                history.loc[:, 'return_1d'] = (history['close'] / history['close'].shift(1) - 1)
                history.loc[:, 'volatility_10d'] = history['return_1d'].rolling(window=10, min_periods=10).std()
                history.loc[:, 'volatility_20d'] = history['return_1d'].rolling(window=20, min_periods=20).std()

                # RSI/MACD(가능하면 close만으로 계산)
                try:
                    import talib as ta
                    history.loc[:, 'rsi'] = ta.RSI(history['close'].values, timeperiod=14)
                    macd, macd_signal, macd_hist = ta.MACD(history['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
                    history.loc[:, 'macd'] = macd
                    history.loc[:, 'macd_signal'] = macd_signal
                    history.loc[:, 'macd_hist'] = macd_hist
                except Exception:
                    pass

                # 3) 다음 루프 입력 피처를 최신 마지막 행으로 교체
                current_features = history.iloc[[-1]][current_features.columns]
                
                # 디버그 로그: 스텝별 핵심 피처 및 예측값 변화 확인
                def _safe_val(df, col):
                    try:
                        return float(df[col].iloc[0]) if col in df.columns and pd.notna(df[col].iloc[0]) else np.nan
                    except Exception:
                        return np.nan
                dbg = {
                    'rtn': return_pred,
                    'prev_close': prev_close if prev_close is not None else np.nan,
                    'next_close': next_close if next_close is not None else np.nan,
                    'price_to_sma_20': _safe_val(history.iloc[[-1]], 'price_to_sma_20'),
                    'bb_position': _safe_val(history.iloc[[-1]], 'bb_position'),
                    'rsi': _safe_val(history.iloc[[-1]], 'rsi'),
                    'macd': _safe_val(history.iloc[[-1]], 'macd'),
                }
                print(f"[predict_next_returns][step={step+1}] rtn={dbg['rtn']:.10f} prev_close={dbg['prev_close']:.6f} next_close={dbg['next_close']:.6f} p2s20={dbg['price_to_sma_20']:.6f} bbpos={dbg['bb_position']:.6f} rsi={dbg['rsi']:.6f} macd={dbg['macd']:.6f}")

            else:
                # 히스토리 없이 운영: close가 있다면 단순 업데이트만 반영
                if 'close' in current_features.columns:
                    prev_close = float(current_features['close'].iloc[0])
                    next_close = prev_close * (1 + return_pred)
                    current_features.loc[:, 'close'] = next_close
                # 간단 파생만 유지
                if 'return_1d' in current_features.columns:
                    current_features.loc[:, 'return_1d'] = return_pred
            
                print(f"[predict_next_returns][step={step+1}] rtn={return_pred:.10f} (no-history mode)")
                
        return predictions
    
    def convert_returns_to_prices(self, base_price, returns):
        """수익률을 가격으로 변환"""
        prices = []
        current_price = base_price
        print(f"current_price = {current_price}")
        for return_rate in returns:
            current_price = current_price * (1 + return_rate)
            prices.append(current_price)
        
        return prices
