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


class StockPredictor:
    """주식 가격 예측 모델"""
    
    def __init__(self, model=None, use_scaler=True, random_state=42):
        self.model = model if model else RandomForestRegressor(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
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
        
        # 수익률 피처
        for period in [1, 3, 5, 10]:
            df_temp[f'return_{period}d'] = (df_temp['close'] / df_temp['close'].shift(period) - 1) * 100
        
        # 변동성
        for period in [10, 20]:
            df_temp[f'volatility_{period}d'] = df_temp[f'return_1d'].rolling(window=period).std()
        
        # 거래량 분석
        df_temp['volume_sma_10'] = df_temp['volume'].rolling(window=10).mean()
        df_temp['volume_ratio'] = df_temp['volume'] / df_temp['volume_sma_10']
        
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
    
    def prepare_features(self, df):
        """피처 준비"""
        df_temp = df.copy()
        
        # 컬럼명을 소문자로 변환
        df_temp.columns = df_temp.columns.str.lower()
        
        df_temp = self.calculate_technical_indicators(df_temp)
        df_temp = self.add_engineered_features(df_temp)
        
        # 타겟 변수 생성 (다음날 수익률)
        df_temp['target'] = (df_temp['close'].shift(-1) / df_temp['close'] - 1) * 100
        
        # NaN 제거
        df_temp = df_temp.dropna()
        
        return df_temp
    
    def get_feature_columns(self, df):
        """피처 컬럼 선택"""
        exclude_cols = ['report_id', 'report_date', 'stock_id', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def train(self, X_train, y_train):
        """모델 학습"""
        # 피처 컬럼 저장
        self.feature_columns = X_train.columns.tolist()
        
        # 교차검증
        cv_mean, cv_std = self.time_series_split_validate(X_train, y_train)
        
        # 스케일링
        if self.use_scaler:
            X_train_scaled = self.scaler.fit_transform(X_train.values)
        else:
            X_train_scaled = X_train.values
        
        # 최종 모델 학습
        self.model.fit(X_train_scaled, y_train)
        
        return self
    
    def predict_next_returns(self, X_last, steps=5):
        """미래 수익률 예측"""
        if self.feature_columns is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        predictions = []
        current_features = X_last.copy()
        
        for step in range(steps):
            # 현재 피처로 수익률 예측
            if self.use_scaler:
                features_scaled = self.scaler.transform(current_features[self.feature_columns].values)
            else:
                features_scaled = current_features[self.feature_columns].values
            
            return_pred = self.model.predict(features_scaled)[0]
            predictions.append(return_pred)
            
            # 다음 스텝을 위한 피처 업데이트 (간단한 방식)
            current_features = current_features.copy()
            
        return predictions
    
    def convert_returns_to_prices(self, base_price, returns):
        """수익률을 가격으로 변환"""
        prices = []
        current_price = base_price
        
        for return_rate in returns:
            current_price = current_price * (1 + return_rate / 100)
            prices.append(current_price)
        
        return prices
