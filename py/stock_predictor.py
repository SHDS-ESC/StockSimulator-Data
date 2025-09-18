# stock_predictor.py
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# 기술분석 라이브러리
import talib as ta

# 데이터베이스 라이브러리
from sqlalchemy import create_engine
import mysql.connector


class DatabaseService:
    """데이터베이스 연결 및 데이터 로드 서비스"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.db_url = f"mysql+mysqlconnector://{db_config['db_id']}:{db_config['db_pwd']}@{db_config['db_host']}/{db_config['db_name']}"
        self.engine = create_engine(self.db_url)
        self._load_base_data()
    
    def _load_base_data(self):
        """기본 데이터 로드"""
        self.report_df = self.load_data_from_db('report')
        self.stock_df = self.load_data_from_db('stock')
    
    def load_data_from_db(self, table_name, query=None):
        """데이터베이스에서 데이터 로드"""
        try:
            if query:
                df = pd.read_sql(sql=query, con=self.engine)
            else:
                df = pd.read_sql_table(table_name=table_name, con=self.engine)
            return df
        except Exception as e:
            print(f"❌ 데이터 로드 오류: {e}")
            return pd.DataFrame()
    
    def get_stock_id(self, ticker):
        """티커로 stock_id 조회"""
        result = self.stock_df[self.stock_df['ticker'] == ticker]['stock_id'].values
        if len(result) == 0:
            raise ValueError(f"티커 '{ticker}'를 찾을 수 없습니다.")
        return result[0]
    
    def get_stock_data(self, ticker):
        """특정 티커의 주식 데이터 조회"""
        stock_id = self.get_stock_id(ticker)
        data = self.report_df[self.report_df['stock_id'] == stock_id].copy()
        
        # 컬럼명 정리 (SQLAlchemy quoted_name 문제 해결)
        new_columns = []
        for col in data.columns:
            if hasattr(col, 'name'):
                new_columns.append(str(col.name))
            else:
                new_columns.append(str(col))
        data.columns = new_columns
        
        data.volume = data.volume.values.astype(np.float64)
        return data


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
        
        # 기본 지표들
        df_temp['rsi'] = ta.RSI(df_temp['close'].values, timeperiod=14)
        df_temp['macd'], df_temp['macd_signal'], df_temp['macd_hist'] = ta.MACD(
            df_temp['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        df_temp['atr'] = ta.ATR(df_temp['high'].values, df_temp['low'].values, 
                               df_temp['close'].values, timeperiod=14)
        df_temp['stoch_k'], df_temp['stoch_d'] = ta.STOCH(
            df_temp['high'].values, df_temp['low'].values, df_temp['close'].values,
            fastk_period=5, slowk_period=3, slowd_period=3)
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


def predict_stock(db_service, ticker, train_days=500, predict_steps=5, today=None):
    """주식 예측 실행 함수"""
    
    if today is None:
        today = dt.date.today()
    
    # 1. 데이터 로드
    stock_data = db_service.get_stock_data(ticker)
    stock_data.set_index('report_date', inplace=True)
    
    # 2. 모델 초기화
    predictor = StockPredictor()
    
    # 3. 피처 준비
    prepared_data = predictor.prepare_features(stock_data)
    
    # 4. 날짜 필터링
    end_date = today + timedelta(days=predict_steps)
    start_date = today - timedelta(days=train_days)
    
    date_mask = (prepared_data.index >= pd.to_datetime(start_date)) & \
                (prepared_data.index <= pd.to_datetime(end_date))
    filtered_data = prepared_data[date_mask]
    
    # 5. 훈련 데이터 선택
    split_idx = int(len(filtered_data) * 1)
    train_data = filtered_data.iloc[:split_idx]
    
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
    
    return {
        'ticker': ticker,
        'base_date': today,
        'last_price': float(last_price),
        'return_predictions': return_predictions,
        'price_predictions': price_predictions,
        'train_data_count': len(X_train),
        'feature_count': len(feature_cols)
    }
