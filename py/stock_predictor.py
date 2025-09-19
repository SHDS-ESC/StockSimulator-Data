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
        
        # 데이터베이스 연결 풀링 및 최적화 설정
        self.engine = create_engine(
            self.db_url,
            pool_size=5,              # 연결 풀 크기
            max_overflow=10,          # 추가 연결 허용
            pool_timeout=30,          # 연결 대기 시간
            pool_recycle=3600,        # 연결 재활용 시간 (1시간)
            echo=False                # SQL 로그 비활성화
        )
        
        # 작은 stock 테이블만 미리 로드 (712개 행)
        print("📋 Stock 테이블 로드 중...")
        self.stock_df = self.load_data_from_db('stock')
        print(f"✅ Stock 데이터 로드 완료: {self.stock_df.shape}")
        
        # report 테이블은 필요할 때만 로드하도록 변경
        self.report_df = None  # 지연 로딩
        self._stock_data_cache = {}  # 티커별 데이터 캐시
    
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
        """특정 티커의 주식 데이터 조회 (캐싱 포함)"""
        # 캐시에서 먼저 확인
        # if ticker in self._stock_data_cache:
        #     print(f"💾 {ticker} 데이터 캐시에서 로드")
        #     return self._stock_data_cache[ticker]
        
        print(f"📊 {ticker} 데이터 데이터베이스에서 로드 중...")
        stock_id = self.get_stock_id(ticker)
        
        # 특정 stock_id에 대한 데이터만 쿼리로 로드 (메모리 효율적)
        query = f"""
        SELECT * FROM report 
        WHERE stock_id = {stock_id}
        ORDER BY report_date
        """
        
        data = self.load_data_from_db('report', query)
        
        if data.empty:
            raise ValueError(f"티커 '{ticker}'에 대한 데이터를 찾을 수 없습니다.")
        
        # 컬럼명 정리 (SQLAlchemy quoted_name 문제 해결)
        new_columns = []
        for col in data.columns:
            if hasattr(col, 'name'):
                new_columns.append(str(col.name))
            else:
                new_columns.append(str(col))
        data.columns = new_columns
        
        data.volume = data.volume.values.astype(np.float64)
        
        # 캐시에 저장
        self._stock_data_cache[ticker] = data
        print(f"✅ {ticker} 데이터 로드 완료: {data.shape}")
        
        return data
    
    def clear_cache(self, ticker=None):
        """캐시 클리어"""
        if ticker:
            if ticker in self._stock_data_cache:
                del self._stock_data_cache[ticker]
                print(f"🗑️ {ticker} 캐시 클리어됨")
        else:
            self._stock_data_cache.clear()
            print("🗑️ 모든 캐시 클리어됨")
    
    def get_cache_info(self):
        """캐시 정보 조회"""
        return {
            "cached_tickers": list(self._stock_data_cache.keys()),
            "cache_count": len(self._stock_data_cache)
        }


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


def predict_stock(db_service, ticker, train_days=500, predict_steps=5, today=None, save_image=True):
    """주식 예측 실행 함수"""
    
    if today is None:
        today = dt.date.today()
    
    # 1. 데이터 로드
    stock_data = db_service.get_stock_data(ticker)
    stock_data.set_index('report_date', inplace=True)
    
    # 인덱스를 datetime으로 변환 (SQL에서 문자열로 가져오는 경우 대비)
    stock_data.index = pd.to_datetime(stock_data.index)
    
    # 2. 모델 초기화
    predictor = StockPredictor()
    
    # 3. 피처 준비
    prepared_data = predictor.prepare_features(stock_data)
    
    # 4. 날짜 필터링
    # end_date = today + timedelta(days=predict_steps)
    today_ts = pd.to_datetime(today)
    today_idx = (abs(stock_data.index - today_ts)).argmin()
    target_idx = today_idx + predict_steps
    if target_idx < len(stock_data.index):
        end_date = stock_data.index[target_idx]
    else:
        부족한_일수 = target_idx - len(stock_data.index) + 1
        end_date = stock_data.index[-1] + timedelta(days=부족한_일수)
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
    pred_dates = []
    today_ts = pd.Timestamp(today)
    close_data = filtered_data['close']
    
    # close_data에서 today 이후의 날짜들을 찾기
    future_dates_in_data = close_data.index[close_data.index > today_ts]
    
    if len(future_dates_in_data) >= len(price_predictions):
        # 충분한 미래 날짜가 데이터에 있는 경우 (과거 예측)
        pred_dates = future_dates_in_data[:len(price_predictions)].tolist()
        print(f"🔍 과거 예측 모드: close_data의 실제 날짜 사용")
    else:
        # 미래 예측의 경우: 기존 패턴 추종 후 단순 증가
        if len(future_dates_in_data) > 0:
            # 일부는 실제 날짜, 나머지는 추정
            pred_dates.extend(future_dates_in_data.tolist())
            last_date = future_dates_in_data[-1]
            remaining_days = len(price_predictions) - len(future_dates_in_data)
        else:
            # 완전 미래 예측: today부터 시작
            last_date = today_ts
            remaining_days = len(price_predictions)
        
        # 나머지 날짜들을 영업일 기준으로 생성
        current_date = last_date
        for i in range(remaining_days):
            current_date += timedelta(days=1)
            # 주말 건너뛰기 (간단한 영업일 계산)
            while current_date.weekday() >= 5:  # 5=토요일, 6=일요일
                current_date += timedelta(days=1)
            pred_dates.append(current_date)
        
        print(f"🔍 미래 예측 모드: 실제 {len(future_dates_in_data)}개 + 추정 {remaining_days}개")
    
    # 11. 차트 생성
    chart_data = None
    if save_image:
        chart_data = create_prediction_charts(ticker, today, price_predictions, filtered_data['close'], pred_dates)
    
    result = {
        'ticker': ticker,
        'base_date': today,
        'last_price': float(last_price),
        'return_predictions': return_predictions,
        'price_predictions': price_predictions,
        'prediction_dates': [date.date() if hasattr(date, 'date') else date for date in pred_dates],  # date 객체로 변환
        'train_data_count': len(X_train),
        'feature_count': len(feature_cols)
    }
    
    # 차트 데이터 추가 (있는 경우)
    if chart_data:
        result.update(chart_data)
    
    return result

def create_prediction_charts(ticker, today, price_predictions, close_prices, pred_dates):
    import matplotlib
    matplotlib.use('Agg')  # GUI 백엔드 비활성화 (서버용)
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import base64
    from io import BytesIO
    
    print('close_prices', close_prices.tail())
    
    try:
        # 한글 폰트 설정 (에러 시 무시)
        try:
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
        except:
            print("⚠️ 한글 폰트 설정 실패, 기본 폰트 사용")

        fig, ax = plt.subplots(figsize=(10, 6))
        
        print(f"🔍 pred_dates 타입: {type(pred_dates[0])}")
        print(f"🔍 close_prices.index 타입: {type(close_prices.index[0])}")
        print(f"🔍 예측 날짜들: {pred_dates}")

        # 1. 전체 데이터 차트
        # 예측 가격
        ax.plot(pred_dates, price_predictions, 'r--',
                label='예측 가격', linewidth=2, marker='o', markersize=6)
        
        # 실제 (학습) 날짜, 가격 (close_prices는 Series)
        ax.plot(close_prices.index, close_prices.values, 'b-',
                label='실제 가격', linewidth=2, marker='.')

        # 제목과 라벨 설정
        ax.set_title(f'{ticker} 주가 예측 (전체 데이터)')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # x축 날짜 포맷팅
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # 레이아웃 조정 및 저장
        plt.tight_layout()
        
        # 1. 파일로 저장
        plt.savefig(f'{ticker}_prediction.png', dpi=100, bbox_inches='tight')
        
        # 2. base64로 인코딩
        buffer_full = BytesIO()
        plt.savefig(buffer_full, format='png', dpi=100, bbox_inches='tight')
        buffer_full.seek(0)
        chart_full_base64 = base64.b64encode(buffer_full.getvalue()).decode('utf-8')
        buffer_full.close()
        
        plt.close(fig)  # 메모리 해제

        print(f"✅ 전체 차트가 '{ticker}_prediction.png' 파일로 저장되었습니다.")
        
        # 3. 최근 30일 차트
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # 최근 30일 데이터만 선택
        recent_30_prices = close_prices.tail(30)
        
        # 예측 가격 (동일)
        ax2.plot(pred_dates, price_predictions, 'r--',
                label='예측 가격', linewidth=2, marker='o', markersize=6)
        
        # 최근 30일 실제 가격
        ax2.plot(recent_30_prices.index, recent_30_prices.values, 'b-',
                label='실제 가격 (최근 30일)', linewidth=2, marker='.')

        # 제목과 라벨 설정
        ax2.set_title(f'{ticker} 주가 예측 (최근 30일)')
        ax2.set_xlabel('날짜')
        ax2.set_ylabel('가격($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # x축 날짜 포맷팅
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # 간격 조정
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 레이아웃 조정 및 저장
        plt.tight_layout()
        
        # 1. 파일로 저장
        plt.savefig(f'{ticker}_prediction_30d.png', dpi=100, bbox_inches='tight')
        
        # 2. base64로 인코딩
        buffer_30d = BytesIO()
        plt.savefig(buffer_30d, format='png', dpi=100, bbox_inches='tight')
        buffer_30d.seek(0)
        chart_30d_base64 = base64.b64encode(buffer_30d.getvalue()).decode('utf-8')
        buffer_30d.close()
        
        plt.close(fig2)  # 메모리 해제

        print(f"✅ 최근 30일 차트가 '{ticker}_prediction_30d.png' 파일로 저장되었습니다.")
        
        # base64 인코딩된 차트 데이터 반환
        return {
            'chart_full': chart_full_base64,
            'chart_30d': chart_30d_base64
        }
        
    except Exception as e:
        print(f"❌ 차트 생성 에러: {e}")
        import traceback
        traceback.print_exc()
        # 에러가 발생해도 함수는 계속 진행
        return None