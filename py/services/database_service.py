"""데이터베이스 서비스"""

import pandas as pd
import numpy as np
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
        if ticker in self._stock_data_cache:
            print(f"💾 {ticker} 데이터 캐시에서 로드")
            return self._stock_data_cache[ticker].copy()
        
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
        
        # report_date를 인덱스로 설정
        if 'report_date' in data.columns:
            data.set_index('report_date', inplace=True)
            data.index = pd.to_datetime(data.index)  # datetime으로 변환
        
        # 캐시에 저장
        self._stock_data_cache[ticker] = data.copy()
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
