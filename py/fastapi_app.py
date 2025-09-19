from typing import Optional, List
from datetime import date, timedelta
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from py.stock_predictor import DatabaseService, predict_stock


app = FastAPI(title="StockSimulator API", version="0.1.0")

# CORS (로컬/노트북/브라우저 테스트용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터베이스 설정
db_config = {
    'db_name': os.getenv('DB_NAME', 'fint'),
    'db_id': os.getenv('DB_ID', 'esc-sangyoon'),
    'db_pwd': os.getenv('DB_PWD', 'teamesc'),
    'db_host': os.getenv('DB_HOST', 'localhost')
}

# 데이터베이스 서비스 초기화
try:
    db_service = DatabaseService(db_config)
    print("✅ 데이터베이스 연결 성공")
except Exception as e:
    print(f"❌ 데이터베이스 연결 실패: {e}")
    db_service = None


class PredictRequest(BaseModel):
    ticker: str = Field(..., description="예상할 티커 (예: NVDA)")
    train_days: int = Field(300, ge=30, le=2000, description="학습에 사용할 과거 일수")
    predict_steps: int = Field(5, ge=1, le=30, description="예측 일수")
    today: Optional[date] = Field(None, description="기준일 (기본: 오늘)")


class PredictPoint(BaseModel):
    day: date  # int에서 date로 변경
    date: date
    return_rate: float
    price: float


class PredictResponse(BaseModel):
    ticker: str
    base_date: date
    last_price: float
    train_data_count: int
    feature_count: int
    predicted: List[PredictPoint]
    chart_full: Optional[str] = Field(None, description="전체 데이터 차트 (base64)")
    chart_30d: Optional[str] = Field(None, description="최근 30일 차트 (base64)")


@app.get("/health")
def health_check():
    """헬스 체크 엔드포인트"""
    db_status = "connected" if db_service is not None else "disconnected"
    return {
        "status": "ok",
        "database": db_status
    }


@app.get("/tickers")
def get_available_tickers():
    """사용 가능한 티커 목록 조회"""
    try:
        if db_service is None:
            raise HTTPException(
                status_code=503, 
                detail="데이터베이스 연결이 설정되지 않았습니다."
            )
        
        tickers = db_service.stock_df['ticker'].tolist()
        return {
            "tickers": tickers,
            "count": len(tickers)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/info")
def get_cache_info():
    """캐시 정보 조회"""
    try:
        if db_service is None:
            raise HTTPException(
                status_code=503, 
                detail="데이터베이스 연결이 설정되지 않았습니다."
            )
        
        return db_service.get_cache_info()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
def clear_cache(ticker: Optional[str] = None):
    """캐시 클리어"""
    try:
        if db_service is None:
            raise HTTPException(
                status_code=503, 
                detail="데이터베이스 연결이 설정되지 않았습니다."
            )
        
        db_service.clear_cache(ticker)
        
        if ticker:
            return {"message": f"{ticker} 캐시가 클리어되었습니다."}
        else:
            return {"message": "모든 캐시가 클리어되었습니다."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    print("request : ", req)

    """
    주식 예측 엔드포인트:
    - 노트북에서 검증된 StockPredictor 모델 사용
    - 데이터베이스에서 실제 주식 데이터 로드
    - 기술분석 지표 및 머신러닝 모델로 예측
    """
    try:
        # 데이터베이스 연결 확인
        if db_service is None:
            raise HTTPException(
                status_code=503, 
                detail="데이터베이스 연결이 설정되지 않았습니다."
            )
        
        # today 기본값 처리
        base_day = req.today or date.today()
        
        # 주식 예측 실행
        result = predict_stock(
            db_service=db_service,
            ticker=req.ticker,
            train_days=req.train_days,
            predict_steps=req.predict_steps,
            today=base_day            
        )
        
        # 예측 결과를 API 응답 형식으로 변환
        predicted_points = []
        prediction_dates = result['prediction_dates']
        for i, (return_rate, price, pred_date) in enumerate(zip(
            result['return_predictions'], 
            result['price_predictions'], 
            prediction_dates
        )):
            predicted_points.append(PredictPoint(
                day=pred_date,  # 계산된 실제 날짜 사용
                date=pred_date,  # 동일한 날짜
                return_rate=return_rate,
                price=price
            ))
        
        print('req, result : ', req, result)
        
        return PredictResponse(
            ticker=result['ticker'],
            base_date=result['base_date'],
            last_price=result['last_price'],
            train_data_count=result['train_data_count'],
            feature_count=result['feature_count'],
            predicted=predicted_points,
            chart_full=result.get('chart_full'),
            chart_30d=result.get('chart_30d')
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류가 발생했습니다: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "py.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )