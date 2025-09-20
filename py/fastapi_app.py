from typing import Optional, List
from datetime import date, timedelta
import os
import logging
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from py.stock_predictor import DatabaseService, predict_stock

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


app = FastAPI(title="StockSimulator API", version="0.1.0")

# 전역 예외 핸들러 추가
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {exc}")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"내부 서버 오류: {str(exc)}",
            "type": type(exc).__name__,
            "request_url": str(request.url)
        }
    )

# CORS (로컬/노트북/브라우저 테스트용)
app.add_middleware(
    CORSMiddleware,
    # 모든 도메인에서의 요청 허용
    allow_origins=["*"],
    # 인증 정보 포함 허용
    allow_credentials=True,
    # 모든 HTTP 메서드 허용
    allow_methods=["*"],
    # 모든 헤더 허용
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
    logger.info("✅ 데이터베이스 연결 성공")
except Exception as e:
    logger.error(f"❌ 데이터베이스 연결 실패: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    db_service = None


class PredictRequest(BaseModel):
    ticker: str = Field(..., description="예상할 티커 (예: NVDA)")
    train_days: int = Field(300, ge=30, le=2000, description="학습에 사용할 과거 일수")
    predict_steps: int = Field(5, ge=1, le=30, description="예측 일수")
    today: Optional[date] = Field(None, description="기준일 (기본: 오늘)")


class PredictPoint(BaseModel):
    day: int
    date: date
    return_rate: float
    price: float


class InvestmentMetrics(BaseModel):
    current_price: float
    predicted_avg_price: float
    predicted_max_price: float
    predicted_min_price: float
    expected_total_return: float
    expected_avg_daily_return: float
    predicted_volatility: float
    upside_probability: float


class RiskMetrics(BaseModel):
    historical_volatility_annualized: float
    predicted_volatility: float
    var_95: float
    max_expected_loss: float
    max_expected_gain: float
    estimated_sharpe_ratio: float


class InvestmentAnalysis(BaseModel):
    recommendation: str
    action: str  # BUY, SELL, HOLD
    confidence: str  # HIGH, MEDIUM, LOW
    score: int
    max_score: int
    min_score: int
    signals: List[str]
    metrics: InvestmentMetrics
    risk_metrics: RiskMetrics


class PredictResponse(BaseModel):
    ticker: str
    base_date: date
    last_price: float
    train_data_count: int
    feature_count: int
    predicted: List[PredictPoint]
    investment_analysis: InvestmentAnalysis
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
    logger.info(f"예측 요청 수신: {req}")
    
    # 요청 세부 정보 로깅
    logger.info(f"Ticker: {req.ticker}, Train days: {req.train_days}, Predict steps: {req.predict_steps}")

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
                day=i+1,
                date=pred_date,
                return_rate=return_rate,
                price=price
            ))
        
        logger.info(f"예측 완료 - Ticker: {result['ticker']}, Last price: {result['last_price']}")
        logger.debug(f"예측 결과 상세: {result}")
        
        # 투자 분석 결과를 Pydantic 모델로 변환
        analysis_data = result['investment_analysis']
        investment_analysis = InvestmentAnalysis(
            recommendation=analysis_data['recommendation'],
            action=analysis_data['action'],
            confidence=analysis_data['confidence'],
            score=analysis_data['score'],
            max_score=analysis_data['max_score'],
            min_score=analysis_data['min_score'],
            signals=analysis_data['signals'],
            metrics=InvestmentMetrics(**analysis_data['metrics']),
            risk_metrics=RiskMetrics(**analysis_data['risk_metrics'])
        )
        
        return PredictResponse(
            ticker=result['ticker'],
            base_date=result['base_date'],
            last_price=result['last_price'],
            train_data_count=result['train_data_count'],
            feature_count=result['feature_count'],
            predicted=predicted_points,
            investment_analysis=investment_analysis,
            chart_full=result.get('chart_full'),
            chart_30d=result.get('chart_30d')
        )
        
    except ValueError as e:
        logger.warning(f"잘못된 요청 파라미터: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"예측 중 예상치 못한 오류 발생: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류가 발생했습니다: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "py.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )