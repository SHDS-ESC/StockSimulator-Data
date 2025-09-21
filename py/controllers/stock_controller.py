"""주식 예측 컨트롤러"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
import datetime as dt

from ..models.dto import MessageResponse, StockPredictionResponse
from ..services.database_service import DatabaseService
from ..services.stock_service import StockService

router = APIRouter(prefix="/stock", tags=["stock"])


def get_db_service():
    """데이터베이스 서비스 의존성"""
    from ..main import get_db_service
    return get_db_service()


def get_stock_service(db_service: DatabaseService = Depends(get_db_service)):
    """주식 서비스 의존성"""
    if db_service is None:
        raise HTTPException(
            status_code=503, 
            detail="데이터베이스 연결이 설정되지 않았습니다."
        )
    return StockService(db_service)


@router.post("/predict/{ticker}", response_model=StockPredictionResponse)
def predict_stock(
    ticker: str,
    train_days: int = Query(500, description="훈련 데이터 기간 (일)"),
    predict_steps: int = Query(5, description="예측 기간 (일)"),
    today: Optional[dt.date] = Query(None, description="기준일 (기본: 오늘, 입력 형식 : yyyy-mm-dd)"),
    save_image: bool = Query(True, description="차트 이미지 생성 여부"),
    stock_service: StockService = Depends(get_stock_service)
):
    """주식 가격 예측"""
    try:
        result = stock_service.predict_stock(
            ticker=ticker,
            train_days=train_days,
            predict_steps=predict_steps,
            today=today,
            save_image=save_image
        )
        
        return StockPredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")


@router.get("/available-tickers")
def get_available_tickers(db_service: DatabaseService = Depends(get_db_service)):
    """사용 가능한 티커 목록 조회"""
    try:
        if db_service is None:
            raise HTTPException(
                status_code=503, 
                detail="데이터베이스 연결이 설정되지 않았습니다."
            )
        
        # stock 테이블에서 티커 목록 조회
        tickers = db_service.stock_df['ticker'].tolist()
        
        return {
            "tickers": sorted(tickers),
            "count": len(tickers)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"티커 목록 조회 실패: {str(e)}")


@router.get("/data/{ticker}")
def get_stock_data(ticker: str, db_service: DatabaseService = Depends(get_db_service)):
    """특정 티커의 주식 데이터 조회"""
    try:
        if db_service is None:
            raise HTTPException(
                status_code=503, 
                detail="데이터베이스 연결이 설정되지 않았습니다."
            )
        
        
        data = db_service.get_stock_data(ticker)
        # 데이터를 JSON 직렬화 가능한 형태로 변환
        result = {
            "ticker": ticker,
            "data_count": len(data),
            "date_range": {
                "start": data.index.min().isoformat(),
                "end": data.index.max().isoformat()
            },
            "columns": data.columns.tolist(),
            "sample_data": data.tail(5).to_dict('records')  # 최근 5개 데이터
        }
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 조회 실패: {str(e)}")
