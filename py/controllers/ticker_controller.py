"""티커 관리 컨트롤러"""

from fastapi import APIRouter, HTTPException, Depends

from ..models.dto import TickersResponse

router = APIRouter(prefix="/tickers", tags=["tickers"])


def get_db_service():
    """데이터베이스 서비스 의존성"""
    from ..main import get_db_service
    return get_db_service()


@router.get("/", response_model=TickersResponse)
def get_available_tickers(db_service = Depends(get_db_service)):
    """사용 가능한 티커 목록 조회"""
    try:
        if db_service is None:
            raise HTTPException(
                status_code=503, 
                detail="데이터베이스 연결이 설정되지 않았습니다."
            )
        
        tickers = db_service.stock_df['ticker'].tolist()
        return TickersResponse(
            tickers=tickers,
            count=len(tickers)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
