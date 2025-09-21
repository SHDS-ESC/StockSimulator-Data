"""캐시 관리 컨트롤러"""

from fastapi import APIRouter, HTTPException, Depends

from ..models.dto import MessageResponse, CacheInfoResponse, SchedulerStatusResponse
from ..services.scheduler_service import SchedulerService

router = APIRouter(prefix="/cache", tags=["cache"])


def get_db_service():
    """데이터베이스 서비스 의존성"""
    from ..main import get_db_service
    return get_db_service()


def get_scheduler_service():
    """스케줄러 서비스 의존성"""
    from ..main import get_scheduler_service
    return get_scheduler_service()


@router.get("/info", response_model=CacheInfoResponse)
def get_cache_info(db_service = Depends(get_db_service)):
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


@router.delete("/", response_model=MessageResponse)
def clear_cache(ticker: str = None, db_service = Depends(get_db_service)):
    """캐시 클리어"""
    try:
        if db_service is None:
            raise HTTPException(
                status_code=503, 
                detail="데이터베이스 연결이 설정되지 않았습니다."
            )
        
        db_service.clear_cache(ticker)
        
        if ticker:
            return MessageResponse(message=f"{ticker} 캐시가 클리어되었습니다.")
        else:
            return MessageResponse(message="모든 캐시가 클리어되었습니다.")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scheduler/status", response_model=SchedulerStatusResponse)
def get_scheduler_status(scheduler_service: SchedulerService = Depends(get_scheduler_service)):
    """스케줄러 상태 조회"""
    return SchedulerStatusResponse(**scheduler_service.get_status())


@router.post("/scheduler/trigger-cache-refresh", response_model=MessageResponse)
def trigger_cache_refresh(scheduler_service: SchedulerService = Depends(get_scheduler_service)):
    """수동으로 캐시 새로고침 실행"""
    try:
        scheduler_service.daily_cache_refresh()
        return MessageResponse(message="캐시 새로고침이 실행되었습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"캐시 새로고침 실행 실패: {str(e)}")


@router.post("/scheduler/trigger-weekly-cleanup", response_model=MessageResponse)
def trigger_weekly_cleanup(scheduler_service: SchedulerService = Depends(get_scheduler_service)):
    """수동으로 주간 정리 실행"""
    try:
        scheduler_service.weekly_cleanup()
        return MessageResponse(message="주간 정리가 실행되었습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"주간 정리 실행 실패: {str(e)}")


@router.post("/scheduler/trigger-market-update", response_model=MessageResponse)
def trigger_market_update(scheduler_service: SchedulerService = Depends(get_scheduler_service)):
    """수동으로 시장 데이터 업데이트 실행"""
    try:
        scheduler_service.market_data_update()
        return MessageResponse(message="시장 데이터 업데이트가 실행되었습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시장 데이터 업데이트 실행 실패: {str(e)}")


@router.post("/scheduler/force-run/{task_name}", response_model=MessageResponse)
def force_run_task(
    task_name: str,
    scheduler_service: SchedulerService = Depends(get_scheduler_service)
):
    """작업 강제 실행"""
    try:
        scheduler_service.force_run_task(task_name)
        return MessageResponse(message=f"{task_name} 작업이 강제로 실행되었습니다.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"작업 강제 실행 실패: {str(e)}")


@router.get("/scheduler/manual-runs")
def get_manual_runs(scheduler_service: SchedulerService = Depends(get_scheduler_service)):
    """수동 실행 시간 조회 (메모리 기반, 서버 재시작 시 초기화)"""
    try:
        status = scheduler_service.get_status()
        return {
            "manual_runs": status.get("manual_runs", {}),
            "message": "수동 실행 시간 정보 (서버 재시작 시 초기화됨)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"수동 실행 시간 조회 실패: {str(e)}")
