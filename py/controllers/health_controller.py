"""헬스 체크 컨트롤러"""

from fastapi import APIRouter, Depends

from ..models.dto import HealthResponse
from ..services.scheduler_service import SchedulerService

router = APIRouter(prefix="/health", tags=["health"])


def get_db_service():
    """데이터베이스 서비스 의존성"""
    from ..main import get_db_service
    return get_db_service()


def get_scheduler_service():
    """스케줄러 서비스 의존성"""
    from ..main import get_scheduler_service
    return get_scheduler_service()


@router.get("/", response_model=HealthResponse)
def health_check(
    db_service = Depends(get_db_service),
    scheduler_service: SchedulerService = Depends(get_scheduler_service)
):
    """헬스 체크 엔드포인트"""
    db_status = "connected" if db_service is not None else "disconnected"
    scheduler_status = "running" if scheduler_service and scheduler_service.is_running else "stopped"
    scheduled_jobs = len(scheduler_service.get_status()["jobs"]) if scheduler_service else 0
    
    return HealthResponse(
        status="ok",
        database=db_status,
        scheduler=scheduler_status,
        scheduled_jobs=scheduled_jobs
    )
