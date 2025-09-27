"""FastAPI 메인 애플리케이션 - 라우트 관리"""

import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config.settings import settings
from .controllers import health_controller, scheduler_controller, stock_controller
from .services.scheduler_service import SchedulerService
from .services.database_service import DatabaseService

# 로깅 설정
logger = settings.setup_logging()

# FastAPI 앱 생성
app = FastAPI(title="StockSimulator API", version="0.1.0")

# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"📥 요청: {request.method} {request.url}")
    
    response = await call_next(request)
    
    logger.info(f"📤 응답: {response.status_code}")
    return response

# 전역 예외 핸들러
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Request validation error: {exc.errors()}")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")
    
    # 요청 바디 로깅
    try:
        body = await request.body()
        if body:
            import json
            try:
                body_json = json.loads(body.decode('utf-8'))
                logger.error(f"📥 실패한 요청 바디: {body_json}")
            except json.JSONDecodeError:
                logger.error(f"📥 실패한 요청 바디 (텍스트): {body.decode('utf-8')[:500]}...")
        else:
            logger.error("📥 실패한 요청 바디: (비어있음)")
    except Exception as e:
        logger.error(f"📥 요청 바디 읽기 실패: {e}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "요청 데이터 검증 실패",
            "errors": exc.errors(),
            "request_url": str(request.url)
        }
    )

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

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 서비스 인스턴스
db_service = None
scheduler_service = SchedulerService()


def initialize_services():
    """서비스 초기화"""
    global db_service
    
    # 데이터베이스 서비스 초기화
    try:
        db_service = DatabaseService(settings.db_config)
        scheduler_service.set_db_service(db_service)
        logger.info("✅ 데이터베이스 연결 성공")
    except Exception as e:
        logger.error(f"❌ 데이터베이스 연결 실패: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        db_service = None
    
    # 스케줄러 시작
    scheduler_service.start()


# 의존성 주입을 위한 함수들
def get_db_service():
    return db_service


def get_scheduler_service():
    return scheduler_service


# 라우터 등록 - 모든 라우트 관리
app.include_router(health_controller.router)
app.include_router(scheduler_controller.router)
app.include_router(stock_controller.router)

# # 등록된 라우트 로깅
# logger.info("📋 등록된 API 라우트:")
# for route in app.routes:
#     if hasattr(route, 'path') and hasattr(route, 'methods'):
#         methods = ', '.join(route.methods)
#         logger.info(f"  {methods} {route.path}")


# 앱 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행"""
    logger.info("🚀 StockSimulator API 시작 중...")
    initialize_services()


# 앱 종료 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행"""
    logger.info("⏹️ StockSimulator API 종료 중...")
    scheduler_service.stop()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "py.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True
    )