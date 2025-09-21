"""스케줄러 서비스"""

import logging
import traceback
import atexit
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

logger = logging.getLogger(__name__)


class SchedulerService:
    """스케줄러 서비스 클래스"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self._db_service = None
        # 간단한 메모리 기반 마지막 실행 시간 추적 (수동 실행용)
        self._manual_run_times = {
            "daily_cache_refresh": None,
            "weekly_cleanup": None,
            "market_data_update": None
        }
    
    def set_db_service(self, db_service):
        """데이터베이스 서비스 설정"""
        self._db_service = db_service
    
    def _should_run_manual_task(self, task_name: str, min_interval_minutes: int = 30) -> bool:
        """수동 작업 실행 여부 확인 (메모리 기반, 서버 재시작 시 초기화)"""
        last_run = self._manual_run_times.get(task_name)
        if not last_run:
            return True
        
        time_diff = datetime.now() - last_run
        return time_diff >= timedelta(minutes=min_interval_minutes)
    
    def _update_manual_run_time(self, task_name: str):
        """수동 실행 시간 업데이트"""
        self._manual_run_times[task_name] = datetime.now()
    
    def daily_cache_refresh(self, force: bool = False):
        """일일 캐시 새로고침 작업"""
        try:
            logger.info("🔄 일일 캐시 새로고침 시작")
            if self._db_service:
                # 모든 캐시 클리어
                self._db_service.clear_cache()
                logger.info("✅ 캐시 새로고침 완료")
            else:
                logger.error("❌ 데이터베이스 서비스가 없어 캐시 새로고침 실패")
        except Exception as e:
            logger.error(f"❌ 캐시 새로고침 중 오류: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def weekly_cleanup(self, force: bool = False):
        """주간 정리 작업"""
        try:
            logger.info("🧹 주간 정리 작업 시작")
            # TODO : 여기에 주간 정리 로직 추가
            # 예: 오래된 로그 파일 삭제, 임시 파일 정리 등
            logger.info("✅ 주간 정리 완료")
        except Exception as e:
            logger.error(f"❌ 주간 정리 중 오류: {e}")

    def market_data_update(self, force: bool = False):
        """시장 데이터 업데이트"""
        try:
            logger.info("📈 시장 데이터 업데이트 시작")
            # 여기에 시장 데이터 업데이트 로직 추가
            
            if not force and not self._should_run_manual_task("market_data_update", 12):
                logger.info("⏭️ 시장 데이터 업데이트 스킵 (12시간 이내 실행됨)")
                return
            
            db_service = self._db_service
            db_service.load_data_from_db("")

            
            
            self._update_manual_run_time("market_data_update")
            
            logger.info("✅ 시장 데이터 업데이트 완료")
        except Exception as e:
            logger.error(f"❌ 시장 데이터 업데이트 중 오류: {e}")

    def setup_schedules(self):
        """스케줄 작업 설정"""
        try:
            # 미국 동부 시간대 설정
            eastern_tz = pytz.timezone('US/Eastern')
            
            # 1. 일일 캐시 새로고침: 월-금, 오후 4시 30분 (미국 주식시장 종료 30분 후)
            self.scheduler.add_job(
                self.daily_cache_refresh,
                CronTrigger(
                    day_of_week='mon-fri',  # 월요일-금요일
                    hour=16,                # 오후 4시
                    minute=30,              # 30분
                    timezone=eastern_tz     # 미국 동부 시간
                ),
                id='daily_cache_refresh',
                replace_existing=True
            )
            
            # 2. 주간 정리: 일요일 새벽 2시
            self.scheduler.add_job(
                self.weekly_cleanup,
                CronTrigger(
                    day_of_week='sun',      # 일요일
                    hour=2,                 # 새벽 2시
                    minute=0,
                    timezone=eastern_tz
                ),
                id='weekly_cleanup',
                replace_existing=True
            )
            
            # 3. 시장 데이터 업데이트: 월-금, 오전 8시 (시장 오픈 전)
            self.scheduler.add_job(
                self.market_data_update,
                CronTrigger(
                    day_of_week='mon-fri',  # 월요일-금요일
                    hour=8,                 # 오전 8시
                    minute=0,
                    timezone=eastern_tz
                ),
                id='market_data_update',
                replace_existing=True
            )
            
            logger.info("📅 스케줄 작업들이 설정되었습니다:")
            logger.info("  - 일일 캐시 새로고침: 월-금 EST 16:30")
            logger.info("  - 주간 정리: 일요일 EST 02:00")
            logger.info("  - 시장 데이터 업데이트: 월-금 EST 08:00")
            
        except Exception as e:
            logger.error(f"❌ 스케줄 설정 실패: {e}")

    def start(self):
        """스케줄러 시작"""
        try:
            self.setup_schedules()
            self.scheduler.start()
            logger.info("🚀 백그라운드 스케줄러 시작됨")
            
            # 앱 종료 시 스케줄러도 종료
            atexit.register(lambda: self.scheduler.shutdown())
            
        except Exception as e:
            logger.error(f"❌ 스케줄러 시작 실패: {e}")

    def stop(self):
        """스케줄러 중지"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("⏹️ 스케줄러 중지됨")

    def get_status(self):
        """스케줄러 상태 조회"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
        
        return {
            "running": self.scheduler.running,
            "jobs": jobs,
            "manual_runs": {
                task: time.isoformat() if time else None 
                for task, time in self._manual_run_times.items()
            }
        }
    
    def force_run_task(self, task_name: str):
        """작업 강제 실행"""
        if task_name == "daily_cache_refresh":
            self.daily_cache_refresh(force=True)
        elif task_name == "weekly_cleanup":
            self.weekly_cleanup(force=True)
        elif task_name == "market_data_update":
            self.market_data_update(force=True)
        else:
            raise ValueError(f"알 수 없는 작업: {task_name}")

    @property
    def is_running(self) -> bool:
        """스케줄러 실행 상태"""
        return self.scheduler.running
