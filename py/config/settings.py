"""애플리케이션 설정"""

import os
import logging
from typing import Dict


class Settings:
    """애플리케이션 설정 클래스"""
    
    def __init__(self):
        # 데이터베이스 설정
        self.db_config = {
            'db_name': os.getenv('DB_NAME', 'fint'),
            'db_id': os.getenv('DB_ID', 'esc-sangyoon'),
            'db_pwd': os.getenv('DB_PWD', 'teamesc'),
            'db_host': os.getenv('DB_HOST', 'localhost')
        }
        
        # 서버 설정
        self.host = os.getenv('HOST', '0.0.0.0')
        self.port = int(os.getenv('PORT', 8000))
        self.reload = os.getenv('RELOAD', 'true').lower() == 'true'
        
        # 로깅 설정
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('LOG_FILE', 'app.log')
        
        # CORS 설정
        self.cors_origins = os.getenv('CORS_ORIGINS', '*').split(',')
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_file, encoding='utf-8')
            ]
        )
        return logging.getLogger(__name__)


# 전역 설정 인스턴스
settings = Settings()
