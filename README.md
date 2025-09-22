# StockSimulator Data

주식 예측 머신러닝 API 서버  
노트북에서 검증된 StockPredictor 모델을 FastAPI로 서비스화

## 주요 기능

- 🤖 **머신러닝 기반 주식 예측**: RandomForest 모델과 기술분석 지표 활용
- 📊 **실시간 데이터베이스 연결**: MySQL에서 S&P500 실시간 데이터 로드
- 🔧 **기술분석 지표**: RSI, MACD, 볼린저밴드, 스토캐스틱 등 20+ 지표
- 🎯 **정확한 예측**: 1~30일간 주식 가격 및 수익률 예측
- 🌐 **RESTful API**: 간편한 HTTP 요청으로 예측 결과 조회

## 설치 및 실행

### 1. 가상환경 설정
```bash
# 가상환경 생성 (처음에만)
python -m venv dataEnv

# 가상환경 활성화
dataEnv/Scripts/activate.ps1   # Windows PowerShell
# 또는
dataEnv/Scripts/activate.bat   # Windows CMD

# 필요 모듈 설치
pip install -r requirements.txt
```

### 2. 데이터베이스 설정
환경변수 또는 기본값 사용:
```bash
# 환경변수 설정 (선택사항)
set DB_HOST=localhost
set DB_NAME=fint
set DB_ID=esc-sangyoon
set DB_PWD=teamesc
```

### 3. API 서버 실행
```bash
# 개발 모드 (자동 재시작)
uvicorn py.main:app --host 0.0.0.0 --port 8000 --reload

# 또는 직접 실행
python -m py.main
```

서버 실행 후 다음 URL들을 확인할 수 있습니다:
- API 문서: http://localhost:8000/docs
- 대체 문서: http://localhost:8000/redoc
- 헬스체크: http://localhost:8000/health

## API 사용법

### 1. 사용 가능한 티커 조회
```bash
curl http://localhost:8000/tickers
```

### 2. 주식 예측 요청
```bash
# 기본 요청 (AAPL, 300일 학습, 5일 예측)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL"}'

# 상세 설정 요청 (Linux/Mac/PowerShell)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "NVDA",
    "train_days": 500,
    "predict_steps": 10,
    "today": "2024-09-17"
  }'

# Windows CMD에서 실행 (한 줄로)
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"ticker\": \"NVDA\", \"train_days\": 500, \"predict_steps\": 10, \"today\": \"2024-09-17\"}"
```

### 3. 응답 예시
```json
{
  "ticker": "AAPL",
  "base_date": "2024-09-17",
  "last_price": 231.78,
  "train_data_count": 300,
  "feature_count": 25,
  "predicted": [
    {
      "day": 1,
      "date": "2024-09-18",
      "return_rate": 0.85,
      "price": 233.75
    },
    ...
  ]
}
```

## 파일 구조

### 🆕 새로운 구조 (현재 사용)
- `py/main.py`: FastAPI 메인 애플리케이션 (라우트 관리)
- `py/controllers/`: API 컨트롤러들
  - `stock_controller.py`: 주식 예측 API
  - `cache_controller.py`: 캐시 관리 API
  - `health_controller.py`: 헬스 체크 API
- `py/services/`: 비즈니스 로직 서비스들
  - `stock_service.py`: 주식 예측 메인 서비스
  - `database_service.py`: 데이터베이스 서비스
  - `prediction_service.py`: 머신러닝 모델 서비스
  - `analysis_service.py`: 투자 분석 서비스
  - `chart_service.py`: 차트 생성 서비스
  - `scheduler_service.py`: 스케줄러 서비스
- `py/models/`: 데이터 전송 객체 (DTO)
- `ipynb/stock_prediction_clean.ipynb`: 원본 노트북 (모델 검증용)
- `requirements.txt`: Python 의존성 목록

## 모델 정보

- **알고리즘**: RandomForest Regressor
- **피처**: 25개 기술분석 지표 + 가격/거래량 데이터
- **예측 대상**: 일일 수익률 → 가격 변환
- **검증 방법**: 시계열 교차검증 (TimeSeriesSplit)

## 개발 정보

```bash
# requirements.txt 갱신
pip freeze > requirements.txt

# 린터 실행
flake8 *.py

# 타입 체크
mypy *.py
```
