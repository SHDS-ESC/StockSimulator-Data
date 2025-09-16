from typing import Optional, List
from datetime import date, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(title="StockSimulator API", version="0.1.0")

# CORS (로컬/노트북/브라우저 테스트용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    ticker: str = Field(..., description="예상할 티커 (예: NVDA)")
    train_days: int = Field(300, ge=30, le=2000, description="학습에 사용할 과거 일수")
    predict_steps: int = Field(5, ge=1, le=30, description="예측 일수")
    today: Optional[date] = Field(None, description="기준일 (기본: 오늘)")


class PredictPoint(BaseModel):
    date: date
    price: float


class PredictResponse(BaseModel):
    ticker: str
    baseline_close: float
    predicted: List[PredictPoint]


@app.get("/health")
def health_check():
    return {"status": "ok"}


def naive_predict_series(
    close_prices: List[float],
    start_date: date,
    predict_steps: int,
) -> List[PredictPoint]:
    """
    간단한 기준선 예측기 (프로덕션 전 노트북 모델 이식 전용 자리표시자):
    - 마지막 종가를 기준으로 선형 드리프트(최근 5일 평균 수익률)를 적용.
    - 실제 모델 준비가 되면 이 함수를 노트북에서 검증된 함수로 교체하세요.
    """
    if not close_prices:
        raise ValueError("close_prices is empty")

    last_close = float(close_prices[-1])
    # 최근 5일 수익률 평균 (데이터 부족 시 0으로 처리)
    rets = []
    for i in range(max(0, len(close_prices) - 5), len(close_prices) - 1):
        prev_p = close_prices[i]
        next_p = close_prices[i + 1]
        if prev_p:
            rets.append((next_p / prev_p) - 1.0)
    drift = sum(rets) / len(rets) if rets else 0.0

    preds: List[PredictPoint] = []
    cur_price = last_close
    for step in range(1, predict_steps + 1):
        cur_price = cur_price * (1.0 + drift)
        preds.append(PredictPoint(date=start_date + timedelta(days=step), price=round(cur_price, 4)))
    return preds


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    예측 엔드포인트 (노트북 코드 이식 전 기본 동작):
    - 실제로는 노트북(`ipynb/stock_prediction_clean.ipynb`)에서 검증한
      `StockPredictor` 로직을 별도 `.py` 모듈로 옮긴 뒤 import 하여 사용하세요.
    - 현재는 yfinance/DB 의존을 제거한 자리표시자 로직으로 응답합니다.
    """
    try:
        # today 기본값 처리
        base_day = req.today or date.today()

        # NOTE: 여기서 노트북 코드를 이식했다면 아래를 교체합니다.
        # - DB 또는 yfinance에서 (base_day - train_days) ~ base_day 데이터 로드
        # - 피처 준비 -> 모델 학습 -> n+1 ~ n+predict_steps 예측
        # 아래는 더미용 Close 배열입니다. 실제로는 DB/CSV에서 가져오세요.
        dummy_close = [100 + i * 0.1 for i in range(req.train_days)]

        preds = naive_predict_series(dummy_close, base_day, req.predict_steps)
        return PredictResponse(
            ticker=req.ticker,
            baseline_close=float(dummy_close[-1]),
            predicted=preds,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


