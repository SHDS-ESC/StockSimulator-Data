"""DTO (Data Transfer Objects) 모델 정의"""

from typing import Optional, List
from datetime import date
from pydantic import BaseModel, Field


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


class StockPredictionResponse(BaseModel):
    ticker: str
    base_date: date
    last_price: float
    return_predictions: List[float]
    price_predictions: List[float]
    prediction_dates: List[date]
    train_data_count: int
    feature_count: int
    investment_analysis: InvestmentAnalysis
    chart_full: Optional[str] = Field(None, description="전체 데이터 차트 (base64)")
    chart_30d: Optional[str] = Field(None, description="최근 30일 차트 (base64)")


class HealthResponse(BaseModel):
    status: str
    database: str
    scheduler: str
    scheduled_jobs: int


class TickersResponse(BaseModel):
    tickers: List[str]
    count: int


class CacheInfoResponse(BaseModel):
    cached_tickers: List[str]
    cache_count: int


class MessageResponse(BaseModel):
    message: str


class SchedulerStatusResponse(BaseModel):
    running: bool
    jobs: List[dict]
