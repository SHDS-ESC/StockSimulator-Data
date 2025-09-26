"""DTO (Data Transfer Objects) 모델 정의"""

from typing import Optional, List
from datetime import date
from pydantic import BaseModel, Field
from pydantic import ConfigDict


def to_camel(s: str) -> str:
    """snake_case를 camelCase로 변환"""
    parts = s.split('_')
    return parts[0] + ''.join(p.title() for p in parts[1:])


class CamelModel(BaseModel):
    """camelCase alias를 지원하는 베이스 모델"""
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True  # camel/snake 모두 입력 허용
    )


class InvestmentMetrics(CamelModel):
    current_price: float
    predicted_avg_price: float
    predicted_max_price: float
    predicted_min_price: float
    expected_total_return: float
    expected_avg_daily_return: float
    predicted_volatility: float
    upside_probability: float


class RiskMetrics(CamelModel):
    historical_volatility_annualized: float
    predicted_volatility: float
    var_95: float
    max_expected_loss: float
    max_expected_gain: float
    estimated_sharpe_ratio: float


class InvestmentAnalysis(CamelModel):
    recommendation: str
    action: str  # BUY, SELL, HOLD
    confidence: str  # HIGH, MEDIUM, LOW
    score: int
    max_score: int
    min_score: int
    signals: List[str]
    metrics: InvestmentMetrics
    risk_metrics: RiskMetrics


class FeatureImportance(CamelModel):
    top_features: List[tuple[str, float]]
    total_features: int
    importance_sum: float
    max_importance: float
    min_importance: float

class StockPredictionResponse(CamelModel):
    ticker: str
    base_date: date
    last_price: float
    return_predictions: List[float]
    price_predictions: List[float]
    prediction_dates: List[date]
    train_data_count: int
    feature_count: int
    feature_importance: Optional[FeatureImportance] = Field(None, description="피처 중요도 정보")
    investment_analysis: InvestmentAnalysis
    chart_full: Optional[str] = Field(None, description="전체 데이터 차트 (base64)")
    chart_brief: Optional[str] = Field(None, description="요약(최근 50일) 차트 (base64)")


class HealthResponse(CamelModel):
    status: str
    database: str
    scheduler: str
    scheduled_jobs: int


class TickersResponse(CamelModel):
    tickers: List[str]
    count: int


class CacheInfoResponse(CamelModel):
    cached_tickers: List[str]
    cache_count: int


class MessageResponse(CamelModel):
    message: str


class SchedulerStatusResponse(CamelModel):
    running: bool
    jobs: List[dict]


class StockPredictionRequest(CamelModel):
    ticker: str = Field(..., description="주식 티커 심볼")
    train_days: int = Field(description="훈련 데이터 기간 (일)")
    predict_steps: int = Field(description="예측 기간 (일)")
    today: Optional[date] = Field(None, description="기준일 (기본: 오늘)")
    save_image: bool = Field(True, description="차트 이미지 생성 여부")
    batch_size: Optional[int] = Field(None, description="배치 크기 (일, 기본값: train_days)")
    step_size: Optional[int] = Field(None, description="슬라이딩 윈도우 스텝 크기 (일, 기본값: train_days)")
    
    # 모델 파라미터 (선택적)
    model_params: Optional[dict] = Field(None, description="LightGBM 모델 파라미터 (기본값 사용 시 생략)")


# ===== 포트폴리오 누적수익률 요청/응답 =====
class TimeValue(CamelModel):
    date: date
    value: float


class PortfolioSpec(CamelModel):
    id: str
    tickers: List[str]
    weights: List[float]


class PortfolioCumulativeReturnsRequest(CamelModel):
    start_date: date = Field(..., alias="startDate")
    end_date: date = Field(..., alias="endDate")
    portfolios: List[PortfolioSpec]
    base_value: float = Field(1.0, alias="baseValue", description="초기가치, 1.0=100과 같은 기준")
    rebalance: Optional[str] = Field(None, description="none|daily|monthly 등 리밸런싱 정책")


class PortfolioSeries(CamelModel):
    id: str
    series: List[TimeValue]


class PortfolioCumulativeReturnsResponse(CamelModel):
    series: List[PortfolioSeries]