# StockSimulator-Data 프로젝트 아키텍처

## 📐 시스템 아키텍처 다이어그램

```mermaid
graph TB
    subgraph "Client Layer"
        Client[클라이언트<br/>Web/Mobile/API]
    end
    
    subgraph "API Layer - FastAPI"
        Main[main.py<br/>FastAPI App]
        Middleware[Middleware<br/>CORS, Logging, Exception]
        
        subgraph "Controllers"
            StockCtrl[stock_controller.py<br/>주식 예측 API]
            HealthCtrl[health_controller.py<br/>헬스 체크]
            SchedulerCtrl[scheduler_controller.py<br/>스케줄러 관리]
        end
    end
    
    subgraph "Service Layer"
        StockSvc[stock_service.py<br/>주식 예측 메인 서비스]
        PredictSvc[prediction_service.py<br/>ML 모델 서비스<br/>LightGBM]
        AnalysisSvc[analysis_service.py<br/>투자 분석 서비스]
        ChartSvc[chart_service.py<br/>차트 생성 서비스]
        DBSvc[database_service.py<br/>데이터베이스 서비스]
        SchedulerSvc[scheduler_service.py<br/>백그라운드 스케줄러]
    end
    
    subgraph "Model Layer"
        DTO[models/dto.py<br/>Pydantic DTOs]
    end
    
    subgraph "Data Layer"
        MySQL[(MySQL Database<br/>S&P500 주식 데이터)]
        Cache[메모리 캐시<br/>티커별 데이터]
    end
    
    subgraph "External Libraries"
        LightGBM[LightGBM<br/>ML 모델]
        TALib[TA-Lib<br/>기술 분석 지표]
        QuantStats[QuantStats<br/>포트폴리오 분석]
        Matplotlib[Matplotlib<br/>차트 생성]
    end
    
    subgraph "Config"
        Settings[config/settings.py<br/>설정 관리]
    end
    
    %% Client to API
    Client -->|HTTP Request| Main
    Main --> Middleware
    Middleware --> StockCtrl
    Middleware --> HealthCtrl
    Middleware --> SchedulerCtrl
    
    %% Controllers to Services
    StockCtrl --> StockSvc
    HealthCtrl --> DBSvc
    HealthCtrl --> SchedulerSvc
    SchedulerCtrl --> SchedulerSvc
    
    %% Service Dependencies
    StockSvc --> PredictSvc
    StockSvc --> AnalysisSvc
    StockSvc --> ChartSvc
    StockSvc --> DBSvc
    SchedulerSvc --> DBSvc
    
    %% Service to External
    PredictSvc --> LightGBM
    PredictSvc --> TALib
    AnalysisSvc --> QuantStats
    ChartSvc --> Matplotlib
    
    %% Data Flow
    DBSvc --> MySQL
    DBSvc --> Cache
    DBSvc -.->|캐시 조회| Cache
    
    %% DTO Usage
    StockCtrl --> DTO
    StockSvc --> DTO
    
    %% Config
    Main --> Settings
    DBSvc --> Settings
    SchedulerSvc --> Settings
    
    %% Styling
    classDef controller fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef service fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef data fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef external fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class StockCtrl,HealthCtrl,SchedulerCtrl controller
    class StockSvc,PredictSvc,AnalysisSvc,ChartSvc,DBSvc,SchedulerSvc service
    class MySQL,Cache,DTO data
    class LightGBM,TALib,QuantStats,Matplotlib external
```

## 🔄 데이터 흐름도

```mermaid
sequenceDiagram
    participant Client
    participant Controller as Stock Controller
    participant StockSvc as Stock Service
    participant PredictSvc as Prediction Service
    participant DBSvc as Database Service
    participant AnalysisSvc as Analysis Service
    participant ChartSvc as Chart Service
    participant DB as MySQL Database
    
    Client->>Controller: POST /predict<br/>{ticker, train_days, predict_steps}
    Controller->>StockSvc: predict_stock(request)
    
    StockSvc->>DBSvc: get_stock_data(ticker)
    DBSvc->>DB: SELECT * FROM report WHERE stock_id=?
    DB-->>DBSvc: 주식 데이터
    DBSvc-->>StockSvc: DataFrame (OHLCV)
    
    StockSvc->>StockSvc: 날짜 범위 계산<br/>피처 준비
    
    StockSvc->>PredictSvc: prepare_returns(data)
    PredictSvc->>PredictSvc: 기술 지표 계산<br/>(RSI, MACD, etc.)
    PredictSvc->>PredictSvc: 피처 엔지니어링<br/>dropna()
    PredictSvc-->>StockSvc: 준비된 데이터
    
    StockSvc->>StockSvc: 슬라이딩 윈도우 학습
    
    loop 각 윈도우
        StockSvc->>PredictSvc: train(X_train, y_train)
        PredictSvc->>PredictSvc: LightGBM 학습
        StockSvc->>PredictSvc: predict_next_returns(steps)
        PredictSvc-->>StockSvc: 예측 결과
    end
    
    StockSvc->>AnalysisSvc: generate_recommendation()
    AnalysisSvc->>AnalysisSvc: 투자 지표 계산<br/>리스크 분석
    AnalysisSvc-->>StockSvc: 투자 추천
    
    StockSvc->>ChartSvc: create_prediction_charts()
    ChartSvc->>ChartSvc: Matplotlib 차트 생성
    ChartSvc-->>StockSvc: Base64 이미지
    
    StockSvc-->>Controller: 예측 결과 + 분석 + 차트
    Controller-->>Client: JSON Response
```

## 🏗️ 클래스 관계도

```mermaid
classDiagram
    class FastAPI {
        +app: FastAPI
        +initialize_services()
        +get_db_service()
        +get_scheduler_service()
    }
    
    class StockController {
        +predict_stock(request)
        +get_available_tickers()
        +get_stock_data(ticker)
        +portfolio_analysis(request)
    }
    
    class StockService {
        -db_service: DatabaseService
        -chart_service: ChartService
        +predict_stock(request)
        +portfolio_analysis(request)
        +_find_nearest_date()
        +_ensure_date_in_index()
        +_sliding_window_predict()
    }
    
    class StockPredictor {
        -model: LightGBM
        -scaler: StandardScaler
        +prepare_returns(df)
        +train(X, y)
        +predict_next_returns(X, steps)
        +calculate_technical_indicators()
        +add_engineered_features()
    }
    
    class InvestmentAnalyzer {
        -ticker: str
        -current_price: float
        -price_predictions: List
        +calculate_metrics()
        +risk_analysis()
        +generate_recommendation()
    }
    
    class DatabaseService {
        -engine: SQLAlchemy Engine
        -stock_df: DataFrame
        -_stock_data_cache: dict
        +get_stock_data(ticker)
        +get_multi_stock_data(tickers)
        +clear_cache()
    }
    
    class ChartService {
        +create_prediction_charts()
        +_create_full_chart()
        +_create_brief_chart()
    }
    
    class SchedulerService {
        -scheduler: BackgroundScheduler
        +start()
        +stop()
        +daily_cache_refresh()
        +market_data_update()
    }
    
    class Settings {
        +db_config: dict
        +host: str
        +port: int
        +setup_logging()
    }
    
    FastAPI --> StockController
    StockController --> StockService
    StockService --> StockPredictor
    StockService --> InvestmentAnalyzer
    StockService --> DatabaseService
    StockService --> ChartService
    DatabaseService --> Settings
    SchedulerService --> DatabaseService
    FastAPI --> SchedulerService
```

## 📊 예측 프로세스 플로우차트

```mermaid
flowchart TD
    Start([예측 요청 시작]) --> LoadData[데이터 로드<br/>DatabaseService]
    LoadData --> CheckCache{캐시에<br/>있나?}
    CheckCache -->|있음| UseCache[캐시 사용]
    CheckCache -->|없음| LoadDB[DB에서 로드]
    LoadDB --> SaveCache[캐시 저장]
    UseCache --> FindDate[날짜 범위 계산<br/>_find_nearest_date]
    SaveCache --> FindDate
    
    FindDate --> PrepareFeatures[피처 준비<br/>prepare_returns]
    PrepareFeatures --> CalcIndicators[기술 지표 계산<br/>RSI, MACD, etc.]
    CalcIndicators --> FeatureEng[피처 엔지니어링<br/>이동평균, 볼린저밴드]
    FeatureEng --> DropNA[dropna]
    
    DropNA --> CheckDates{날짜 유효성<br/>확인}
    CheckDates -->|날짜 제거됨| AdjustDates[날짜 조정<br/>_ensure_date_in_index]
    CheckDates -->|정상| SlidingWindow[슬라이딩 윈도우<br/>예측 시작]
    AdjustDates --> SlidingWindow
    
    SlidingWindow --> WindowLoop{윈도우<br/>반복}
    WindowLoop --> TrainModel[모델 학습<br/>LightGBM]
    TrainModel --> Predict[재귀적 예측<br/>predict_next_returns]
    Predict --> Validate[성능 검증<br/>MAE, RMSE]
    Validate --> WindowLoop
    
    WindowLoop -->|완료| FinalPred[최종 예측<br/>마지막 윈도우]
    FinalPred --> Analysis[투자 분석<br/>InvestmentAnalyzer]
    Analysis --> CalcMetrics[지표 계산<br/>수익률, 리스크]
    CalcMetrics --> GenerateRec[추천 생성<br/>BUY/SELL/HOLD]
    
    GenerateRec --> CreateChart[차트 생성<br/>ChartService]
    CreateChart --> ReturnResult[결과 반환]
    ReturnResult --> End([종료])
```

## 🔌 API 엔드포인트 구조

```mermaid
graph LR
    subgraph "API Endpoints"
        A[POST /predict<br/>주식 예측]
        B[GET /tickers<br/>티커 목록]
        C[GET /data/{ticker}<br/>주식 데이터]
        D[POST /portfolio/analysis<br/>포트폴리오 분석]
        E[GET /health<br/>헬스 체크]
        F[GET /scheduler/status<br/>스케줄러 상태]
    end
    
    subgraph "Services"
        S1[StockService]
        S2[DatabaseService]
        S3[SchedulerService]
    end
    
    A --> S1
    B --> S2
    C --> S2
    D --> S1
    E --> S2
    E --> S3
    F --> S3
```

## 🗄️ 데이터베이스 스키마

```mermaid
erDiagram
    STOCK ||--o{ REPORT : has
    STOCK {
        int stock_id PK
        string ticker
        string company_name
    }
    REPORT {
        int report_id PK
        int stock_id FK
        date report_date
        float open
        float high
        float low
        float close
        float volume
    }
```

## 🔄 피드백 루프 (슬라이딩 윈도우 검증)

```mermaid
graph TB
    subgraph "슬라이딩 윈도우 검증 프로세스"
        Start([학습 데이터 시작]) --> Window1[윈도우 1<br/>학습/검증]
        Window1 --> Metrics1[성능 메트릭<br/>MAE, RMSE, 방향정확도]
        Metrics1 --> Window2[윈도우 2<br/>학습/검증]
        Window2 --> Metrics2[성능 메트릭]
        Metrics2 --> Window3[윈도우 3<br/>...]
        Window3 --> WindowN[윈도우 N<br/>최종]
        WindowN --> MetricsN[성능 메트릭]
        MetricsN --> Aggregate[메트릭 집계<br/>평균 계산]
        Aggregate --> FinalPred[최종 예측<br/>마지막 윈도우 결과]
        FinalPred --> End([완료])
    end
    
    style Window1 fill:#e3f2fd
    style Window2 fill:#e3f2fd
    style Window3 fill:#e3f2fd
    style WindowN fill:#c8e6c9
    style FinalPred fill:#fff9c4
```

## 📦 컴포넌트 의존성

```mermaid
graph TD
    subgraph "Core Components"
        Main[main.py]
        Config[config/settings.py]
    end
    
    subgraph "API Layer"
        Ctrl1[stock_controller]
        Ctrl2[health_controller]
        Ctrl3[scheduler_controller]
    end
    
    subgraph "Business Logic"
        Svc1[stock_service]
        Svc2[prediction_service]
        Svc3[analysis_service]
        Svc4[chart_service]
        Svc5[database_service]
        Svc6[scheduler_service]
    end
    
    subgraph "Data Models"
        DTO[models/dto.py]
    end
    
    Main --> Config
    Main --> Ctrl1
    Main --> Ctrl2
    Main --> Ctrl3
    
    Ctrl1 --> Svc1
    Ctrl1 --> DTO
    Ctrl2 --> Svc5
    Ctrl2 --> Svc6
    Ctrl3 --> Svc6
    
    Svc1 --> Svc2
    Svc1 --> Svc3
    Svc1 --> Svc4
    Svc1 --> Svc5
    Svc6 --> Svc5
    
    Svc1 --> DTO
```

---

## 📝 다이어그램 설명

### 1. 시스템 아키텍처 다이어그램
- 전체 시스템의 계층 구조와 컴포넌트 간 관계를 보여줍니다
- 클라이언트부터 데이터베이스까지의 전체 흐름을 시각화합니다

### 2. 데이터 흐름도
- 예측 요청이 들어와서 결과가 반환될 때까지의 시퀀스를 보여줍니다
- 각 서비스 간의 상호작용을 시간 순서대로 표현합니다

### 3. 클래스 관계도
- 주요 클래스와 그들의 메서드, 속성을 보여줍니다
- 클래스 간 의존성을 명확히 합니다

### 4. 예측 프로세스 플로우차트
- 예측 프로세스의 단계별 흐름을 보여줍니다
- 조건부 분기와 반복 프로세스를 포함합니다

### 5. API 엔드포인트 구조
- 모든 API 엔드포인트와 연결된 서비스를 보여줍니다

### 6. 데이터베이스 스키마
- 데이터베이스 테이블과 관계를 보여줍니다

### 7. 피드백 루프
- 슬라이딩 윈도우 검증 프로세스를 보여줍니다

### 8. 컴포넌트 의존성
- 컴포넌트 간 의존성 관계를 보여줍니다
