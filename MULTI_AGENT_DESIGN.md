# Multi-Agent System 설계 제안

## 🎯 제안하신 구조의 장점
- ✅ LLM을 활용한 투자 조언 생성
- ✅ 예측 결과를 자연어로 해석
- ✅ 사용자 친화적인 출력

## 🚀 개선된 Multi-Agent 아키텍처 제안

### Agent 구성

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                    │
│              (작업 조율 및 결과 통합)                      │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Prediction  │  │  Analysis   │  │  Advisory   │
│   Agent     │  │   Agent     │  │   Agent     │
│             │  │             │  │             │
│ - LightGBM  │  │ - LLM 기반  │  │ - LLM 기반  │
│ - RF        │  │   해석      │  │   조언 생성  │
│ - LSTM      │  │ - 메트릭    │  │ - 리스크    │
│             │  │   분석      │  │   평가      │
└─────────────┘  └─────────────┘  └─────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ Validation      │
              │ Agent           │
              │                 │
              │ - 결과 검증     │
              │ - 피드백 생성   │
              │ - 성능 평가     │
              └─────────────────┘
```

### 1. Prediction Agent (기존 시스템 확장)
**역할**: 주가 예측 수행
- **현재**: LightGBM 기반 예측
- **확장**: 
  - 여러 모델 Agent (LightGBM Agent, RandomForest Agent, LSTM Agent)
  - 각 Agent가 독립적으로 예측 수행
  - 앙상블 전략 Agent로 결과 통합

**출력**:
```json
{
  "agent_id": "prediction_lightgbm",
  "predictions": [...],
  "confidence": 0.85,
  "metrics": {...}
}
```

### 2. Analysis Agent (LLM 기반)
**역할**: 예측 결과 해석 및 분석
- **입력**: Prediction Agent들의 결과
- **LLM 활용**:
  - 예측 결과를 자연어로 해석
  - 기술적 지표 분석
  - 시장 상황 컨텍스트 분석
  - 트렌드 패턴 식별

**프롬프트 예시**:
```
당신은 주식 분석 전문가입니다. 다음 예측 결과를 분석해주세요:

예측 결과:
- 현재 가격: $150
- 예측 평균 가격: $155
- 예상 수익률: +3.3%
- 변동성: 2.1%
- RSI: 65
- MACD: 양수 전환

이 결과를 바탕으로:
1. 기술적 분석 관점에서 해석
2. 주요 리스크 요인 식별
3. 시장 상황과의 연관성 분석
```

**출력**:
```json
{
  "agent_id": "analysis_llm",
  "interpretation": "RSI가 65로 과매수 구간에 접근하고 있으며...",
  "risk_factors": ["변동성 증가 가능성", "시장 조정 우려"],
  "market_context": "최근 기술주 상승세와 연관...",
  "confidence": 0.78
}
```

### 3. Advisory Agent (LLM 기반)
**역할**: 투자 조언 생성
- **입력**: 
  - Prediction Agent 결과
  - Analysis Agent 해석
  - 사용자 프로필 (리스크 성향, 투자 목표 등)
  - 포트폴리오 현황

**프롬프트 엔지니어링**:
```
당신은 투자 자문 전문가입니다. 다음 정보를 바탕으로 투자 조언을 제공해주세요:

[예측 결과]
[분석 결과]
[사용자 프로필]
- 리스크 성향: 보수적/공격적
- 투자 목표: 단기/장기
- 현재 포트폴리오: [...]

다음 형식으로 조언을 제공해주세요:
1. 투자 의사결정 (BUY/SELL/HOLD)
2. 근거 설명
3. 권장 투자 비중
4. 리스크 관리 방안
5. 대안 투자 옵션
```

**출력**:
```json
{
  "agent_id": "advisory_llm",
  "recommendation": {
    "action": "BUY",
    "confidence": "MEDIUM",
    "rationale": "기술적 지표상 상승 가능성이 있으나...",
    "suggested_allocation": 0.15,
    "risk_management": [
      "손절매 가격: $145",
      "목표가: $160",
      "최대 투자 비중: 20%"
    ],
    "alternatives": ["비슷한 섹터의 다른 종목 고려"]
  }
}
```

### 4. Validation Agent
**역할**: 결과 검증 및 피드백 생성
- **입력**: 모든 Agent의 결과
- **기능**:
  - 예측 결과 일관성 검증
  - Agent 간 결과 비교
  - 성능 메트릭 평가
  - 피드백 생성

**출력**:
```json
{
  "agent_id": "validation",
  "consistency_score": 0.82,
  "agent_agreement": {
    "prediction_agents": 0.75,
    "analysis_advisory": 0.88
  },
  "feedback": {
    "strengths": ["일관된 상승 예측", "낮은 변동성"],
    "concerns": ["과도한 낙관적 전망 가능성"],
    "suggestions": ["추가 검증 데이터 필요"]
  }
}
```

### 5. Orchestrator Agent
**역할**: 전체 워크플로우 조율
- Agent 간 통신 관리
- 작업 순서 결정
- 결과 통합
- 에러 처리 및 재시도

## 🔄 피드백 루프 설계

### 1. 실시간 피드백
```
예측 실행 → 결과 검증 → 성능 평가 → 모델 파라미터 조정
```

### 2. 장기 피드백
```
과거 예측 저장 → 실제 결과와 비교 → Agent 성능 평가 → 
Prompt 개선 / 모델 재학습 → 다음 예측에 반영
```

### 3. 사용자 피드백
```
사용자 평가 (좋음/나쁨) → 피드백 수집 → 
Advisory Agent 프롬프트 개선 → 더 나은 조언 생성
```

## 📊 Context/Prompt Engineering 전략

### 1. 동적 프롬프트 생성
```python
def build_analysis_prompt(prediction_result, market_context, user_profile):
    base_prompt = """
    당신은 주식 분석 전문가입니다.
    
    [예측 결과]
    {prediction_summary}
    
    [시장 상황]
    {market_context}
    
    [사용자 프로필]
    {user_profile}
    
    다음을 분석해주세요:
    1. 기술적 분석
    2. 펀더멘털 분석
    3. 리스크 평가
    """
    
    return base_prompt.format(
        prediction_summary=format_prediction(prediction_result),
        market_context=get_market_context(),
        user_profile=user_profile
    )
```

### 2. Few-shot Learning
- 과거 성공적인 분석 사례를 예시로 포함
- 유사한 상황의 과거 분석 결과 활용

### 3. Chain-of-Thought
- 단계별 추론 과정을 요청
- 중간 단계 결과를 다음 Agent에 전달

## 🎯 Graph RAG 통합 가능성

### 주식 관계 그래프 구축
```
주식 노드
  ├─ 섹터 관계
  ├─ 공급망 관계
  ├─ 경쟁사 관계
  └─ 상관관계 (가격 움직임)
```

### 활용 방법
1. **관련 주식 정보 검색**
   - 분석 대상 주식과 연관된 주식 정보를 그래프에서 검색
   - 섹터/산업 트렌드 반영

2. **컨텍스트 확장**
   - 단일 주식 분석 → 섹터 전체 분석
   - 경쟁사 성과 비교

3. **포트폴리오 최적화**
   - 주식 간 상관관계를 그래프로 시각화
   - 분산 투자 전략 수립

## 💡 구현 우선순위

### Phase 1: 기본 Multi-Agent 구조
1. ✅ Prediction Agent (기존 시스템)
2. ➕ Analysis Agent (LLM 기반 해석)
3. ➕ Advisory Agent (LLM 기반 조언)
4. ➕ Orchestrator Agent

### Phase 2: 피드백 루프
1. ➕ Validation Agent
2. ➕ 성능 추적 시스템
3. ➕ 프롬프트 개선 자동화

### Phase 3: 고급 기능
1. ➕ Graph RAG 통합
2. ➕ Text-to-SQL (자연어 쿼리)
3. ➕ 다중 모델 앙상블

## 🔧 기술 스택 제안

### LLM 통합
- **OpenAI GPT-4 / Claude**: 고품질 분석 및 조언
- **Llama 2/3**: 오픈소스 대안
- **LangChain**: Agent 프레임워크
- **LlamaIndex**: RAG 구현

### Agent 프레임워크
- **LangGraph**: Agent 워크플로우 관리
- **AutoGen**: Multi-Agent 대화 시스템

### 평가 시스템
- **LangSmith**: LLM 애플리케이션 평가
- **Custom Metrics**: Agent별 성능 추적

## 📝 예상 효과

### JD 요구사항 충족
1. ✅ **Multi Agent System**: 여러 Agent 협업 구조
2. ✅ **피드백 루프**: Validation Agent를 통한 지속적 개선
3. ✅ **Context/Prompt Engineering**: 동적 프롬프트 생성
4. ✅ **LLM 활용**: 분석 및 조언 생성
5. 🔄 **Graph RAG**: 확장 가능 (Phase 3)
6. 🔄 **Text-to-SQL**: 확장 가능 (Phase 3)

### 사용자 경험 개선
- 단순 숫자 예측 → 이해하기 쉬운 자연어 설명
- 투자 조언의 근거 명확화
- 개인화된 투자 전략 제안

### 시스템 개선
- Agent별 성능 추적 가능
- 프롬프트 개선을 통한 점진적 성능 향상
- 다양한 모델/전략 비교 가능
