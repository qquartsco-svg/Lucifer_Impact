# LuciferEngine 🌑
### 완전 독립형 혜성 궤도·충돌 탐색 시스템

CookiieBrain `L0_solar._06_lucifer_impact`에서 출발한 독립 상용화 엔진.
혜성·소행성의 **궤도 전파 → 충돌 탐지 → 피해 추정** 전 파이프라인을 단일 패키지로 제공.

---

## 아키텍처

```
lucifer_engine/
├── orbit/              케플러 궤도 역학 + N-체 수치 전파
│   ├── kepler.py         6요소 ↔ 상태벡터, 케플러 방정식
│   ├── propagator.py     RK4 N-체 적분기 (행성 섭동 포함)
│   └── __init__.py
│
├── detection/          충돌 탐지 파이프라인
│   ├── moid.py           MOID (최소 궤도 교차 거리)
│   ├── bplane.py         B-plane 충돌 기하학
│   ├── probability.py    Monte Carlo + Torino / Palermo Scale
│   └── __init__.py
│
├── effects/            충돌 물리 모델
│   ├── energy.py         충돌 에너지 + 환경 델타
│   ├── crater.py         Pi-group 크레이터 스케일링 (Holsapple 1993)
│   ├── tsunami.py        해양 충돌 쓰나미 (Ward & Asphaug 2000)
│   └── __init__.py
│
├── io/                 데이터 인터페이스
│   ├── sbdb.py           JPL SBDB JSON + MPC 혜성 궤도요소 파싱
│   └── __init__.py
│
├── engine.py           LuciferEngine 오케스트레이터
├── cli.py              커맨드라인 인터페이스
└── __init__.py
```

---

## 빠른 시작

### Python API

```python
from lucifer_engine import LuciferEngine

# 내장 혜성 (Halley, Chicxulub, Lucifer)
engine = LuciferEngine.from_builtin("Chicxulub")
report = engine.full_analysis(years_ahead=0, is_ocean=False)
print(report.summary())
```

```python
# 직접 궤도요소 입력
engine = LuciferEngine.from_elements(
    name="2029 XB1",
    a=2.5, e=0.75, i_deg=15.0,
    raan_deg=42.0, argp_deg=130.0, M0_deg=0.0,
    diameter_km=3.0, rho_gcm3=0.6,
)
report = engine.full_analysis(years_ahead=50, n_mc=20_000)
print(report.summary())
```

### CLI

```bash
# 설치
pip install -e ".[all]"

# 내장 혜성 분석
lucifer-analyze --comet Halley --years 80

# 해양 충돌 (쓰나미 포함)
lucifer-analyze --comet Lucifer --ocean --years 0 --v 25 --theta 60

# JSON 출력
lucifer-analyze --comet Chicxulub --json > chicxulub_report.json

# 직접 궤도요소 입력
lucifer-analyze --elements --name "2030 AB1" --a 3.0 --e 0.8 --i 20 \
                --raan 50 --argp 100 --M0 0 --D 2.5 --years 30
```

---

## 물리 모델 개요

| 모듈 | 방정식 | 기반 문헌 |
|------|--------|----------|
| `orbit/kepler.py` | M = E − e·sin E (Newton-Raphson) | Gauss, 1809 |
| `orbit/propagator.py` | r̈ = −GM/r³·r + Σ행성섭동 + a_ng | Marsden (1973) |
| `detection/moid.py` | min‖r₁(ν₁) − r₂(ν₂)‖ (L-BFGS-B) | Gronchi (2005) |
| `detection/bplane.py` | B = h×Ŝ / v∞, r_cap = R⊕√(1+2GM/Rv²) | Öpik (1951) |
| `detection/probability.py` | PS = log₁₀(p / f_b·T) | Chesley & Chodas (2002) |
| `effects/energy.py` | E = ½mv², E_eff = E·sin²θ | — |
| `effects/crater.py` | π_V = K₁(π₂+K₂π₃)^α · π₄^β | Holsapple (1993) |
| `effects/tsunami.py` | H₀ = 0.021·(E/ρ_w·g·r₀³)^(1/3)·r₀ | Ward & Asphaug (2000) |

---

## 위험 척도

### Torino Scale (0–10)
- **0**: 충돌 가능성 없음
- **1–4**: 주의 감시 필요
- **5–7**: 심각한 지역/광역 위협
- **8–10**: 전지구적 재앙

### Palermo Scale
- **< −2**: 무시 가능 (배경 충돌 수준 이하)
- **−2 ~ 0**: 정상 감시 대상
- **> 0**: 즉각 대응 필요

---

## 설치

```bash
cd ENGINE_HUB/00_PLANET_LAYER/Lucifer_Engine
pip install -e .                # 기본 (numpy, scipy)
pip install -e ".[network]"     # JPL 실시간 API 포함
pip install -e ".[all]"         # 전체
```

---

## CookiieBrain 연결

```
CookiieBrain/
└── L0_solar/
    └── _06_lucifer_impact/    ← 루시퍼 서사 트리거 (노아 홍수 이후)
        ├── impact_estimator.py   (간략 계산기)
        └── __init__.py           (LuciferEngine 있으면 이 엔진 우선 사용)

ENGINE_HUB/
└── 00_PLANET_LAYER/
    └── Lucifer_Engine/           ← 이 패키지 (완전 독립)
```

`L0_solar._06_lucifer_impact.__init__.py`는 `lucifer_engine`이 설치되어 있으면 자동으로 이 엔진을 사용:

```python
try:
    from lucifer_engine import LuciferEngine  # 이 패키지
except ImportError:
    from .impact_estimator import estimate_impact  # 폴백
```

---

## 로드맵

- [ ] JPL SBDB 실시간 API 연동 (`requests` 기반)
- [ ] 궤도 시각화 (`matplotlib` 3D 궤도 플롯)
- [ ] `RK45` 적응 스텝 적분기
- [ ] 가상 소행성 임팩터 라이브러리
- [ ] 대기권 진입·파편화 모델 (Chyba 1993)
