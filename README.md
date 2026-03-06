# LuciferEngine 🌑
### 완전 독립형 혜성 궤도·충돌 탐색 시스템 — 배포 계약 문서 v1.0.0

CookiieBrain `L0_solar._06_lucifer_impact` 서사를 물리적으로 구현하기 위해 설계된 독립 상용화 엔진.
**궤도 전파 → 충돌 탐지 → 피해 추정** 전 파이프라인을 단일 패키지로 제공.

> 이 문서는 "아키텍처 소개"가 아닌 **계약 문서**다.  
> 입력 단위, 좌표계, 확률 모델 가정, 리포트 스키마까지 명시함으로써  
> 결과의 재현·서명·검증이 가능하도록 설계되었다.

---

## 아키텍처

```
lucifer_engine/
├── orbit/              케플러 궤도 역학 + N-체 수치 전파
│   ├── kepler.py         6요소 ↔ 상태벡터, 케플러 방정식 (Newton-Raphson)
│   ├── propagator.py     RK4 N-체 적분기 (행성 섭동 + 혜성 비중력 가속)
│   └── __init__.py
│
├── detection/          충돌 탐지 파이프라인
│   ├── moid.py           MOID — 격자 탐색 + L-BFGS-B 정밀화
│   ├── bplane.py         B-plane 충돌 기하학 + 포획 반지름
│   ├── probability.py    Monte Carlo + Torino Scale + Palermo Scale
│   └── __init__.py
│
├── effects/            충돌 물리 모델
│   ├── energy.py         충돌 에너지 + 환경 델타 (해수면, 기압, 극-적도 ΔT)
│   ├── crater.py         Holsapple (1993) Pi-group 크레이터 스케일링
│   ├── tsunami.py        Ward & Asphaug (2000) 해양 쓰나미 전파
│   └── __init__.py
│
├── io/                 데이터 인터페이스
│   ├── sbdb.py           JPL SBDB JSON + MPC 혜성 궤도요소 파싱
│   └── __init__.py
│
├── engine.py           LuciferEngine 오케스트레이터 + FullReport
├── cli.py              커맨드라인 인터페이스
└── __init__.py
```

---

## 빠른 시작

```python
from lucifer_engine import LuciferEngine

engine = LuciferEngine.from_builtin("Lucifer")
report = engine.full_analysis(
    years_ahead=5,
    n_mc=10_000,
    mc_seed=42,          # 재현 키
    is_ocean=True,
    v_kms=25.0,
    theta_deg=45.0,
    water_depth_km=4.0,
    coast_dist_km=1000.0,
)
print(report.summary())
# report.to_dict() → JSON 직렬화 (재현 계약 포함)
```

```bash
# CLI
lucifer-analyze --comet Lucifer --ocean --years 5 --v 25 --theta 45 --json
```

---

## 1. Elements Contract — 궤도요소 계약

> 이 계약이 지켜지지 않으면 전파·탐지·확률 결과 모두 무의미하다.

| 파라미터 | 단위 | 설명 |
|---------|------|------|
| `a` | **AU** | 반장축 (타원: a > 0, 쌍곡선: a = q/(e-1)) |
| `e` | 무차원 | 이심률 (타원: 0 ≤ e < 1, 쌍곡선: e > 1) |
| `i` | **rad** | 궤도경사각 (내부 저장) |
| `raan` | **rad** | 승교점 적경 Ω (내부 저장) |
| `argp` | **rad** | 근점인수 ω (내부 저장) |
| `M0` | **rad** | `epoch_jd` 기준 평균근점각 M (내부 저장) |
| `epoch_jd` | **JD (TDB)** | default = 2451545.0 (J2000.0) |

**① 기준계:** 태양 중심 (heliocentric), 좌표 원점 = 태양 질량 중심  
**② 각도 단위 계약:** `OrbitalElements` 내부는 전부 **radian**.  
`from_elements()` API는 `i_deg`, `raan_deg`, `argp_deg`, `M0_deg`로 **degree 입력 → 내부에서 rad 자동 변환**.  
`OrbitalElements`를 직접 생성할 때는 반드시 radian 입력.  
**③ 좌표계:** ICRF / 적도 관성계 (equatorial J2000)  
**④ 시간 척도:** TDB (Barycentric Dynamical Time)  
**⑤ 태양 GM:** `k² = 0.01720209895² AU³/day²` (Gauss 상수, IAU 1976)  
**⑥ 질량:** 태양 = 1 M☉ (행성 섭동은 절대 GM 사용, `PLANET_GM` 딕셔너리)

```python
# 리포트에서 확인
report.orbit.frame      # → "ICRF/equatorial-J2000"
report.orbit.time_scale # → "TDB (Barycentric Dynamical Time)"
report.orbit.gm_sun     # → "0.01720209895² AU³/day² (Gauss k²)"

# API 사용 예: degree 입력 (내부에서 rad 변환)
engine = LuciferEngine.from_elements(
    "2029 XB1", a=2.5, e=0.75,
    i_deg=15.0, raan_deg=42.0, argp_deg=130.0, M0_deg=0.0,
)
```

---

## 2. Propagation Model — 전파 모델 계약

### 케플러 전파 (`mode="kepler"`)

- 행성 섭동 **없음**, 혜성 비중력 가속 **없음**
- 정확도: 단기(< 10년) 혜성 궤도에서 ≲ 1 AU 오차
- 속도: 매우 빠름 (수치 적분 없음)

### N-체 전파 (`mode="nbody"`, RK4 고정 스텝)

기본 포함 섭동체: **Jupiter, Saturn, Earth, Mars** (변경 가능)

```python
from lucifer_engine import PropagatorConfig
cfg = PropagatorConfig(
    dt_day=1.0,
    planets=["Jupiter", "Saturn", "Earth", "Mars", "Uranus", "Neptune"],
    include_nongrav=True,
)
engine = LuciferEngine.from_builtin("Halley", propagator=cfg)
```

| 항목 | 상태 | 비고 |
|------|------|------|
| 행성 위치 | 간략 케플러 (VSOP87 서브셋) | JPL DE 미적용, 단기 정밀도 ~0.01 AU |
| 태양-달 분리 | **없음** | 지구-달 합산 GM 사용 |
| 상대론 보정 | **없음** | — |
| 태양 복사압 (SRP) | **없음** | 소형 천체 장기 전파 시 오차 원인 |
| 비중력 가속 | **있음** (선택) | Marsden (1973), A1/A2/A3 파라미터 |
| 적분기 | RK4 고정 스텝 | RK45 적응 스텝 미구현 (로드맵) |

**정확도 한계 고지:** JPL Horizons 대비 장기(> 50년) N-체 전파 오차는 수 AU 수준이 될 수 있다. 궤도 결정 목적으로는 JPL 경로 연동 권장.

---

## 3. Detection Contract — 탐지 모델 계약

### MOID (최소 궤도 교차 거리)

```
알고리즘: 180×180 격자 탐색 → L-BFGS-B 정밀화
```

**⚠ 중요 한계:** MOID는 다중 최솟값 함수다. 현재 구현은 **멀티스타트 미적용**.  
단일 최적화 해가 전역 최솟값임을 보장하지 않는다.  
→ MOID ≈ 0 (< 0.01 AU) 구간은 멀티스타트 또는 분석적 보완 권장.

```python
report.detection.moid_method
# → "grid-180×180 + L-BFGS-B  ※다중 최솟값 주의: 멀티스타트 미적용"
```

**③ PHA (Potentially Hazardous) 판정 계약:**  
소행성 기준: H < 22 mag **AND** MOID < 0.05 AU (IAU Working Group 정의).  
혜성은 절대 등급 기준이 불안정하므로 본 엔진은 **MOID < 0.05 AU** 조건만 사용.  
`report.detection.pha_threshold_au = 0.05` (변경 불가, 계약값).  
**지구 반지름:** 6.371×10³ km = 4.258×10⁻⁵ AU (충돌 판정 기준선)

### B-plane 기하

- **중력집중(gravitational focusing):** Öpik (1951) 포획 반지름  
  `r_cap = R⊕ · √(1 + 2·GM⊕ / (R⊕ · v∞²))`  
- **B 벡터:** 쌍곡선 점근선 수직 평면 위 최근접 거리 벡터  
- **입력:** 지구 중심 기준 상대 위치/속도 (태양 중심 → 지구 중심 변환 필요)

---

## 4. Probability Contract — 확률 모델 계약

### Monte Carlo 샘플링 가정

| 항목 | 값 |
|------|---|
| 샘플링 방식 | 6요소 가우시안 섭동 (대각 공분산, 상관 없음) |
| 기본 σ | a: 1e-5 AU, e: 1e-6, i/Ω/ω: 1e-5~1e-6 rad, M0: 1e-4 rad |
| σ 의미 | 궤도 결정 정보가 없을 때 쓰는 **보수적 기본값** (실제 오차와 무관할 수 있음). 실측 궤도 공분산이 있으면 `OrbitalUncertainty.cov6x6` 직접 입력 권장. |
| 충돌 판정 | 각 샘플의 MOID < R⊕ (지구 반지름) |
| 시드 | 명시적 `mc_seed` 파라미터 (기본 42), report.config에 기록 |

```python
report.probability.mc_seed      # → 재현 키
report.probability.n_mc_samples # → 샘플 수
report.probability.n_impacts    # → 충돌 카운트
report.probability.sampling_model
# → "Gaussian 6-element perturbation (diagonal cov)"
```

**⚠ 한계:** 공분산 행렬이 대각이므로 요소 간 상관(예: a-e 결합 불확도)은 미반영.  
정밀 궤도 결정값(6×6 공분산)이 있으면 `OrbitalUncertainty.cov6x6`에 직접 입력 가능.

### Torino Scale

| 등급 | 의미 |
|------|------|
| 0 | 충돌 가능성 없음 / 무시 가능 |
| 1–2 | 주의 감시 |
| 3–4 | 가까운 접근, 확률 상승 |
| 5–7 | 심각한 지역/광역 위협 |
| 8–10 | 전지구적 재앙 (대멸종급) |

기준: Binzel (2000) IAU 결의안, 2005 개정  
계산: `p_impact × E_eff_MT` 곱과 에너지 임계값 조합

### Palermo Scale

```
PS = log₁₀(p / (f_b · T))
f_b = 10^(0.5616 - 0.8026·log₁₀E)  [배경 충돌률, Chesley & Chodas 2002]
T   = years_to_encounter
```

| PS | 의미 |
|----|------|
| < −2 | 무시 가능 (배경 이하) |
| −2 ~ 0 | 정상 감시 대상 |
| > 0 | 즉각 대응 필요 |

---

## 5. Effects Model — 충돌 영향 모델

### 에너지 + 환경 델타 (`effects/energy.py`)

```
E_total = ½mv²        m = ρ · (4/3)π(D/2)³
E_eff   = E_total · sin²θ   (수직 성분만 지면에 전달)
```

**⑤ 대기권 감속·파편화 미적용 고지:**  
`v_kms`는 **대기권 진입 전 속도가 아닌 지표 충돌 속도로 가정**한다.  
실제로 소형 천체(D ≲ 0.3 km)는 대기권 통과 중 대부분 소멸하므로 E_eff가 과대 추정된다.  
대기권 진입·파편화 모델(Chyba 1993)은 로드맵 4순위로 예정.  
D > 1 km 천체에서는 대기권 손실 < 10%이므로 현재 모델이 실용적으로 유효하다.

**환경 델타 입력 계약:**

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `D_km` | — | 충돌체 직경 [km] |
| `rho_gcm3` | 0.6 | 밀도 (혜성~0.5, 암석~3.0) |
| `v_kms` | 20.0 | 충돌 속도 [km/s] |
| `theta_deg` | 45.0 | 입사각 [°] (수직=90°) |
| `composition` | "ice" | ice/rock/iron/mixed |

### 크레이터 스케일링 (`effects/crater.py`)

Holsapple & Housen (2007) Pi-group 스케일링.  
지배 레짐 자동 판단 (중력 vs. 강도).

**추가 입력:**

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `rho_t` | 2.7e3 kg/m³ | 타깃 밀도 (화강암) |
| `Y_Pa` | 1e7 Pa | 타깃 강도 (연암 1e6, 경암 1e8) |
| `target_type` | "rock" | rock / sediment / ocean |
| `water_depth_km` | 0.0 | 해양 충돌 시 수심 [km] |

### 쓰나미 모델 (`effects/tsunami.py`)

Ward & Asphaug (2000) 에너지 스케일링 + 원형 파 감쇠.

**⑥ 쓰나미 r₀ 정의:**  
`r₀ = D_impactor / 2` [m] — 충돌체 반지름을 초기 파 생성 반경으로 사용.  
초기 파고: `H₀ = 0.021 · (E / ρ_w·g·r₀³)^(1/3) · r₀`  
전파 감쇠: `H(r) = H₀ · (r₀ / r)^1`  (2D 원형 파, 분산 미적용)

**필수 입력 계약:**

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `is_ocean` | False | 해양 충돌 여부 (False면 쓰나미 계산 생략) |
| `water_depth_km` | 4.0 | 충돌 해역 수심 [km] |
| `coast_dist_km` | 1000.0 | 대상 해안선까지 거리 [km] |

**모델 한계:** 대륙붕 효과, 굴절, 분산 미적용. 처오름은 지형 증폭 2배 근사. 정밀 해석에는 MOST/COMCOT 등 수치 파랑 모델 병행 권장.

---

## 6. Report Schema — 리포트 스키마 계약

모든 `FullReport`는 재현 가능하다. `report.config`에 버전·시드·입력 전체가 기록됨.

```python
report.orbit          # ReportOrbit — 입력 요소 + 좌표계/단위 계약
report.detection      # ReportDetection — MOID, 방법, PHA 판정
report.probability    # ReportProbability — MC 결과, seed, 배경률 출처
report.config         # ReportConfig — 재현용 전체 파라미터
report.impact         # ImpactResult — 에너지, 환경 델타
report.crater         # CraterResult — Holsapple 크레이터
report.tsunami        # TsunamiResult — 쓰나미 (ocean=True 시)
report.notes          # 경고/한계 메시지
```

```python
# 재현 예시
import json
d = report.to_dict()
print(d["config"]["mc_seed"])      # → 777
print(d["config"]["engine_version"]) # → "1.0.0"

# 같은 seed로 동일 결과 재현
report2 = engine.full_analysis(**{
    k: d["config"][k] for k in
    ["years_ahead", "n_mc_samples", "mc_seed",
     "is_ocean", "v_impact_kms", "theta_deg"]
})
# years_ahead → years_ahead, n_mc_samples → n_mc, v_impact_kms → v_kms
```

---

## 7. Lucifer Impact → Effects 플로우 구조

```
┌─────────────────────────────────────────────────────────────┐
│  LuciferEngine  full_analysis()  파이프라인                  │
│                                                             │
│  [io/]           혜성 레코드 로드                            │
│   CometRecord ──────────────────────────────────────────┐  │
│                                                          │  │
│  [orbit/]        궤도 전파                               ↓  │
│   OrbitalElements → kepler / nbody → (t, r, v) array    │  │
│                                          │               │  │
│  [detection/]    충돌 탐지               ↓               │  │
│   moid.py  ─── MOID < 0.05 AU?          │               │  │
│   bplane.py ── |B| < r_cap?             │               │  │
│   probability.py ← Monte Carlo (seed)   │               │  │
│         │                               │               │  │
│         ↓                               ↓               │  │
│   RiskAssessment                  (전파 궤도)            │  │
│   (p_impact, Torino, Palermo)           │               │  │
│         │                               │               │  │
│  [effects/]      충돌 영향 ←────────────┘               │  │
│   energy.py  → ImpactResult (E_eff_MT, ΔT, ΔSL, ΔP)   │  │
│   crater.py  → CraterResult (D_final, depth, regime)   │  │
│   tsunami.py → TsunamiResult (H0, H_coast, run_up)     │  │
│         │                                               │  │
│         └───────────────────────────────────────────────┘  │
│                         ↓                                   │
│                   FullReport                                │
│           (orbit + detection + probability +                │
│            impact + crater + tsunami + config)              │
└─────────────────────────────────────────────────────────────┘
```

**플로우 검증: 구조가 맞다.**

| 단계 | 입력 | 출력 | 독립성 |
|------|------|------|--------|
| `io` → `orbit` | `CometRecord.elements` | 상태벡터 배열 | 독립 실행 가능 |
| `orbit` → `detection` | `OrbitalElements` (MOID는 전파 불필요) | `RiskAssessment` | 독립 실행 가능 |
| `detection` → `effects` | 에너지 계산만 필요, 확률과 독립 | `ImpactResult/Crater/Tsunami` | **독립 실행 가능** |
| `effects` → `FullReport` | 모든 서브 결과 집약 | JSON-직렬화 가능 계약 | — |

핵심: **detection과 effects는 서로 독립적**이다. MOID/확률 없이 에너지만 계산 가능.  
이것이 CookiieBrain에서 `impact_only` 폴백이 성립하는 이유.

---

## 8. CookiieBrain 연결 — 정확한 폴백 계약  

```
CookiieBrain/
└── L0_solar/
    └── _06_lucifer_impact/
        ├── impact_estimator.py   ← effects-only 폴백 (에너지·환경 델타만)
        └── __init__.py           ← LuciferEngine 있으면 full pipeline 우선
```

```python
# _06_lucifer_impact/__init__.py
try:
    from lucifer_engine import LuciferEngine  # orbit + detection + effects
    LUCIFER_MODE = "full_pipeline"
except ImportError:
    from .impact_estimator import estimate_impact  # effects-only fallback
    LUCIFER_MODE = "impact_only"
```

**역할 분리 계약:**

| 모드 | 계산 범위 | 서사 트리거 |
|------|-----------|------------|
| `full_pipeline` | 궤도 → MOID → 확률 → 에너지/크레이터/쓰나미 | 충돌 확률 p > 임계값 도달 시 Noah에 주입 |
| `impact_only` | 에너지/크레이터/쓰나미만 (충돌은 기정사실) | 충격 강도 수치만 Noah에 주입 |

**Noah 홍수 연결 — 임계점 외부 충격 모델:**

LuciferEngine은 "노아 홍수의 원인"을 주장하지 않는다.  
이 엔진은 **임계점 시스템에 외부 충격을 주입하는 수치 입력기**다.

```
LuciferEngine.effects.energy.E_eff_MT
    → Noah 레이어의 compute_effective_instability()에 impulse_shock으로 주입
    → FirmamentLayer 붕괴 임계(0.85) 초과 여부를 동역학적으로 결정
    → 하드코딩 없음. 에너지가 작으면 임계를 못 넘고 서사가 다르게 전개됨
```

---

## 9. 서사적 의미 — LuciferEngine이 CookiieBrain에서 하는 일

LuciferEngine은 CookiieBrain `L0_solar` 서사의 6번째이자 마지막 장 `_06_lucifer_impact`를 물리적으로 구동한다.

```
L0_solar 서사 흐름:
  _01_beginnings         태초의 상태 초기화
  _02_creation_days      창조 6일 — 행성계 형성
  _03_eden_os_underworld 에덴 / 내부 역학
  _04_firmament_era      궁창 레이어 안정화
  _05_noah_flood         임계점 접근 (FirmamentLayer instability)
  _06_lucifer_impact  ←  LuciferEngine이 여기에 연결됨
       │
       └── E_eff_MT → Noah.compute_effective_instability() → impulse_shock
           → FirmamentLayer 붕괴 임계(0.85) 초과 여부를 수치로 결정
```

**서사와 물리의 분리 원칙:**  
LuciferEngine은 "루시퍼가 충돌했다"는 결론을 내리지 않는다.  
오직 **주어진 궤도요소에서 E_eff_MT와 확률을 계산해 넘겨줄 뿐**이다.  
에너지가 충분히 크면 Noah 레이어가 임계를 넘어 홍수 서사로 분기되고,  
에너지가 작으면 FirmamentLayer가 버텨 다른 서사가 전개된다.  
서사는 물리 결과의 함수다 — 하드코딩 없음.

---

## 10. 확장성 및 활용 가능성

### 독립 모듈로서의 확장

LuciferEngine은 CookiieBrain 없이도 독립적으로 동작한다. 가능한 확장:

| 활용 분야 | 방법 | 추가 필요 |
|----------|------|----------|
| **행성 방어 시뮬레이터** | MOID + Torino Scale 실시간 계산 | JPL SBDB 실시간 API |
| **교육용 충돌 시뮬레이터** | CLI `lucifer-analyze` 직접 사용 | — (현재 완성) |
| **게임/인터랙티브 서사 엔진** | `full_analysis()` → JSON → 이벤트 트리거 | 프론트엔드 연결 |
| **기후 모델 입력** | `delta_pole_eq_K`, `delta_H2O_canopy` 추출 | 기후 모델 커넥터 |
| **소행성 채굴 선택** | 궤도 전파 + MOID로 접근 가능 소행성 필터링 | `io/sbdb.py` API 연동 |
| **다중 천체 스윕** | `LuciferEngine.from_elements()` 루프 | — (현재 완성) |

### CookiieBrain 레이어 확장 포인트

```python
# 어떤 레이어든 LuciferEngine 결과를 구독할 수 있다
report = engine.full_analysis(...)

# L3_memory: 충격 이벤트를 MemoryWell로 기록
memory.record_event("lucifer_impact", report.to_dict())

# L4_analysis: 충격파 에너지를 통계역학 레이어로 분석
from L4_analysis.Layer_1 import analyze_transition
analyze_transition(impulse=report.impact.E_eff_MT)

# L1_dynamics: N-체 전파 결과를 Phase_B 다중 우물 퍼텐셜에 주입
from L1_dynamics.Phase_B import MultiWellPotential
pot.inject_perturbation(report.orbit.a_au, report.impact.E_eff_MT)
```

### 상용화 경로

```
현재 상태  → pip install lucifer-engine (로컬)
1단계      → PyPI 배포 + JPL 실시간 API
2단계      → REST API 서버 (FastAPI) + Swagger 문서
3단계      → 웹 대시보드 (궤도 시각화 + 위험 등급 실시간 표시)
4단계      → 행성 방어 기관 데이터 파이프라인 연동 (ESA NEO, NASA CNEOS)
```

---

## 12. 물리 모델 근거

| 모듈 | 방정식 | 기반 문헌 |
|------|--------|----------|
| `orbit/kepler.py` | M = E − e·sin E (Newton-Raphson) | Gauss, 1809 |
| `orbit/propagator.py` | r̈ = −GM/r³·r + Σ행성섭동 + a_ng | Marsden (1973) 비중력 |
| `detection/moid.py` | min‖r₁(ν₁) − r₂(ν₂)‖ | Gronchi (2005) 참조 |
| `detection/bplane.py` | B = h×Ŝ/v∞, r_cap = R⊕√(1+2GM/Rv²) | Öpik (1951) |
| `detection/probability.py` | PS = log₁₀(p / f_b·T) | Chesley & Chodas (2002) |
| `effects/energy.py` | E = ½mv², E_eff = E·sin²θ | — |
| `effects/crater.py` | π_V = K₁(π₂+K₂π₃)^α · π₄^β | Holsapple (1993, 2007) |
| `effects/tsunami.py` | H₀ = 0.021·(E/ρ_w·g·r₀³)^(1/3)·r₀ | Ward & Asphaug (2000) |

---

## 13. 설치

```bash
cd ENGINE_HUB/00_PLANET_LAYER/Lucifer_Engine
pip install -e .                # 기본 (numpy ≥ 1.24, scipy ≥ 1.10)
pip install -e ".[network]"     # JPL SBDB 실시간 API (requests)
pip install -e ".[all]"         # 전체 (network + matplotlib)
```

---

## 14. 로드맵

| 우선순위 | 항목 | 이유 |
|---------|------|------|
| 1 | MOID 멀티스타트 | 전역 최솟값 보장 (현재 근사) |
| 2 | RK45 적응 스텝 적분기 | 장기 전파 정밀도 향상 |
| 3 | JPL SBDB 실시간 API | 실제 혜성 데이터 자동 연동 |
| 4 | 대기권 진입·파편화 (Chyba 1993) | 소형 천체 공중 폭발 모델 |
| 5 | matplotlib 3D 궤도 시각화 | 분석 결과 시각화 |
| 6 | OrbitalUncertainty 전체 6×6 공분산 | 상관 불확도 반영 |
