"""effects.tsunami — 해양 충돌 쓰나미 모델

해양에 충돌하는 천체가 생성하는 쓰나미 파고·전파 계산.

모델:
  - Mader (1998) 분산 파고 감쇠 모델
  - Ward & Asphaug (2000) 에너지 기반 파고 스케일링
  - 천해/심해 전파 구분

물리:
  H₀(r₀)  초기 파고   [m]  at r₀ = 충돌 지점 반경
  H(r)     전파 파고   [m]  at 거리 r [km]
  c        파속       [m/s] = √(g·h)  (천해파 근사)
  T        파주기      [s]  ≈ 2π·r₀ / c₀
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from math import pi, sqrt, exp, log10
from typing import List, Tuple


@dataclass
class TsunamiParams:
    """쓰나미 계산 입력."""
    E_eff_MT:      float          # 유효 충돌 에너지 [메가톤 TNT]
    D_impactor_km: float          # 충돌체 직경 [km]
    water_depth_km: float = 4.0   # 충돌 해역 평균 수심 [km]
    target_coast_km: float = 1000.0  # 대상 해안선까지 거리 [km]


@dataclass
class TsunamiResult:
    """쓰나미 계산 결과."""
    H0_m:         float   # 초기 파고 (충돌 지점) [m]
    H_coast_m:    float   # 해안 도달 파고 [m]
    wave_period_s: float  # 주기 [s]
    wave_speed_kms: float # 파속 [km/s]
    run_up_m:     float   # 최대 처오름 높이 [m]  (지형 증폭 2배 근사)
    t_arrival_min: float  # 해안 도달 시간 [분]
    inundation_category: str   # 피해 등급


def compute_tsunami(p: TsunamiParams) -> TsunamiResult:
    """Ward & Asphaug (2000) + Mader (1998) 쓰나미 추정.

    References
    ----------
    Ward & Asphaug (2000) GRL 27(24)
    Mader (1998) Sci Tsunami Hazards 16(1)
    """
    g = 9.81
    _MT_TO_J = 4.184e15

    E_J = p.E_eff_MT * _MT_TO_J
    h   = p.water_depth_km * 1e3       # [m]
    r0  = (p.D_impactor_km * 1e3) / 2.0  # 충돌 반경 [m]

    # 파속 (천해파 근사)
    c_ms  = sqrt(g * h)                 # [m/s]
    c_kms = c_ms / 1e3

    # 초기 파고 (Ward & Asphaug 스케일링)
    # H₀ ≈ 0.021 · (E/ρ_w·g·r₀³)^(1/3) · r₀   [m]
    rho_w = 1025.0   # [kg/m³]
    pi_E  = E_J / (rho_w * g * r0**3)
    H0_m  = 0.021 * pi_E**(1.0/3.0) * r0

    # 전파 감쇠: H(r) = H₀ · (r₀/r)^n
    # 심해 분산 감쇠: n ≈ 1.0 for cylindrical wave
    # 얕은 해역 접근 시 증폭 효과는 run_up에서 처리
    r_coast_m = p.target_coast_km * 1e3
    n_decay   = 1.0   # 2D 원형 파 감쇠 지수
    H_coast   = H0_m * (r0 / max(r_coast_m, r0)) ** n_decay

    # 파주기
    T_s = 2.0 * pi * r0 / c_ms

    # 해안 도달 시간
    t_arrive_s   = r_coast_m / c_ms
    t_arrive_min = t_arrive_s / 60.0

    # 처오름 높이 (run-up): 지형 증폭 약 2배 (초기 근사)
    # 더 정밀한 값은 그린 함수 모델 필요
    run_up = 2.0 * H_coast

    # 피해 등급
    if run_up < 1.0:
        category = "경미 (1m 미만, 항구 피해)"
    elif run_up < 5.0:
        category = "중등 (1-5m, 해안가 침수)"
    elif run_up < 20.0:
        category = "심각 (5-20m, 광역 해안 파괴)"
    elif run_up < 100.0:
        category = "대재앙 (20-100m, 해안 도시 壊滅)"
    else:
        category = "전지구 위협 (100m+, 내륙 침수)"

    return TsunamiResult(
        H0_m=H0_m,
        H_coast_m=H_coast,
        wave_period_s=T_s,
        wave_speed_kms=c_kms,
        run_up_m=run_up,
        t_arrival_min=t_arrive_min,
        inundation_category=category,
    )


def tsunami_propagation_profile(
    H0_m: float,
    r0_km: float,
    r_max_km: float,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """거리별 파고 프로필 계산.

    Returns
    -------
    distances_km : ndarray
    heights_m    : ndarray
    """
    r0_m = r0_km * 1e3
    r_arr = np.linspace(r0_km, r_max_km, n_points) * 1e3
    H_arr = H0_m * (r0_m / np.maximum(r_arr, r0_m))
    return r_arr / 1e3, H_arr


__all__ = [
    "TsunamiParams", "TsunamiResult",
    "compute_tsunami", "tsunami_propagation_profile",
]
