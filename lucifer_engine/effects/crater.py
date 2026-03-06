"""effects.crater — Pi-group 크레이터 스케일링 법칙

Holsapple & Housen (2007) Pi-group 스케일링.
과도기 크레이터 → 최종 크레이터 직경 계산.

스케일링 무차원수:
  π₂ = g·a / v²          (중력 vs 관성)
  π₃ = Y / (ρ_t·v²)      (강도 vs 관성)
  π_V = m / (ρ_t · a³)   (질량)
  π_4 = ρ_i / ρ_t        (밀도비)

결과:
  D_transient: 과도기 크레이터 직경 [km]
  D_final:     최종 크레이터 직경 [km]  (중력 붕괴 보정)
  D_rim:       테두리 직경 [km]
  depth:       크레이터 깊이 [km]
  melt_mass:   용융 암석 질량 [kg]
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from math import pi, sin, radians, log10, sqrt, exp
from typing import Optional


@dataclass
class CraterParams:
    """크레이터 스케일링 입력."""
    D_km:        float          # 충돌체 직경 [km]
    rho_i:       float = 0.6e3  # 충돌체 밀도 [kg/m³]  (혜성 ~600)
    v_kms:       float = 20.0   # 충돌 속도 [km/s]
    theta_deg:   float = 45.0   # 입사각 [°]
    rho_t:       float = 2.7e3  # 타깃 밀도 [kg/m³]  (화강암 ~2700)
    g_ms2:       float = 9.81   # 중력 가속도 [m/s²]
    Y_Pa:        float = 1e7    # 타깃 강도 [Pa]  (연암: 1e6, 경암: 1e8)
    target_type: str   = "rock" # 'rock', 'sediment', 'ocean'
    water_depth_km: float = 0.0 # 해양 충돌 시 수심 [km]


@dataclass
class CraterResult:
    """크레이터 스케일링 결과."""
    D_transient_km:  float   # 과도기 크레이터 직경 [km]
    D_final_km:      float   # 최종 크레이터 직경 [km]
    D_rim_km:        float   # 테두리 외경 [km]
    depth_km:        float   # 크레이터 깊이 [km]
    melt_volume_km3: float   # 용융 암석 부피 [km³]
    melt_mass_kg:    float   # 용융 암석 질량 [kg]
    regime:          str     # 'gravity' or 'strength'
    ejecta_thick_1km: float  # 크레이터 반경 1배 지점 분출물 두께 [m]
    notes:           str = ""


def compute_crater(p: CraterParams) -> CraterResult:
    """Holsapple Pi-group 스케일링으로 크레이터 크기 계산.

    References
    ----------
    Holsapple (1993) AREPS 21:333-373
    Collins et al. (2005) Meteor. Planet. Sci. 40(6):817-840
    """
    # 단위 변환
    a_m   = (p.D_km * 1e3) / 2.0    # 충돌체 반경 [m]
    v_ms  = p.v_kms * 1e3           # 속도 [m/s]
    theta = radians(p.theta_deg)

    # 입사각 보정 속도 (수직 성분)
    v_vert = v_ms * sin(theta)

    # Pi 무차원수
    pi2 = p.g_ms2 * a_m / v_vert**2          # 중력 Pi
    pi3 = p.Y_Pa / (p.rho_t * v_vert**2)      # 강도 Pi
    pi4 = p.rho_i / p.rho_t                   # 밀도비

    # Holsapple (1993) 계수 (암석 타깃 기준)
    # π_V = K₁ · (π₂ + K₂·π₃)^(-3ν/(2+μ)) · π₄^((2+μ-6ν)/(3(2+μ)))
    if p.target_type == "sediment":
        mu, nu, K1, K2 = 0.55, 0.4, 0.32, 0.52
    else:  # rock
        mu, nu, K1, K2 = 0.41, 0.4, 0.20, 0.52

    # 지배 레짐 판단
    pi_ratio = pi2 / (K2 * pi3) if pi3 > 0 else 1e10
    if pi_ratio > 1.0:
        regime = "gravity"
    else:
        regime = "strength"

    # 과도기 크레이터 부피 (Pi 스케일링)
    # V_tr = (m/ρ_t) · π_V
    m_i    = p.rho_i * (4.0/3.0) * pi * a_m**3
    exp_v  = -3.0 * nu / (2.0 + mu)
    pi_sum = pi2 + K2 * pi3
    pi_V   = K1 * pi_sum**exp_v * pi4**((2.0+mu-6.0*nu) / (3.0*(2.0+mu)))
    V_tr   = (m_i / p.rho_t) * pi_V

    # 과도기 크레이터 직경 (원통 근사: V ≈ 0.5 · D_tr²/4 · depth_tr)
    # D_tr ≈ (8/pi · V_tr)^(1/3) 경험식
    D_tr_m = (8.0 * V_tr / pi) ** (1.0/3.0)
    D_tr_km = D_tr_m / 1e3

    # 해양 충돌 보정: 수심이 충돌체 직경보다 크면 크레이터 크기 감소
    if p.target_type == "ocean" and p.water_depth_km > 0:
        wf = min(1.0, p.D_km / (2.0 * p.water_depth_km))
        D_tr_km *= wf

    # 최종 크레이터 직경 (중력 붕괴)
    # D_final / D_tr = 1.17 · (D_tr / D_Q)^0.13   [D_Q: 전이 직경 ≈ 3.2 km 지구]
    D_Q = 3.2   # km (지구 기준 전이 직경)
    if D_tr_km > D_Q:
        D_final_km = 1.17 * D_tr_km * (D_tr_km / D_Q) ** 0.13
    else:
        D_final_km = D_tr_km

    # 테두리 직경: D_rim ≈ 1.25 · D_final
    D_rim_km = 1.25 * D_final_km

    # 크레이터 깊이 [km]
    # 단순 비 depth/D_final ≈ 0.2 (복잡 크레이터) ~ 0.3 (단순 크레이터)
    if D_final_km > 4.0:
        depth_km = 0.2 * D_final_km
    else:
        depth_km = 0.3 * D_final_km

    # 용융 암석 부피 (Grieve & Cintala 1992)
    # V_melt ≈ C · E_eff / H_melt
    E_eff_J   = 0.5 * m_i * v_vert**2
    H_melt_J  = p.rho_t * 3.5e6   # 용융 엔탈피 [J/m³] (화강암 ~3.5 MJ/m³ 근사)
    V_melt_m3 = 0.002 * E_eff_J / H_melt_J
    V_melt_km3 = V_melt_m3 / 1e9
    melt_mass  = V_melt_m3 * p.rho_t

    # 1-크레이터반경 지점 분출물 두께
    # T_ej(r) ≈ 0.14 · D_tr^4 / r^3  (Housen & Holsapple 2011)
    r_1x_m = D_final_km * 1e3
    T_ej_m = 0.14 * D_tr_m**4 / r_1x_m**3 if r_1x_m > 0 else 0.0

    notes = f"레짐={regime}, π₂={pi2:.3e}, π₃={pi3:.3e}"

    return CraterResult(
        D_transient_km=D_tr_km,
        D_final_km=D_final_km,
        D_rim_km=D_rim_km,
        depth_km=depth_km,
        melt_volume_km3=V_melt_km3,
        melt_mass_kg=melt_mass,
        regime=regime,
        ejecta_thick_1km=T_ej_m,
        notes=notes,
    )


__all__ = ["CraterParams", "CraterResult", "compute_crater"]
