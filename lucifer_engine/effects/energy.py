"""effects.energy — 충돌 에너지 및 환경 델타 계산

기존 L0_solar._06_lucifer_impact.impact_estimator 고도화.

물리 모델:
  E_total = ½ · m · v²            (운동 에너지)
  m = ρ · (4/3)π(D/2)³           (구형 혜성 질량)
  E_eff = E_total · sin²(θ)       (입사각 보정)

환경 델타:
  해수면 상승:  증발된 해수 → 수증기 → 강수 순환
  기압 변화:   충격파 최대 과압
  극-적도 온도차: 에어로졸 부하
  대기 수증기: 혜성 워터 캐노피 (Noah 모델 연장)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from math import pi, sin, cos, radians, log10, exp
from typing import Optional


@dataclass
class ImpactParams:
    """충돌체 파라미터.

    Parameters
    ----------
    D_km       : 직경 [km]
    rho_gcm3   : 밀도 [g/cm³]  (혜성: ~0.5, 암석: ~3.0, 철: ~8.0)
    v_kms      : 충돌 속도 [km/s]  (지구 평균: ~20 km/s)
    theta_deg  : 입사각 [°]  (수직=90°, 수평=0°)
    h_km       : 대기권 진입 고도 [km]  (기본 80 km)
    lat_deg    : 충돌 위도 [°]  (-90 ~ +90)
    lon_deg    : 충돌 경도 [°]  (-180 ~ +180)
    composition: 'ice', 'rock', 'iron', 'mixed'
    """
    D_km:        float
    rho_gcm3:    float = 0.6
    v_kms:       float = 20.0
    theta_deg:   float = 45.0
    h_km:        float = 80.0
    lat_deg:     float = 0.0
    lon_deg:     float = 0.0
    composition: str   = "ice"


@dataclass
class ImpactResult:
    """충돌 영향 계산 결과."""
    # 에너지
    E_total_J:        float   # 전체 운동 에너지 [J]
    E_total_MT:       float   # [메가톤 TNT]
    E_eff_J:          float   # 유효 에너지 (입사각 보정) [J]
    E_eff_MT:         float   # [메가톤 TNT]
    mass_kg:          float   # 질량 [kg]

    # 대기·기후 델타
    delta_H2O_canopy: float   # 대기 수증기 증가 [kg]
    delta_pressure_atm: float # 충격파 최대 과압 [atm]  (충돌 지점)
    delta_sea_level_m:  float # 해수면 상승 기여 [m]
    delta_pole_eq_K:    float # 극-적도 온도차 변화 [K]  (에어로졸)
    ejecta_mass_kg:     float # 분출물 총 질량 [kg]
    shock_strength:     str   # 충격 강도 레이블

    # 구조적
    crater_d_km:        float = 0.0  # (energy.py는 간략 추정, crater.py에서 정밀 계산)
    fireball_r_km:      float = 0.0  # 화구 반경 [km]
    blast_r_km:         float = 0.0  # 충격파 건물 파괴 반경 (1 psi) [km]


_MT_TO_J = 4.184e15   # 1 메가톤 TNT → J
_J_TO_MT = 1.0 / _MT_TO_J


def estimate_impact(params: ImpactParams) -> ImpactResult:
    """충돌 에너지 + 환경 델타 전체 계산.

    Parameters
    ----------
    params : ImpactParams

    Returns
    -------
    ImpactResult
    """
    # ── 기본 파라미터 변환 ──────────────────────────────
    R_m = (params.D_km * 1e3) / 2.0           # 반경 [m]
    rho_kgm3 = params.rho_gcm3 * 1e3          # [kg/m³]
    v_ms     = params.v_kms * 1e3              # [m/s]
    theta_r  = radians(params.theta_deg)

    # ── 질량 ────────────────────────────────────────────
    mass_kg  = rho_kgm3 * (4.0/3.0) * pi * R_m**3

    # ── 에너지 ──────────────────────────────────────────
    E_total_J  = 0.5 * mass_kg * v_ms**2
    E_eff_J    = E_total_J * sin(theta_r)**2   # 수직 성분만 지면에 전달
    E_total_MT = E_total_J * _J_TO_MT
    E_eff_MT   = E_eff_J   * _J_TO_MT

    # ── 충격파 최대 과압 (지표 충돌 기준) ────────────────
    # 간략 Holsapple 공식: ΔP ≈ C · (E_eff/r³)^(1/3) at r=1 km
    # 참조 스케일링 지수
    C_shock = 1.1e6   # 경험 상수 [Pa / (J/m³)^(1/3)]
    r_ref_m = 1000.0  # 1 km
    delta_P_Pa  = C_shock * (E_eff_J / r_ref_m**3) ** (1.0/3.0)
    delta_P_atm = delta_P_Pa / 101325.0

    # ── 수증기 캐노피 (해양 충돌) ────────────────────────
    # 충돌 에너지가 해수 증발에 쓰이는 분율 f_vap ≈ 0.01 ~ 0.1
    if params.composition in ("ice", "mixed"):
        f_vap = 0.05
    else:
        f_vap = 0.02
    L_vap      = 2.257e6          # 물 기화 잠열 [J/kg]
    delta_H2O  = (f_vap * E_eff_J) / L_vap   # [kg]

    # ── 해수면 상승 (증발 → 재강수 가정, 전지구 분포) ────
    A_ocean_m2 = 3.61e14          # 지구 해양 면적 [m²]
    rho_water  = 1025.0           # [kg/m³]
    delta_sl_m = (delta_H2O / rho_water) / A_ocean_m2

    # ── 극-적도 온도차 (에어로졸 부하) ───────────────────
    # 에어로졸 질량 ~ 1e-4 × E_eff [MT]  (경험 추정)
    aerosol_mt = 1e-4 * E_eff_MT
    # 온도 변화 ΔT ≈ -3·log10(aerosol+1) [K]
    delta_T_K  = -3.0 * log10(aerosol_mt + 1.0)

    # ── 분출물 ───────────────────────────────────────────
    # Croft (1985): m_ejecta ≈ 0.1 × (E/g·D_crater³)^0.7 … 간략화
    g_ms2       = 9.81
    D_cr_m_est  = 0.0087 * (E_eff_J / (rho_kgm3 * g_ms2)) ** (1.0/3.24)
    ejecta_mass = 0.1 * rho_kgm3 * (D_cr_m_est / 2.0)**3

    # ── 화구 반경 ────────────────────────────────────────
    # 간략: R_fireball ≈ 0.05 · E_eff_MT^(1/3)  [km]
    R_fireball_km = 0.05 * E_eff_MT ** (1.0/3.0)

    # ── 충격파 건물 파괴 반경 (1 psi ≈ 6900 Pa) ────────
    # ΔP·r³ = const → r ∝ E^(1/3)
    # 1 psi 파괴: R ≈ 0.28 · E_eff_MT^(1/3)  [km]
    R_blast_km = 0.28 * E_eff_MT ** (1.0/3.0)

    # ── 충격 강도 레이블 ─────────────────────────────────
    if E_eff_MT < 1:
        shock = "대기권 내 폭발 (소규모)"
    elif E_eff_MT < 1e3:
        shock = "지역 대재앙 (도시 규모)"
    elif E_eff_MT < 1e6:
        shock = "광역 대재앙 (대륙 규모)"
    elif E_eff_MT < 1e8:
        shock = "전지구 위협 (문명 위협)"
    else:
        shock = "대멸종급 (공룡 소행성 이상)"

    # 간략 크레이터 직경 [km]
    D_cr_km = D_cr_m_est / 1000.0

    return ImpactResult(
        E_total_J=E_total_J,  E_total_MT=E_total_MT,
        E_eff_J=E_eff_J,      E_eff_MT=E_eff_MT,
        mass_kg=mass_kg,
        delta_H2O_canopy=delta_H2O,
        delta_pressure_atm=delta_P_atm,
        delta_sea_level_m=delta_sl_m,
        delta_pole_eq_K=delta_T_K,
        ejecta_mass_kg=ejecta_mass,
        shock_strength=shock,
        crater_d_km=D_cr_km,
        fireball_r_km=R_fireball_km,
        blast_r_km=R_blast_km,
    )


def impact_from_dict(d: dict) -> ImpactResult:
    """딕셔너리 입력으로 충돌 계산."""
    return estimate_impact(ImpactParams(**d))


__all__ = [
    "ImpactParams", "ImpactResult", "estimate_impact", "impact_from_dict",
    "_MT_TO_J", "_J_TO_MT",
]
