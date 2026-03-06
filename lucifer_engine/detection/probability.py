"""detection.probability — 충돌 확률 + Torino/Palermo Scale

Monte Carlo 궤도 불확도 전파 → 충돌 확률 계산.
표준 위험 척도:
  - Torino Scale  (0-10, 정수)
  - Palermo Scale (연속 실수, -∞ ~ +∞)

Reference:
  Chesley & Chodas 2002, AAS 02-176
  Milani et al. 2005, Icarus 173
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from math import log10, log, pi, exp, sqrt

from .moid import MOIDResult, _AU_TO_KM, _EARTH_RADIUS_AU
from .bplane import BPlaneResult, GM_EARTH_KM3, R_EARTH_KM


@dataclass
class RiskAssessment:
    """충돌 위험 평가 결과."""
    p_impact: float             # 충돌 확률 (0-1)
    energy_mt: float            # 충돌 에너지 [메가톤 TNT]
    torino_scale: int           # Torino Scale (0-10)
    palermo_scale: float        # Palermo Scale
    palermo_label: str          # 해석 문자열
    torino_description: str     # Torino Scale 설명
    years_to_impact: float      # 충돌까지 남은 시간 [년]
    n_mc_samples: int           # Monte Carlo 샘플 수
    n_impacts: int              # 충돌 카운트


# Torino Scale 정의 (0-10)
_TORINO_TABLE = [
    (0,  "충돌 가능성 없음 / 무시 가능"),
    (1,  "일상적 감시 필요, 충돌 가능성 극히 낮음"),
    (2,  "가까운 접근, 주의 요망 (0.01 이상 p_impact·E ≥ 1)"),
    (3,  "가까운 접근, 10% 이상 충돌 확률"),
    (4,  "가까운 접근, 높은 충돌 확률"),
    (5,  "상당히 큰 천체, 심각한 지역 피해 가능"),
    (6,  "큰 천체, 전지구 재앙 가능성"),
    (7,  "매우 큰 천체, 전지구 재앙 거의 확실"),
    (8,  "확실한 지역 대재앙 (소행성 충돌)"),
    (9,  "확실한 지역 대재앙 (매우 큰 소행성)"),
    (10, "확실한 전지구 대재앙"),
]

# 배경 충돌률 (기준: 메가톤 에너지 E, 연 단위)
# log10 N(>E) = a - b·log10(E)  형태
# 추정: a=0.5616, b=0.8026  (Chesley & Chodas 2002)
_BKG_A = 0.5616
_BKG_B = 0.8026


def background_rate(energy_mt: float) -> float:
    """배경 충돌 발생률 [events/year] for energy ≥ energy_mt [MT]."""
    if energy_mt <= 0:
        return 1.0
    return 10.0 ** (_BKG_A - _BKG_B * log10(energy_mt))


def palermo_scale(p_impact: float, energy_mt: float,
                  years_to_impact: float) -> float:
    """Palermo Scale PS = log10(p / (f_b · T)).

    f_b: 배경 충돌률, T: 충돌까지 시간.
    """
    if p_impact <= 0 or years_to_impact <= 0:
        return -99.0
    fb = background_rate(energy_mt)
    return log10(p_impact / (fb * years_to_impact))


def torino_scale(p_impact: float, energy_mt: float) -> int:
    """Torino Scale 0-10 계산.

    p × E 제품 및 에너지 임계값 기반.
    """
    if p_impact <= 0:
        return 0

    pE = p_impact * energy_mt  # [확률 × MT]

    # Torino Scale 규칙표 (근사)
    if energy_mt < 1e-3 or p_impact < 1e-8:
        return 0
    elif energy_mt < 1.0:          # < 1 MT: 대기권에서 소멸
        if p_impact < 0.01: return 0
        elif p_impact < 0.5: return 1
        else:                return 2
    elif energy_mt < 1e3:          # 1 ~ 1000 MT: 지역 피해
        if pE < 1e-4:    return 0
        elif pE < 1e-2:  return 1
        elif pE < 1e-1:  return 2
        elif pE < 1.0:   return 3
        elif p_impact < 0.5: return 4
        else:            return 5
    elif energy_mt < 1e6:          # 1000 MT ~ 1 GT: 광역 피해
        if pE < 1e-2:    return 1
        elif pE < 1.0:   return 4
        elif p_impact < 0.5: return 6
        else:            return 7
    else:                           # > 1 GT: 전지구 위협
        if pE < 1.0:     return 4
        elif p_impact < 0.5: return 8
        elif p_impact < 0.99: return 9
        else:            return 10


@dataclass
class OrbitalUncertainty:
    """궤도 공분산 (6×6 상태벡터 공분산 행렬 또는 간략 σ₆ 벡터)."""
    sigma_a:    float = 1e-5   # AU
    sigma_e:    float = 1e-6
    sigma_i:    float = 1e-6   # rad
    sigma_raan: float = 1e-5   # rad
    sigma_argp: float = 1e-5   # rad
    sigma_M0:   float = 1e-4   # rad
    cov6x6: Optional[np.ndarray] = None   # 전체 공분산 (있으면 우선 사용)


def monte_carlo_impact(
    el_mean,               # OrbitalElements 평균값
    uncertainty: OrbitalUncertainty,
    jd_encounter: float,   # 예상 접근 JD
    years_to_impact: float,
    kinetic_energy_mt: float,
    n_samples: int = 10_000,
    seed: int = 42,
) -> RiskAssessment:
    """Monte Carlo 궤도 불확도 전파로 충돌 확률 계산.

    각 샘플: 6요소 가우시안 섭동 → 지구 MOID 계산 → 충돌 판정.
    """
    from ..orbit.kepler import OrbitalElements, elements_to_state
    from .moid import moid_from_comet

    rng = np.random.default_rng(seed)
    sigmas = np.array([
        uncertainty.sigma_a, uncertainty.sigma_e, uncertainty.sigma_i,
        uncertainty.sigma_raan, uncertainty.sigma_argp, uncertainty.sigma_M0,
    ])

    n_impacts = 0
    for _ in range(n_samples):
        dv = rng.normal(0, sigmas)
        el_s = OrbitalElements(
            a    = max(el_mean.a    + dv[0], 0.01),
            e    = max(min(el_mean.e + dv[1], 0.9999), 0.0),
            i    = el_mean.i    + dv[2],
            raan = el_mean.raan + dv[3],
            argp = el_mean.argp + dv[4],
            M0   = el_mean.M0   + dv[5],
            epoch_jd=el_mean.epoch_jd,
        )
        moid = moid_from_comet(el_s)
        if moid.is_earth_crosser:
            n_impacts += 1

    p_impact = n_impacts / n_samples
    ts       = torino_scale(p_impact, kinetic_energy_mt)
    ps       = palermo_scale(p_impact, kinetic_energy_mt, years_to_impact)

    # Palermo 라벨
    if ps < -2:
        label = "무시 가능 (배경 수준 이하)"
    elif ps < 0:
        label = "배경 수준 (정상 감시)"
    elif ps < 2:
        label = "주의 요망 (배경 이상)"
    else:
        label = "위험 (즉각 대응 필요)"

    return RiskAssessment(
        p_impact=p_impact,
        energy_mt=kinetic_energy_mt,
        torino_scale=ts,
        palermo_scale=ps,
        palermo_label=label,
        torino_description=_TORINO_TABLE[min(ts, 10)][1],
        years_to_impact=years_to_impact,
        n_mc_samples=n_samples,
        n_impacts=n_impacts,
    )


def bplane_probability(bp: BPlaneResult,
                       sigma_b_km: float,
                       kinetic_energy_mt: float,
                       years_to_impact: float) -> RiskAssessment:
    """B-plane 불확도 타원에서 충돌 확률 계산.

    B-plane 내 오차 타원 면적과 포획 원 면적 비율로 p_impact 추정.

    p ≈ (π r_cap²) / (2π σ_B · σ_B_T)  (가우시안 근사)
    """
    r_cap = bp.r_capture_km
    if sigma_b_km <= 0:
        sigma_b_km = r_cap * 0.1

    # 2D 가우시안에서 원 안에 들어올 확률
    # p = 1 - exp(-r_cap² / (2σ²))
    p_impact = 1.0 - exp(-r_cap**2 / (2.0 * sigma_b_km**2))

    if bp.will_impact:
        p_impact = max(p_impact, 0.5)

    ts = torino_scale(p_impact, kinetic_energy_mt)
    ps = palermo_scale(p_impact, kinetic_energy_mt, years_to_impact)

    if ps < -2:
        label = "무시 가능 (배경 수준 이하)"
    elif ps < 0:
        label = "배경 수준 (정상 감시)"
    elif ps < 2:
        label = "주의 요망 (배경 이상)"
    else:
        label = "위험 (즉각 대응 필요)"

    return RiskAssessment(
        p_impact=p_impact,
        energy_mt=kinetic_energy_mt,
        torino_scale=ts,
        palermo_scale=ps,
        palermo_label=label,
        torino_description=_TORINO_TABLE[min(ts, 10)][1],
        years_to_impact=years_to_impact,
        n_mc_samples=0,
        n_impacts=0,
    )


__all__ = [
    "RiskAssessment", "OrbitalUncertainty",
    "monte_carlo_impact", "bplane_probability",
    "torino_scale", "palermo_scale", "background_rate",
    "_TORINO_TABLE",
]
